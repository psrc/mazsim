### Import modules and run model steps
# import numpy as np
import pandas as pd
import copy
import sys

# urbansim imports
from urbansim.models import dcm
from urbansim.models import util
from urbansim_templates import modelmanager as mm
import orca

# local imports
import mazsim.data_loader
import mazsim.variables

# Calibration-related imports
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.numpy as np

import matplotlib.pyplot as plt


def get_calib_targets(target_type):
    var_dict = {
        'employment': {
            'lcm':'elcm',
            'str_replace':'',
            'sum_cols': ['2011', '2021'],
        },
        'households': {
            'lcm':'hlcm',
            'str_replace':'(recent_mover == 1) & ',
            'sum_cols': ['hhlds11', 'hhlds21'],
        },
        'residential_units': {
            'lcm':'rdplcm',
            'str_replace':'(year_built > 2010) & ',
            'sum_cols': ['residential_units'],
        },
    }

    # empty tract df
    idx = list(
        orca.get_table('blocks').to_frame('tract_id')['tract_id'].unique()
    )
    df = pd.DataFrame(index = idx)

    # get target data
    data = orca.get_table(f'calib_targets_{target_type}').local
    data['tract_id'] = data['tract_id'].astype(str).str.zfill(11)
    data = data.fillna(0)

    if target_type == 'employment':
        data.rename(columns={'agg_sector_id':'aggr_sector_id'}, inplace=True)
    if target_type == 'households':
        data.rename(columns={'qrt_income':'income_quartile', 'with_children':'children'}, inplace=True)
        data['age_of_head'] = np.where(data['60_or_older']==1,61,59)

    lcm = var_dict[target_type]['lcm']

    for segment in orca.get_injectable(f'{lcm}_step_names'):
        filter_ = mm.get_step(segment).chooser_filters.replace(var_dict[target_type]['str_replace'],'')
        data.loc[data.query(filter_).index, 'segment'] = segment.replace(lcm, '')

    sum_cols = var_dict[target_type]['sum_cols']

    data_gr = (
        data.groupby(['tract_id', 'segment'])[sum_cols]
        .sum().unstack().reindex(df.index).fillna(0)
    )

    for model_name in orca.get_injectable(f'{lcm}_step_names'):
        segment = model_name.replace(lcm, "")
        if len(sum_cols)==2:
            df[f'growth_{segment}'] = (
                data_gr[(sum_cols[1],segment)] - data_gr[(sum_cols[0],segment)]
            )
        if len(sum_cols)==1:
            df[f'growth_{segment}'] = data_gr[(sum_cols[0],segment)]
        df[f'growth_{segment}'] = np.clip(df[f'growth_{segment}'], 0, None)
        df[f'growth_perc_{segment}'] = df[f'growth_{segment}']/df[f'growth_{segment}'].sum()

    return {'calib_raw':data,'calib_clean':df}


def standardize_coefficients(model):
    expression = model.model_expression[:-4].split(' + ')
    standardized = [
        var if var.startswith('st_') else 'st_' + var for var in expression
    ]
    model.model_expression = util.str_model_expression(standardized, add_constant=False)
    model.fit()

def model_calibration(
        segment, agent, calib_targets, aggr_growth, max_iter=10000, tol=1e-13,
        step_size=0.001, std=False, county_calib=False
    ):

    print(f"\nCalibrating segment: {segment}\n")

    # get model segment id
    segment_id = segment.replace(f'{agent}lcm','')

    # re-fit model with standardized coefficients if desired
    m = copy.copy(mm.get_step(segment))

    if std:
        standardize_coefficients(m)

    # set id geog to tract
    idx_geog_col = 'tract_id'

    # get model expression and parameters
    expvar_names = m.model_expression[:-4].split(' + ')
    fitted_parameters = m.fitted_parameters

    # add county-level indicators if desired
    extra_calib_cols = []
    if county_calib:
        extra_calib_cols += [
            'county_id_is_08001',
            'county_id_is_08005',
            'county_id_is_08013',
            'county_id_is_08014',
            'county_id_is_08019',
            'county_id_is_08031',
            'county_id_is_08035',
            'county_id_is_08039',
            'county_id_is_08047',
            'county_id_is_08059',
            'county_id_is_08123'
        ]
        if std:
            extra_calib_cols = ['st_' + col for col in extra_calib_cols]

    calib_expvars = expvar_names + extra_calib_cols

    capacity_dict = {
        'e':'vacant_job_spaces',
        'h':'vacant_residential_units',
        'rdp':'du_spaces',
    }

    capacity_var = capacity_dict[agent]
    
    # get exp vars and vacant job spaces by block with tract id
    tracking_cols = [capacity_var, idx_geog_col]
    buildings = orca.get_table('blocks').to_frame(calib_expvars + tracking_cols)

    # tracking: vacant units and tract id
    tracking_table = buildings[tracking_cols]

    # get array of vacant capacity by block
    vacant_capacity = tracking_table[capacity_var].copy()
    vacant_capacity[vacant_capacity < 0] = 0
    vacant_capacity = vacant_capacity.values

    # expvars: explanatory variables by block
    expvar_table = buildings[calib_expvars]
    x = np.transpose(expvar_table.to_numpy())

    # format parameters (weights)
    param_scale = 0.001
    extra_calib_cols_init = np.random.randn(len(extra_calib_cols)) * param_scale
    w = np.concatenate((np.array(fitted_parameters), extra_calib_cols_init))
    w = w.reshape((1, len(w)))

    # Get unique tract ids by block
    calib_geog_id = np.copy(tracking_table[idx_geog_col])
    unique_calib_geogs = np.unique(calib_geog_id)
    unique_calib_geogs_map = (
        pd.Series(unique_calib_geogs).reset_index().set_index(0)['index']
    )
    calib_geog_id_idx = (
        pd.Series(tracking_table[idx_geog_col])
        .map(unique_calib_geogs_map).fillna(0).values
    )
    unique_calib_geog_id_idx = np.unique(calib_geog_id_idx)
    geog_idxs = {}
    for geog in unique_calib_geog_id_idx:
        geog_idxs[geog] = calib_geog_id_idx == geog

    # Get segment-level target shares from calibration data
    target_shares = calib_targets[f'growth_perc_{segment_id}'].values



    # Define calibration functions
    ## exponentiate logits to get probabilities
    def softmax(utilities):
        exp_utility = np.exp(utilities)
        sum_expu_across_submodels = np.sum(exp_utility, axis=1, keepdims=True)
        proba = exp_utility / sum_expu_across_submodels
        return proba

    ## calculate probabilities w/ or w/o capacity constraint
    def calc_probas(weights, capacity_weight=True):
        logits = np.dot(weights, x)
        probas = softmax(logits)
        if capacity_weight:
            probas = probas * vacant_capacity
            probas = probas / probas.sum()
        return probas

    ## Turn probabilities into growth shares based on aggregate growth target
    def capacity_constrained_allocation(weights, capacity_weight=True):
        probas = calc_probas(weights, capacity_weight=capacity_weight)
        probas = probas.reshape(probas.shape[1])
        allocated_growth = np.zeros(probas.shape[0])
        for i in range(10):
            amount_to_allocate = aggr_growth - np.sum(allocated_growth)
            if amount_to_allocate <= 0: break

            expected_growth_alt = probas * amount_to_allocate
            allocated_growth = allocated_growth + expected_growth_alt
            allocated_growth = np.clip(allocated_growth, 0, vacant_capacity)

            can_grow_mask = allocated_growth < vacant_capacity
            capacity_mask = np.zeros(can_grow_mask.shape[0])
            capacity_mask[can_grow_mask] = 1.0
            probas = probas * capacity_mask
            if capacity_weight:
                probas = probas * vacant_capacity
            probas = probas / probas.sum()

        return allocated_growth

    ## sum growth by tract
    def growth_sum_by_geog(expected_growth_by_alternative):
        geog_growths = []
        for geog in unique_calib_geog_id_idx:
            idx_geog = geog_idxs[geog]
            expected_geog_growth = np.sum(expected_growth_by_alternative[idx_geog])
            geog_growths.append(expected_geog_growth)

        expected_growth_by_geog = np.array(geog_growths)
        return expected_growth_by_geog

    ## mean squared error
    def mse(target, predicted):
        error = target - predicted
        squared_error = error**2
        return np.mean(squared_error)

    ## capacity-constrained loss function
    def capacity_lcm_loss(weights, i=None):
        allocated_growth = capacity_constrained_allocation(weights)
        growth_proportions = allocated_growth / allocated_growth.sum()
        predicted_distrib = growth_sum_by_geog(growth_proportions)
        mse_score = mse(target_shares, predicted_distrib)
        return mse_score

    def capacity_corr_with_observed(weights):
        allocated_growth = capacity_constrained_allocation(weights)
        growth_proportions = allocated_growth / allocated_growth.sum()
        predicted_distrib = growth_sum_by_geog(growth_proportions)
        return np.corrcoef(target_shares, predicted_distrib)[0][1]


    # train calibrated coefficients
    training_loss_grad = grad(capacity_lcm_loss)
    trained_params = w

    loss = capacity_lcm_loss(trained_params)
    corr = capacity_corr_with_observed(trained_params)
    print(f'\nPre-calibration: {segment}')
    print(f"MSE: {loss}")
    print(f"Corr: {corr}")

    losses = [loss]
    for i in range(max_iter):
        sys.stdout.write(f"\riteration: {i+1}")
        sys.stdout.flush()
        trained_params = adam(
            training_loss_grad, trained_params, step_size=step_size, num_iters=1
        )
        loss_i  = capacity_lcm_loss(trained_params)
        losses.append(loss_i)
        improvement = losses[i] - loss_i
        if improvement < tol:
            break
    
    # get post-calibration stats
    loss = capacity_lcm_loss(trained_params)
    corr = capacity_corr_with_observed(trained_params)
    print(f'\nPost-calibration: {segment}')
    print(f"MSE: {loss}")
    print(f"Corr: {corr}\n")

    plt.plot(losses)
    plt.title(f'Segment: {segment}; step size: {step_size}; tolerance: {tol}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    # save new parameters to model data
    m.fitted_parameters = list(map(float, list(trained_params.flatten())))
    m.summary_table = None
    m.name = segment + '_calib'

    # # register model (will overwrite previous model)
    # mm.register(m)

    return m