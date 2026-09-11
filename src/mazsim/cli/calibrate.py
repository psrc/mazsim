"""Calibrate the location choice (HLCM/JLCM/HULCM) models defined in calibrate.yaml against calib_targets tables."""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import orca
import yaml
from autograd import grad
from autograd.misc.optimizers import adam
import autograd.numpy as np
import pandas as pd
from urbansim.models import util
from urbansim_templates import modelmanager as mm

# side-effect imports: registers the load_data/build_networks/register_variables/setup_lcms orca steps
from mazsim import data_loader, variables, models

# maps each lcm name to the agent-letter/target-type conventions used by _model_calibration
LCM_AGENTS = {
    "hlcm": "h",
    "jlcm": "j",
    "hulcm": "hu",
}
TARGET_COLS = {
    "household": "households_target",
    "job": "jobs_target",
    "housing_unit": "units_target",
}


def _load_calibrate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "calibrate.yaml"
    return yaml.safe_load(config_path.read_text())


def _load_config(project_dir: Path, key: str) -> dict[str, Any]:
    return _load_calibrate_yaml(project_dir)[key]


def _get_calib_targets(target_type: str) -> dict[str, pd.DataFrame]:
    """Build per-tract growth targets/shares for a target type (household/job/housing_unit)."""
    var_dict = {
        "job": {
            "lcm": "jlcm",
            "str_replace": "",
            "target_col": "jobs_target",
        },
        "household": {
            "lcm": "hlcm",
            "str_replace": "(recent_mover == 1) & ",
            "target_col": "households_target",
        },
        "housing_unit": {
            "lcm": "hulcm",
            "str_replace": "(year_built > 2010) & ",
            "target_col": "units_target",
        },
    }

    # empty tract df
    idx = list(orca.get_table("blocks").to_frame("tract_id")["tract_id"].unique())
    df = pd.DataFrame(index=idx)

    # get target data
    data = orca.get_table(f"{target_type}_calib_targets").local
    data = data.fillna(0)

    lcm = var_dict[target_type]["lcm"]
    target_col = var_dict[target_type]["target_col"]

    for segment in orca.get_injectable(f"{lcm}_step_names"):
        filter_ = mm.get_step(segment).chooser_filters.replace(var_dict[target_type]["str_replace"], "")
        data.loc[data.query(filter_).index, "segment"] = segment.replace(lcm, "")

    data_gr = data.groupby(["tract_id", "segment"])[target_col].sum().unstack().reindex(df.index).fillna(0)

    for model_name in orca.get_injectable(f"{lcm}_step_names"):
        segment = model_name.replace(lcm, "")
        df[f"growth_{segment}"] = np.clip(data_gr[segment], 0, None)
        df[f"growth_perc_{segment}"] = df[f"growth_{segment}"] / df[f"growth_{segment}"].sum()

    return {"calib_raw": data, "calib_clean": df}


def _standardize_coefficients(model) -> None:
    expression = model.model_expression[:-4].split(" + ")
    standardized = [var if var.startswith("st_") else "st_" + var for var in expression]
    model.model_expression = util.str_model_expression(standardized, add_constant=False)
    model.fit()


def _model_calibration(
    segment,
    agent,
    calib_targets,
    aggr_growth,
    max_iter=10000,
    tol=1e-13,
    step_size=0.001,
    std=False,
    county_calib=False,
    param_scale=0.001,
    max_alloc_iter=10,
):
    """Calibrate one registered LCM sub-model's coefficients to match target growth shares by tract."""
    print(f"\nCalibrating segment: {segment}\n")

    # get model segment id
    segment_id = segment.replace(f"{agent}lcm", "")

    # re-fit model with standardized coefficients if desired
    m = copy.copy(mm.get_step(segment))

    if std:
        _standardize_coefficients(m)

    # set id geog to tract
    idx_geog_col = "tract_id"

    # get model expression and parameters
    expvar_names = m.model_expression[:-4].split(" + ")
    fitted_parameters = m.fitted_parameters

    # add county-level indicators if desired, one per distinct county_id present in blocks
    extra_calib_cols = []
    if county_calib:
        county_ids = orca.get_table("blocks").to_frame("county_id")["county_id"].unique()
        extra_calib_cols += [f"county_id_is_{county_id}" for county_id in county_ids]
        if std:
            extra_calib_cols = ["st_" + col for col in extra_calib_cols]

    calib_expvars = expvar_names + extra_calib_cols

    capacity_dict = {
        "j": "vacant_job_spaces",
        "h": "vacant_housing_units",
        "hu": "vacant_hu_spaces",
    }

    capacity_var = capacity_dict[agent]

    # get exp vars and vacant job spaces by block with tract id
    tracking_cols = [capacity_var, idx_geog_col]
    buildings = orca.get_table("blocks").to_frame(calib_expvars + tracking_cols)

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
    extra_calib_cols_init = np.random.randn(len(extra_calib_cols)) * param_scale
    w = np.concatenate((np.array(fitted_parameters), extra_calib_cols_init))
    w = w.reshape((1, len(w)))

    # Get unique tract ids by block
    calib_geog_id = np.copy(tracking_table[idx_geog_col])
    unique_calib_geogs = np.unique(calib_geog_id)
    unique_calib_geogs_map = pd.Series(unique_calib_geogs).reset_index().set_index(0)["index"]
    calib_geog_id_idx = pd.Series(tracking_table[idx_geog_col]).map(unique_calib_geogs_map).fillna(0).values
    unique_calib_geog_id_idx = np.unique(calib_geog_id_idx)
    geog_idxs = {}
    for geog in unique_calib_geog_id_idx:
        geog_idxs[geog] = calib_geog_id_idx == geog

    # Get segment-level target shares from calibration data
    target_shares = calib_targets[f"growth_perc_{segment_id}"].values

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
        for i in range(max_alloc_iter):
            amount_to_allocate = aggr_growth - np.sum(allocated_growth)
            if amount_to_allocate <= 0:
                break

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
    print(f"\nPre-calibration: {segment}")
    print(f"MSE: {loss}")
    print(f"Corr: {corr}")

    losses = [loss]
    for i in range(max_iter):
        sys.stdout.write(f"\riteration: {i+1}")
        sys.stdout.flush()
        trained_params = adam(training_loss_grad, trained_params, step_size=step_size, num_iters=1)
        loss_i = capacity_lcm_loss(trained_params)
        losses.append(loss_i)
        improvement = losses[i] - loss_i
        if improvement < tol:
            break

    # get post-calibration stats
    loss = capacity_lcm_loss(trained_params)
    corr = capacity_corr_with_observed(trained_params)
    print(f"\nPost-calibration: {segment}")
    print(f"MSE: {loss}")
    print(f"Corr: {corr}\n")

    # save new parameters to model data
    m.fitted_parameters = list(map(float, list(trained_params.flatten())))
    m.summary_table = None
    m.name = segment + "_calib"
    return m


def _calibrate_lcm(project_dir: Path, lcm: str) -> None:
    """Calibrate every registered sub-model for one LCM against its calib_targets table and re-register it."""
    mm.initialize(Path.joinpath(project_dir, "configs"))
    calibrate_config = _load_calibrate_yaml(project_dir)
    lcm_config = calibrate_config[lcm]

    calib_dict = _get_calib_targets(lcm_config["target_type"])
    df_calib = calib_dict["calib_clean"]
    raw_data = calib_dict["calib_raw"]

    target_col = TARGET_COLS[lcm_config["target_type"]]
    aggr_growth = raw_data[target_col].sum() / (calibrate_config["base_year"] - calibrate_config["historic_year"])

    for segment in orca.get_injectable(f"{lcm}_step_names"):
        m = _model_calibration(
            segment=segment,
            agent=LCM_AGENTS[lcm],
            calib_targets=df_calib,
            aggr_growth=aggr_growth,
            max_iter=lcm_config.get("max_iter", 10000),
            tol=lcm_config.get("tol", 1e-13),
            step_size=lcm_config.get("step_size", 0.001),
            std=lcm_config.get("std", False),
            county_calib=lcm_config.get("county_calib", False),
            param_scale=lcm_config.get("param_scale", 0.001),
            max_alloc_iter=lcm_config.get("max_alloc_iter", 10),
        )
        mm.register(m)  # overwrites any previously registered model of the same name


@orca.step("calibrate_hlcm")
def calibrate_hlcm(project_dir: Path) -> None:
    """Calibrate every HLCM sub-model against household_calib_targets and register the results."""
    _calibrate_lcm(project_dir, "hlcm")


@orca.step("calibrate_jlcm")
def calibrate_jlcm(project_dir: Path) -> None:
    """Calibrate every JLCM sub-model against job_calib_targets and register the results."""
    _calibrate_lcm(project_dir, "jlcm")


@orca.step("calibrate_hulcm")
def calibrate_hulcm(project_dir: Path) -> None:
    """Calibrate every HULCM sub-model against housing_unit_calib_targets and register the results."""
    _calibrate_lcm(project_dir, "hulcm")


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )

def run(args):
    """Run the orca steps listed under calibration_steps in calibrate.yaml."""
    project_dir = Path(args.configs_dir).parent
    calibration_steps = _load_calibrate_yaml(project_dir)["calibration_steps"]
    orca.add_injectable("project_dir", project_dir)
    orca.add_injectable('running_calibrate', True) # overrides calibrated status to False for running calibration
    orca.run(calibration_steps)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))
