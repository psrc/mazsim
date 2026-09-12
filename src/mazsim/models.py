import copy
import numpy as np
import orca
import pandas as pd

from urbansim_templates import modelmanager as mm
from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLDiscreteChoiceModel, SegmentedMNLDiscreteChoiceModel, \
    GrowthRateTransition, transition, relocation


def add_subregion_to_element(element,subregion):
    if element:
        if isinstance(element, str):
            element = '({})'.format(element) + ' & (subregion_id == {})'.format('\'' + subregion + '\'')
        else:
            element = element.append('subregion_id == {}'.format('\'' + subregion + '\''))
    else:
        element = '(subregion_id == {})'.format('\'' + subregion + '\'')
    return element


def generate_subregion_model(subregion, segment):
    model_object = copy.copy(mm.get_step(segment.split('.')[0]))
    model_object.name = model_object.name + '_{}'.format(subregion)
    model_object.out_alt_filters = add_subregion_to_element(model_object.out_alt_filters,subregion)
    model_object.out_chooser_filters = add_subregion_to_element(model_object.out_chooser_filters,subregion)

    mm.register(model_object, save_to_disk = False) #for debugging can be set to True
    return model_object.name


@orca.step('setup_lcms')
def setup_lcms():
    """Register hlcm/jlcm/hulcm step names from the 'submodel_list'/'submodel_list_calib' injectable."""
    subregional_ct_dict = {'hlcm':orca.get_injectable('hh_ct_type'),
                            'hulcm':orca.get_injectable('hh_ct_type'),
                            'jlcm':orca.get_injectable('job_ct_type')}

    # calibration runs always fit the uncalibrated submodels; simulation runs use the
    # calibrated submodels only when 'calibrated' is set in settings.yaml
    running_calibrate = orca.get_injectable('running_calibrate')
    use_calibrated_submodels = orca.get_injectable('calibrated') and not running_calibrate
    submodel_list_name = 'submodel_list_calib' if use_calibrated_submodels else 'submodel_list'
    models_from_yaml = orca.get_injectable(submodel_list_name)

    models = []
    hreloc_models = []
    reg_jlcms =[]
    for model, segments in models_from_yaml.items():
        if model in ['hlcm', 'jlcm', 'hulcm']:
            for segment in segments:
                if subregional_ct_dict[model] == 'sub_ct':
                    subregions = orca.get_table('blocks').to_frame('subregion_id')['subregion_id'].unique()
                    for subregion in subregions:
                        models.append(generate_subregion_model(subregion, segment))
                else:
                    models.append(segment.split('.')[0])
            if model in ['hlcm']:
                for segment in segments:
                    hreloc_models.append(segment.split('.')[0])
            if model in ['jlcm']:
                for segment in segments:
                    reg_jlcms.append(segment.split('.')[0])
    orca.add_injectable('hlcm_step_names', sorted(
        [x for x in models if 'hlcm' in x], reverse=True))
    orca.add_injectable('jlcm_step_names', sorted(
        [x for x in models if 'jlcm' in x], reverse=True))
    orca.add_injectable('hulcm_step_names', sorted(
        [x for x in models if ('hulcm' in x) and (x[0] != 'n')], reverse=True))
    # orca.add_injectable(
    #     'hrelcm_step_names', sorted(hreloc_models, reverse=True)
    # )
    orca.add_injectable(
        'reg_jlcm_step_names', sorted(reg_jlcms, reverse=True)
    )


@orca.step('setup_hupm')
def setup_hupm():
    models_from_yaml = orca.get_injectable('submodel_list')
    models = []
    for model, segments in models_from_yaml.items():
        if model in ['hupm_rent', 'hupm_value']:
            for segment in segments:
                models.append(segment.split('.')[0])
    orca.add_injectable('price_models', models)


def update_linked_table(tbl, col_name, added, copied, removed):
    """
    Copy and update rows in a table that has a column referencing another
    table that has had rows added via copying.

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to update with new or removed rows.
    col_name : str
        Name of column in `table` that corresponds to the index values
        in `copied` and `removed`.
    added : pandas.Index
        Indexes of rows that are new in the linked table.
    copied : pandas.Index
        Indexes of rows that were copied to make new rows in linked table.
    removed : pandas.Index
        Indexes of rows that were removed from the linked table.

    Returns
    -------
    updated : pandas.DataFrame

    """

    # handle removals
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]
    removed = table.loc[table[col_name].isin(set(removed))]
    if (added is None or len(added) == 0):
        return table

    # map new IDs to the IDs from which they were copied
    id_map = pd.concat(
        [pd.Series(copied, name=col_name), pd.Series(added, name='temp_id')],
        axis=1)

    # join to linked table and assign new id
    new_rows = id_map.merge(table, on=col_name)
    new_rows.drop(col_name, axis=1, inplace=True)
    new_rows.rename(columns={'temp_id': col_name}, inplace=True)

    # index the new rows
    starting_index = table.index.values.max() + 1
    new_rows.index = np.arange(starting_index, starting_index + len(new_rows),
                               dtype=int)

    return pd.concat([table, new_rows])


def full_transition(agents, agent_controls, year, location_fname,
                    linked_tables={}, set_year_built=False):
    """
    Run a transition model based on control totals specified in the usual
    UrbanSim way

    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    year : int
        The year, which will index into the controls
    settings : dict
        Contains the configuration for the transition model - is specified
        down to the yaml level with a "total_column" which specifies the
        control total and an "add_columns" param which specified which
        columns to add when calling to_frame (should be a list of the columns
        needed to do the transition
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with dict of
        {'table_name':(DataFrameWrapper, 'link_id')}
    set_year_built: boolean
        Indicates whether to update 'year_built' columns with current simulation year

    Returns
    -------
    Nothing
    """
    try:
        ct = agent_controls.to_frame()
    except:
        ct = agent_controls.copy()
    ct = ct.replace(-1, 99999999)

    agent_df = agents.to_frame(agents.local_columns)
    print("Total agents before transition: {}".format(len(agent_df)))
    tran = transition.TabularTotalsTransition(ct, 'total')
    updated, added, copied, removed = tran.transition(agent_df, year)
    updated.loc[added, location_fname] = "-1"
    if set_year_built:
        updated.loc[added, 'year_built'] = year

    updated_links = {}
    for table_name, (table, col) in linked_tables.items():
        # logger.debug('updating linked table {}'.format(table_name))
        updated_links[table_name] = \
            update_linked_table(table, col, added, copied, removed)
        orca.add_table(table_name, updated_links[table_name])

    print("Total agents after transition: {}".format(len(updated)))
    orca.add_table(agents.name, updated[agents.local_columns])


def simple_transition(tbl, rate, location_fname, linked_tables={},
                      set_year_built=False):
    """
    Run a simple growth rate transition model on the table passed in

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    linked_tables : dict, optional
        Sets the tables linked to new or removed agents to be updated with dict of
        {'table_name':(DataFrameWrapper, 'link_id')}
    location_fname : str
        The field name in the resulting dataframe to set to "-1" (to unplace
        new agents)

    Returns
    -------
    Nothing
    """
    transition = GrowthRateTransition(rate)
    df_base = tbl.to_frame(tbl.local_columns)

    print("%d agents before transition" % len(df_base.index))
    df, added, copied, removed = transition.transition(df_base, None)
    print("%d agents after transition" % len(df.index))

    df.loc[added, location_fname] = "-1"

    if set_year_built:
        df.loc[added, 'year_built'] = orca.get_injectable('year')

    updated_links = {}
    for table_name, (table, col) in linked_tables.items():
        # logger.debug('updating linked table {}'.format(table_name))
        updated_links[table_name] = \
            update_linked_table(table, col, added, copied, removed)
        orca.add_table(table_name, updated_links[table_name])

    orca.add_table(tbl.name, df)

@orca.step('households_transition_basic')
def households_transition_basic(households, persons):
    growth_rate = orca.get_injectable(
        'household_growth_rate') if 'household_growth_rate' in orca.list_injectables() else .01
    print('Running household transition with %s percent growth rate' %
          (growth_rate * 100.0))
    return simple_transition(households, growth_rate, "block_id",
                             linked_tables={
                                 'persons': (persons, 'household_id')})

@orca.step('jobs_transition_basic')
def jobs_transition_basic(jobs):
    growth_rate = orca.get_injectable(
        'job_growth_rate') if 'job_growth_rate' in orca.list_injectables() else .01
    print('Running job transition with %s percent growth rate' %
          (growth_rate * 100.0))
    return simple_transition(jobs, growth_rate, "block_id")

@orca.step('households_transition')
def households_transition(households, persons, year):
    if 'household_growth_rate' in orca.list_injectables():
        households_transition_basic(households, persons)
    elif 'annual_household_control_totals' in orca.list_tables():
        household_controls = orca.get_table('annual_household_control_totals')
        full_transition(households, household_controls, year, 'block_id',
                        linked_tables={
                            'persons': (persons, 'household_id')})
    else:
        households_transition_basic(households, persons)

@orca.step('jobs_transition')
def jobs_transition(jobs, year):
    if 'job_growth_rate' in orca.list_injectables():
        jobs_transition_basic(jobs)
    elif 'annual_job_control_totals' in orca.list_tables():
        job_controls = orca.get_table('annual_job_control_totals')
        full_transition(jobs, job_controls, year, 'block_id')
    else:
        jobs_transition_basic(jobs)