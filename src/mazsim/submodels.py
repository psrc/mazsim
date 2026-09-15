import copy
from pathlib import Path

import numpy as np
import orca
import pandas as pd

from urbansim_templates import modelmanager as mm
# side-effect import: registers the templates modelmanager needs to rebuild saved steps
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep
from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLDiscreteChoiceModel, SegmentedMNLDiscreteChoiceModel, \
    GrowthRateTransition, transition, relocation

from mazsim import config


@orca.step('initialize_submodels')
def initialize_submodels(project_dir):
    """Point modelmanager at the project's submodels directory, where every saved step yaml lives."""
    submodels_dir = Path(project_dir) / 'configs' / 'submodels'
    submodels_dir.mkdir(parents=True, exist_ok=True)
    mm.initialize(submodels_dir)


def add_subregion_to_element(element, subregion, subregion_column):
    # subregion ids are integers in this model, so the value is not quoted
    condition = '{} == {}'.format(subregion_column, int(subregion))
    if not element:
        return '({})'.format(condition)
    if isinstance(element, str):
        return '({}) & ({})'.format(element, condition)
    return list(element) + [condition]


def generate_subregion_model(subregion, segment, subregion_column):
    model_object = copy.copy(mm.get_step(segment.split('.')[0]))
    model_object.name = model_object.name + '_{}'.format(subregion)
    model_object.out_alt_filters = add_subregion_to_element(model_object.out_alt_filters, subregion, subregion_column)
    model_object.out_chooser_filters = add_subregion_to_element(model_object.out_chooser_filters, subregion, subregion_column)

    mm.register(model_object, save_to_disk = False) #for debugging can be set to True
    return model_object.name


@orca.step('setup_lcms')
def setup_lcms(project_dir):
    """Register a "<group>_step_names" injectable for every group in submodel_list.yaml."""
    cfg = config.load_yaml('submodel_list.yaml', project_dir)
    groups = cfg['submodel_groups']
    subregion_source = cfg['subregion_source']
    subregion_column = subregion_source['column']

    # calibration runs always fit the uncalibrated submodels; simulation runs use the
    # calibrated submodels only when 'calibrated' is set in settings.yaml
    # 'running_calibrate' is only injected by the calibrate command
    running_calibrate = orca.is_injectable('running_calibrate') and orca.get_injectable('running_calibrate')
    use_calibrated_submodels = orca.get_injectable('calibrated') and not running_calibrate
    submodel_list_name = 'submodel_list_calib' if use_calibrated_submodels else 'submodel_list'
    models_from_yaml = orca.get_injectable(submodel_list_name)

    for group, segments in models_from_yaml.items():
        if group not in groups:
            continue
        group_config = groups[group]
        unsegmented = [segment.split('.')[0] for segment in segments]

        if orca.get_injectable(group_config['ct_type']) == 'sub_ct':
            subregions = orca.get_table(subregion_source['table']).to_frame(subregion_column)[subregion_column].unique()
            step_names = [generate_subregion_model(subregion, segment, subregion_column)
                          for segment in segments for subregion in subregions]
        else:
            step_names = list(unsegmented)

        orca.add_injectable(f'{group}_step_names', sorted(step_names, reverse=True))

        unsegmented_injectable = group_config.get('unsegmented_injectable')
        if unsegmented_injectable:
            orca.add_injectable(unsegmented_injectable, sorted(unsegmented, reverse=True))


@orca.step('setup_hupm')
def setup_hupm(project_dir):
    """Load the saved housing unit price models so the rent and value steps can run them."""
    initialize_submodels(project_dir)


def _price_models(project_dir=None):
    return config.load_yaml('submodel_list.yaml', project_dir)['price_models']


@orca.step('rent_price_model')
def rent_price_model(project_dir):
    """Predict the configured rent variable from its fitted hedonic regression."""
    mm.get_step(_price_models(project_dir)['rent_price_model']).run()


@orca.step('value_price_model')
def value_price_model(project_dir):
    """Predict the configured value variable from its fitted hedonic regression."""
    mm.get_step(_price_models(project_dir)['value_price_model']).run()


@orca.step('clip_price_data')
def clip_price_data(project_dir):
    """The hedonics are linear and can predict negative prices, so hold them at zero."""
    clip = _price_models(project_dir)['clip']
    table = orca.get_table(clip['table'])
    for column in clip['columns']:
        table.update_col(column, table[column].clip(lower=0))


@orca.step('household_lcms')
def household_lcms():
    steps = orca.get_injectable('hlcm_step_names')
    orca.run(steps, [orca.get_injectable('iter_var')])

@orca.step('job_lcms')
def job_lcms():
    steps = orca.get_injectable('jlcm_step_names')
    orca.run(steps, [orca.get_injectable('iter_var')])

@orca.step('housing_unit_lcms')
def housing_unit_lcms():
    steps = orca.get_injectable('hulcm_step_names')
    orca.run(steps, [orca.get_injectable('iter_var')])
