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


def initialize_submodels(project_dir):
    """Point modelmanager at the project's submodels directory, where every saved step yaml lives."""
    submodels_dir = Path(project_dir) / 'configs' / 'submodels'
    submodels_dir.mkdir(parents=True, exist_ok=True)
    mm.initialize(submodels_dir)


def add_subregion_to_element(element, subregion):
    # subregion_id is a 6-digit integer county id in this model, so the value is not quoted
    condition = 'subregion_id == {}'.format(int(subregion))
    if not element:
        return '({})'.format(condition)
    if isinstance(element, str):
        return '({}) & ({})'.format(element, condition)
    return list(element) + [condition]


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
def setup_hupm(project_dir):
    """Load the saved housing unit price models so the rent and value steps can run them."""
    initialize_submodels(project_dir)


@orca.step('rent_price_model')
def rent_price_model():
    """Predict block-level residential rent from the fitted hupm_rent regression."""
    mm.get_step('hupm_rent').run()


@orca.step('value_price_model')
def value_price_model():
    """Predict block-level residential value from the fitted hupm_value regression."""
    mm.get_step('hupm_value').run()


@orca.step('clip_price_data')
def clip_price_data(blocks):
    """The hedonics are linear and can predict negative prices, so hold them at zero."""
    for column in ('res_rent', 'res_value'):
        blocks.update_col(column, blocks[column].clip(lower=0))


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


