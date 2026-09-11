import orca
import os
import yaml
from urbansim.utils import misc
from urbansim_templates import modelmanager as mm
import copy


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
    hlcm_step_names = []
    jlcm_step_names = []
    hulcm_step_names = []
    location_choice_models = {}
    models = []
    subregion_models = []

    if orca.get_injectable('calibrated'):
        yaml_file = 'yaml_configs_calib.yaml'
        calib_option = 'calib'
    else:
        yaml_file = 'yaml_configs.yaml'
        calib_option = 'non_calib'

    with open(os.path.join(misc.configs_dir(), yaml_file)) as f:
        config = yaml.safe_load(f)
    orca.add_injectable('yaml_configs', config)

    subregional_ct_dict = {'hlcm':orca.get_injectable('hh_ct_type'),
                            'hulcm':orca.get_injectable('hh_ct_type'),
                            'jlcm':orca.get_injectable('emp_ct_type')}

    models_from_yaml = orca.get_injectable('yaml_configs')
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