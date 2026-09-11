import orca
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