import copy
from pathlib import Path

import numpy as np
import orca
import pandas as pd

from urbansim_templates import modelmanager as mm
# side-effect import: registers the templates modelmanager needs to rebuild saved steps
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep
from urbansim_templates.utils import get_data
from urbansim.models import RegressionModel, SegmentedRegressionModel, \
    MNLDiscreteChoiceModel, SegmentedMNLDiscreteChoiceModel, \
    GrowthRateTransition, transition, relocation
from urbansim.models.util import columns_in_formula

from mazsim import config
from mazsim.control_totals import UNPLACED, refresh_subregion
from mazsim.data_loader import OBSERVED_PREFIX


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


def _check_model_data(model):
    """Fail with the offending column name if a choice model's explanatory data has gaps.

    patsy drops rows with missing values from the design matrix, which choicemodels then
    reports only as a reshape error on an array of the wrong size.
    """
    expression_columns = set(columns_in_formula(model.model_expression))
    sides = [
        ('choosers', model.out_choosers, model.choosers, model.out_chooser_filters),
        ('alternatives', model.out_alternatives, model.alternatives, model.out_alt_filters),
    ]

    problems = []
    for side, tables, fallback_tables, filters in sides:
        df = get_data(tables=tables, fallback_tables=fallback_tables, filters=filters,
                      model_expression=model.model_expression)
        for column in sorted(expression_columns.intersection(df.columns)):
            missing = int(df[column].isna().sum())
            if missing:
                problems.append('  {}.{}: {:,} of {:,} rows missing'.format(
                    side, column, missing, len(df)))

    if problems:
        raise ValueError('{} cannot run, its explanatory data has missing values:\n{}'.format(
            model.name, '\n'.join(problems)))


def _run_lcms(step_names_injectable):
    steps = orca.get_injectable(step_names_injectable)
    for step_name in steps:
        _check_model_data(mm.get_step(step_name))
    orca.run(steps, [orca.get_injectable('iter_var')])


@orca.step('household_lcms')
def household_lcms():
    _run_lcms('hlcm_step_names')

@orca.step('job_lcms')
def job_lcms():
    _run_lcms('jlcm_step_names')

@orca.step('housing_unit_lcms')
def housing_unit_lcms():
    _run_lcms('hulcm_step_names')


def _segment_block_probabilities(lcm_group, segment_column):
    """Per-block probability of each segment, from the calibrated sub-models' fitted utilities.

    Returns a blocks-by-segment frame whose rows sum to 1, i.e. P(segment | block).
    """
    # the subregion-suffixed copies made by setup_lcms share their parents' coefficients,
    # so the unsegmented models give the same probabilities with far less work
    step_names = orca.get_injectable(f'reg_{lcm_group}_step_names') \
        if orca.is_injectable(f'reg_{lcm_group}_step_names') \
        else orca.get_injectable(f'{lcm_group}_step_names')

    blocks = orca.get_table('blocks')
    utilities = {}
    for step_name in step_names:
        model = mm.get_step(step_name)
        # model expressions are built by util.str_model_expression and always end in ' - 1'
        expl_vars = model.model_expression[:-4].split(' + ')
        design = blocks.to_frame(expl_vars)[expl_vars].to_numpy(dtype=float)
        segment = _segment_value(model.chooser_filters, segment_column)
        utilities[segment] = design @ np.asarray(model.fitted_parameters, dtype=float)

    utility_df = pd.DataFrame(utilities, index=blocks.index)
    # softmax within a segment gives P(block | segment)
    exp_utility = np.exp(utility_df - utility_df.max())
    block_given_segment = exp_utility / exp_utility.sum(axis=0)

    # renormalizing across segments turns that into the block's segment mix
    return block_given_segment.div(block_given_segment.sum(axis=1), axis=0)


def _segment_value(chooser_filters, segment_column):
    """Pull the segment id a sub-model is fit on out of its "<segment_column> == <value>" filter."""
    filters = [chooser_filters] if isinstance(chooser_filters, str) else list(chooser_filters or [])
    for condition in ' & '.join(filters).split('&'):
        left, _, right = condition.partition('==')
        if left.strip(' ()') == segment_column:
            return int(right.strip(' ()'))
    raise ValueError(f'No "{segment_column} == <value>" condition in chooser_filters {chooser_filters!r}.')


def _grow_pool(df, table_name, shortfall, year_built_column, year):
    """Clone rows to cover a shortfall the unplaced pool cannot meet, and return the enlarged frame."""
    print(f'WARNING: the unplaced {table_name} pool is {shortfall:,} short of the observed '
          f'{year} totals; cloning that many rows to cover it.')
    df, added, _ = transition.add_rows(df, shortfall)
    df.loc[added, orca.get_injectable('geography_id')] = UNPLACED
    if year_built_column:
        df.loc[added, year_built_column] = year
    return df


def _place_type(type_name, spec, year):
    """Force each block's row count for one agent table onto its observed value."""
    table_name = spec['table']
    segment_column = spec['segment_column']
    location_column = orca.get_injectable('geography_id')

    table = orca.get_table(table_name)
    df = table.to_frame(table.local_columns)
    # unit_type_id is stored as a float, so it needs coercing to match the probability columns
    segments = table.to_frame([segment_column])[segment_column].astype(int)

    observed = orca.get_table('blocks').to_frame([f'{OBSERVED_PREFIX}{type_name}'])
    observed = observed[f'{OBSERVED_PREFIX}{type_name}'].dropna()
    current = df[location_column].value_counts().reindex(observed.index, fill_value=0)
    delta = observed.astype(int) - current

    probabilities = _segment_block_probabilities(spec['lcm_group'], segment_column)

    # surpluses are released first so the freed agents are available to the deficit blocks
    surplus = -delta[delta < 0]
    locations = df[location_column]
    in_surplus = locations[locations.isin(surplus.index)]
    # 1/p evicts the segments the model considers least likely for this block
    weights = 1.0 / _segment_weights(probabilities, in_surplus, segments)
    evicted = _sample_within_blocks(in_surplus, weights, surplus)
    df.loc[evicted, location_column] = UNPLACED

    # a shuffled order keeps the blocks drawn last from being biased by what the pool has left
    deficit = delta[delta > 0].sample(frac=1)
    pool = df.index[df[location_column] == UNPLACED]
    shortfall = int(deficit.sum()) - len(pool)
    if shortfall > 0:
        df = _grow_pool(df, table_name, shortfall, spec.get('year_built_column'), year)
        # the clones have to reach orca before a computed segment column can be read back for them
        orca.add_table(table_name, df)
        table = orca.get_table(table_name)
        segments = table.to_frame([segment_column])[segment_column].astype(int)
        pool = df.index[df[location_column] == UNPLACED]

    placements = _draw_from_pool(pool, segments.loc[pool].to_numpy(), deficit, probabilities)
    df.loc[placements.index, location_column] = placements.to_numpy()

    orca.add_table(table_name, df)
    # the pool is drawn from regionwide, so rows can land in a different subregion than they left
    refresh_subregion(table_name)
    print(f'Observed {year} {type_name}: placed {len(placements):,} and unplaced {len(evicted):,} '
          f'across {len(delta[delta != 0]):,} blocks.')

    return surplus.index


def _segment_weights(probabilities, locations, segments):
    """Look each agent's own (block, segment) probability out of the blocks-by-segment frame."""
    rows = probabilities.index.get_indexer(locations.to_numpy())
    columns = probabilities.columns.get_indexer(segments.loc[locations.index].to_numpy())
    found = (rows >= 0) & (columns >= 0)

    weights = np.zeros(len(locations))
    weights[found] = probabilities.to_numpy()[rows[found], columns[found]]
    return weights


def _sample_within_blocks(locations, weights, counts):
    """Draw `counts[block]` of each block's agents, proportional to `weights`, without replacement.

    Ranking one Exp(1)/weight key per agent samples every block in a single pass, which is the
    same draw `numpy.random.choice(replace=False, p=...)` makes one block at a time.
    """
    if locations.empty:
        return locations.index[:0]

    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0)
    keys = np.random.exponential(size=len(weights)) / np.where(weights > 0, weights, np.finfo(float).tiny)

    blocks = locations.to_numpy()
    order = np.lexsort((keys, blocks))
    ranked = blocks[order]
    rank = np.arange(len(ranked)) - np.searchsorted(ranked, ranked, side='left')

    return locations.index[order[rank < counts.reindex(ranked).to_numpy()]]


def _draw_from_pool(pool, pool_segments, deficit, probabilities):
    """Assign unplaced agents to the deficit blocks, weighting each by its segment's block probability.

    The weights vary only by segment, so drawing an agent without replacement is the same as drawing
    a segment with probability proportional to (weight x agents it has left) and popping a member of
    it. That keeps the cost proportional to the agents placed rather than to the pool size.
    """
    queues, cursors, available = [], [], []
    for segment in probabilities.columns:
        members = pool.to_numpy()[pool_segments == segment]
        np.random.shuffle(members)
        queues.append(members)
        cursors.append(0)
        available.append(len(members))

    segment_count = len(queues)
    weights = np.nan_to_num(probabilities.reindex(deficit.index).to_numpy(), nan=0.0, posinf=0.0)
    wanted = deficit.to_numpy()
    total_wanted = int(wanted.sum())
    draws = np.random.random(total_wanted).tolist()

    chosen_ids = np.empty(total_wanted, dtype=pool.dtype)
    chosen_blocks = np.empty(total_wanted, dtype=deficit.index.dtype)
    filled = 0

    for row, block_id in enumerate(deficit.index.to_numpy()):
        block_weights = weights[row].tolist()
        if sum(block_weights) <= 0:
            block_weights = [1.0] * segment_count
        odds = [block_weights[s] * available[s] for s in range(segment_count)]
        total = sum(odds)

        for _ in range(int(wanted[row])):
            if total <= 0:
                break
            draw = draws[filled] * total
            running = 0.0
            for s in range(segment_count):
                running += odds[s]
                if draw < running:
                    break
            # float drift can overshoot the last segment, so fall back to the fullest queue
            if available[s] == 0:
                s = max(range(segment_count), key=available.__getitem__)

            chosen_ids[filled] = queues[s][cursors[s]]
            chosen_blocks[filled] = block_id
            filled += 1
            cursors[s] += 1
            available[s] -= 1
            odds[s] -= block_weights[s]
            total -= block_weights[s]

    return pd.Series(chosen_blocks[:filled], index=pd.Index(chosen_ids[:filled], name=pool.name))


def _reconcile_agents(spec, blocks_to_check):
    """Unplace agents at random wherever removing spaces left a block with more agents than rows."""
    location_column = orca.get_injectable('geography_id')
    agents_name = spec['agents']
    agents = orca.get_table(agents_name)
    agent_df = agents.to_frame(agents.local_columns)

    spaces = orca.get_table(spec['spaces'])[location_column].value_counts()
    occupied = agent_df[location_column].value_counts()
    excess = occupied.reindex(blocks_to_check, fill_value=0) - spaces.reindex(blocks_to_check, fill_value=0)
    excess = excess[excess > 0]

    locations = agent_df[location_column]
    over_housed = locations[locations.isin(excess.index)]
    displaced = _sample_within_blocks(over_housed, np.ones(len(over_housed)), excess)

    if displaced.empty:
        return

    agent_df.loc[displaced, location_column] = UNPLACED
    orca.add_table(agents_name, agent_df)

    for linked_name, key in (spec.get('linked_tables') or {}).items():
        linked = orca.get_table(linked_name)
        linked_df = linked.to_frame(linked.local_columns)
        linked_df.loc[linked_df[key].isin(displaced), location_column] = UNPLACED
        orca.add_table(linked_name, linked_df)

    print(f'Unplaced {len(displaced):,} {agents_name} left without a space by the observed removals.')


@orca.step('place_observed_data')
def place_observed_data():
    cfg = config.load_yaml('observed_data.yaml')
    if not cfg.get('place_observed_data'):
        return

    variables = cfg['types_to_place']
    year = orca.get_injectable('year')
    obs_year_match = []
    for variable in variables:
        if orca.get_injectable(f'observed_{variable}_year') == year:
            obs_year_match.append(variable)
    if not obs_year_match:
        return

    placement = cfg['placement']
    for type_name in obs_year_match:
        spec = placement[type_name]
        emptied_blocks = _place_type(type_name, spec, year)

        reconcile = spec.get('reconcile')
        if reconcile:
            _reconcile_agents({**reconcile, 'spaces': spec['table']}, emptied_blocks)

