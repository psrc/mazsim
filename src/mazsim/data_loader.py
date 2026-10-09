"""Validate the project's CSV tables against its data_model.py schemas and register them with orca."""

import zipfile
from pathlib import Path
import numpy as np
import orca
import pandana as pdna  # type: ignore[import-not-found]
import pandas as pd
import pandera.pandas as pa
from urbansim.utils import misc

from mazsim import config, geography

# observed counts land on the base geography as obs_<type>; variables.yaml aggregates them to sum_obs_<type>
OBSERVED_PREFIX = "obs_"

# data_sources.yaml entries with this prefix stand in for the regular table when calibrating
HISTORY_YEAR_PREFIX = "history_year_"


def _validate(table_models, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against `table_name`'s pandera schema, collecting every failure before raising."""
    if table_name not in table_models:
        raise KeyError(f"{table_name}: no schema in the project's data model TABLE_MODELS.")
    try:
        return table_models[table_name].validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"{table_name}: schema validation failed\n{exc.failure_cases}") from exc


def _active_sources(settings: dict) -> list[tuple[str, str, bool]]:
    """Resolve data_sources.yaml into (table_name, filename, from_history_year) for the current run.

    `history_year_<table>` entries are used only when calibrating, where each one is registered
    under the regular table name in place of the regular file, so the project's data model,
    variables, and sub-models apply to it unchanged. Every other run ignores them.
    """
    calibrating = orca.is_injectable("running_calibrate") and orca.get_injectable("running_calibrate")
    sources = [next(iter(entry.items())) for entry in settings.get("data_sources", [])]

    if not calibrating:
        return [(name, filename, False) for name, filename in sources
                if not name.startswith(HISTORY_YEAR_PREFIX)]

    history = {name[len(HISTORY_YEAR_PREFIX):]: filename
               for name, filename in sources if name.startswith(HISTORY_YEAR_PREFIX)}
    active = [(name, filename, False) for name, filename in sources
              if not name.startswith(HISTORY_YEAR_PREFIX) and name not in history]
    active += [(name, filename, True) for name, filename in history.items()]
    return active


def _check_base_geography_ids(history_tables: set[str], base_table: str, base_id: str) -> None:
    """Check that rows keyed by the base geography id point at blocks that exist.

    The history year tables carry their own base ids, which need not match the regular ones, and
    the variables reindex onto the blocks by id, so a stray id would silently turn into a zero or
    a NaN. Mismatches in a history year table are errors; elsewhere, e.g. in observed_data, which
    is not swapped, they are warnings.
    """
    base_index = orca.get_table(base_table).local.index
    for table_name in orca.list_tables():
        if table_name == base_table:
            continue
        table = orca.get_table(table_name).local
        if base_id in table.columns:
            ids = table[base_id]
        elif table.index.name == base_id:
            ids = table.index.to_series()
        else:
            continue

        stray = ids[~ids.isin(base_index)].unique()
        if len(stray) == 0:
            continue
        message = (f"{table_name}: {len(stray):,} {base_id} value(s) not found in {base_table}, "
                   f"e.g. {list(stray[:5])}.")
        if table_name in history_tables:
            raise ValueError(f"{message} The history year tables must share one set of {base_id}s.")
        print(f"WARNING: {message}")


def register_tables(project_dir: Path) -> None:
    """Load every table in data_sources.yaml, validate it against the project's data model, and register it."""
    data_dir = project_dir / orca.get_injectable('data_dir')
    data_model = config.load_data_model(project_dir)

    settings = config.load_yaml("data_sources.yaml", project_dir)

    history_tables = set()
    for table_name, filename, from_history_year in _active_sources(settings):
        df = pd.read_csv(data_dir / filename)
        df = _validate(data_model.TABLE_MODELS, table_name, df)

        index_col = data_model.TABLE_INDEXES.get(table_name)
        if index_col:
            df = df.set_index(index_col)

        orca.add_table(table_name, df)
        if from_history_year:
            history_tables.add(table_name)

    if history_tables:
        print(f"Using history year tables: {sorted(history_tables)}")
        variables = config.load_yaml("variables.yaml", project_dir)
        base_table, base_id = geography.base_geography(variables)
        _check_base_geography_ids(history_tables, base_table, base_id)
    orca.add_injectable("history_year_tables", history_tables)


def register_aggregation_table(table_name, table_id, base_table):
    """
    Generator function for tables representing aggregate geography.
    """
    @orca.table(table_name, cache=True)
    def func():
        geog_ids = orca.get_table(base_table)[table_id].value_counts().index.values
        df = pd.DataFrame(index=geog_ids)
        df.index.name = table_id
        return df
    return func


def _register_observed_data(project_dir):
    """Pivot the observed table onto the base geography and register each type's latest year."""
    cfg = config.load_yaml("observed_data.yaml", project_dir)
    type_col, year_col = cfg["type_column"], cfg["year_column"]
    id_col = cfg["id_column"]

    df = orca.get_table(cfg["table"]).local
    latest_years = df.groupby(type_col)[year_col].max()
    target = orca.get_table(cfg["target_table"])
    history_tables = orca.get_injectable("history_year_tables") if orca.is_injectable("history_year_tables") else set()
    target_is_history_year = cfg["target_table"] in history_tables
    if target_is_history_year and cfg["table"] not in history_tables:
        print(f"WARNING: {cfg['table']} is keyed by the regular {cfg['target_table']} ids, so it is not "
              f"attached to the history year {cfg['target_table']}; add history_year_{cfg['table']} "
              f"to data_sources.yaml to use it.")

    for obs_type in cfg["types"]:
        if obs_type not in latest_years:
            raise ValueError(f"{cfg['table']} has no rows of type {obs_type!r}.")
        year = latest_years[obs_type].item()
        orca.add_injectable(f"observed_{obs_type}_year", year)

        col_name = f"{OBSERVED_PREFIX}{obs_type}"
        if target_is_history_year and cfg["table"] not in history_tables:
            # the observed ids belong to the regular geography; the same id can be a different
            # place in the history year table, so nothing is attached rather than the wrong values
            orca.add_column(cfg["target_table"], col_name, pd.Series(np.nan, index=target.index))
            continue
        observed = (
            df.loc[(df[type_col] == obs_type) & (df[year_col] == year)]
            .set_index(id_col)[cfg["value_column"]]
            .rename(col_name)
        )
        if id_col in target.columns:
            # the observed ids key a column on the target table rather than its index, e.g. the
            # base geography id when the target table is indexed by a finer id of its own
            observed = misc.reindex(observed, target[id_col])
        orca.add_column(cfg["target_table"], col_name, observed)


def _missing_table_files(project_dir) -> list[Path]:
    """Files this run will load from data_sources.yaml that are missing from disk."""
    cfg = config.load_yaml("data_sources.yaml", project_dir)
    data_dir_path = Path.joinpath(project_dir, orca.get_injectable('data_dir'))
    paths = (data_dir_path / file_name for _, file_name, _ in _active_sources(cfg))
    return [path for path in paths if not path.exists()]


def check_for_missing_tables(project_dir):
    """Return True if any table this run will load from data_sources.yaml is missing from disk."""
    return bool(_missing_table_files(project_dir))


def unzip_data_archive(project_dir):
    """Extract each data archive that covers a missing table; leftover gaps are caught by schema validation later.

    An archive is extracted into its own folder, so a scenario's baseline archive (an absolute path in
    the merged data_sources.yaml) restores the baseline's data dir and `scenario_data_archive` the scenario's.
    """
    missing_dirs = {path.parent for path in _missing_table_files(project_dir)}
    if not missing_dirs:
        return

    cfg = config.load_yaml("data_sources.yaml", project_dir)
    data_dir_path = Path.joinpath(project_dir, orca.get_injectable('data_dir'))
    for key in ("data_archive", "scenario_data_archive"):
        if not cfg.get(key):
            continue
        archive_path = data_dir_path / cfg[key]
        if not archive_path.exists() or archive_path.parent not in missing_dirs:
            continue

        print(f"Extracting data archive: {archive_path}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(archive_path.parent)


@orca.step("load_data")
def load_data():
    """Load and register all project tables with orca."""
    project_dir = orca.get_injectable('project_dir')
    unzip_data_archive(project_dir)
    register_tables(project_dir)

    variables = config.load_yaml("variables.yaml", project_dir)
    base_table = variables["base_geography"]["table"]
    # tables a project supplies itself (zones, say) are already registered and are left alone
    registered_tables = set(orca.list_tables())
    for geography_name, geography_id in geography.geography_level_keys(variables).items():
        if geography_name == base_table or geography_name in registered_tables:
            continue
        register_aggregation_table(geography_name, geography_id, base_table)

    _register_observed_data(project_dir)

@orca.step()
def build_networks(project_dir):
    cfg = config.load_yaml("networks.yaml", project_dir)

    try:
        pdna.network.reserve_num_graphs(cfg["reserve_num_graphs"])
    except Exception:
        pass

    x_col, y_col, node_id_col = cfg["x_column"], cfg["y_column"], cfg["node_id_column"]
    from_col, to_col = cfg["from_column"], cfg["to_column"]

    nodes = orca.get_table(cfg["nodes_table"]).local
    edges = orca.get_table(cfg["edges_table"]).local
    print('Number of nodes is %s.' % len(nodes))
    print('Number of edges is %s.' % len(edges))
    net = pdna.Network(nodes[x_col], nodes[y_col], edges[from_col], edges[to_col],
                        edges[[cfg["weight_column"]]], twoway=False)

    precompute_distance = cfg["precompute_distance"]
    print('Precomputing network for distance %s.' % precompute_distance)
    print('Network precompute starting.')
    net.precompute(precompute_distance)
    print('Network precompute done.')

    b = orca.get_table(cfg["onto_table"]).local
    b[node_id_col] = net.get_node_ids(b[x_col], b[y_col])
    orca.add_injectable("net", net)
    for poi_table in cfg.get("poi_tables", []):
        get_node_ids(net, poi_table, x_col, y_col, node_id_col)

    edge_type_col = cfg.get("edge_type_column")
    if not edge_type_col or edge_type_col not in edges.columns:
        return

    for edge_type in edges[edge_type_col].unique():
        to_nodes = edges[edges[edge_type_col] == edge_type][to_col].values
        from_nodes = edges[edges[edge_type_col] == edge_type][from_col].values
        relevant_nodes = np.unique(np.concatenate([to_nodes, from_nodes]))
        b['%s_node' % edge_type] = b[node_id_col].isin(relevant_nodes).astype('int').astype('float')

    for flag_name, edge_type_flags in cfg.get("node_flags", {}).items():
        present = [flag for flag in edge_type_flags if flag in b.columns]
        if not present:
            print(f'Skipping {flag_name}: none of its edge types are present.')
            continue
        b[flag_name] = (b[present].sum(axis=1) > 0).astype(int).astype('float')

def get_node_ids(net, table, x_col, y_col, node_id_col):
    table_df = orca.get_table(table).to_frame([x_col, y_col])
    table_df[node_id_col] = net.get_node_ids(table_df[x_col], table_df[y_col])
    orca.add_column(table, node_id_col, table_df[node_id_col], cache = True, cache_scope = 'forever')