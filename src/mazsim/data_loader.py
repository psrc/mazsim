"""Validate the project's CSV tables against its data_model.py schemas and register them with orca."""

import zipfile
from pathlib import Path
import numpy as np
import orca
import pandana as pdna  # type: ignore[import-not-found]
import pandas as pd
import pandera.pandas as pa

from mazsim import config

# observed counts land on the base geography as obs_<type>; variables.yaml aggregates them to sum_obs_<type>
OBSERVED_PREFIX = "obs_"


def _validate(table_models, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against `table_name`'s pandera schema, collecting every failure before raising."""
    if table_name not in table_models:
        raise KeyError(f"{table_name}: no schema in the project's data model TABLE_MODELS.")
    try:
        return table_models[table_name].validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"{table_name}: schema validation failed\n{exc.failure_cases}") from exc


def register_tables(project_dir: Path) -> None:
    """Load every table in data_sources.yaml, validate it against the project's data model, and register it."""
    data_dir = project_dir / orca.get_injectable('data_dir')
    data_model = config.load_data_model(project_dir)

    settings = config.load_yaml("data_sources.yaml", project_dir)

    for entry in settings["data_sources"]:
        ((table_name, filename),) = entry.items()

        df = pd.read_csv(data_dir / filename)
        df = _validate(data_model.TABLE_MODELS, table_name, df)

        index_col = data_model.TABLE_INDEXES.get(table_name)
        if index_col:
            df = df.set_index(index_col)

        orca.add_table(table_name, df)


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

    df = orca.get_table(cfg["table"]).local
    latest_years = df.groupby(type_col)[year_col].max()

    for obs_type in cfg["types"]:
        if obs_type not in latest_years:
            raise ValueError(f"{cfg['table']} has no rows of type {obs_type!r}.")
        year = latest_years[obs_type].item()
        orca.add_injectable(f"observed_{obs_type}_year", year)

        col_name = f"{OBSERVED_PREFIX}{obs_type}"
        observed = (
            df.loc[(df[type_col] == obs_type) & (df[year_col] == year)]
            .set_index(cfg["id_column"])[cfg["value_column"]]
            .rename(col_name)
        )
        orca.add_column(cfg["target_table"], col_name, observed)


def check_for_missing_tables(project_dir):
    """Return True if any table listed in data_sources.yaml is missing from disk."""
    cfg = config.load_yaml("data_sources.yaml", project_dir)
    data_dir_path = Path.joinpath(project_dir, orca.get_injectable('data_dir'))
    for source in cfg.get("data_sources", []):
        for _, file_name in source.items():
            if not (data_dir_path / file_name).exists():
                return True
    return False

def unzip_data_archive(project_dir):
    """Extract the data archive if any tables are missing; leftover gaps are caught by schema validation later."""
    if not check_for_missing_tables(project_dir):
        return

    cfg = config.load_yaml("data_sources.yaml", project_dir)
    data_archive_file = cfg.get("data_archive")
    if not data_archive_file:
        return

    data_dir_path = Path.joinpath(project_dir, orca.get_injectable('data_dir'))
    archive_path = Path.joinpath(data_dir_path, data_archive_file)
    if not archive_path.exists():
        return

    print(f"Extracting data archive: {archive_path}")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir_path)


@orca.step("load_data")
def load_data():
    """Load and register all project tables with orca."""
    project_dir = orca.get_injectable('project_dir')
    unzip_data_archive(project_dir)
    register_tables(project_dir)

    variables = config.load_yaml("variables.yaml", project_dir)
    base_table = variables["base_geography"]["table"]
    for geography_name, geography_id in variables["geographic_levels"]:
        if geography_name == base_table:
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