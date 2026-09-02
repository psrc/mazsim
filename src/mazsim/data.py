"""Validate project CSV tables against data_schema.yaml (via pandera) and register them with orca."""

from pathlib import Path
from typing import Any
import pandana as pdna

import numpy as np
import orca
import pandas as pd
import pandera.pandas as pa
import yaml

_DTYPE_MAP: dict[str, Any] = {
    "int64": "int64",
    "float64": "float64",
    "str": "str",
    "bool": "bool",
}

_CHECK_KEYS = ("ge", "le", "gt", "lt")

# Geographic level
orca.add_injectable('geography_id', 'block_id')

def _build_column(col_schema: dict[str, Any], enums: dict[str, Any]) -> pa.Column:
    """Build a pandera Column from a data_schema.yaml column entry."""
    checks = []

    enum_name = col_schema.get("enum")
    if enum_name:
        allowed_values = sorted((enums.get(enum_name) or {}).keys())
        if allowed_values:
            checks.append(pa.Check.isin(allowed_values))

    for key in _CHECK_KEYS:
        bound = col_schema.get(key)
        if bound is not None:
            checks.append(getattr(pa.Check, key)(bound))

    return pa.Column(
        _DTYPE_MAP[col_schema["dtype"]],
        checks=checks,
        nullable=col_schema.get("nullable", False),
        coerce=True,
    )


def _build_schema(table_schema: dict[str, Any], enums: dict[str, Any]) -> pa.DataFrameSchema:
    """Build a pandera DataFrameSchema from a data_schema.yaml table entry, dropping unlisted columns."""
    columns = {
        col_name: _build_column(col_schema, enums) for col_name, col_schema in table_schema["columns"].items()
    }
    return pa.DataFrameSchema(columns, strict="filter")


def _validate(table_name: str, df: pd.DataFrame, table_schema: dict[str, Any], enums: dict[str, Any]) -> pd.DataFrame:
    """Validate `df` against `table_name`'s schema, collecting every failure before raising."""
    schema = _build_schema(table_schema, enums)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"{table_name}: schema validation failed\n{exc.failure_cases}") from exc


def register_tables(project_dir: Path) -> None:
    """Load settings.yaml/data_schema.yaml, validate each table, and register it with orca."""
    configs_dir = project_dir / "configs"
    data_dir = project_dir / "data"

    settings = yaml.safe_load((configs_dir / "settings.yaml").read_text())
    schemas = yaml.safe_load((configs_dir / "data_schema.yaml").read_text())
    enums = schemas.get("enums", {})

    for entry in settings["data_sources"]:
        ((table_name, filename),) = entry.items()
        table_schema = schemas[table_name]

        df = pd.read_csv(data_dir / filename)
        df = _validate(table_name, df, table_schema, enums)

        index_col = table_schema.get("index")
        if index_col:
            df = df.set_index(index_col)

        if table_name == "blocks":
            # variable_generators.make_density_var hardcodes the column name "sum_acres", so the
            # source column that gets aggregated must be named "acres" rather than "acres_land".
            df = df.rename(columns={"acres_land": "acres"})

        orca.add_table(table_name, df)


@orca.step("load_data")
def load_data(project_dir):
    """Load and register all project tables with orca."""
    register_tables(project_dir)


@orca.step()
def build_networks(blocks, nodes, edges, project_dir):
    network_config = yaml.safe_load((project_dir / "configs" / "networks.yaml").read_text())

    try:
        pdna.network.reserve_num_graphs(network_config["reserve_num_graphs"])
    except Exception:
        pass

    nodes, edges = nodes.local, edges.local
    print('Number of nodes is %s.' % len(nodes))
    print('Number of edges is %s.' % len(edges))
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                        edges[["weight"]], twoway=False)

    precompute_distance = network_config["precompute_distance"]
    print('Precomputing network for distance %s.' % precompute_distance)
    print('Network precompute starting.')
    net.precompute(precompute_distance)
    print('Network precompute done.')

    b = blocks.local
    b['node_id'] = net.get_node_ids(b['x'], b['y'])
    orca.add_injectable("net", net)
    get_node_ids(net, "transit_stops")

    # Adding edge type variables
    if 'edge_type' in edges.columns:
        for edge_type in edges.edge_type.unique():
            to_nodes = edges[edges.edge_type == edge_type]['to'].values
            from_nodes = edges[edges.edge_type == edge_type]['from'].values
            relevant_nodes = np.unique(np.concatenate([to_nodes, from_nodes]))
            b['%s_node' % edge_type] = b.node_id.isin(relevant_nodes).astype('int').astype('float')
        try:
            major_edge_types = network_config["major_edge_types"]
            major_no_highway_types = network_config["major_no_highway_edge_types"]
            b['motorways_node'] = (b[['motorway_node', 'motorway_link_node']].sum(axis=1) > 0).astype(int).astype('float')
            b['major_road_node'] = (b[major_edge_types].sum(axis=1) > 0).astype(int).astype('float')
            b['major_no_highway_node'] = (b[major_no_highway_types].sum(axis=1) > 0).astype(int).astype('float')
        except KeyError:
            print('Edge type not available')

def get_node_ids(net, table):
    table_df = orca.get_table(table).to_frame(['x', 'y'])
    table_df['node_id'] = net.get_node_ids(table_df['x'], table_df['y'])
    orca.add_column(table, 'node_id', table_df['node_id'], cache = True, cache_scope = 'forever')


def register_aggregation_table(table_name, table_id):
    """
    Generator function for tables representing aggregate geography.
    """
    @orca.table(table_name, cache=True)
    def func(blocks):
        geog_ids = blocks[table_id].value_counts().index.values
        df = pd.DataFrame(index=geog_ids)
        df.index.name = table_id
        return df
    return func

# Aggregate-geography tables
aggregate_geos = [('tracts', 'tract_id'),
                  ('block_groups', 'block_group_id'),
                  ('counties', 'county_id'),
                  ('zones', 'zone_id')]
for geog in aggregate_geos:
    register_aggregation_table(geog[0], geog[1])