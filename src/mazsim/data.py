"""Validate project CSV tables against data_schema.yaml and register them with orca."""

import operator
from pathlib import Path
from typing import Any
import pandana as pdna

import numpy as np
import orca
import pandas as pd
import yaml

_DTYPE_MAP: dict[str, type] = {
    "int64": int,
    "float64": float,
    "str": str,
    "bool": bool,
}

_CONSTRAINT_OPS = {
    "ge": operator.ge,
    "le": operator.le,
    "gt": operator.gt,
    "lt": operator.lt,
}

# Geographic level
orca.add_injectable('geography_id', 'block_id')

def _coerce_column(col_name: str, series: pd.Series, dtype_name: str, nullable: bool) -> pd.Series:
    """Cast `series` to the schema dtype, raising if nulls or bad values are present."""
    null_mask = series.isna()
    if null_mask.any():
        if not nullable:
            raise ValueError(f"{col_name}: {int(null_mask.sum())} null values not allowed")
        casted = series.copy()
        try:
            casted[~null_mask] = series[~null_mask].astype(_DTYPE_MAP[dtype_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{col_name}: cannot cast to {dtype_name} ({exc})") from exc
        return casted

    try:
        return series.astype(_DTYPE_MAP[dtype_name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{col_name}: cannot cast to {dtype_name} ({exc})") from exc


def _check_enum(col_name: str, series: pd.Series, enum_name: str, enums: dict[str, Any]) -> None:
    """Raise if any non-null value in `series` falls outside the named enum's keys."""
    allowed_values = set((enums.get(enum_name) or {}).keys())
    if not allowed_values:
        return
    bad_mask = ~series.isin(allowed_values) & series.notna()
    if bad_mask.any():
        bad_values = sorted(series[bad_mask].unique())[:10]
        raise ValueError(f"{col_name}: {int(bad_mask.sum())} values not in enum {enum_name} (e.g. {bad_values})")


def _check_constraints(col_name: str, series: pd.Series, col_schema: dict[str, Any]) -> None:
    """Raise if any non-null value in `series` violates a ge/le/gt/lt bound from the schema."""
    for key, op in _CONSTRAINT_OPS.items():
        bound = col_schema.get(key)
        if bound is None:
            continue
        bad_mask = ~op(series, bound) & series.notna()
        if bad_mask.any():
            raise ValueError(f"{col_name}: {int(bad_mask.sum())} values violate {key}={bound}")


def _validate(table_name: str, df: pd.DataFrame, table_schema: dict[str, Any], enums: dict[str, Any]) -> pd.DataFrame:
    """Validate every column of `df` against `table_name`'s schema using vectorized pandas checks."""
    columns_schema = table_schema["columns"]
    expected_columns = list(columns_schema.keys())
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{table_name}: missing columns {sorted(missing_columns)}")

    df = df[expected_columns].copy()
    for col_name, col_schema in columns_schema.items():
        series = _coerce_column(col_name, df[col_name], col_schema["dtype"], col_schema.get("nullable", False))
        df[col_name] = series

        enum_name = col_schema.get("enum")
        if enum_name:
            _check_enum(col_name, series, enum_name, enums)

        _check_constraints(col_name, series, col_schema)

    return df


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
def build_networks(blocks, nodes, edges):
    try:
        pdna.network.reserve_num_graphs(2)
    except:
        pass

    nodes, edges = nodes.local, edges.local
    print('Number of nodes is %s.' % len(nodes))
    print('Number of edges is %s.' % len(edges))
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                           edges[["weight"]],twoway=False)

    precompute_distance = 3000
    print('Precomputing network for distance %s.' % precompute_distance)
    print('Network precompute starting.')
    net.precompute(precompute_distance)
    print('Network precompute done.')

    b = blocks.local
    b['node_id'] = net.get_node_ids(b['x'], b['y'])
    orca.add_injectable("net", net)

    orca.add_table("blocks", b)
    if 'transit_stops' in orca.list_tables():
        get_node_ids(net, 'transit_stops')

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