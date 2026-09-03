"""Validate project CSV tables against data_model.py schemas (via pandera) and register them with orca."""

from pathlib import Path

import numpy as np
import orca
import pandana as pdna  # type: ignore[import-not-found]
import pandas as pd
import pandera.pandas as pa
import yaml

from mazsim.data_model import TABLE_INDEXES, TABLE_MODELS

# Geographic level
orca.add_injectable('geography_id', 'block_id')


def _validate(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against `table_name`'s pandera schema, collecting every failure before raising."""
    model = TABLE_MODELS[table_name]
    try:
        return model.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"{table_name}: schema validation failed\n{exc.failure_cases}") from exc


def register_tables(project_dir: Path) -> None:
    """Load settings.yaml, validate each table against its data_model.py schema, and register it with orca."""
    configs_dir = project_dir / "configs"
    data_dir = project_dir / "data"

    settings = yaml.safe_load((configs_dir / "settings.yaml").read_text())

    for entry in settings["data_sources"]:
        ((table_name, filename),) = entry.items()

        df = pd.read_csv(data_dir / filename)
        df = _validate(table_name, df)

        index_col = TABLE_INDEXES.get(table_name)
        if index_col:
            df = df.set_index(index_col)

        if table_name == "blocks":
            # variable_generators.make_density_var hardcodes the column name "sum_acres", so the
            # source column that gets aggregated must be named "acres" rather than "acres_land".
            df = df.rename(columns={"acres_land": "acres"})

        orca.add_table(table_name, df)


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


@orca.step("load_data")
def load_data(project_dir):
    """Load and register all project tables with orca."""
    register_tables(project_dir)
    # Aggregate-geography tables
    aggregate_geos = [('tracts', 'tract_id'),
                    ('block_groups', 'block_group_id'),
                    ('counties', 'county_id'),
                    ('zones', 'zone_id')]
    for geog in aggregate_geos:
        register_aggregation_table(geog[0], geog[1])


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