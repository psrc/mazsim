"""Register derived orca variables: geography ids, aggregations, ratios, disaggregations, skims, and pandana access."""

from pathlib import Path
from typing import Any

import numpy as np
import orca
import pandas as pd
import yaml
from urbansim.utils import misc
from variable_generators import generators


def _load_config(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "variables.yaml"
    return yaml.safe_load(config_path.read_text())


def register_geography_ids() -> None:
    """Register county_id/tract_id/block_group_id columns derived from the blocks index."""

    @orca.column("blocks", "county_id", cache=True)
    def county_id(blocks):
        blocks = blocks.to_frame(blocks.local_columns)
        return pd.Series(blocks.index.values, index=blocks.index).astype(str).str.slice(0, 6).astype("int64")

    @orca.column("blocks", "tract_id", cache=True)
    def tract_id(blocks):
        blocks = blocks.to_frame(blocks.local_columns)
        return pd.Series(blocks.index.values, index=blocks.index).astype(str).str.slice(0, 12).astype("int64")

    @orca.column("blocks", "block_group_id", cache=True)
    def block_group_id(blocks):
        blocks = blocks.to_frame(blocks.local_columns)
        return pd.Series(blocks.index.values, index=blocks.index).astype(str).str.slice(0, 13).astype("int64")


def register_block_variables() -> None:
    """Register derived block-level variables"""

    @orca.column("blocks", "housing_unit_capacity", cache=True)
    def housing_unit_capacity(blocks, block_capacity):
        return block_capacity.housing_unit_capacity.reindex(blocks.index).fillna(0).astype("int32")

    @orca.column("blocks", "job_capacity", cache=True)
    def job_capacity(blocks, block_capacity):
        return block_capacity.job_capacity.reindex(blocks.index).fillna(0).astype("int32")


def register_household_variables() -> None:
    """Register derived household variables"""

    @orca.column("households", "income_quartile", cache=True, cache_scope="iteration")
    def income_quartile(households):
        return pd.qcut(households.income, 4, labels=False) + 1


def register_job_variables(config: dict[str, Any]) -> None:
    """Register derived job variables"""
    aggr_sector_map = config["aggr_sector_map"]

    @orca.column("jobs", "aggr_sector_id", cache=True, cache_scope="iteration")
    def aggr_sector_id(jobs):
        return jobs.sector_id.map(aggr_sector_map)


def fillna_median(series: pd.Series) -> pd.Series:
    return series.fillna(series.median())


def register_agent_geography_ids(config: dict[str, Any]) -> None:
    """Broadcast block_group_id/tract_id/county_id/zone_id from blocks onto agent tables via block_id."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]
    agents = [agent for agent in config["variables_to_aggregate"] if agent != "blocks"]

    for agent in agents:
        agent_columns = orca.get_table(agent).columns
        for _, geography_id in geographic_levels:
            if geography_id in agent_columns:
                continue
            generators.make_disagg_var("blocks", agent, geography_id, "block_id", name_based_on_geography=False)


def register_aggregation_variables(config: dict[str, Any], generated_variables: set[str]) -> None:
    """Register total_<agent> size variables and mean/median/std/sum attribute variables at each geography."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]
    aggregation_functions = config["aggregation_functions"]
    variables_to_aggregate = config["variables_to_aggregate"]
    sum_vars = set(config["sum_vars"])

    for agent, variables in variables_to_aggregate.items():
        for geography_name, geography_id in geographic_levels:
            if geography_name == agent:
                continue

            generators.make_size_var(agent, geography_name, geography_id)
            generated_variables.add("total_" + agent)

            for var in variables:
                for aggregation_function in aggregation_functions:
                    if aggregation_function == "sum":
                        if var not in sum_vars:
                            continue
                        generators.make_agg_var(agent, geography_name, geography_id, var, aggregation_function)
                    else:
                        generators.make_agg_var(
                            agent, geography_name, geography_id, var, aggregation_function, fillna_median
                        )
                    generated_variables.add(aggregation_function + "_" + var)


def register_proportion_variables(config: dict[str, Any], generated_variables: set[str]) -> None:
    """Register prop_<var>_<category> variables for discrete variables with more than 5000 occurrences."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]
    discrete_variables = config["discrete_variables"]

    for agent, discrete_vars in discrete_variables.items():
        agents = orca.get_table(agent)
        for var in discrete_vars:
            agents_by_cat = agents[var].value_counts()
            cats_to_measure = agents_by_cat[agents_by_cat > 5000].index.values
            for cat in cats_to_measure:
                for geography_name, geography_id in geographic_levels:
                    generators.make_proportion_var(agent, geography_name, geography_id, var, cat)
                generated_variables.add("prop_%s_%s" % (var, int(cat)))


def register_ratio_and_density_variables(config: dict[str, Any], generated_variables: set[str]) -> None:
    """Register jobs/households and households/housing_units ratios, plus density_<agent>, at each geography."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]
    discrete_variables = config["discrete_variables"]

    for geography_name, _ in geographic_levels:
        generators.make_ratio_var("jobs", "households", geography_name)
        generated_variables.add("ratio_jobs_to_households")

        generators.make_ratio_var("households", "housing_units", geography_name)
        generated_variables.add("ratio_households_to_housing_units")

        for agent in discrete_variables:
            generators.make_density_var(agent, geography_name)
            generated_variables.add("density_%s" % agent)


def register_block_disaggregations(config: dict[str, Any], generated_variables: set[str]) -> None:
    """Disaggregate geography-level, node-level, zone-level, and block-group-level variables down to blocks."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]

    for geography_name, geography_id in geographic_levels:
        if geography_name == "blocks":
            continue
        for var in generated_variables:
            generators.make_disagg_var(geography_name, "blocks", var, geography_id)

    for var in orca.get_table("nodes").columns:
        if var not in ("x", "y"):
            generators.make_disagg_var("nodes", "blocks", var, "node_id")

    if "travel_data" in orca.list_tables():
        for var in orca.get_table("zones").columns:
            generators.make_disagg_var("zones", "blocks", var, "zone_id", name_based_on_geography=True)

    for var in orca.get_table("block_groups").columns:
        generators.make_disagg_var("block_groups", "blocks", var, "block_group_id", name_based_on_geography=True)


def register_geographic_dummies(config: dict[str, Any]) -> None:
    """Register block-level dummy columns for each distinct value of the configured geography columns."""
    for geog_var in config["geog_vars_to_dummify"]:
        geog_ids = np.unique(orca.get_table("blocks")[geog_var])
        for geog_id in geog_ids:
            generators.make_dummy_variable("blocks", geog_var, geog_id)


def register_skim_zone_variable(table_name: str, column_name: str, tt: int, var: str, column_time: str):
    """Register a zone-level column summing `var` reachable within `tt` minutes via the `column_time` skim."""

    @orca.column(table_name, column_name, cache=True, cache_scope="iteration")
    def column_func(travel_data, zones):
        data = misc.compute_range(travel_data.to_frame(), zones[var], column_time, tt, agg=np.sum)
        return pd.Series(data, index=zones.index).fillna(0)

    return column_func


def register_skim_variables(config: dict[str, Any]) -> None:
    """Register zone-level accessibility variables for every travel-time/skim-column/target-variable combination."""
    skims = config["skims"]
    for column_time in skims["columns"]:
        for tt in skims["travel_times"]:
            for var in skims["variables"]:
                column_name = f"{var}_{tt}_minutes_{column_time}"
                register_skim_zone_variable("zones", column_name, tt, var, column_time)


def register_pandana_access_variable(
    column_name: str,
    onto_table: str,
    pois_table: str,
    variable_to_summarize: str,
    distance: int,
    agg_type: str = "sum",
    decay: str = "linear",
    log: bool = True,
):
    """Register a pandana network-distance accessibility column."""

    @orca.column(onto_table, column_name, cache=True, cache_scope="iteration")
    def column_func():
        net = orca.get_injectable("net")
        table = orca.get_table(pois_table).to_frame(["node_id", variable_to_summarize])
        df = orca.get_table(onto_table).to_frame("node_id")
        net.set(table.node_id, variable=table[variable_to_summarize])
        results = net.aggregate(distance, type=agg_type, decay=decay)
        if log:
            results = np.log1p(results)
        return misc.reindex(results, df.node_id)

    return column_func


def register_accessibility_dummy(table: str, col_name: str, variable: str):
    """Register a 0/1 dummy column marking where `variable` is greater than zero."""

    @orca.column(table, col_name, cache=True, cache_scope="iteration")
    def func():
        df = orca.get_table(table).to_frame(variable)
        return (df[variable] > 0).astype("int32")

    return func


def register_pandana_variables(config: dict[str, Any]) -> None:
    """Register pandana-based accessibility variables (and their dummies) across distances/decays."""
    pandana_config = config["pandana"]
    distances = range(
        pandana_config["distances"]["start"], pandana_config["distances"]["stop"], pandana_config["distances"]["step"]
    )
    agg_types = pandana_config["agg_types"]
    decay_types = pandana_config["decay_types"]
    variables_to_aggregate = pandana_config["variables_to_aggregate"]

    for distance in distances:
        for decay in decay_types:
            for variable in variables_to_aggregate:
                for agg_type in agg_types:
                    var_name = "_".join([variable, agg_type, str(distance), decay])
                    register_pandana_access_variable(
                        var_name, "blocks", "blocks", variable, distance, agg_type=agg_type, decay=decay
                    )
                    register_pandana_access_variable(
                        "without_log_" + var_name,
                        "blocks",
                        "blocks",
                        variable,
                        distance,
                        agg_type=agg_type,
                        decay=decay,
                        log=False,
                    )

            var_name = f"transit_stop_sum_{distance}_{decay}"
            register_pandana_access_variable(
                var_name, "blocks", "transit_stops", "hct", distance, agg_type="sum", decay=decay, log=False
            )
            register_accessibility_dummy("blocks", "is_" + var_name, var_name)


def register_ln_variable(table_name: str, column_to_ln: str):
    """Register a log1p-transformed version of a column."""
    new_col_name = "ln_" + column_to_ln

    @orca.column(table_name, new_col_name, cache=True, cache_scope="iteration")
    def column_func():
        return np.log1p(orca.get_table(table_name)[column_to_ln])

    return column_func


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def register_standardized_variable(table_name: str, column_to_s: str):
    """Register a standardized (z-score) version of a column."""
    new_col_name = "st_" + column_to_s

    @orca.column(table_name, new_col_name, cache=True, cache_scope="iteration")
    def column_func():
        return standardize(orca.get_table(table_name)[column_to_s])

    return column_func


def register_log_and_standardized_variables() -> None:
    """Register ln_/st_/st_ln_ versions of every non-id block column."""
    block_columns = orca.get_table("blocks").columns

    for var in block_columns:
        if var.startswith(("ln", "st", "st_ln")) or var.endswith("_id"):
            continue
        ln_version = "ln_" + var
        st_version = "st_" + var
        st_ln_version = "st_ln_" + var
        if ln_version not in block_columns:
            register_ln_variable("blocks", var)
        if st_version not in block_columns:
            register_standardized_variable("blocks", var)
        if st_ln_version not in block_columns:
            register_standardized_variable("blocks", ln_version)


@orca.step("register_variables")
def register_variables(project_dir: Path) -> None:
    """Register every derived orca variable: geography ids, aggregations, ratios, disaggregations, skims, and pandana access."""
    config = _load_config(project_dir)

    register_geography_ids()
    register_agent_geography_ids(config)
    register_block_variables()
    register_household_variables()
    register_job_variables(config)

    generated_variables: set[str] = set()
    register_aggregation_variables(config, generated_variables)
    register_proportion_variables(config, generated_variables)
    register_ratio_and_density_variables(config, generated_variables)
    register_geographic_dummies(config)

    # Must run before register_block_disaggregations, which reads zones.columns to
    # decide which zone-level variables (including these skims) to disaggregate to blocks.
    register_skim_variables(config)
    register_block_disaggregations(config, generated_variables)
    register_pandana_variables(config)
    register_log_and_standardized_variables()