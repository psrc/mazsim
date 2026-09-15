"""Register derived orca variables: geography ids, aggregations, ratios, disaggregations, skims, and pandana access."""

from pathlib import Path
from typing import Any

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
from variable_generators import generators

from mazsim.config import load_yaml


DERIVED_VARIABLE_GENERATORS = {
    "join": generators.make_join_var,
    "difference": generators.make_difference_var,
    "sum": generators.make_sum_var,
    "quantile": generators.make_quantile_var,
    "map": generators.make_map_var,
    "threshold": generators.make_threshold_var,
    "constant": generators.make_constant_var,
    "expr": generators.make_expression_var,
}


def _load_config(project_dir: Path) -> dict[str, Any]:
    return load_yaml("variables.yaml", project_dir)


def register_geography_ids(config: dict[str, Any]) -> None:
    """Register the parent geography id columns carved out of the base geography's index."""
    table = config["base_geography"]["table"]

    for spec in config["geography_ids"]:
        start, stop = spec["slice"]
        generators.make_index_slice_var(
            table, spec["name"], start, stop, dtype=spec.get("dtype", "int64")
        )


def register_derived_variables(config: dict[str, Any]) -> None:
    """Register every entry in the config's derived_variables list through its generator kind."""
    for entry in config.get("derived_variables", []):
        spec = dict(entry)
        table = spec.pop("table")
        name = spec.pop("name")
        kind = spec.pop("kind")

        if kind not in DERIVED_VARIABLE_GENERATORS:
            raise ValueError(
                f"{table}.{name}: unknown derived variable kind {kind!r}; "
                f"expected one of {sorted(DERIVED_VARIABLE_GENERATORS)}"
            )

        # a string `mapping` refers to a top-level key holding the lookup
        if isinstance(spec.get("mapping"), str):
            spec["mapping"] = config[spec["mapping"]]

        DERIVED_VARIABLE_GENERATORS[kind](table, name, **spec)


def fillna_median(series: pd.Series) -> pd.Series:
    return series.fillna(series.median())


def register_agent_geography_ids(config: dict[str, Any]) -> None:
    """Broadcast each geography id from the base geography onto the agent tables via the base id."""
    base_table = config["base_geography"]["table"]
    base_id = config["base_geography"]["id"]
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]
    agents = [agent for agent in config["variables_to_aggregate"] if agent != base_table]

    for agent in agents:
        agent_columns = orca.get_table(agent).columns
        for _, geography_id in geographic_levels:
            if geography_id in agent_columns:
                continue
            generators.make_disagg_var(base_table, agent, geography_id, base_id, name_based_on_geography=False)


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
    """Register the configured agent ratios, plus density_<agent>, at each geography."""
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]

    for geography_name, _ in geographic_levels:
        for numerator, denominator in config["ratio_variables"]:
            generators.make_ratio_var(numerator, denominator, geography_name)
            generated_variables.add(f"ratio_{numerator}_to_{denominator}")

        for agent in config["density_agents"]:
            generators.make_density_var(agent, geography_name)
            generated_variables.add("density_%s" % agent)


def register_base_geography_disaggregations(config: dict[str, Any], generated_variables: set[str]) -> None:
    """Disaggregate every generated variable, plus the configured extra tables, down to the base geography."""
    base_table = config["base_geography"]["table"]
    geographic_levels = [tuple(g) for g in config["geographic_levels"]]

    for geography_name, geography_id in geographic_levels:
        if geography_name == base_table:
            continue
        for var in generated_variables:
            generators.make_disagg_var(geography_name, base_table, var, geography_id)

    for spec in config["disaggregations"]:
        required = spec.get("requires_table")
        if required is not None and required not in orca.list_tables():
            continue

        from_table = spec["from"]
        exclude = set(spec.get("exclude", []))
        for var in orca.get_table(from_table).columns:
            if var in exclude:
                continue
            generators.make_disagg_var(
                from_table,
                base_table,
                var,
                spec["key"],
                name_based_on_geography=spec.get("name_based_on_geography", True),
            )


def register_geographic_dummies(config: dict[str, Any]) -> None:
    """Register base-geography dummy columns for each distinct value of the configured geography columns."""
    base_table = config["base_geography"]["table"]

    for geog_var in config["geog_vars_to_dummify"]:
        geog_ids = np.unique(orca.get_table(base_table)[geog_var])
        for geog_id in geog_ids:
            generators.make_dummy_variable(base_table, geog_var, geog_id)


def register_skim_zone_variable(zone_table: str, skim_table: str, column_name: str, tt: int, var: str, column_time: str):
    """Register a zone-level column summing `var` reachable within `tt` minutes via the `column_time` skim."""

    @orca.column(zone_table, column_name, cache=True, cache_scope="iteration")
    def column_func():
        skims = orca.get_table(skim_table).to_frame()
        zones = orca.get_table(zone_table)
        data = misc.compute_range(skims, zones[var], column_time, tt, agg=np.sum)
        return pd.Series(data, index=zones.index).fillna(0)

    return column_func


def register_skim_variables(config: dict[str, Any]) -> None:
    """Register zone-level accessibility variables for every travel-time/skim-column/target-variable combination."""
    skims = config["skims"]
    zone_table = skims["zone_table"]
    skim_table = skims["table"]

    for column_time in skims["columns"]:
        for tt in skims["travel_times"]:
            for var in skims["variables"]:
                column_name = f"{var}_{tt}_minutes_{column_time}"
                register_skim_zone_variable(zone_table, skim_table, column_name, tt, var, column_time)


def register_pandana_access_variable(
    column_name: str,
    onto_table: str,
    pois_table: str,
    variable_to_summarize: str,
    distance: int,
    node_column: str = "node_id",
    agg_type: str = "sum",
    decay: str = "linear",
    log: bool = True,
):
    """Register a pandana network-distance accessibility column."""

    @orca.column(onto_table, column_name, cache=True, cache_scope="iteration")
    def column_func():
        net = orca.get_injectable("net")
        table = orca.get_table(pois_table).to_frame([node_column, variable_to_summarize])
        df = orca.get_table(onto_table).to_frame(node_column)
        net.set(table[node_column], variable=table[variable_to_summarize])
        results = net.aggregate(distance, type=agg_type, decay=decay)
        if log:
            # signed log: aggregates of signable quantities (e.g. sum_income) can fall
            # below -1, where plain log1p yields NaN and corrupts the design matrix
            results = np.sign(results) * np.log1p(np.abs(results))
        return misc.reindex(results, df[node_column])

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
    onto_table = pandana_config["onto_table"]
    node_column = pandana_config["node_column"]
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
                        var_name, onto_table, onto_table, variable, distance,
                        node_column=node_column, agg_type=agg_type, decay=decay,
                    )
                    register_pandana_access_variable(
                        "without_log_" + var_name,
                        onto_table,
                        onto_table,
                        variable,
                        distance,
                        node_column=node_column,
                        agg_type=agg_type,
                        decay=decay,
                        log=False,
                    )

            for poi in pandana_config.get("poi_variables", []):
                var_name = poi["name_template"].format(distance=distance, decay=decay)
                register_pandana_access_variable(
                    var_name,
                    onto_table,
                    poi["pois_table"],
                    poi["variable"],
                    distance,
                    node_column=node_column,
                    agg_type=poi.get("agg_type", "sum"),
                    decay=decay,
                    log=poi.get("log", True),
                )
                dummy_prefix = poi.get("dummy_prefix")
                if dummy_prefix:
                    register_accessibility_dummy(onto_table, dummy_prefix + var_name, var_name)


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


def register_log_and_standardized_variables(config: dict[str, Any]) -> None:
    """Register ln_/st_/st_ln_ versions of every eligible column on the transform table."""
    transforms = config["transforms"]
    table = transforms["table"]
    skip_prefixes = tuple(transforms["skip_prefixes"])
    skip_suffixes = tuple(transforms["skip_suffixes"])
    columns = orca.get_table(table).columns

    for var in columns:
        if var.startswith(skip_prefixes) or var.endswith(skip_suffixes):
            continue
        ln_version = "ln_" + var
        st_version = "st_" + var
        st_ln_version = "st_ln_" + var
        if ln_version not in columns:
            register_ln_variable(table, var)
        if st_version not in columns:
            register_standardized_variable(table, var)
        if st_ln_version not in columns:
            register_standardized_variable(table, ln_version)


@orca.step("register_variables")
def register_variables(project_dir: Path) -> None:
    """Register every derived orca variable: geography ids, aggregations, ratios, disaggregations, skims, and pandana access."""
    config = _load_config(project_dir)

    register_geography_ids(config)
    register_agent_geography_ids(config)
    register_derived_variables(config)

    generated_variables: set[str] = set()
    register_aggregation_variables(config, generated_variables)
    register_proportion_variables(config, generated_variables)
    register_ratio_and_density_variables(config, generated_variables)
    register_geographic_dummies(config)

    # Must run before register_base_geography_disaggregations, which reads the zone table's columns
    # to decide which zone-level variables (including these skims) to disaggregate down.
    register_skim_variables(config)
    register_base_geography_disaggregations(config, generated_variables)
    register_pandana_variables(config)
    register_log_and_standardized_variables(config)