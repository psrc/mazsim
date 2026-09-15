"""Grow the agent and space tables each simulation year to match the control totals in control_totals.yaml."""

import orca
import pandas as pd
from urbansim.developer import developer
from urbansim.models import GrowthRateTransition, transition

from mazsim import config

# geography ids are integers in this model, so newly created agents are flagged as unplaced with -1
UNPLACED = -1

# control totals use -1 to mean "no upper bound" on a segmentation column
NO_UPPER_BOUND = 99999999

# settings.yaml values of a *_ct_type setting that mean "apply the totals per subregion"
SUBREGIONAL_CT_TYPES = ("sub_ct", "subregional", "subregion")


def _config():
    return config.load_yaml("control_totals.yaml")


@orca.injectable("year")
def year():
    """Current simulation year, falling back to the base year outside of an orca iteration."""
    iter_var = orca.get_injectable("iter_var") if "iter_var" in orca.list_injectables() else None
    return iter_var if iter_var is not None else orca.get_injectable("base_year")


def is_subregional(ct_type: str) -> bool:
    return str(ct_type).lower() in SUBREGIONAL_CT_TYPES


@orca.step("subregional_ct")
def subregional_ct():
    """Normalize the control totals to the subregion column and stamp it onto the agent tables."""
    cfg = _config()
    subregion = cfg["subregion"]
    subregion_col, source_col = subregion["column"], subregion["source_column"]

    for spec in cfg["transitions"].values():
        type_injectable = spec["ct_type"]
        table_name = spec["controls_table"]
        requested = orca.get_injectable(type_injectable)
        ct_type = "reg_ct"

        if table_name in orca.list_tables():
            ct = orca.get_table(table_name).to_frame().rename(columns={source_col: subregion_col})
            orca.add_table(table_name, ct)
            if is_subregional(requested):
                if subregion_col not in ct.columns:
                    raise RuntimeError(
                        f"{type_injectable} is '{requested}' but {table_name} has no {source_col} column "
                        f"to segment on."
                    )
                ct_type = "sub_ct"

        orca.add_injectable(type_injectable, ct_type)
        print(f"Registered injectable: {type_injectable}: {ct_type}")

    if not any(is_subregional(orca.get_injectable(spec["ct_type"])) for spec in cfg["transitions"].values()):
        return

    for table_name in subregion["tables"]:
        table = orca.get_table(table_name)
        # write the subregion back as a local column so transitions carry it on cloned agents
        df = table.local.copy()
        df[subregion_col] = table.to_frame([source_col])[source_col]
        orca.add_table(table_name, df)


def update_linked_table(tbl, col_name, added, copied, removed):
    """Keep a child table (e.g. persons) in sync after rows were cloned from or dropped out of its parent."""
    table = tbl.local
    table = table.loc[~table[col_name].isin(set(removed))]

    if added is None or len(added) == 0:
        return table

    index_name = table.index.name
    id_map = pd.concat(
        [pd.Series(copied, name=col_name).reset_index(drop=True),
         pd.Series(added, name="temp_id").reset_index(drop=True)],
        axis=1,
    )
    new_rows = id_map.merge(table.reset_index(), on=col_name)
    new_rows = new_rows.drop(columns=[col_name, index_name or "index"])
    new_rows = new_rows.rename(columns={"temp_id": col_name})

    starting_index = table.index.max() + 1
    new_rows.index = pd.RangeIndex(starting_index, starting_index + len(new_rows), name=index_name)

    return pd.concat([table, new_rows])


def _apply_linked_tables(linked_tables, added, copied, removed):
    for table_name, (table, col) in linked_tables.items():
        updated_linked = update_linked_table(table, col, added, copied, removed)
        orca.add_table(table_name, updated_linked)
        print(f"{table_name} now has {len(updated_linked):,} rows.")


def _flag_new_agents(df, added, location_fname, year_built_column, current_year):
    if len(added) == 0:
        return df
    df.loc[added, location_fname] = UNPLACED
    if year_built_column:
        df.loc[added, year_built_column] = current_year
    return df


def prepare_control_totals(agent_controls, totals_column, ct_type):
    """Reshape a control total table into the year-indexed, `total`-valued frame TabularTotalsTransition wants."""
    try:
        ct = agent_controls.to_frame()
    except AttributeError:
        ct = agent_controls.copy()

    if ct.index.name == "year":
        ct = ct.reset_index()

    ct = ct.replace(-1, NO_UPPER_BOUND)
    ct = ct.rename(columns={totals_column: "total"})

    if not is_subregional(ct_type):
        # a regional run ignores the subregional segmentation and matches one total per year
        ct = ct.groupby("year", as_index=False)["total"].sum()

    return ct.set_index("year")


def control_total_transition(agents, agent_controls, totals_column, ct_type, current_year,
                             location_fname, linked_tables=None, year_built_column=None):
    """Add or remove agents so each control total segment matches its target for `current_year`."""
    linked_tables = linked_tables or {}
    ct = prepare_control_totals(agent_controls, totals_column, ct_type)

    if current_year not in ct.index:
        raise ValueError(
            f"No control totals for {agents.name} in {current_year}; "
            f"available years are {ct.index.min()}-{ct.index.max()}."
        )

    agent_df = agents.to_frame(agents.local_columns)
    # segmentation columns such as subregion_id are computed columns, not part of the agent table itself
    for col in ct.columns.drop("total"):
        if col not in agent_df.columns:
            agent_df[col] = agents[col]

    print(f"{agents.name} has {len(agent_df):,} rows before the transition.")

    tran = transition.TabularTotalsTransition(ct, "total")
    updated, added, copied, removed = tran.transition(agent_df, current_year)

    updated = _flag_new_agents(updated, added, location_fname, year_built_column, current_year)
    _apply_linked_tables(linked_tables, added, copied, removed)

    print(f"{agents.name} has {len(updated):,} rows after the transition.")
    orca.add_table(agents.name, updated[agents.local_columns])


def growth_rate_transition(tbl, rate, current_year, location_fname, linked_tables=None, year_built_column=None):
    """Grow a table by a flat annual rate when no control totals are available."""
    linked_tables = linked_tables or {}
    df_base = tbl.to_frame(tbl.local_columns)
    print(f"{tbl.name} has {len(df_base):,} rows before the transition.")

    df, added, copied, removed = GrowthRateTransition(rate).transition(df_base, None)

    df = _flag_new_agents(df, added, location_fname, year_built_column, current_year)
    _apply_linked_tables(linked_tables, added, copied, removed)

    print(f"{tbl.name} has {len(df):,} rows after the transition.")
    orca.add_table(tbl.name, df)


def _run_transition(agents, agents_name, rate_injectable, controls_table, totals_column, ct_type,
                    current_year, linked_tables=None, year_built_column=None):
    """Dispatch to the growth rate transition when a rate is configured, otherwise to the control totals."""
    location_fname = orca.get_injectable("geography_id")

    if rate_injectable and rate_injectable in orca.list_injectables():
        rate = orca.get_injectable(rate_injectable)
        print(f"Transitioning {agents_name} by the configured growth rate of {rate * 100:.2f}%.")
        growth_rate_transition(agents, rate, current_year, location_fname,
                               linked_tables=linked_tables, year_built_column=year_built_column)
        return

    if controls_table in orca.list_tables():
        control_total_transition(agents, orca.get_table(controls_table), totals_column, ct_type,
                                 current_year, location_fname, linked_tables=linked_tables,
                                 year_built_column=year_built_column)
        return

    raise RuntimeError(
        f"Cannot transition {agents_name}: set the '{rate_injectable}' setting in settings.yaml "
        f"or add the '{controls_table}' table to data_sources.yaml."
    )


def _run_configured_transition(step_name, current_year):
    """Look the step's tables and columns up in control_totals.yaml, then run its transition."""
    cfg = _config()
    spec = cfg["transitions"][step_name]
    agents_name = spec["agents"]
    linked_tables = {
        name: (orca.get_table(name), key)
        for name, key in (spec.get("linked_tables") or {}).items()
    }

    _run_transition(
        orca.get_table(agents_name),
        agents_name,
        spec.get("growth_rate"),
        spec["controls_table"],
        spec["totals_column"],
        orca.get_injectable(spec["ct_type"]),
        current_year,
        linked_tables=linked_tables,
        year_built_column=cfg["year_built_column"] if spec.get("set_year_built") else None,
    )


@orca.step("household_control_totals")
def household_control_totals(year):
    """Match the household count to the annual household control totals, cloning persons along with households."""
    _run_configured_transition("household_control_totals", year)


@orca.step("job_control_totals")
def job_control_totals(year):
    """Match the job count to the annual job control totals."""
    _run_configured_transition("job_control_totals", year)


def current_vacancy(agent_table, space_table):
    return 1 - (len(orca.get_table(agent_table)) / len(orca.get_table(space_table)))


def _target_vacancy_rate(agent_table, space_table, rate_injectable):
    if rate_injectable in orca.list_injectables():
        return orca.get_injectable(rate_injectable)
    # without a configured target, hold vacancy where it is so no spaces get built
    return current_vacancy(agent_table, space_table)


@orca.step("housing_unit_control_totals")
def housing_unit_control_totals(year):
    """Build enough new spaces to hit the target vacancy rate given the current agent count."""
    cfg = _config()
    spec = cfg["vacancy_transitions"]["housing_unit_control_totals"]
    subregion_col = cfg["subregion"]["column"]
    year_built_column = cfg["year_built_column"]

    agents = orca.get_table(spec["agents"])
    spaces = orca.get_table(spec["spaces"])
    ct_type = orca.get_injectable(spec["ct_type"])

    location_fname = orca.get_injectable("geography_id")
    vacancy_rate = _target_vacancy_rate(spec["agents"], spec["spaces"], spec["vacancy_rate"])

    if not is_subregional(ct_type):
        target_new_spaces = developer.Developer.compute_units_to_build(
            len(agents), len(spaces), vacancy_rate)
        if target_new_spaces <= 0:
            print(f"Current {spec['spaces']} vacancy already meets the target, none built.")
            return
        growth_rate = target_new_spaces / len(spaces)
        print(f"Growing {spec['spaces']} by {growth_rate * 100:.2f}%.")
        growth_rate_transition(spaces, growth_rate, year, location_fname,
                               year_built_column=year_built_column)
        return

    agent_subregions = agents[subregion_col]
    space_subregions = spaces[subregion_col]
    targets = []
    for subregion in space_subregions.unique():
        number_agents = (agent_subregions == subregion).sum()
        number_agent_spaces = (space_subregions == subregion).sum()
        subregion_rate = vacancy_rate[subregion] if isinstance(vacancy_rate, dict) else vacancy_rate
        target_new_spaces = developer.Developer.compute_units_to_build(
            number_agents, number_agent_spaces, subregion_rate)
        targets.append({"year": year, subregion_col: subregion,
                        "total": number_agent_spaces + target_new_spaces})

    control_total_transition(spaces, pd.DataFrame(targets), "total", ct_type, year,
                             location_fname, year_built_column=year_built_column)
