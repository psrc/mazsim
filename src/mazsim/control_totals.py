"""Grow the household, job, and housing unit tables each simulation year to match control totals."""

import orca
import pandas as pd
from urbansim.developer import developer
from urbansim.models import GrowthRateTransition, transition

# block_id is an integer in this model, so newly created agents are flagged as unplaced with -1
UNPLACED = -1

# control totals use -1 to mean "no upper bound" on a segmentation column
NO_UPPER_BOUND = 99999999

# settings.yaml values of hh_ct_type/job_ct_type that mean "apply the totals per subregion"
SUBREGIONAL_CT_TYPES = ("sub_ct", "subregional", "subregion")

# the settings.yaml control total type injectable that governs each control total table
CT_TYPE_TABLES = {
    "hh_ct_type": "annual_household_control_totals",
    "job_ct_type": "annual_job_control_totals",
}

# tables that carry a subregion_id once a subregional run has been set up
SUBREGION_TABLES = ("households", "jobs", "housing_units", "blocks")


@orca.injectable("year")
def year():
    """Current simulation year, falling back to the base year outside of an orca iteration."""
    iter_var = orca.get_injectable("iter_var") if "iter_var" in orca.list_injectables() else None
    return iter_var if iter_var is not None else orca.get_injectable("base_year")


def is_subregional(ct_type: str) -> bool:
    return str(ct_type).lower() in SUBREGIONAL_CT_TYPES


@orca.step("subregional_ct")
def subregional_ct():
    """Normalize the control totals to subregion_id and stamp subregion_id onto the agent tables."""
    for type_injectable, table_name in CT_TYPE_TABLES.items():
        requested = orca.get_injectable(type_injectable)
        ct_type = "reg_ct"

        if table_name in orca.list_tables():
            ct = orca.get_table(table_name).to_frame().rename(columns={"county_id": "subregion_id"})
            orca.add_table(table_name, ct)
            if is_subregional(requested):
                if "subregion_id" not in ct.columns:
                    raise RuntimeError(
                        f"{type_injectable} is '{requested}' but {table_name} has no county_id column "
                        f"to segment on."
                    )
                ct_type = "sub_ct"

        orca.add_injectable(type_injectable, ct_type)
        print(f"Registered injectable: {type_injectable}: {ct_type}")

    if not any(is_subregional(orca.get_injectable(name)) for name in CT_TYPE_TABLES):
        return

    for table_name in SUBREGION_TABLES:
        table = orca.get_table(table_name)
        # write subregion_id back as a local column so transitions carry it on cloned agents
        df = table.local.copy()
        df["subregion_id"] = table.to_frame(["county_id"])["county_id"]
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


def _flag_new_agents(df, added, location_fname, set_year_built, current_year):
    if len(added) == 0:
        return df
    df.loc[added, location_fname] = UNPLACED
    if set_year_built:
        df.loc[added, "year_built"] = current_year
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
                             location_fname, linked_tables=None, set_year_built=False):
    """Add or remove agents so each control total segment matches its target for `current_year`."""
    linked_tables = linked_tables or {}
    ct = prepare_control_totals(agent_controls, totals_column, ct_type)

    if current_year not in ct.index:
        raise ValueError(
            f"No control totals for {agents.name} in {current_year}; "
            f"available years are {ct.index.min()}-{ct.index.max()}."
        )

    agent_df = agents.to_frame(agents.local_columns)
    # segmentation columns such as county_id are computed columns, not part of the agent table itself
    for col in ct.columns.drop("total"):
        if col not in agent_df.columns:
            agent_df[col] = agents[col]

    print(f"{agents.name} has {len(agent_df):,} rows before the transition.")

    tran = transition.TabularTotalsTransition(ct, "total")
    updated, added, copied, removed = tran.transition(agent_df, current_year)

    updated = _flag_new_agents(updated, added, location_fname, set_year_built, current_year)
    _apply_linked_tables(linked_tables, added, copied, removed)

    print(f"{agents.name} has {len(updated):,} rows after the transition.")
    orca.add_table(agents.name, updated[agents.local_columns])


def growth_rate_transition(tbl, rate, current_year, location_fname, linked_tables=None, set_year_built=False):
    """Grow a table by a flat annual rate when no control totals are available."""
    linked_tables = linked_tables or {}
    df_base = tbl.to_frame(tbl.local_columns)
    print(f"{tbl.name} has {len(df_base):,} rows before the transition.")

    df, added, copied, removed = GrowthRateTransition(rate).transition(df_base, None)

    df = _flag_new_agents(df, added, location_fname, set_year_built, current_year)
    _apply_linked_tables(linked_tables, added, copied, removed)

    print(f"{tbl.name} has {len(df):,} rows after the transition.")
    orca.add_table(tbl.name, df)


def _run_transition(agents, agents_name, rate_injectable, controls_table, totals_column, ct_type,
                    current_year, linked_tables=None, set_year_built=False):
    """Dispatch to the growth rate transition when a rate is configured, otherwise to the control totals."""
    location_fname = orca.get_injectable("geography_id")

    if rate_injectable in orca.list_injectables():
        rate = orca.get_injectable(rate_injectable)
        print(f"Transitioning {agents_name} by the configured growth rate of {rate * 100:.2f}%.")
        growth_rate_transition(agents, rate, current_year, location_fname,
                               linked_tables=linked_tables, set_year_built=set_year_built)
        return

    if controls_table in orca.list_tables():
        control_total_transition(agents, orca.get_table(controls_table), totals_column, ct_type,
                                 current_year, location_fname, linked_tables=linked_tables,
                                 set_year_built=set_year_built)
        return

    raise RuntimeError(
        f"Cannot transition {agents_name}: set the '{rate_injectable}' setting in settings.yaml "
        f"or add the '{controls_table}' table to data_sources.yaml."
    )


@orca.step("household_control_totals")
def household_control_totals(households, persons, year, hh_ct_type):
    """Match the household count to the annual household control totals, cloning persons along with households."""
    _run_transition(
        households,
        "households",
        "household_growth_rate",
        "annual_household_control_totals",
        "total_number_of_households",
        hh_ct_type,
        year,
        linked_tables={"persons": (persons, "household_id")},
    )


@orca.step("job_control_totals")
def job_control_totals(jobs, year, job_ct_type):
    """Match the job count to the annual job control totals."""
    _run_transition(
        jobs,
        "jobs",
        "job_growth_rate",
        "annual_job_control_totals",
        "total_number_of_jobs",
        job_ct_type,
        year,
    )


def current_vacancy(agent_table, space_table):
    return 1 - (len(orca.get_table(agent_table)) / len(orca.get_table(space_table)))


def _target_vacancy_rate():
    if "housing_vacancy_rate" in orca.list_injectables():
        return orca.get_injectable("housing_vacancy_rate")
    # without a configured target, hold vacancy where it is so no units get built
    return current_vacancy("households", "housing_units")


@orca.step("housing_unit_control_totals")
def housing_unit_control_totals(households, housing_units, year, hh_ct_type):
    """Build enough new housing units to hit the target vacancy rate given the current household count."""
    location_fname = orca.get_injectable("geography_id")
    vacancy_rate = _target_vacancy_rate()

    if not is_subregional(hh_ct_type):
        target_new_spaces = developer.Developer.compute_units_to_build(
            len(households), len(housing_units), vacancy_rate)
        if target_new_spaces <= 0:
            print("Current housing vacancy already meets the target, no units built.")
            return
        growth_rate = target_new_spaces / len(housing_units)
        print(f"Growing housing units by {growth_rate * 100:.2f}%.")
        growth_rate_transition(housing_units, growth_rate, year, location_fname, set_year_built=True)
        return

    household_subregions = households["subregion_id"]
    unit_subregions = housing_units["subregion_id"]
    targets = []
    for subregion in unit_subregions.unique():
        number_agents = (household_subregions == subregion).sum()
        number_agent_spaces = (unit_subregions == subregion).sum()
        subregion_rate = vacancy_rate[subregion] if isinstance(vacancy_rate, dict) else vacancy_rate
        target_new_spaces = developer.Developer.compute_units_to_build(
            number_agents, number_agent_spaces, subregion_rate)
        targets.append({"year": year, "subregion_id": subregion,
                        "total": number_agent_spaces + target_new_spaces})

    control_total_transition(housing_units, pd.DataFrame(targets), "total", hh_ct_type, year,
                             location_fname, set_year_built=True)
