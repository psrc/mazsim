import orca
import pandas as pd


#----------------------------------------------------------------------------------------
# Custom block variables
#----------------------------------------------------------------------------------------

@orca.column('blocks', 'vacant_housing_units')
def vacant_housing_units(blocks, households):
    return blocks.total_housing_units.sub(
        households.block_id.value_counts(), fill_value=0)

# household lcm capacity variable
@orca.column("blocks", "housing_unit_capacity", cache=True)
def housing_unit_capacity(blocks, block_capacity):
    return block_capacity.housing_unit_capacity.reindex(blocks.index).fillna(0).astype("int32")

# housing unit lcm capacity variable
@orca.column('blocks', 'vacant_hu_spaces')
def vacant_hu_spaces(blocks, housing_units):
    return blocks.housing_unit_capacity.sub(
        housing_units.block_id.value_counts(), fill_value=0).clip(lower=0)

@orca.column("blocks", "job_capacity", cache=True)
def job_capacity(blocks, block_capacity):
    return block_capacity.job_capacity.reindex(blocks.index).fillna(0).astype("int32")

# job lcm capacity variable
@orca.column('blocks', 'vacant_job_spaces', cache=False, cache_scope = 'step')
def vacant_job_spaces(blocks, jobs):
    return blocks.job_capacity.sub(
        jobs.block_id.value_counts(), fill_value=0).clip(lower=0)


#----------------------------------------------------------------------------------------
# Custom household variables
#----------------------------------------------------------------------------------------

@orca.column("households", "income_quartile", cache=True, cache_scope="iteration")
def income_quartile(households):
    return pd.qcut(households.income, 4, labels=False) + 1


#----------------------------------------------------------------------------------------
# Custom housing unit variables
#----------------------------------------------------------------------------------------

@orca.column("housing_units", "built_after_2010", cache=True, cache_scope="iteration")
def built_after_2010(housing_units):
    return (housing_units.year_built >= 2010).astype("int8")

@orca.column("housing_units", "built_prev_10_years", cache=True, cache_scope="iteration")
def built_prev_10_years(housing_units):
    year = orca.get_injectable("year")
    return (housing_units.year_built >= year - 10).astype("int8")

#----------------------------------------------------------------------------------------
# Custom job variables
#----------------------------------------------------------------------------------------

# Maps 2-digit NAICS sector_id to an aggregated sector category for the jobs.aggr_sector_id column.
aggr_sector_map = {
  11: 1,  # Agriculture, Forestry, Fishing and Hunting
  21: 1,  # Mining, Quarrying, and Oil and Gas Extraction
  22: 1,  # Utilities
  23: 1,  # Construction
  3133: 2,  # Manufacturing
  42: 2,  # Wholesale Trade
  4445: 3,  # Retail Trade
  4849: 2,  # Transportation and Warehousing
  51: 4,  # Information
  52: 4,  # Finance and Insurance
  53: 4,  # Real Estate and Rental and Leasing
  54: 4,  # Professional, Scientific, and Technical Services
  55: 4,  # Management of Companies and Enterprises
  56: 4,  # Administrative and Support and Waste Management and Remediation Services
  61: 5,  # Educational Services
  62: 5,  # Health Care and Social Assistance
  71: 3,  # Arts, Entertainment, and Recreation
  72: 3,  # Accommodation and Food Services
  81: 4,  # Other Services (except Public Administration)
  98: 5,  # Government
}

@orca.column("jobs", "aggr_sector_id", cache=True, cache_scope="iteration")
def aggr_sector_id(jobs):
    return jobs.sector_id.map(aggr_sector_map)