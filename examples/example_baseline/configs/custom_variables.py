import orca
import pandas as pd


#----------------------------------------------------------------------------------------
# Custom block variables
#----------------------------------------------------------------------------------------

@orca.column('blocks', 'vacant_housing_units')
def vacant_housing_units(blocks, households):
    return blocks.total_housing_units.sub(
        households.maz_id.value_counts(), fill_value=0)

# household lcm capacity variable
@orca.column("blocks", "housing_unit_capacity", cache=True, cache_scope="forever")
def housing_unit_capacity(blocks, block_capacity):
    return block_capacity.housing_unit_capacity.reindex(blocks.index).fillna(0).astype("int32")

# housing unit lcm capacity variable
@orca.column('blocks', 'vacant_hu_spaces')
def vacant_hu_spaces(blocks, housing_units):
    return blocks.housing_unit_capacity.sub(
        housing_units.maz_id.value_counts(), fill_value=0).clip(lower=0)

@orca.column("blocks", "job_capacity", cache=True, cache_scope="forever")
def job_capacity(blocks, block_capacity):
    return block_capacity.job_capacity.reindex(blocks.index).fillna(0).astype("int32")

# job lcm capacity variable
@orca.column('blocks', 'vacant_job_spaces')
def vacant_job_spaces(blocks, jobs):
    return blocks.job_capacity.sub(
        jobs.maz_id.value_counts(), fill_value=0).clip(lower=0)


TRANSIT_AREA_TRACTS = [
    153053060200,
    153053061100,
    153053061200,
    153053061300,
    153053061400,
    153053061501,
    153053061502,
    153053061601,
    153053061602,
    153053061700,
    153053940002
]

@orca.column("blocks", "transit_area", cache=True, cache_scope="forever")
def transit_area(blocks):
    return blocks["tract_id"].isin(TRANSIT_AREA_TRACTS).astype("int32")

#----------------------------------------------------------------------------------------
# Custom household variables
#----------------------------------------------------------------------------------------

@orca.column("households", "income_quartile", cache=True, cache_scope="iteration")
def income_quartile(households):
    return pd.qcut(households.income, 4, labels=False) + 1

@orca.column("households", "cars_per_adult")
def cars_per_adult(households):
    return households.cars.div(households.adults).fillna(0)


#----------------------------------------------------------------------------------------
# Custom housing unit variables
#----------------------------------------------------------------------------------------

@orca.column("housing_units", "built_after_2010", cache=True, cache_scope="step")
def built_after_2010(housing_units):
    return (housing_units.year_built >= 2010).astype("int8")

@orca.column("housing_units", "built_prev_10_years", cache=True, cache_scope="step")
def built_prev_10_years(housing_units):
    year = orca.get_injectable("year")
    return (housing_units.year_built >= year - 10).astype("int8")

@orca.column("housing_units", "sf", cache=True, cache_scope="step")
def sf(housing_units):
    return housing_units.unit_type_id.isin([1])

@orca.column("housing_units", "middle", cache=True, cache_scope="step")
def middle(housing_units):
    return housing_units.unit_type_id.isin([2])

@orca.column("housing_units", "mf", cache=True, cache_scope="step")
def mf(housing_units):
    return housing_units.unit_type_id.isin([3])

@orca.column("housing_units", "sf_built_prev_10_years", cache=True, cache_scope="step")
def sf_built_prev_10_years(housing_units):
    return (sf(housing_units) & built_prev_10_years(housing_units)).astype("int8")

@orca.column("housing_units", "middle_built_prev_10_years", cache=True, cache_scope="step")
def middle_built_prev_10_years(housing_units):
    return (middle(housing_units) & built_prev_10_years(housing_units)).astype("int8")

@orca.column("housing_units", "mf_built_prev_10_years", cache=True, cache_scope="step")
def mf_built_prev_10_years(housing_units):
    return (mf(housing_units) & built_prev_10_years(housing_units)).astype("int8")

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

#----------------------------------------------------------------------------------------
# Custom person variables
#----------------------------------------------------------------------------------------

@orca.column("persons", "worker_aggr_sector", cache=True, cache_scope="iteration")
def worker_aggr_sector(persons):
    return persons.industry.map(aggr_sector_map)


# Maps 2-digit SOC occupational codes (XX-0000) to an aggregated occupation category
# for the persons.worker_aggr_occ column.
#   1 = Management, business, and financial
#   2 = Professional and technical
#   3 = Service
#   4 = Sales and office
#   5 = Natural resources, construction, and maintenance
#   6 = Production and transportation
aggr_occ_map = {
  11: 1,  # Management
  13: 1,  # Business and Financial Operations
  15: 2,  # Computer and Mathematical
  17: 2,  # Architecture and Engineering
  19: 2,  # Life, Physical, and Social Science
  21: 2,  # Community and Social Service
  23: 2,  # Legal
  25: 2,  # Educational Instruction and Library
  29: 2,  # Healthcare Practitioners and Technical
  31: 3,  # Healthcare Support
  33: 3,  # Protective Service
  35: 3,  # Food Preparation and Serving Related
  37: 3,  # Building and Grounds Cleaning and Maintenance
  39: 3,  # Personal Care and Service
  41: 4,  # Sales and Related
  43: 4,  # Office and Administrative Support
  45: 5,  # Farming, Fishing, and Forestry
  47: 5,  # Construction and Extraction
  49: 5,  # Installation, Maintenance, and Repair
  51: 6,  # Production
  53: 6,  # Transportation and Material Moving
}


@orca.column("persons", "worker_aggr_occ", cache=True, cache_scope="iteration")
def worker_aggr_occ(persons):
    return persons.occupation.map(aggr_occ_map)