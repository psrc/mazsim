"""
Data Model for Project CSV Inputs

Defines the enumerated variables and pandera schemas used to validate each
table listed under `data_sources` in settings.yaml before it is registered
with orca. Customize the enums/fields here as the input data changes.
"""

from __future__ import annotations

from enum import IntEnum

import pandera.pandas as pa


class BuildingType(IntEnum):
    """Structure type of a housing unit/household's building."""

    MOBILE_HOME = 1
    SINGLE_FAMILY_DETACHED = 2
    SINGLE_FAMILY_ATTACHED = 3
    MULTI_FAMILY_2_UNITS = 4
    MULTI_FAMILY_3_4_UNITS = 5
    MULTI_FAMILY_5_9_UNITS = 6
    MULTI_FAMILY_10_19_UNITS = 7
    MULTI_FAMILY_20_49_UNITS = 8
    MULTI_FAMILY_50_PLUS_UNITS = 9
    BOAT_RV_VAN = 10


class Tenure(IntEnum):
    """Whether a household owns or rents its unit."""

    OWNER = 1
    RENTER = 2


class SectorId(IntEnum):
    """Employment sector of a job."""

    AGRICULTURE = 11
    MINING = 21
    UTILITIES = 22
    CONSTRUCTION = 23
    MANUFACTURING = 3133
    WHOLESALE_TRADE = 42
    RETAIL = 4445
    TRANSPORTATION_AND_WAREHOUSING = 4849
    INFORMATION = 51
    FINANCE_AND_INSURANCE = 52
    REAL_ESTATE = 53
    PROFESSIONAL_SERVICES = 54
    MANAGEMENT = 55
    ADMINISTRATIVE_SERVICES = 56
    EDUCATION = 61
    HEALTH_CARE = 62
    ARTS_AND_ENTERTAINMENT = 71
    ACCOMMODATION_AND_FOOD_SERVICES = 72
    OTHER_SERVICES = 81
    GOVERNMENT = 98


class UnitTypeId(IntEnum):
    """Tenure/structure grouping of a housing unit."""

    SINGLE_FAMILY_OWNER = 1
    SINGLE_FAMILY_RENTER = 2
    MULTI_FAMILY_OWNER = 3
    MULTI_FAMILY_RENTER = 4


class Race(IntEnum):
    """Race of a person/household head."""

    WHITE = 1
    BLACK = 2
    AMERICAN_INDIAN = 3
    ALASKA_NATIVE = 4
    AMERICAN_INDIAN_OR_ALASKA_NATIVE = 5
    ASIAN = 6
    NATIVE_HAWAIIAN_PACIFIC_ISLANDER = 7
    OTHER = 8
    TWO_OR_MORE_RACES = 9


class Sex(IntEnum):
    """Sex of a person."""

    MALE = 1
    FEMALE = 2


class Industry(IntEnum):
    """Industry a person works in."""

    NA = 0
    AGRICULTURE_MINING = 1121
    CONSTRUCTION = 23
    MANUFACTURING = 3133
    WHOLESALE = 42
    RETAIL = 4445
    TRANSPORTATION_WAREHOUSING_UTILITIES = 484922
    INFORMATION = 51
    FINANCE_REAL_ESTATE = 5253
    PROFESSIONAL_MANAGEMENT_ADMIN = 5456
    EDUCATION_HEALTHCARE = 6162
    ARTS_ENTERTAINMENT_ACCOMMODATION_FOOD = 7172
    OTHER_SERVICES = 81
    GOVERNMENT = 98


class Occupation(IntEnum):
    """Occupation of a person."""

    NA = 0
    MANAGEMENT = 11
    BUSINESS = 13
    COMPUTER = 15
    ENGINEERING = 17
    LIFE_SCIENCE = 19
    COMMUNITY_SERVICE = 21
    LEGAL = 23
    EDUCATIONAL = 25
    ARTS = 27
    HEALTHCARE_PRACTITIONER = 29
    HEALTHCARE_SUPPORT = 31
    PROTECTIVE_SERVICE = 33
    FOOD_PREPARATION = 35
    BUILDING_GROUNDS_CLEANING = 37
    PERSONAL_CARE = 39
    SALES = 41
    OFFICE_ADMINISTRATIVE = 43
    FARMING_FISHING_FORESTRY = 45
    CONSTRUCTION_EXTRACTION = 47
    INSTALLATION_MAINTENANCE_REPAIR = 49
    PRODUCTION = 51
    TRANSPORTATION_MATERIAL_MOVING = 53
    UNEMPLOYED = 99


class IncomeQuartile(IntEnum):
    """Household income quartile within its geography."""

    LOWEST = 1
    LOWER_MIDDLE = 2
    UPPER_MIDDLE = 3
    HIGHEST = 4


class Blocks(pa.DataFrameModel):
    """
    Census block land use/parcel data.

    block_id: unique 15-digit census block ID
    zone_id: MAZ the block belongs to
    x, y: block centroid coordinates
    res_rent: imputed/observed residential rent
    res_value: imputed/observed residential value
    rent_impute, value_impute: whether rent/value was imputed rather than observed
    acres_land: land area of the block, in acres
    """

    block_id: int = pa.Field(unique=True, ge=0)
    zone_id: int = pa.Field(ge=0)
    x: float = pa.Field()
    y: float = pa.Field()
    res_rent: float = pa.Field(ge=0)
    res_value: float = pa.Field(ge=0)
    rent_impute: bool = pa.Field()
    value_impute: bool = pa.Field()
    acres_land: float = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class Households(pa.DataFrameModel):
    """
    Synthetic households.

    household_id: unique household ID
    puma_id, tract_id, block_id: geographies the household resides in
    hh_id: source household ID from the synthesis process
    persons: number of people in the household
    building_type: structure type of the household's building, see BuildingType
    tenure: owner/renter status, see Tenure
    income: annual household income in $
    rent, home_value: monthly rent / home value, if applicable
    children, adults, workers: counts of persons by group within the household
    cars: number of vehicles owned
    age_of_head: age of the household head
    race_of_head: race of the household head, see Race
    hispanic_status_of_head: whether the household head is Hispanic
    recent_mover: whether the household moved in the last 4 years
    wgtp: household survey weight
    serialno: source survey serial number
    year_built: year the household's building was built
    """

    household_id: int = pa.Field(unique=True, ge=0)
    puma_id: int = pa.Field(ge=0)
    tract_id: int = pa.Field(ge=0)
    block_id: int = pa.Field(ge=0)
    hh_id: int = pa.Field(ge=0)
    persons: int = pa.Field(ge=0)
    building_type: int = pa.Field(isin=BuildingType)
    tenure: int = pa.Field(isin=Tenure)
    income: float = pa.Field()
    rent: float = pa.Field(ge=0)
    home_value: float = pa.Field(ge=0)
    children: int = pa.Field(ge=0)
    adults: int = pa.Field(ge=0)
    workers: int = pa.Field(ge=0)
    cars: int = pa.Field(ge=0)
    age_of_head: int = pa.Field(ge=0)
    race_of_head: int = pa.Field(isin=Race)
    hispanic_status_of_head: bool = pa.Field()
    recent_mover: bool = pa.Field()
    wgtp: float = pa.Field(ge=0)
    serialno: str = pa.Field()
    year_built: int = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class Jobs(pa.DataFrameModel):
    """
    Synthetic jobs.

    job_id: unique job ID
    sector_id: employment sector, see SectorId
    block_id: block the job is located in
    """

    job_id: int = pa.Field(unique=True, ge=0)
    sector_id: int = pa.Field(isin=SectorId)
    block_id: int = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class HousingUnits(pa.DataFrameModel):
    """
    Synthetic housing units.

    unit_id: unique housing unit ID
    block_id: block the unit is located in
    year_built: year the unit's building was built
    unit_type_id: tenure/structure grouping, see UnitTypeId
    """

    unit_id: int = pa.Field(unique=True, ge=0)
    block_id: int = pa.Field(ge=0)
    year_built: int = pa.Field()
    unit_type_id: float = pa.Field(isin=UnitTypeId)

    class Config:
        strict = "filter"
        coerce = True


class Persons(pa.DataFrameModel):
    """
    Synthetic persons.

    puma_id, tract_id, block_id: geographies the person resides in
    hh_id: source household ID from the synthesis process
    household_id: household ID of the person
    age: person age
    is_child, is_adult, is_worker, is_student: person status flags
    work_from_home: whether the person works from home
    is_hispanic: whether the person is Hispanic
    race: person race, see Race
    sex: person sex, see Sex
    industry: person's industry, see Industry
    occupation: person's occupation, see Occupation
    pwgtp: person survey weight
    serialno: source survey serial number
    per_num: person number within the household
    """

    puma_id: int = pa.Field(ge=0)
    tract_id: int = pa.Field(ge=0)
    block_id: int = pa.Field(ge=0)
    hh_id: int = pa.Field(ge=0)
    household_id: int = pa.Field(ge=0)
    age: int = pa.Field(ge=0)
    is_child: bool = pa.Field()
    is_adult: bool = pa.Field()
    is_worker: bool = pa.Field()
    is_student: bool = pa.Field()
    work_from_home: bool = pa.Field()
    is_hispanic: bool = pa.Field()
    race: int = pa.Field(isin=Race)
    sex: int = pa.Field(isin=Sex)
    industry: int = pa.Field(isin=Industry)
    occupation: int = pa.Field(isin=Occupation)
    pwgtp: float = pa.Field(ge=0)
    serialno: str = pa.Field()
    per_num: int = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class TransitStops(pa.DataFrameModel):
    """
    Transit stop locations.

    x, y: stop coordinates
    hct: high-capacity transit flag/type
    """

    x: float = pa.Field()
    y: float = pa.Field()
    hct: int = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class Nodes(pa.DataFrameModel):
    """
    Pandana network nodes.

    id: unique node ID
    x, y: node coordinates
    """

    id: int = pa.Field(unique=True, ge=0)
    x: float = pa.Field()
    y: float = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class Edges(pa.DataFrameModel):
    """
    Pandana network edges.

    from, to: node IDs the edge connects
    weight: edge impedance (e.g. distance)
    edge_type: functional class of the edge
    """

    from_: int = pa.Field(ge=0, alias="from")
    to: int = pa.Field(ge=0)
    weight: float = pa.Field(ge=0)
    edge_type: str = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class JobCalibTargets(pa.DataFrameModel):
    """
    Job calibration targets by tract.

    tract_id: unique tract ID
    jobs_2020, jobs_2010: observed job counts
    jobs_target: target job count
    """

    tract_id: int = pa.Field(unique=True, ge=0)
    jobs_2020: float = pa.Field(ge=0)
    jobs_2010: float = pa.Field(ge=0)
    jobs_target: float = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class HouseholdCalibTargets(pa.DataFrameModel):
    """
    Household calibration targets by tract and income quartile.

    tract_id: tract ID (repeated per income_quartile)
    income_quartile: household income quartile, see IncomeQuartile
    households_2010, households_2020: observed household counts
    households_target: target household count
    """

    tract_id: int = pa.Field(ge=0)
    income_quartile: int = pa.Field(isin=IncomeQuartile)
    households_2010: float = pa.Field(ge=0)
    households_2020: float = pa.Field(ge=0)
    households_target: float = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class HousingUnitCalibTargets(pa.DataFrameModel):
    """
    Housing unit calibration targets by tract and unit type.

    tract_id: tract ID (repeated per unit_type_id)
    unit_type_id: tenure/structure grouping, see UnitTypeId
    units_2010, units_2020: observed unit counts
    units_target: target unit count
    """

    tract_id: int = pa.Field(ge=0)
    unit_type_id: float = pa.Field(isin=UnitTypeId)
    units_2010: float = pa.Field(ge=0)
    units_2020: float = pa.Field(ge=0)
    units_target: float = pa.Field()

    class Config:
        strict = "filter"
        coerce = True


class TravelData(pa.DataFrameModel):
    """
    Zone-to-zone skims.

    from_zone_id, to_zone_id: zone pair the record applies to
    am_single_vehicle_to_work_travel_time: AM single-occupant vehicle travel time, in minutes
    """

    from_zone_id: int = pa.Field(ge=0)
    to_zone_id: int = pa.Field(ge=0)
    am_single_vehicle_to_work_travel_time: float = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class BlockCapacity(pa.DataFrameModel):
    """
    Development capacity by block.

    block_id: unique block ID
    job_capacity: maximum number of jobs the block can hold
    housing_unit_capacity: maximum number of housing units the block can hold
    """

    block_id: int = pa.Field(unique=True, ge=0)
    job_capacity: int = pa.Field(ge=0)
    housing_unit_capacity: int = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class AnnualHouseholdControlTotals(pa.DataFrameModel):
    """
    Annual household control totals by county.

    county_id: county the control total applies to
    year: year the control total applies to
    total_number_of_households: target number of households
    """

    county_id: int = pa.Field()
    year: int = pa.Field()
    total_number_of_households: int = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


class AnnualEmploymentControlTotals(pa.DataFrameModel):
    """
    Annual employment control totals by county.

    county_id: county the control total applies to
    year: year the control total applies to
    total_number_of_jobs: target number of jobs
    """

    county_id: int = pa.Field()
    year: int = pa.Field()
    total_number_of_jobs: int = pa.Field(ge=0)

    class Config:
        strict = "filter"
        coerce = True


# Maps each settings.yaml data_sources table name to its pandera schema.
TABLE_MODELS: dict[str, type[pa.DataFrameModel]] = {
    "blocks": Blocks,
    "households": Households,
    "jobs": Jobs,
    "housing_units": HousingUnits,
    "persons": Persons,
    "transit_stops": TransitStops,
    "nodes": Nodes,
    "edges": Edges,
    "job_calib_targets": JobCalibTargets,
    "household_calib_targets": HouseholdCalibTargets,
    "housing_unit_calib_targets": HousingUnitCalibTargets,
    "travel_data": TravelData,
    "block_capacity": BlockCapacity,
    "annual_household_control_totals": AnnualHouseholdControlTotals,
    "annual_employment_control_totals": AnnualEmploymentControlTotals,
}

# Maps each table to the column(s) orca should set as its index, after validation.
TABLE_INDEXES: dict[str, str | list[str]] = {
    "blocks": "block_id",
    "households": "household_id",
    "jobs": "job_id",
    "housing_units": "unit_id",
    "nodes": "id",
    "job_calib_targets": "tract_id",
    "household_calib_targets": "tract_id",
    "housing_unit_calib_targets": "tract_id",
    "travel_data": ["from_zone_id", "to_zone_id"],
    "block_capacity": "block_id",
}
