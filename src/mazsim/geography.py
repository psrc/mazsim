"""Resolve the project's geography from variables.yaml: the base geography and each level's join key.

The base geography is the finest location agents are placed at. Its table is indexed by
`base_geography.id` -- the same id agents carry in their `geography_id` column (settings.yaml)
and that location choice models write as their choices. Any finer id of that table, such as
block_id when the base geography is maz_id, is an ordinary column on it.

Each level in `geographic_levels` is joined on the id column it declares, which has to be carried
on the base geography's table -- either as a raw column (zone_id) or as an id carved off a finer
id by `geography_ids` (county_id, tract_id, block_group_id). The level holding the base
geography's own rows always resolves to the base id whatever id it declares, so a project can
change its base geography in one place.

Use :func:`geography_level_keys` wherever a level's id column is needed, so the resolution rules
stay in one place.
"""

from typing import Any


def base_geography(config: dict[str, Any]) -> tuple[str, str]:
    """The (table, id) of the finest geography, whose table is indexed by that id."""
    try:
        return config["base_geography"]["table"], config["base_geography"]["id"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "variables.yaml needs base_geography: {table: <base table>, id: <location column>}, "
            "e.g. {table: blocks, id: maz_id}."
        ) from exc


def geography_level_keys(config: dict[str, Any]) -> dict[str, str]:
    """The id column each level in `config["geographic_levels"]` is joined on.

    Returns the levels in declaration order as name -> id column. The base geography's own
    level resolves to the base id; every other level to the id it declares, which must be a
    column on the base geography's table.
    """
    base_table, base_id = base_geography(config)
    return {
        geography_name: base_id if geography_name == base_table else declared_id
        for geography_name, declared_id in config["geographic_levels"]
    }
