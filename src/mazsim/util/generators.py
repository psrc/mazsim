"""
BSD 3-Clause License

Copyright (c) 2020, UrbanSim Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from __future__ import print_function
import re

import numpy as np
import pandas as pd
import operator
import re
from typing import Any, Callable

import orca

from urbansim.utils import misc
from urbansim.models import util

try:
    import pandana
except ImportError:
    pass

COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")

def make_agg_var(agent, geog, geog_id, var_to_aggregate, agg_function, how_fillna=None):
    """
    Generator function for aggregation variables. Registers with orca.
    """
    var_name = agg_function + '_' + var_to_aggregate

    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent)
        print('Calculating {} of {} for {}'
              .format(var_name, agent, geog))

        groupby = agents[var_to_aggregate].groupby(agents[geog_id])
        if agg_function == 'mean':
            values = groupby.mean().fillna(0)
        if agg_function == 'median':
            values = groupby.median().fillna(0)
        if agg_function == 'std':
            values = groupby.std().fillna(0)
        if agg_function == 'sum':
            values = groupby.sum().fillna(0)
        if agg_function == 'max':
            values = groupby.max().fillna(0)
        if agg_function == 'min':
            values = groupby.min().fillna(0)

        locations_index = orca.get_table(geog).index
        series = pd.Series(data=values, index=locations_index)

        # Fillna.
        # For certain functions, must add other options,
        # like puma value or neighboring value
        if how_fillna is not None:
            series = how_fillna(series)
        else:
            if agg_function == 'sum':
                series = series.fillna(0)
            else:
                series = series.ffill()
                series = series.bfill()

        return series

    return func


def make_disagg_var(from_geog_name, to_geog_name, var_to_disaggregate,
                    from_geog_id_name, name_based_on_geography=True):
    """
    Generator function for disaggregating variables. Registers with orca.
    """
    if name_based_on_geography:
        var_name = from_geog_name + '_' + var_to_disaggregate
    else:
        var_name = var_to_disaggregate

    @orca.column(to_geog_name, var_name, cache=True, cache_scope='iteration')
    def func():
        print('Disaggregating {} to {} from {}'
              .format(var_to_disaggregate, to_geog_name, from_geog_name))

        from_geog = orca.get_table(from_geog_name)
        to_geog = orca.get_table(to_geog_name)
        return misc.reindex(from_geog[var_to_disaggregate],
                            to_geog[from_geog_id_name]).fillna(0)

    return func


def make_size_var(agent, geog, geog_id, cache=True, cache_scope='step', prefix_agent='total'):
    """
    Generator function for size variables. Registers with orca.
    """
    var_name = prefix_agent + '_' + agent

    @orca.column(geog, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        agents = orca.get_table(agent)
        print('Calculating number of {} for {}'.format(agent, geog))

        size = agents[geog_id].value_counts()

        locations_index = orca.get_table(geog).index
        series = pd.Series(data=size, index=locations_index)
        series = series.fillna(0)

        return series

    return func


def make_proportion_var(agent, geog, geog_id, target_variable, target_value, prefix_agent='total'):
    """
    Generator function for proportion variables. Registers with orca.
    """
    try:
        var_name = 'prop_%s_%s' % (target_variable, int(target_value))
    except Exception:
        var_name = 'prop_%s_%s' % (target_variable, target_value)

    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent).to_frame(
            columns=[target_variable, geog_id])
        locations = orca.get_table(geog)
        print('Calculating proportion {} {} for {}'
              .format(target_variable, target_value, geog))

        agent_subset = agents[agents[target_variable] == target_value]
        series = (agent_subset.groupby(geog_id).size()
                  * 1.0
                  / locations[prefix_agent + '_' + agent])
        series = series.fillna(0)
        return series

    return func


def make_dummy_variable(agent, geog_var, geog_id):
    """
    Generator function for spatial dummy. Registers with orca.
    """
    # cache_scope
    try:
        var_name = geog_var + '_is_' + str(geog_id)
    except Exception:
        var_name = geog_var + '_is_' + str(int(geog_id))

    @orca.column(agent, var_name, cache=True, cache_scope='iteration')
    def func():
        agents = orca.get_table(agent)
        return (agents[geog_var] == geog_id).astype('int32')

    return func


def make_ratio_var(agent1, agent2, geog, prefix1='total', prefix2='total'):
    """
    Generator function for ratio variables. Registers with orca.
    """
    var_name = 'ratio_%s_to_%s' % (agent1, agent2)

    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        locations = orca.get_table(geog)
        print('Calculating ratio of {} to {} for {}'
              .format(agent1, agent2, geog))

        series = (locations[prefix1 + '_' + agent1]
                  * 1.0
                  / (locations[prefix2 + '_' + agent2] + 1.0))
        series = series.fillna(0)
        return series

    return func


def make_density_var(agent, geog, prefix_agent='total'):
    """
    Generator function for density variables. Registers with orca.
    """
    var_name = 'density_%s' % (agent)

    @orca.column(geog, var_name, cache=True, cache_scope='iteration')
    def func():
        locations = orca.get_table(geog)

        print('Calculating density of {} for {}'.format(agent, geog))

        series = locations[prefix_agent + '_' + agent] * 1.0 / (
            locations['sum_acres'] + 1.0)

        series = series.fillna(0)
        return series

    return func


def make_access_var(name, agent, target_variable=False, target_value=False,
                    radius=1000, agg_function='sum', decay='flat', log=True,
                    filters=False):
    """
    Generator function for accessibility variables. Registers with orca.
    """

    @orca.column('nodes', name, cache=True, cache_scope='iteration')
    def func(net):
        print('Calculating {}'.format(name))

        nodes = pd.DataFrame(index=net.node_ids)
        flds = [target_variable] if target_variable else []

        if target_value:
            flds += util.columns_in_filters(
                ["{} == {}".format(target_variable, target_value)])

        if filters:
            flds += util.columns_in_filters(filters)
        flds.append('node_id')

        df = orca.get_table(agent).to_frame(flds)

        if target_value:
            df = util.apply_filter_query(df, [
                "{} == {}".format(target_variable, target_value)])
        if filters:
            df = util.apply_filter_query(df, filters)

        net.set(df['node_id'],
                variable=df[target_variable] if target_variable else None)
        nodes[name] = net.aggregate(radius, type=agg_function, decay=decay)

        if log:
            nodes[name] = nodes[name].apply(eval('np.log1p'))
        return nodes[name]

    return func


def _finalize(series: pd.Series, fillna: Any = None, clip_lower: Any = None, dtype: str | None = None) -> pd.Series:
    if fillna is not None:
        series = series.fillna(fillna)
    if clip_lower is not None:
        series = series.clip(lower=clip_lower)
    if dtype is not None:
        series = series.astype(dtype)
    return series


def _resolve_operand(table_name: str, spec: Any) -> pd.Series:
    """Resolve an arithmetic operand: a column name, a scalar, or a {count_of, key} agent tally."""
    if isinstance(spec, dict):
        agent = orca.get_table(spec["count_of"])
        return agent[spec["key"]].value_counts()
    if isinstance(spec, str):
        return orca.get_table(table_name)[spec]
    return spec


def make_join_var(table, var_name, from_table, from_column, on=None,
                  fillna=0, dtype=None, cache=True, cache_scope="iteration"):
    """Generator function for attaching a column from another table. Registers with orca.

    `on=None` aligns on the target table's index; otherwise `on` names a key column on the
    target table holding `from_table`'s index values.
    """

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Joining {from_column} onto {table} from {from_table}")
        source = orca.get_table(from_table)[from_column]
        target = orca.get_table(table)
        if on is None:
            series = source.reindex(target.index)
        else:
            series = misc.reindex(source, target[on])
        return _finalize(series, fillna=fillna, dtype=dtype)

    return func


def make_difference_var(table, var_name, subtract_from, subtract, fill_value=0,
                        clip_lower=None, dtype=None, cache=False, cache_scope="step"):
    """Generator function for a difference of two operands. Registers with orca."""

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        left = _resolve_operand(table, subtract_from)
        right = _resolve_operand(table, subtract)
        series = left.sub(right, fill_value=fill_value) if isinstance(left, pd.Series) else left - right
        return _finalize(series, clip_lower=clip_lower, dtype=dtype)

    return func


def make_sum_var(table, var_name, operands, fill_value=0, dtype=None,
                 cache=False, cache_scope="step"):
    """Generator function for a sum of two or more operands. Registers with orca."""

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        series = _resolve_operand(table, operands[0])
        for spec in operands[1:]:
            other = _resolve_operand(table, spec)
            series = series.add(other, fill_value=fill_value) if isinstance(series, pd.Series) else series + other
        return _finalize(series, dtype=dtype)

    return func


def make_quantile_var(table, var_name, source, bins, offset=0, dtype=None,
                      cache=True, cache_scope="iteration"):
    """Generator function for quantile bins of a column. Registers with orca."""

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        series = pd.qcut(orca.get_table(table)[source], bins, labels=False) + offset
        return _finalize(series, dtype=dtype)

    return func


def make_map_var(table, var_name, source, mapping, fillna=None, dtype=None,
                 cache=True, cache_scope="iteration"):
    """Generator function for recoding a column through a lookup. Registers with orca."""

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        series = orca.get_table(table)[source].map(mapping)
        return _finalize(series, fillna=fillna, dtype=dtype)

    return func


def make_threshold_var(table, var_name, source, op, value, dtype="int8",
                       cache=True, cache_scope="iteration"):
    """Generator function for a 0/1 dummy from a comparison. Registers with orca."""
    if op not in COMPARISONS:
        raise ValueError(f"{var_name}: unsupported comparison {op!r}; expected one of {sorted(COMPARISONS)}")

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        series = COMPARISONS[op](orca.get_table(table)[source], value)
        return _finalize(series, dtype=dtype)

    return func


def make_constant_var(table, var_name, value, dtype="int32", cache=True, cache_scope="iteration"):
    """Generator function for a constant-valued column. Registers with orca."""

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        index = orca.get_table(table).index
        return _finalize(pd.Series(np.full(len(index), value), index=index), dtype=dtype)

    return func


def make_index_slice_var(table, var_name, start, stop, dtype="int64", cache=True, cache_scope="forever"):
    """Generator function for an id carved out of the table's own index. Registers with orca.

    Nesting geographies such as census blocks encode their parent ids as a prefix of the child id.
    """

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        index = orca.get_table(table).index
        series = pd.Series(index.values, index=index).astype(str).str.slice(start, stop)
        return _finalize(series, dtype=dtype)

    return func


def make_expression_var(table, var_name, expr, dtype=None, cache=True, cache_scope="iteration"):
    """Generator function for a pandas expression over the table's own columns. Registers with orca.

    Evaluated with DataFrame.eval, which parses to a restricted expression AST -- never builtin eval.
    """

    @orca.column(table, var_name, cache=cache, cache_scope=cache_scope)
    def func():
        print(f"Calculating {var_name} for {table}")
        target = orca.get_table(table)
        referenced = [c for c in dict.fromkeys(_IDENTIFIER.findall(expr)) if c in target.columns]
        df = target.to_frame(referenced)
        series = df.eval(expr, global_dict={}, local_dict={})
        return _finalize(series, dtype=dtype)

    return func
