"""
Functions for helping make pandas parallel.
"""
from functools import partial

import numpy as np
import pandas as pd

from .parallel_easy import map_easy


###############################################################################
# Globals
###############################################################################

###############################################################################
# Functions
###############################################################################


def groupby_to_scalar_to_series(df_or_series, func, n_jobs, **groupby_kwargs):
    """
    Returns a parallelized, simplified, and restricted version of:
    df_or_series.groupby(**groupby_kwargs).apply(func)

    Works ONLY for the simple case that .apply(func) would yield a Series
    of length equal to the number of groups, in other words, func applied
    to each group is a scalar.

    Parameters
    ----------
    df_or_series : DataFrame or Series
        This is what is grouped
    func : Function
        Applied to each group using func(df_or_series)
        Should return one single value (e.g. string or number)
        Must be picklable:  A lambda function will not work!
    groupby_kwargs : Keyword args
        Passed directly to DataFrame.groupby to determine groups.
        The most common one is "by", e.g.
            by='a'
            by=my_grouper_function
            by=my_grouping_list_of_labels

    Returns
    -------
    result : Series
        Index is the group names
        Values are func(group) iterated over every group

    Examples
    --------
    >>> from rosetta.parallel.pandas_easy import groupby_to_series
    >>> df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
    >>> df
       a  b
    0  6  4
    1  2  5
    2  2  6
    >>> groupby_to_series(df, max, n_jobs, by='a')
    2    b
    6    b

    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    >>> labels = ['a', 'a', 'b', 'b']
    >>> groupby_to_series(s, max, 1, by=labels)
    a    2
    b    4
    """
    grouped = df_or_series.groupby(**groupby_kwargs)
    apply_func = partial(_get_label_values, func, False)

    labels_values = map_easy(apply_func, grouped, n_jobs)
    labels, values = zip(*labels_values)

    return pd.Series(values, index=labels)


def groupby_to_series_to_frame(
    frame, func, n_jobs, use_apply=True, **groupby_kwargs):
    """
    A parallel function somewhat similar DataFrame.groupby.apply(func).

    For each group in df_or_series.groupby(**groupby_kwargs), compute
    func(group) or group.apply(func) and, assuming each result is a series,
    flatten each series then paste them together.

    Parameters
    ----------
    frame : DataFrame
    func : Function
        Applied to each group using func(df_or_series)
        Must be picklable:  A lambda function will not work!
    use_apply : Boolean
        If True, use group.apply(func)
        If False, use func(group)
    groupby_kwargs : Keyword args
        Passed directly to DataFrame.groupby to determine groups.
        The most common one is "by", e.g.
            by='a'
            by=my_grouper_function
            by=my_grouping_list_of_labels

    Returns
    -------
    result : DataFrame
        Index is the group names
        Values are func(group) iterated over every group, then pasted together

    Examples
    --------
    >>> from rosetta.parallel.pandas_easy import groupby_to_series_to_frame
    >>> df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]})
    >>> labels = ['g1', 'g1', 'g2']
    # Result and benchmark will be equal...despite the fact that you can't
    # do df.groupby(labels).apply(np.mean)
    >>> benchmark = df.groupby(labels).mean()
    >>> result = groupby_to_series_to_frame(
    ...    df, np.mean, 1, use_apply=True, by=labels)
    >>> print result
        a    b
    g1  4  4.5
    g2  2  6.0
    """
    grouped = frame.groupby(**groupby_kwargs)
    apply_func = partial(_get_label_values, func, use_apply)

    # For every group, get the label (group name) and the values
    # (output of apply_func)
    labels_values = map_easy(apply_func, grouped, n_jobs)
    labels, values = zip(*labels_values)

    # Since each value is a series, concat along axis 1 to make a short
    # and fat frame, then take transpose
    concatted = pd.concat(values, axis=1).T

    # Set the index
    if hasattr(groupby_kwargs['by'], 'name'):
        indexname = groupby_kwargs['by'].name
    elif isinstance(groupby_kwargs['by'], basestring):
        indexname = groupby_kwargs['by']
    else:
        indexname = None
    concatted.index = pd.Index(labels, name=indexname)

    return concatted


def _get_label_values(func, use_apply, name_and_group):
    """
    Returns a tuple of a name, func(group) for this name_and_group.
    Used since .groupby() returns an iterator over the pairs (name, group).

    Parameters
    ----------
    func : Function
        Must be picklable:  A lambda function will not work!
    name_and_group : Tuple
        name, group
    use_apply : Boolean
        If True, use group.apply(func)
        If False, use func(group)

    Returns
    -------
    name : the group name/label
        Same as the 'name' passed in
    value : Either group.apply(func) or func(group)
    """
    name, group = name_and_group

    value = group.apply(func) if use_apply else func(group)

    return name, value


if __name__ == '__main__':
    # Can't get doctest to work with multiprocessing...
    pass
