"""
Common math functions.
"""
import numpy as np
import pandas as pd
from numpy.random import choice

from scipy.sparse import isspmatrix

def pandas_to_ndarray_wrap(X, copy=True):
    """
    Converts X to a ndarray and provides a function to help convert back
    to pandas object.

    Parameters
    ----------
    X : Series/DataFrame/ndarray
    copy : Boolean
        If True, return a copy.

    Returns
    -------
    Xvals : ndarray
        If X is a Series/DataFrame, then Xvals = X.values,
        if ndarray, Xvals = X
    F : Function
        F(Xvals) = X
    """
    if copy:
        X = X.copy()

    if isinstance(X, pd.Series):
        return X.values, lambda Z: pd.Series(np.squeeze(Z), index=X.index)
    elif isinstance(X, pd.DataFrame):
        return X.values, lambda Z: pd.DataFrame(
            Z, index=X.index, columns=X.columns)
    elif isinstance(X, np.ndarray) or isspmatrix(X):
        return X, lambda Z: Z
    else:
        raise ValueError("Unhandled type: %s" % type(X))


def subsample_arr(arr, N=None, frac_keep=None):
    """
    Subsample a Series, DataFrame, or ndarray along axis 0.

    Parameters
    ----------
    arr : Series, DataFrame, or ndarray
    N : Integer
        Number of samples to keep
    frac_keep : Real in [0, 1]
        Fraction of samples to keep

    Returns
    -------
    subsampled : Series, DataFrame, or ndarray
        A copy
    """
    # Input checking
    assert ((N is None) and (frac_keep is not None)) \
        or ((N is not None) and (frac_keep is None))

    #
    if N is None:
        N = int(len(arr) * frac_keep)

    if isinstance(arr, np.ndarray):
        index = choice(range(len(arr)), size=N, replace=False)
        return arr[np.ix_(index)]
    elif isinstance(arr, pd.Series) or isinstance(arr, pd.DataFrame):
        index = choice(arr.index, size=N, replace=False)
        return arr.ix[index]
    else:
        raise ValueError("arr of unhandled type:  %s" % type(arr))


def get_item_names(data):
    """
    If DataFrame, return columns, if Series, return index.
    """
    if isinstance(data, pd.Series):
        items = data.index
    elif isinstance(data, pd.DataFrame):
        items = data.columns
    else:
        raise TypeError("Argument type %s is a type not handled" % type(data))

    return items


def series_to_frame(data):
    """
    If length(N) Series, return an N x 1 Frame with name equal to the series
    name.  If frame, passthrough.

    Parameters
    ----------
    data : pandas Series or DataFrame.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    else:
        raise ValueError("type(data) = %s is not handled" % type(data))

    return data

