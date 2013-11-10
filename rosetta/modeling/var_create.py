import scipy as sp
import numpy as np

from rosetta.modeling import eda


# Define the logit function that acts on series to get the logit of the series
def logit(series):
    """
    Logit for Pandas.Series
    """
    logseries = np.log(series)
    return logseries / (1 - logseries)


def logit_of_mean(series):
    """
    Logit of the mean of a pandas series
    """
    mean = series.mean()
    return logit(mean)


def sigmoidize(x, scale=5, mid=None):
    """
    Returns y, a sigmoidal version of the variable x.
    y = sig((x - mid) / scale)
    sig(z) = exp(z) / (1 + exp(z))

    Parameters
    ----------
    x : np.ndarray, ndim=1
    scale : positive real number
    mid : real number
        If None, use the mean
    """
    y = x.copy()
    if not mid:
        mid = y.mean()

    z = (x - mid) / float(scale)
    exp_z = np.exp(z)

    return exp_z / (1 + exp_z)


def standardize(x):
    """
    Standardizes a Series or DataFrame.
    """
    return (x - x.mean()) / x.std()


def build_xy_for_linearize(
    x, y, bins=10, Y_reducer=np.mean, x_lims=None, endpoints=None):
    """
    Return x and y for use in linearization.  Use with var_create.interp.

    Parameters
    ----------
    x : Pandas.Series
    y : Pandas.Series
    bins : positive integer
        Number of bins for x
    Y_reducer : Function
        Used to reduce Y in each of the bins.  E.g np.mean, logit_of_mean.
    x_lims : 2-tuple, (xmin, xmax)
        Rescaled x will be constant outside of this range.
        Choose xmin, xmax such that you have enough data in the interval
        (xmin, xmax)
    endpoints : Array-like
        [xmin, xmax, ymin, ymax].  Makes sure F(xmin) = ymin, etc...

    Returns
    -------
    x : Array
        The bin midpoints (with adjunstments at the ends)
    y : Array
        y reduced in the bins
    """
    # Trim the range of x used
    x_actualmin = x.min()
    x_actualmax = x.max()
    if x_lims:
        # We need to keep the actual max/min
        mask = (x > x_lims[0]) & (x < x_lims[1])
    else:
        mask = np.ones(len(x), dtype=bool)

    # reduced_Y is the reduced y values with an index equal to the x midpoints
    reduced_Y, _ = eda.reducedY_vs_binnedX(
        x[mask], y[mask], Y_reducer=Y_reducer, bins=bins)

    # If we don't convert to float, we get an object series...
    x_midpts = reduced_Y.index.values.astype('float')

    # Stick on the endpoints
    if endpoints is None:
        endpoints = x_actualmin, x_actualmax, reduced_Y[0], reduced_Y[-1]
    x_extended = np.r_[endpoints[0], x_midpts, endpoints[1]]
    y_extended = np.r_[endpoints[2], reduced_Y, endpoints[3]]

    return x_extended, y_extended


def interp(x, y, t=1, scaling=None):
    """
    Return interpolation helpers for x and y.
    See build_xy_for_linearize for use in linearization.

    Parameters
    ----------
    x : Array-like
    y : Array-like
    t : Real number in [0, 1]
        With F(x) the linearization function, re-set F(x) = t*F(x) + (1-t)*x
    scaling : String
        If None, the output is not rescaled and Y_reducer(bin_j) = x_j where
            x_j is the midpoint of bin_j.
        If 'standardize', then output will have zero mean and unit variance
        If 'unit', then output will be on the interval [0, 1]

    Returns
    -------
    F_x : Pandas.Series
        A rescaled version of x.
    F : Function that will rescale x
        Cannot be pickled... :(

    Examples
    --------
    x4linear, y4linear = vc.build_xy_for_linearize(y_score, y)
    F_x, F = interp(x4linear, y4linear)
    """
    # Interpolate to get our first try at F
    F_1 = sp.interpolate.interp1d(x, y, kind='linear')

    # Reshape
    F_2 = lambda x: t * F_1(x) + (1 - t) * x

    # Scaling
    F_2_x = F_2(x)
    if scaling is not None:
        if scaling == 'standardize':
            F_3 = lambda x: (F_2(x) - F_2_x.mean()) / F_2_x.std()
        elif scaling == 'unit':
            F_3 = lambda x: (
                F_2(x) - F_2_x.min()) / (F_2_x.max() - F_2_x.min())
        else:
            raise ValueError("Unknown scaling passed: %s" % scaling)
    else:
        F_3 = F_2

    return F_3(x), F_3
