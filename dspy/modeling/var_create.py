import pandas as pd
import numpy as np
from numpy import linalg
import scipy as sp
import matplotlib.pyplot as plt

from dspy.modeling import eda


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
