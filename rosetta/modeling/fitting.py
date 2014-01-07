"""
Common functions for fitting regression and classification models.
"""
import numpy as np
from numpy import linalg
import pandas as pd

from rosetta import common_math


def get_relative_error(reality, estimate):
    """
    Compares estimate to reality and returns the the mean-square error:

    |estimate - reality|_F / |reality|_F  where F is the Frobenius norm.
    """
    diff = estimate - reality

    return linalg.norm(diff) / linalg.norm(reality)


def get_R2(Y, Y_hat):
    """
    Gets the coefficient of determination R^2.
    """
    diff_Y_Ybar = Y - Y.mean()
    SStotal = diff_Y_Ybar.dot(diff_Y_Ybar)

    diff_Y_Yhat = Y - Y_hat
    SSerr = diff_Y_Yhat.dot(diff_Y_Yhat)

    return 1 - SSerr / float(SStotal)


def get_MSerr(Y, Y_hat):
    """
    Gets the mean square error
    """
    err = Y_hat - Y

    return err.dot(err) / len(Y)


def standardize(df, dont_standardize=None):
    """
    Parameters
    ----------
    df : pandas DataFrame
        Contains independent variables
    dont_standardize : List
        Names of variables to not standardize

    Returns
    -------
    Tuple of DataFrames: standardized_df, stats_df
        standardized_df is the standardized version of df
        stats_df contains the mean and std of the variables
    """
    dont_standardize = [] if dont_standardize is None else dont_standardize

    ## Get the mean and std and then the standardized dataframe
    mu = df.mean()
    sigma = df.std()

    standardized_df = (df - mu) / sigma

    ## If necessary, don't change some columns
    for name in dont_standardize:
        standardized_df[name] = df[name]

    return standardized_df


class CoefficientConverter(object):
    """
    For [un]standardizing/winsorizing coefficients and data.
    CoefficientConverter is initialized with one dataset, from this the
    standardization/winsorization rules are learned.
    The functions can be applied to other datasets.

    Standardization part of module provides the fundamental relation:
        X.dot(self.unstandardize_params(w_st)) = self.standardize(X).dot(w_st)

    WORKFLOW 1
    1)  Initialize with a DataFrame.  From this frame we learn the rules.
    2)  To fit, we use self.transform to transform a (possibly) new DataFrame.
        This fit results in a set of "transformed params" w_tr
    3)  To predict Y_hat corresponding to new input X, we first compute
        X_tr = self.transform(X), and then use X_tr.dot(w_tr)

    WORKFLOW 2  (standardization only!!)
    1)  Initialize with a dataframe.  From this frame we learn the
        standardization rules.
    2a) To fit, we use self.standardize to standardize a (possibly) new
        DataFrame.  This fit results in a set of "standardized params" w_st.
    2b) We obtain the "unstandardized params"
        w = self.unstandardized_params(w_st)
    3)  To predict Y_hat corresponding to new input X, we use X.dot(w)
    """
    def __init__(
        self, df, ones_column=None, dont_standardize=[], dont_winsorize=[],
        lower_quantile=0, upper_quantile=1, max_std=np.inf):
        """
        Parameters
        ----------
        df :  Pandas.DataFrame
            We learn the standardization rules from this df.
        ones_column : String
            Name of a column that is all ones.  This is required to use
            self.unstandardize_params().  You can have more than one column of
            all ones, but only one can be specified here.
        dont_standardize : List
            Names of variables that we will not standardize.
        dont_winsorize : List
            Names of variables that we will not winsorize.
        upper_quantile : Real number in [0, 1]
            The upper quantile above which we trim
        lower_quantile : Real number in [0, 1]
            The lower quantile below which we trim
        max_std : Non-negative real
            Trim values that are more than max_std standard deviations away
            from the mean This is done after quantile trimming.
        """
        self.known_columns = list(df.columns)

        self.dont_winsorize = dont_winsorize
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.max_std = max_std

        self.dont_standardize = dont_standardize

        # List of coefficients that should be standardized/winsorized
        self._should_standardize = list(df.columns.diff(dont_standardize))
        self._should_winsorize = list(df.columns.diff(dont_winsorize))

        # Initialize the rules
        self.stats = self._get_stats(df)
        self.clip_levels = self._get_clip_levels(df)

        # Get a list of columns that were constant.  Error check against
        # dont_standardize
        self._const_columns = self._get_const_columns()

        self._ones_column = self._verify_ones_column(ones_column, df)

    def _get_stats(self, df):
        """
        Creates self.stats, a DataFrame holding mean and standard deviation
        for all fields in df.
        """
        # Create self.stats
        stats = pd.DataFrame({'mu': df.mean(), 'sigma': df.std()})

        return stats

    def _get_clip_levels(self, df):
        def func(s):
            return _get_clip_levels_series(
                s, self.lower_quantile, self.upper_quantile, self.max_std)

        items = common_math.get_item_names(df)
        sw = items.intersection(self._should_winsorize)
        levels = pd.Series(
            np.nan * np.ones(len(items)), index=items).astype('O')
        if len(sw) > 0:
            # This cast to float prevents a mixed data type frame...which can
            # cause apply to act in a funny manner
            levels[sw] = df[sw].astype('float').apply(func)

        return levels

    def _get_const_columns(self):
        """
        Returns a list of columns that were constant.  Note that the first of
        these is the only one that will be unstandardized in
        self.unstandardize_params
        """
        _const_columns = list(self.stats[self.stats.sigma == 0].index)
        # Make sure any and all constant columns were specified as columns
        # to notstandardize
        for col in _const_columns:
            assert col in self.dont_standardize, (
                "Variable %s is constant, but was not specified in the "
                "dont_standardize initialization kwarg" % col)

        return _const_columns

    def _verify_ones_column(self, ones_column, df):
        """
        If ones_column is indeed a column of ones, returns True, otherwise
        raises ValueError.
        """
        if ones_column is None:
            return None
        elif all(df[ones_column] == 1):
            return ones_column
        else:
            raise ValueError(
                "The initialization parameter ones_column = %s is not in fact "
                "a column of all ones!" % ones_column)

    def _check_compatible(self, data):
        """
        Raises ValueError if the columns/index of the DataFrame/Series "data"
        are not contained in self.known_columns.

        In this case, we don't know how to standardize/unstandardize/winsorize
        data, so we must raise an exception.
        """
        diff = common_math.get_item_names(data).diff(self.known_columns)
        if diff:
            raise ValueError(
                "Data contained items we don't know how to work with:  %s"
                % diff)

    def standardize(self, data):
        """
        Returns a standardized version of data.

        Parameters
        ----------
        data : pandas Series or DataFrame

        Notes
        -----
        data is standardized according to the rules that self was initialized
        with, i.e. the rules implicit in self.stats.
        """
        self._check_compatible(data)

        # Convenience
        stats = self.stats

        standardized = data.copy().astype('float')
        if self._should_standardize:
            ss = common_math.get_item_names(data).intersection(
                self._should_standardize)
            standardized[ss] = (data[ss] - stats.mu[ss]) / stats.sigma[ss]

        return standardized

    def unstandardize_params(self, w_st):
        """
        Returns "w", an unstandardized version of w_st so that
        X.dot(w) = self.standardize(X).dot(w_st)

        Parameters
        ----------
        w_st : Pandas.Series
            Index is names of variables
            Values are the fitted parameter values
        """
        self._check_compatible(w_st)
        assert self._ones_column, (
            "Specify a ones_column during initialization if you want to "
            "unstandardize")

        ## We will return this Series
        w = w_st.copy().astype('float')

        # ss = "should standardize"
        ss = common_math.get_item_names(w_st).intersection(
            self._should_standardize)

        ## Unstandardize colums that were standardized
        if len(ss) > 0:
            w_st_part_only = w_st[ss]
            sigma = self.stats.sigma[ss]
            w[ss] = w_st_part_only / sigma

        # Unstandardize the constant.  Add the "excess" to self._ones_column
        if len(ss) > 0:
            mu = self.stats.mu[ss]
            w[self._ones_column] -= (mu * w_st_part_only / sigma).sum()

        return w

    def winsorize(self, data):
        """
        Winsorize the data using the rules determined during initialization.
        """
        self._check_compatible(data)

        def func(series):
            lower, upper = self.clip_levels[series.name]
            return np.maximum(lower, np.minimum(upper, series))

        # sw = "should winsorize"
        sw = common_math.get_item_names(data).intersection(
            self._should_winsorize)
        winsorized = data.copy()
        if len(sw) > 0:
            winsorized[sw] = winsorized[sw].apply(func)

        return winsorized

    def transform(self, data):
        """
        Winsorize then standardize data.  Returns a copy.
        """
        return self.standardize(self.winsorize(data))


def _get_clip_levels_series(series, lower_quantile, upper_quantile, max_std):
    """
    Gets clip levels for winsorization.
    """
    # Quantile trimming
    upper_q_value = series.quantile(upper_quantile)
    lower_q_value = series.quantile(lower_quantile)

    # Std trimming
    mu = series.mean()
    sigma = series.std()
    upper_s_value = mu + max_std * sigma
    lower_s_value = mu - max_std * sigma

    return max(lower_q_value, lower_s_value), min(upper_q_value, upper_s_value)


def winsorize(series, lower_quantile=0, upper_quantile=1, max_std=np.inf):
    """
    Truncate all items in series that are in extreme quantiles.

    Parameters
    ----------
    series : pandas.Series.  Real valued.
    upper_quantile : Real number in [0, 1]
        The upper quantile above which we trim
    lower_quantile : Real number in [0, 1]
        The lower quantile below which we trim
    max_std : Non-negative real
        Trim values that are more than max_std standard deviations away
        from the mean

    Returns
    -------
    winsorized_series : pandas.Series

    Notes
    -----
    Trimming according to max_std is done AFTER quantile trimming.
    I.e. the std is computed on the series that has already been trimmed by
    quantile.
    """
    lower, upper = _get_clip_levels_series(
        series, lower_quantile, upper_quantile, max_std)

    return np.maximum(lower, np.minimum(upper, series))
