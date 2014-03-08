import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def plot_corr_grid(
    corr, cluster=True, cluster_method='weighted', distance_fun=None, ax=None,
    **fig_kwargs):
    """
    Plot a correlation matrix as a grid.  Uses scipy.cluster.hierarchy.linkage
    to compute clusters based on distance between samples.  
    
    Since correlation is passed in, this correlation must be converted to a
    distance (using distance_fun).  The default distance_fun makes highly
    correlated points have low distance, and vice versa.

    Parameters
    ----------
    corr : numpy ndarray or pandas DataFrame
        corr[i, j] is the correlation (should be between -1 and 1) of samples
        i and j.
    cluster : Boolean
        If True, reorder the matrix putting correlated entries nearby.
    distance_fun : Function
        inter-variable distance = distance_fun(corr).
        If None, use (1 - corr) / 2.
    ax : matplotlib AxesSubplot instance
        If None, use pl.gca()
    cluster_method : String
        Method to use to amalgomate clusters.
        Either 'single', 'complete', 'average', or 'weighted'.
        See scipy.cluster.hierarchy.linkage for details.

    Returns
    -------
    fig : matplotlib figure instance
    """
    fig_kwargs.setdefault('figsize', (8, 8))

    if distance_fun is None:
        distance_fun = lambda c: (1 - c) / 2.

    # Convert to a DataFrame in all cases.
    if not isinstance(corr, pd.DataFrame):
        names = range(len(corr))
        corr = pd.DataFrame(corr, index=names, columns=names)
    else:
        names = corr.index.tolist()

    # If you're clustering, reorder the matrix.
    if cluster:
        corr = corr.copy()
        dist = distance_fun(corr)
        Z = linkage(squareform(dist.values), method=cluster_method)
        idx_order = dendrogram(Z, no_plot=True)['ivl']
        names = [names[int(i)] for i in idx_order]
        corr = corr.reindex(index=names, columns=names)

    if ax is None:
        ax = pl.gca()
    fig = sm.graphics.plot_corr(corr, xnames=names, ynames=names, ax=ax)
    fig.set_size_inches(fig_kwargs['figsize'])

    return fig


def plot_corr_dendrogram(
    corr, cluster_method='weighted', **dendrogram_kwargs):
    """
    Plot a correlation matrix as a dendrogram (on the current axes).
    Uses scipy.cluster.hierarchy.linkage
    to compute clusters based on distance between samples.  
    
    Since correlation is passed in, this correlation must be converted to a
    distance (using distance_fun).  The default distance_fun makes highly
    correlated points have low distance, and vice versa.

    Parameters
    ----------
    corr : numpy ndarray or pandas DataFrame
        corr[i, j] is the correlation (should be between -1 and 1) of samples
        i and j.
    cluster_method : String
        Method to use to amalgomate clusters.
        Either 'single', 'complete', 'average', or 'weighted'.
        See scipy.cluster.hierarchy.linkage for details.
    dendrogram_kwargs : Additional kwargs
        Pass to the call of scipy.cluster.hierarchy.dendrogram()
    """
    # Convert to a DataFrame in all cases.
    if not isinstance(corr, pd.DataFrame):
        names = range(len(corr))
    else:
        names = corr.index.tolist()
        corr = corr.values

    dist = (1 - corr) / 2.
    Z = linkage(squareform(dist), method=cluster_method)

    dendrogram(Z, labels=names, **dendrogram_kwargs)


def plot_scatterXY(x, y, stride=1, plot_XequalsY=False, ax=None, **plt_kwargs):
    """
    Plot a XY scatter plot of two Series.

    Parameters
    ----------
    x, y : Pandas.Series
    stride : Positive integer
        If stride == n, then plot only every nth point
    plot_XequalsY : Boolean
        If True, plot the line X = Y in red.
    plt_kwargs : Additional kwargs to pass to plt.scatter
    """
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        xname = x.name if x.name else 'X'
    else:
        xname = 'X'

    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        yname = y.name if y.name else 'Y'
    else:
        yname = 'Y'

    x = x[::stride]
    y = y[::stride]
    if 'c' in plt_kwargs:
        plt_kwargs['c'] = plt_kwargs['c'][::stride]

    if plot_XequalsY:
        plt.plot(x, x, 'r-')

    plt.scatter(x, y, **plt_kwargs)
    plt.xlabel(xname)
    plt.ylabel(yname)


def reducedY_vs_binnedX(
    x, y, Y_reducer=np.mean, X_reducer='midpoint', bins=10, quantiles=False,
    labels=None):
    """
    Bin X and, inside every bin, apply Y_reducer to the Y values.

    Parameters
    ----------
    x : Pandas.Series with numeric data
    y : Pandas.Series with numeric data = 0 or 1
    Y_reducer : function
        Used to aggregate the Y values in every bin
    X_reducer : function or 'midpoint'
        Used to aggregate the X values in every bin.  This gives us the bin
        labels that are used as the indices.
        If 'midpoint', then use the bin midpoint.
    bins : Positive Integer, optional
        Number of bins to divide series
    quantiles : Boolean, optional
        If True, bin data using quantiles rather than an evenly divided range
    labels : List-like, with len(labels) = len(x), optional
        If given, use these labels to bin X rather than bins.

    Returns
    -------
    y_reduced : Series
        The reduced y values with an index equal to the reduced X
    count_X : Series
        The number of X variables in each bin.  Index is the reduced value.

    Examples
    --------
    Suppose Y is binary.  Then to compute P[Y=1|X=x] (for x inside the bins),
    as well as #[X=x], use:

    P_Y_g_X, count_X = eda.reducedY_vs_binnedX(x, y, Y_reducer=np.mean)
    """
    ## Get the labels that are also used to group the x data
    if labels is None:
        labels = get_labels(x, bins=bins, quantiles=quantiles)

    ## Reduce Y
    y_reduced = y.groupby([labels]).agg(Y_reducer)

    ## Get the sizes of the bins
    count_X = x.groupby([labels]).size()

    ## Rename the indices
    if X_reducer != 'midpoint':
        x_reduced = x.groupby([labels]).agg(X_reducer)
        reindex_map = {i: x_reduced[i] for i in x_reduced.index}
        y_reduced = y_reduced.rename(reindex_map)
        count_X = count_X.rename(reindex_map)

    ## Rename the axis
    count_X.index.name = x.name
    count_X.name = '#[X=x]'
    y_reduced.index.name = x.name
    y_reduced.name = y.name

    return y_reduced, count_X


def plot_reducedY_vs_binnedX(
    x, y, Y_reducer=np.mean, X_reducer=np.mean, bins=10, quantiles=False,
    plot_count_X=False, **plt_kwargs):
    """
    Bin X and, inside every bin, apply Y_reducer to the Y values.  Then plot.

    Parameters
    ----------
    x : Pandas.Series with numeric data
    y : Pandas.Series with numeric data = 0 or 1
    Y_reducer : function
        Used to aggregate the Y values in every bin
    X_reducer : function
        Used to aggregate the X values in every bin.  This gives us the bin
        labels that are used as the indices.
    bins : Positive Integer, optional
        Number of bins to divide series
    quantiles : Boolean, optional
        If True, bin data using quantiles rather than an evenly divided range
    plot_count_X : Boolean, optional
        If True, plot count_X versus x in a separate subplot
    **kwargs : Extra keywordargs passed to plot

    Examples
    --------
    Suppose Y is binary.  Then to plot P[Y=1|X=x] (for x inside the bins),
    as well as #[X=x], use:

    eda.plot_reducedY_vs_binnedX(x, y, Y_reducer=np.mean, plot_count_X=True)
    """
    y_reduced, count_X = reducedY_vs_binnedX(
        x, y, Y_reducer, X_reducer, bins, quantiles)

    # Set a default figure size
    plt_kwargs.setdefault('figsize', (10, 5))
    # We handle the subplots ourselves
    plt_kwargs['subplots'] = False

    # If plot_count_X, then we are plotting a dataframe rather than a series,
    # and there are different key word args available.
    if plot_count_X:
        if quantiles:
            print "Warning!  plot_count_X is meaningless if quantiles==True"
        fig, axes = plt.subplots(1, 2, figsize=plt_kwargs['figsize'])
        y_reduced.plot(ax=axes[0], **plt_kwargs)
        count_X.plot(ax=axes[1], title=count_X.name, **plt_kwargs)
    else:
        y_reduced.plot(**plt_kwargs)


def get_labels(series, bins=10, quantiles=False):
    """
    Divides series into bins and returns labels corresponding to midpoints of
    bins.

    Parameters
    ----------
    series : Pandas.Series of numeric data
    bins : Positive Integer, optional
        Number of bins to divide series
    quantiles : Boolean, optional
        If True, bin data using quantiles rather than an evenly divided range
    """
    cutfun = pd.qcut if quantiles else pd.cut
    levels = cutfun(series, bins)
    labels = np.zeros(len(levels))
    for i, lev in enumerate(levels):
        # NaN label occurs sometimes, just use as-is
        if isinstance(lev, float):
            assert np.isnan(lev)
            labels[i] = lev
        else:
            start = lev.split(',')[0][1:]
            end = lev.split(',')[1][:-1]
            mid = (float(start) + float(end)) / 2
            labels[i] = mid

    return labels


def hist_cols(
    df, cols_to_plot, num_cols, num_rows, figsize=None, **kwargs):
    """
    Plots histograms of columns of a DataFrame as subplots in one big plot.
    Handles nans and extreme values in a "graceful" manner by removing them
    and reporting their occurance.

    Parameters
    ----------
    df : Pandas DataFrame
    cols_to_plot : List
        Column names of df that will be plotted
    num_cols, num_rows : Positive integers
        Number of columns and rows in the plot
    figsize : (x, y) tuple, optional
        Size of the figure
    **kwargs : Keyword args to pass on to plot
    """
    num_figures = len(cols_to_plot)
    num_plots = num_figures / (num_cols * num_rows)
    if num_plots * num_cols * num_rows < num_figures:
        num_plots += 1
    # Plot the cols
    old_figure_index = -1
    for item_index, col_name in enumerate(cols_to_plot):
        # Set up the subplot
        figure_index = item_index / (num_rows * num_cols)
        if figure_index != old_figure_index:
            #plt.figure(figure_index)
            plt.figure(figsize=figsize)
            plt.clf()
            plt.suptitle('Histograms %d' % figure_index)
            old_figure_index = figure_index
        subplot_index = item_index % (num_rows * num_cols)
        plt.subplot(num_rows, num_cols, subplot_index)
        # Plot
        col = df[col_name]
        hist_one_col(col)


def hist_one_col(col):
    """
    Plots a histogram one column.  Handles nans and extreme values in a
    "graceful" manner.
    """
    nan_idx = np.isnan(col)
    mean, std = col.mean(), col.std()
    extreme_idx = np.fabs(col - mean) > 10 * std
    normal_idx = np.logical_not(extreme_idx) * np.logical_not(nan_idx)

    total_count = len(col)
    nan_frac = nan_idx.sum() / float(total_count)
    extreme_frac = extreme_idx.sum() / float(total_count)

    if normal_idx.sum() > 0:
        col[normal_idx].hist(bins=50, normed=True)

    plt.title(
        '%s.    extreme: %.3f, nan: %.3f' % (col.name, extreme_frac, nan_frac))
