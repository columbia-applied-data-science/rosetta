"""
Helpful module for plotting predictions
"""
import pylab as pl
pl.ion()
import numpy as np

###############################################################################
#  Globals
###############################################################################
EPS = 1e-5

###############################################################################
#  Plotter2D objects
###############################################################################


class BasePlotter2D(object):
    """
    Abstract base class for 2D plotters.  Not to be used directly.
    """
    def __init__(self):
        pass

    def plot(
        self, clf, X, y, mode='predict', contourf_kwargs={},
        scatter_kwargs={}):
        """
        Plot levelsets of clf then plot the X/y data.

        Parameters
        ----------
        clf : Trained sklearn classifier
        mode : 'predict', 'predict_proba'
            If 'predict', plot the 0/1 levelsets using clf.predict
            If 'predct_proba', plot a contour plot of clf.predict_proba.
        contourf_kwargs : Dict
            kwargs passed to pylab.contourf
        scatter_kwargs : Dict
            kwargs passed to pylab.scatter
        """
        if self.box_ends is not None:
            box_ends = self.box_ends
        else:
            box_ends = (
                X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())

        # Calling self.plot_levelsets also calls self._revise_vmin_vmax
        self.plot_levelsets(
            clf, box_ends=box_ends, mode=mode, **contourf_kwargs)

        # The colorbar must be added after plot_levelsets, before plot_data
        # If 0/1 predictions, the colorbar is misleading
        if mode == 'predict_proba':
            pl.colorbar()

        self.plot_data(X, y, **scatter_kwargs)

    def plot_levelsets(
        self, clf, box_ends=None, mode='predict', **contourf_kwargs):
        """
        Plot level sets of the model clf.

        Parameters
        ----------
        clf : Trained sklearn model
        box_ends : 4-tuple
            xmin, xmax, ymin, ymax
            plot levelsets within box defined by box_ends
        box_ends : 4-tuple
            xmin, xmax, ymin, ymax
            plot levelsets within box defined by box_ends
            Over-rides self.box_ends if self.box_ends is not set.
        mode : 'predict', 'predict_proba'
            If 'predict', plot the levelsets using clf.predict
            If 'predct_proba', plot a contour plot of clf.predict_proba.
        contourf_kwargs : Keyword arguments
            Passed to pylab.contourf
        """
        # Define box_ends
        if box_ends is None:
            assert self.box_ends is not None
            box_ends = self.box_ends

        # Define defaults and arguments depending on mode
        # func is the function that gives the levelsets
        if mode == 'predict':
            func = lambda xx, yy: clf.predict(np.c_[xx.ravel(), yy.ravel()])
        elif mode == 'predict_proba':
            func = lambda xx, yy: clf.predict_proba(
                np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            raise ValueError("mode %s is not handled" % mode)

        # Set up the colors and grid
        cmap = pl.cm.get_cmap(self.cmap)
        xx, yy = self._create_meshgrid(box_ends)

        # Get the predictions
        Z = func(xx, yy)
        Z = Z.reshape(xx.shape)

        self._revise_vmin_vmax(Z)
        N = min(50, len(np.unique(Z)))
        contourf_kwargs.setdefault('vmin', self.vmin)
        contourf_kwargs.setdefault('vmax', self.vmax)

        pl.contourf(xx, yy, Z, N, cmap=cmap, **contourf_kwargs)

        self._add_x_names()

    def _revise_vmin_vmax(self, values):
        """
        Resets self.vmin, self.vmax to the new smallest/largest values seen.
        """
        values = np.asarray(values)

        self.vmin = min(getattr(self, 'vmin', np.inf), values.min())
        self.vmax = max(getattr(self, 'vmax', -np.inf), values.max())

    def _add_x_names(self):
        """
        Add x_names to the current axes.
        """
        if self.x_names:
            pl.xlabel(self.x_names[0], fontsize='large')
            pl.ylabel(self.x_names[1], fontsize='large')

    def _create_meshgrid(self, box_ends):
        """
        Creates a 200 x 200 meshgrid of points bounded by the box_ends.
        """
        xmin, xmax, ymin, ymax = box_ends
        xstep = (xmax - xmin) / 200.
        ystep = (ymax - ymin) / 200.

        xx, yy = np.meshgrid(
            np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))

        return xx, yy


class ClassifierPlotter2D(BasePlotter2D):
    """
    For plotting 2D classifiers.

    Initialize the ClassifierPlotter2D, then train different (2-d) classifiers
    on different data sets and plot the data and level sets of the classifier.
    """
    def __init__(
        self, y_markers=None, y_names=None, x_names=None, cmap='PuBu',
        box_ends=None):
        """
        Parameters
        ----------
        y_markers : List
            E.g. ['x', 'o'] to plot the first y value as an 'x' and the second
            as 'o'.
        y_names : List
            E.g. ['negative', 'positive']
        x_names : List
            E.g. ['height', 'weight'] if x1 is 'height' and x2 is 'weight'
        cmap : String
            Use this colormap (from pylab.cm) for plots.
            append a '_r' to reverse color.  Use tab completion in IPython
            on pylab.cm. to see all possible maps, or goto:
            http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        box_ends : 4-tuple
            xmin, xmax, ymin, ymax
            plot levelsets within box defined by box_ends
        """
        self.y_markers = y_markers
        self.y_names = y_names
        self.x_names = x_names
        self.cmap = cmap
        self.box_ends = box_ends

    def plot_data(self, X, y, **scatter_kwargs):
        """
        Plot the (X, y) data as a bunch of labeled markers.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
            Numeric data only
        scatter_kwargs : Keyword arguments
            Passed to pylab.scatter()
        """
        # Set some sensible defaults
        scatter_kwargs.setdefault('s', 100)
        scatter_kwargs.setdefault('linewidths', 2)
        scatter_kwargs.setdefault('color', 'k')
        scatter_kwargs.setdefault('facecolors', 'none')

        # All possible values of y
        classes = sorted(list(np.unique(y)))

        # Plot each class with a different marker
        for i, yval in enumerate(classes):
            marker = (
                self.y_markers[i] if self.y_markers else 'ox+*01234567'
                [i % 12])
            label = self.y_names[i] if self.y_names else yval
            idx = np.where(y == yval)
            pl.scatter(
                X[idx, 0], X[idx, 1], marker=marker, label=label,
                **scatter_kwargs)

        self._add_x_names()
        pl.legend()


class RegressorPlotter2D(BasePlotter2D):
    """
    For plotting 2D regressors.

    Initialize the RegressorPlotter2D, then train different (2-d) regressors
    on different data sets and plot the data and level sets of the regressor.
    """
    def __init__(self, x_names=None, y_name=None, cmap='PuBu', box_ends=None):
        """
        Parameters
        ----------
        x_names : List
            E.g. ['height', 'weight'] if x1 is 'height' and x2 is 'weight'
        cmap : String
            Use this colormap (from pylab.cm) for plots.
            append a '_r' to reverse color.  Use tab completion in IPython
            on pylab.cm. to see all possible maps, or goto:
            http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
        box_ends : 4-tuple
            xmin, xmax, ymin, ymax
            plot levelsets within box defined by box_ends
        """
        self.x_names = x_names
        self.y_name = y_name
        self.cmap = cmap
        self.box_ends = box_ends

    def plot_data(self, X, y, **scatter_kwargs):
        """
        Plot the (X, y) data as filled in circles.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
            Numeric data only
        scatter_kwargs : Keyword arguments
            Passed to pylab.scatter()
        """
        self._revise_vmin_vmax(y)

        # Set some sensible defaults
        cmap_ = pl.cm.get_cmap(self.cmap)
        scatter_kwargs.setdefault('cmap', cmap_)
        scatter_kwargs.setdefault('marker', 'o')
        scatter_kwargs.setdefault('s', 100)
        scatter_kwargs.setdefault('linewidths', 2)
        scatter_kwargs.setdefault('color', 'k')
        scatter_kwargs.setdefault('vmin', self.vmin)
        scatter_kwargs.setdefault('vmax', self.vmax)

        pl.scatter(
            X[:, 0], X[:, 1], c=y, label=self.y_name, **scatter_kwargs)

        self._add_x_names()
        if self.y_name:
            pl.legend()

