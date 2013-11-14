"""
An example script plotting some regressors using prediction_plotter.
"""
import numpy as np
import matplotlib.pylab as pl
pl.ion()

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.gaussian_process import GaussianProcess

from rosetta.modeling import prediction_plotter


###############################################################################
# Make training data
###############################################################################
# X data
N = 10 # Number of data points
x1min, x1max, x2min, x2max = 0., 200., 1., 20.
x1 = np.random.randint(x1min, x1max+1, size=N)
x2 = np.random.randint(x2min, x2max+1, size=N)
X = np.c_[x1, x2]

# y data
# y is bigger near the center of the support of X
# noise is added to y
x1mid = (x1max - x1min) / 2.
x2mid = (x2max - x2min) / 2.
center = np.array([x1mid, x2mid])
width1, width2 = x1mid/2, x2mid/2

noise_level = 0.2
product = ((X[:, 0] - center[0]) / width1)**4 + ((X[:, 1] - center[1]) / width2)**4
y = np.exp(- product / 2. ) + noise_level * np.random.randn(N)

###############################################################################
# Initialize the plotter
###############################################################################
plotter = prediction_plotter.RegressorPlotter2D(
    x_names=['age', 'height'], y_name='measured-data')


###############################################################################
# Linear Regression
###############################################################################
pl.figure(1)
pl.clf()

clf = LinearRegression().fit(X, y)

plotter.plot(clf, X, y)
pl.title("LinearRegression")
pl.colorbar()

###############################################################################
# K Nearest Neighbors
###############################################################################
pl.figure(2)
pl.clf()

n_neighbors = max(2, N/10)
clf = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)

plotter.plot(clf, X, y)
pl.title("KNeighborsRegressor")
pl.colorbar()

###############################################################################
# Support Vector Regressor (SVR, a.k.a. RVM)
###############################################################################
pl.figure(4)
pl.clf()

clf = svm.SVR(gamma=0.01).fit(X, y)

plotter.plot(clf, X, y)
pl.title("svm.SVR")
pl.colorbar()

###############################################################################
# Gaussian Process Models
###############################################################################
pl.figure(5)
pl.clf()

clf = GaussianProcess(regr='linear', theta0=2).fit(X, y)

plotter.plot(clf, X, y)
pl.title("GaussianProcess")
pl.colorbar()
