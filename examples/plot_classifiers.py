"""
An example script plotting some classifiers using prediction_plotter.
"""
import numpy as np
import matplotlib.pylab as pl
pl.ion()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from rosetta.modeling import prediction_plotter


###############################################################################
# Make training data
###############################################################################
# X data
N = 50 # Number of data points
x1min, x1max, x2min, x2max = 0., 200., 1., 20.
x1 = np.random.randint(x1min, x1max+1, size=N)
x2 = np.random.randint(x2min, x2max+1, size=N)
X = np.c_[x1, x2]

# y data
# Probability y = 1 is highest near the mid-points of X
x1mid = (x1max - x1min) / 2.
x2mid = (x2max - x2min) / 2.
center = np.array([x1mid, x2mid])
width1, width2 = x1mid/2, x2mid/2

product = (
    ((X[:, 0] - center[0]) / width1)**4 + ((X[:, 1] - center[1]) / width2)**4)
pdf_arr = np.exp(- product / 2. )
y = (np.random.rand(N) < pdf_arr).astype('int')

# Names
x_names = ['doc-length', 'num-recipients']
y_names = ['non-relevant', 'relevant']
y_markers = ['x', 'o']


###############################################################################
# Initialize the plotter
###############################################################################
plotter = prediction_plotter.ClassifierPlotter2D(
    y_markers=y_markers, y_names=y_names, x_names=x_names)

###############################################################################
# Decision Tree
###############################################################################
pl.figure(1)
pl.clf()

clf = DecisionTreeClassifier().fit(X, y)

plotter.plot(clf, X, y)
pl.title("DecisionTreeClassifier")

###############################################################################
# Random Forest
###############################################################################
pl.figure(2)
pl.clf()

clf = RandomForestClassifier(n_estimators=100, max_features=None).fit(X, y)

plotter.plot(clf, X, y, mode='predict_proba', contourf_kwargs={'alpha':0.8})
pl.title("RandomForestClassifier")


###############################################################################
# Logistic Regression
###############################################################################
pl.figure(3)
pl.clf()

clf = LogisticRegression(penalty='l2', C=1000).fit(X, y)

plotter.plot(clf, X, y, mode='predict_proba')
pl.title("LogisticRegression")


###############################################################################
# SVM
###############################################################################
pl.figure(4)
pl.clf()

clf = svm.SVC(gamma=0.01).fit(X, y)

plotter.plot(clf, X, y)
pl.title("svm.SVC")


###############################################################################
# K Nearest Neighbors
###############################################################################
pl.figure(5)
pl.clf()

clf = KNeighborsClassifier(n_neighbors=N/10).fit(X, y)

plotter.plot(clf, X, y)
pl.title("KNeighborsClassifier")
