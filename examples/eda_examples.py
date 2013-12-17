"""
Examples using rosetta.modeling.eda
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as pl

from numpy.random import randn, rand

from rosetta.modeling import eda


###############################################################################
# X-Y plotting
###############################################################################
N = 1000

# Make a linear income vs. age relationship.
age = pd.Series(100 * rand(N))
age.name = 'age'
income = 10 * age + 10 * randn(N)
income.name = 'income'

# The relationship E[Y | X=x] is linear
pl.figure(1); pl.clf()
eda.plot_reducedY_vs_binnedX(age, income)


# Make a sigmoidal P[is_manager | X=x] relationship
def sigmoid(x):
    x_st = 5 * (x - x.mean()) / x.std()
    return np.exp(x_st) / (1 + np.exp(x_st))

is_manager = (rand(N) < sigmoid(age)).astype('int')
is_manager.name = 'is_manager'
pl.figure(2); pl.clf()
eda.plot_reducedY_vs_binnedX(age, is_manager)


###############################################################################
# Correlation matrix plotting
###############################################################################

# P[has_porsche] is higher for young rich people
has_porsche = (rand(N) < sigmoid(income - 0.5 * age)).astype('int')
has_porsche.name = 'has_porsche'

has_hotwheels = (rand(N) < sigmoid(-age)).astype('int')
has_hotwheels.name = 'has_hotwheels'

all_vars = pd.concat(
    [age, income, is_manager, has_porsche, has_hotwheels], axis=1)
corr = all_vars.corr()

fig = pl.figure(3); pl.clf()
eda.plot_corr_grid(corr)

fig = pl.figure(4); pl.clf()
eda.plot_corr_dendrogram(corr)
