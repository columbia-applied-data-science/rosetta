"""
Examples using rosetta.modeling.eda
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as pl

from rosetta.modeling import eda

N = 1000

# Make a linear income vs. age relationship.
age = pd.Series(100 * np.random.rand(N))
age.name = 'age'
income = 10 * age + 10 * np.random.randn(N)

# The relationship E[Y | X=x] is linear
pl.figure(1); pl.clf()
eda.plot_reducedY_vs_binnedX(age, income)


# Make a sigmoidal P[cancer | X=x] relationship
def sigmoid(x):
    x_st = 5 * (x - x.mean()) / x.std()
    return np.exp(x_st) / (1 + np.exp(x_st))

has_cancer = (np.random.rand(N) < sigmoid(age)).astype('int')
pl.figure(2); pl.clf()
eda.plot_reducedY_vs_binnedX(age, has_cancer)
