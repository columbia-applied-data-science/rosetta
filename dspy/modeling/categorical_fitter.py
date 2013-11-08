"""
Various functions for fitting categorical models.
Put functions specific to logistic regression in multinomial_fitter
"""
import copy

import numpy as np

from sklearn.cross_validation import StratifiedKFold

from .. import common_math


def predict_proba_cv(clf, X, y, n_folds=5):
    """
    Returns an out-of-sample clf.predict_proba(X, y).

    Parameters
    ----------
    clf : sklearn classifier with a predict_proba method
    X : 2-D numpy array or DataFrame
    y : 1-D numpy array or Series
        Use this along with StratifiedKFold to determine splits.
    n_folds: int

    Returns
    -------
    probas : np.ndarray or series

    Note
    ----
    After folds are created, each training sets need to be large enough to
    contain at least one member of each class.  This happens iff the number
    of members of each class is less than or equal to n_folds.
    """
    X, _ = common_math.pandas_to_ndarray_wrap(X)
    y, _ = common_math.pandas_to_ndarray_wrap(y)

    n_classes = len(np.unique(y))

    # We don't want to re-fit our original classifier (changing its coeff_).
    clf = copy.deepcopy(clf)

    cv = StratifiedKFold(y, n_folds=n_folds)
    probas = np.nan * np.ones((len(y), n_classes))
    for i, (train, test) in enumerate(cv):
        if len(np.unique(y[train])) < n_classes:
            raise ValueError(
                "Training set did not contain samples from all classes."
                "Try decreasing the number of folds or use more data")
        probas[test, :] = clf.fit(X[train], y[train]).predict_proba(X[test])

    return probas
