from __future__ import division

import numpy as np

from utils import compute_errors
from sklearn.naive_bayes import GaussianNB


def fit(training_data, training_labels):
    clf = GaussianNB()
    clf.fit(training_data, training_labels)

    return clf


def test_logistic_model(clf, test_data, test_labels):
    preds = clf.predict(test_data)
    errors = compute_errors(preds, [t[0] for t in test_labels])

    return len(errors) / test_labels.shape[0]