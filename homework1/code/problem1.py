from __future__ import division
import random

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.io import loadmat
import matplotlib.pyplot as plt


ocr = loadmat('../data/ocr.mat')

testdata = ocr['testdata']
testlabels = ocr['testlabels']


def generate_training_data(sample_size, collection_size):
    """ Generates a collection of training data/label dicts

    Args:
        sample_size: int of the number of samples
            to include in each training dict
        collection_size: int of number of dicts
            desired in the collection

    Returns:
        collection: list of dicts with training data / labels

    """
    collection = []
    for sample in range(collection_size):
        sel = random.sample(xrange(60000), sample_size)
        data = ocr['data'][sel]
        labels = ocr['labels'][sel]
        collection.append({'data':data, 'labels':labels})

    return collection


def compute_error_rates(sample_size):
    """ Computes the errors rates for a collection of
    training datasets
    """
    actual = [lab[0] for lab in testlabels]
    error_rates = []
    for training in generate_training_data(sample_size, 10):
        trainingdata = training['data']
        traininglabels = training['labels']
        preds = nearest_neighbors(testdata, trainingdata, traininglabels)
        error_rate = compute_error_rate(preds, actual)
        error_rates.append(error_rate)

    return error_rates


def compute_error_rate(preds, actual):
    """ Computes the error rate of a vector of
    predicted labels
    """
    zipped = zip(preds, actual)
    errors = [tup for tup in zipped if tup[0] != tup[1]]

    return len(errors) / len(preds)


def euclidean_dist(X, Y):
    """ Computes the euclidean distance

    Args:
        X: matrix of training data
        Y: matrix of test data

    Returns:
        dists: matrix of euclidean distances
    """
    a = np.dot(X, X.T)
    b = np.dot(X, Y.T)
    c = np.dot(Y, Y.T)
    ad = np.tile(np.diagonal(a), (Y.shape[0] ,1))
    cd = np.tile(np.diagonal(c), (1, X.shape[0]))

    return ad.T - 2 * b + cd


def nearest_neighbors(testdata, trainingdata, traininglabels):
    dists = euclidean_dist(trainingdata, testdata)
    nnixs = np.argmin(dists, axis=1)
    preds = [traininglabels[ix][0] for ix in nnixs]

    return preds


def main():
    sample_sizes = [1000, 2000, 4000, 8000]
    errors = []
    for sample in sample_sizes:
        error = compute_error_rates(size)
        errors.append(error)

    errordf = pd.DataFrame(errors, colums=sample_sizes)
    melted = pd.melt(errordf)
    melted.columns = ['sample_size', 'error']
    sns.set_style("whitegrid")
    sns.pointplot(x='sample_size', y='error',
                  data=melted, markers=[''], ci=68)


if __name__ == '__main__':
    main()

