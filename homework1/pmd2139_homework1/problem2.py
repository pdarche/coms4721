from __future__ import division
import random

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.io import loadmat
import matplotlib.pyplot as plt


news = loadmat('../data/news.mat')
vocab = open('../data/vocab.txt').readlines()
vocab = np.array([w[:-1] for w in vocab])

# 7505 x 61188 sparse matrix
p2testdata = news['testdata'].toarray()
# 7505 x 1 array
p2testlabels = news['testlabels']

# 11269 x 61188 sparse matrix
p2trainingdata = news['data'].toarray()
# 11269 x 1 array
p2traininglabels = news['labels']


def loglikelihoods(Mu, x):
    """ Computes the loglikelihod of a vector for each class

    Args:
        Mu: 20 x d matrix of estimated parameter values
        x: 1 x d vector of test values

    Returns:
        likelihoods: 20 x d matrix of likelihood values
    """
    return x.dot(np.log(Mu)) + (1-x).dot(np.log(1-Mu))


def posteriors(loglikelihoods, priors):
    """ Computes the posterior
    Args:
        loglikelihoods: Matrix of loglikelihood values
        priors: 1 x d dimensional vector of priors

    Returns:
        posteriors: Matrix of posterior values

    TODO: clean up the transpose
    """
    return loglikelihoods.T + np.log(priors)


def fit(X, Y, laplace=False):
    """ Create parameter weightings.
    Args:
        X: matrix of training feature vectors
        Y: vector of labels

    TODO: Assumes labels and training data are sorted!
    """
    breakpoints = np.where(Y[:-1] != Y[1:])[0]
    Boards = np.array(np.split(X, breakpoints))
    if laplace:
        mus = np.array([(board.sum(axis=0) + 1) / (board.shape[0] + 2) for board in Boards])
    else:
        mus = np.array([board.mean(axis=1) for board in Boards])

    return mus


def predict(params, testdata, priors):
    """ Predicts categories based on fitted parameters
    Args:
        params: 20 x 61188 matrix of parameter estimates
        testdata: n x 611888 matrix of test vectors to classify
        priors: vector of priors

    Returns:
        preds: vector of class predictions

    """
    likelihoods = np.apply_along_axis(loglikelihoods, 1, params, testdata)
    posts = posteriors(likelihoods, priors)
    preds = posts.argmax(axis=1)

    return preds + 1 # adjust the indicies to match the labels


def part_a(params, priors):
    training_preds = predict(params, p2trainingdata, priors)
    training_errors = compute_error_rate(training_preds,
                                         [l[0] for l in p2traininglabels])
    test_preds = predict(params, p2testdata, priors)
    test_errors = compute_error_rate(test_preds,
                                     [l[0] for l in p2testlabels])

    print """
        Training Error: {}
        Test Error:     {}
    """.format(training_errors, test_errors)


def part_b(params, vocab):
    alphas = ((np.log(params) - np.log(1-params)).T + np.log(priors))
    tta = np.argsort(alphas.T)[:, :20]

    top_words = []
    for arr in tta:
        top_word = list(vocab[arr])
        top_words.append(top_word)

    twa = np.apply_along_axis(lambda x: vocab[x], 0, tta)


def main():
    breakpoints = np.where(p2traininglabels[:-1] != p2traininglabels[1:])[0]
    Boards = np.array(np.split(p2trainingdata.toarray(), breakpoints))
    priors = np.array([(board.shape[0] / Boards.shape[0]) / 100 for board in Boards])
    params = fit(p2trainingdata, p2traininglabels, laplace=True)

    part_a(params, priors)
    part_b(params, vocab)


if __name__ == '__main__':
    main()

