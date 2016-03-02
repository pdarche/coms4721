from __future__ import division

import numpy as np

from utils import compute_errors


class AveragedPerceptron(object):
    def __init__(self):
        self.num_iterations = 64

    def _prep_examples(self, X, y):
        return zip(X, y)

    def fit(self, training_data, training_labels):
        examples = self._prep_examples(training_data, training_labels)
        weights = np.zeros(examples[0][0].shape)
        cweights = np.zeros(examples[0][0].shape)
        bias = 0
        counter = 1

        for iteration in range(0, self.num_iterations):
            np.random.shuffle(examples)
            for features, label in examples:
                if np.dot(features, weights) <= 0:
                    weights = weights + (label * features)
                    cweights = cweights + (label * counter * features)
                counter += 1

        self.model = weights - ((1/counter) * cweights)

    def predict(self, test_data):
        return np.array([self._predict_one(features)
                         for features in test_data])

    def _predict_one(self, features):
        activation = np.dot(features, self.model)
        if activation > 0:
            return 1
        else:
            return -1


def avg_perceptron_fit(num_iterations, examples):
    weights = np.zeros(examples[0][0].shape)
    cweights = np.zeros(examples[0][0].shape)
    bias = 0
    counter = 1

    for iteration in range(0, num_iterations):
        np.random.shuffle(examples)
        for features, label in examples:
            if np.dot(features, weights) <= 0:
                # update the weights for this iteration
                weights = weights + (label * features)
                # update the cached weights
                cweights = cweights + (label * counter * features)
            counter += 1

    return weights - ((1/counter) * cweights)


def predict(features,  weights):
    """
    Predicts a label (1, -1) given a vector
    of features and weights

    Args:
        features:
        weights:

    Return:
        prediction: int of -1 or 1
    """
    prediction = np.dot(features, weights)
    if prediction > 0:
        return 1
    else:
        return -1

def update_weights(prediction, label, features, weights):
    """
    Args:
        prediction: int of predicted label (1 or -1)
        label: the true label of the data point (1 or -1)
        features: numpy array of feature values
        weights: 1d numpy array of weights for the features

    Returns:
        weights:
    """
    if prediction != label:
        weights = weights + (label * features)

    return weights


def perceptron_fit(examples):
    """
    Generates a vector of weights

    Args:
        examples: vector of feature, label tuples

    Returns:
        weights: d-dimensional vector of weights
    """
    weights = np.zeros(examples[0][0].shape)
    for features, label in examples:
        prediction = predict(features, weights)
        weights = update_weights(prediction, label[0],
                                 features, weights)
    return weights


def test_model(testdata, testlabels, trained_weights):
    """ Generates predictions from a trained weight vector """
    preds = [predict(features, trained_weights)
             for features in testdata]
    errors = compute_errors(preds, [t[0] for t in testlabels])

    return len(errors) / testlabels.shape[0]

