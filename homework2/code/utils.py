from __future__ import division
import itertools


def compute_errors(predictions, labels):
    """ Generates a list of indexes of misclassified
    examples
    """
    zipped = zip(predictions, labels)
    errors = [ix for ix, tup in enumerate(zipped)
              if tup[0] != tup[1]]

    return errors


def test_scikits_model(clf, test_data, test_labels):
    preds = clf.predict(test_data)
    errors = compute_errors(preds, [t[0] for t in test_labels])

    return len(errors) / test_labels.shape[0]


def combinations(x):
    combos = np.array(list(itertools.combinations(x,2)))
    x_prime = np.array([x1 * x2.T for x1, x2 in combos])

    return x_prime


def expand_features(original_features):
    squared = original_features ** 2
    combos = combinations(features)
    return np.concatenate([original_features, squared, combos])