from __future__ import division


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