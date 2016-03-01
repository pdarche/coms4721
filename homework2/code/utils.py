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
    combos = np.array(list(itertools.combinations(x, 2)))
    x_prime = np.array([x1 * x2.T for x1, x2 in combos])

    return x_prime


def expand_features(original_features):
    squared = original_features ** 2
    combos = combinations(features)
    return np.concatenate([original_features, squared, combos])


def k_fold_cross_validation(data, labels, num_folds):
    data_folds = np.array_split(data, num_folds)
    label_folds = np.array_split(labels, num_folds)
    data_combos = np.array(list(itertools.combinations(data_folds, 9)))
    label_combos = np.array(list(itertools.combinations(label_folds, 9)))

    for fold_num in range(num_folds):
        test_ix = (num_folds - 1) - fold_num
        train_data = np.concatenate(data_combos[fold_num])
        train_labels = np.concatenate(label_combos[fold_num])
        test_data = data_folds[test_ix]
        test_labels = label_folds[test_ix]
        training = {'data': train_data, 'labels': train_labels}
        test = {'data': test_data, 'labels': test_labels}

        yield (training, test)


def score_models(models):
    for name, Model in models:
        errors = []
        num_folds = 10
        for training, testing in k_fold_cross_validation(spam_data, spam_labels, num_folds):
            clf = Model()
            clf.fit(training['data'], training['labels'])
            error = test_scikits_model(clf, testing['data'], testing['labels'])
            errors.append(error)
        avg_error = np.sum(errors) / num_folds

        yield (name, avg_error, errors)

