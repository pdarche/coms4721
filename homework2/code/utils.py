from __future__ import division

import itertools


def compute_errors(predictions, labels):
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
        classifiers = []
        num_folds = 10
        for training, testing in k_fold_cross_validation(spam_data, spam_labels, num_folds):
            if 'Expanded' in name:
                training['data'] = np.array([expand_features(x) for x in training['data']])
                testing['data'] = np.array([expand_features(x) for x in testing['data']])
            clf = Model()
            clf.fit(training['data'], training['labels'])
            error = test_scikits_model(clf, testing['data'], testing['labels'])
            errors.append(error)
            classifiers.append(clf)
        avg_error = np.sum(errors) / num_folds

        yield (name, avg_error, errors, classifiers)


def select_classifer(scored_models):
    errors = np.concatenate([s[2] for s in scored_models])
    classifiers = np.concatenate([s[3] for s in scored_models])
    min_error_ix = np.argmin(errors)
    name_ix = min_error_ix//10
    clf_name = [m[0] for m in scored_models][name_ix]
    return (clf_name, errors[min_error_ix], classifiers[min_error_ix])


def cross_validation_error_rates(scored_models):
    for scored in scored_models:
        print "{} Cross Validation Error Rate: {}".format(scored[0], scored[1])


def best_classifier_training_error_rate(scored_models):
    clf_name, clf_error, clf = select_classifer(scored_models)

    print "%s Training Error Rate: %.4f" % (clf_name, clf_error)


def best_classfier_test_error_rate(scored_models, test_data, test_labels):
    clf_name, clf_error, clf = select_classifer(scored_models)
    if 'Expanded' in clf_name:
        test_data = np.array([expand_features(x) for x in test_data])
    test_error = test_scikits_model(clf, test_data, test_labels)

    print "%s Testing Error Rate %.4f" % (clf_name, clf_error)



