"""
Main Driver Script for the Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classfiers other than AveragedPerceptron are from scikits-learn
LinearDiscriminantAnalysis      : http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
QuadraticDiscriminantAnalysis   : http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
LogisticRegression              : http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Code for the AveragedPerceptron can be found in avgperceptron.py
Code for cross-validation and other methods is in utils.py
"""
import warnings


from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from avgperceptron import AveragedPerceptron

from utils import *

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    spam = loadmat('../data/spam_fixed.mat')

    models = [
        ('Averaged Perceptron', AveragedPerceptron),
        ('Logistic Regression', LogisticRegression),
        ('QDA', QuadraticDiscriminantAnalysis),
        ('LDA', LinearDiscriminantAnalysis),
        ('Averaged Perceptron Expanded', AveragedPerceptron),
        ('Logistic Regression Expanded', LogisticRegression)
    ]
    print "Scoring Models"
    scored_models = list(score_models(models, spam['data'], spam['labels']))

    # 1. Cross-validation error rates for all methods
    cross_validation_error_rates(scored_models)

    # 2. Training error rate of the classifier learned by the selected method (and state which method was chosen)
    best_classifier_training_error_rate(scored_models)

    # 3. Test error rate for the learned classifier
    best_classfier_test_error_rate(
        scored_models, spam['testdata'], spam['testlabels'])

