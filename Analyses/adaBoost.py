#-*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
"""
Ada Boost Classifier

Meta-parameters:
    base_estimator : object, optional (default=DecisionTreeClassifier)
    The base estimator from which the boosted ensemble is built.
    Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.

    n_estimators : integer, optional (default=50)
    The maximum number of estimators at which boosting is terminated.
    In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
    Learning rate shrinks the contribution of each classifier by learning_rate.
    There is a trade-off between learning_rate and n_estimators.

    algorithm : ‘SAMME’, ‘SAMME.R’, optional (default=’SAMME.R’)
    If ‘SAMME.R’ then use the SAMME.R real boosting algorithm. base_estimator must support calculation of class probabilities.
    If ‘SAMME’ then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically converges faster than SAMME,
    achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used by np.random.


"""

def train_classifier(xTrain_s, yTrain_s, kwargs):
    """
    Train a naive baise classifier on xTrain and yTrain and return the trained
    classifier
    """
    if type(xTrain_s) != list:
        classifier_s = AdaBoostClassifier(**kwargs)
        classifier_s.fit(xTrain_s, yTrain_s)

    else:
        classifier_s = train_classifier_8(xTrain_s, yTrain_s, kwargs)

    return classifier_s

def train_classifier_8(xsTrain_s, yTrain_s, kwargs):
    """
    performs the training and returns the predictors
    """
    # If we work with the splitted dataset:

    classifier_s = []

    for n in range(len(xsTrain_s)):
        # Training:
        classifier = train_classifier(xsTrain_s[n], yTrain_s[n], kwargs)
        classifier_s.append(classifier)

    return classifier_s

def predict_proba(classifier_s, dataset_s):
    """
    Given a dataset and a classifier, compute the proba prediction
    This function can be use for validation as well as for the test.
    """
    if type(classifier_s) != list:
        # Probability of being in each label
        proba_predicted_s = classifier_s.predict_proba(dataset_s) #[:,1]

    else:
        proba_predicted_s = predict_proba_8(classifier_s, dataset_s)

    return proba_predicted_s

def predict_proba_8(classifier_s, dataset_s):
    """
    Predict the output of this classifier on the the dataset divided in 8 groups
    """

    # If we work with the splitted dataset:
    proba_predicted_s = []

    for n in range(len(dataset_s)):
        proba_predicted = predict_proba(classifier_s[n], dataset_s[n])
        proba_predicted_s.append(proba_predicted)

    return proba_predicted_s


def get_classification_error(y_predicted_s, y_true_s, normalize= True):

    if type(y_predicted_s) == list:
        prediction_error_s = []

        for n in range(len(y_predicted_s)):
            prediction_error_s.append(accuracy_score(y_true_s[n],
                                                     y_predicted_s[n],
                                                     normalize=normalize))
    else:
        prediction_error_s = accuracy_score(y_true_s, y_predicted_s,
                                            normalize=normalize)

    return prediction_error_s
