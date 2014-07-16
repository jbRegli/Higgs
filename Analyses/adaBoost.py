#-*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import AdaBoostClassifier

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

def classifier(xTrain, yTrain, base_estimators, n_estimators, learning_rate, algorithm, random_state):
    """
    Train a ada classifier on xTrain and yTrain and return the trained
    classifier
    """
    ada = AdaBoostClassifier() 
    ada.fit(xTrain, yTrain)

    return ada


def prediction(predictor, testset):
    """
    Given a dataset and a classifier, compute the prediction (label and proba - if
    available).
    This function can be use for validation as well as for the test.
    """
    # Label prediction:
    label_predicted = predictor.predict(testset)
    # Probability of being in each label
    proba_predicted = predictor.predict_proba(testset)

    return label_predicted, proba_predicted


def get_yPredicted_s(xsTrain_s, yTrain_s, xsValidation_s, base_estimators = None, \
                    n_estimators = 50, learning_rate = 1., algorithm = 'SAMME.R', random_state = None):
    """
    Perform the training and the prediction on the 8 sub-sets
    """
    # If we work with the splitted dataset:
    if type(xsTrain_s) == list:
        predictor_s = []
        yPredicted_s = []
        yProba_s = []

        for n in range(len(xsTrain_s)):
            # Training:
            clf = classifier(xsTrain_s[n], yTrain_s[n], base_estimators = base_estimators, \
                            n_estimators = n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)

            # Prediction:
            label_predicted, proba_predicted = prediction(clf, xsValidation_s[n])

            predictor_s.append(clf)
            yPredicted_s.append(label_predicted)
            yProba_s.append(proba_predicted)

    else:
        # Training:
        predictor_s = classifier(xsTrain_s, yTrain_s)

        #Prediction:
        yPredicted_s, yProba_s = prediction(predictor_s, xsValidation_s)

    return predictor_s, yPredicted_s, yProba_s


def get_test_prediction(predictor_s, xsTest_s):
    """
    Predict the output of this classifier on the test set
    """

    # If we work with the splitted dataset:
    if type(xsTest_s) == list:
        test_prediction_s = []
        test_proba_s = []

        for n in range(len(xsTest_s)):
            label_predicted, proba_predicted = prediction(predictor_s[n],
                                                            xsTest_s[n])

            test_prediction_s.append(label_predicted)
            test_proba_s.append(np.max(proba_predicted,axis=1))

    else:
        test_prediction_s , proba_predicted = prediction(predictor_s, xsTest_s)

        test_proba_s = proba_predicted[1]

    return test_prediction_s, test_proba_s


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
