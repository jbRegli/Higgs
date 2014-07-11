import numpy as np
from sklearn.naive_bayes import GaussianNB

"""
Naive bayse classifier

Meta-parameters:
    NONE
"""


def classifier(xTrain, yTrain):
    """
    Train a naive baise classifier on xTrain and yTrain and return the trained
    classifier
    """
    gnb = GaussianNB()
    gnb.fit(xTrain, yTrain)

    return gnb


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


def get_yPredicted_s(xsTrain_s, yTrain_s, xsValidation_s):
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
            gnb = classifier(xsTrain_s[n], yTrain_s[n])

            # Prediction:
            label_predicted, proba_predicted = prediction(gnb, xsValidation_s[n])

            predictor_s.append(gnb)
            yPredicted_s.append(label_predicted)
            yProba_s.append(proba_predicted)

    else:
        # Training:
        predictor_s = classifier(xsTrain_s, yTrain_s)

        #Prediction:
        yPredicted_s, yProba_s = prediction(predictor_s, xsValidation_s)

    return yPredicted_s, yProba_s


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

        test_proba_s = np.max(proba_predicted,axis=1)

    return test_prediction_s, test_proba_s
