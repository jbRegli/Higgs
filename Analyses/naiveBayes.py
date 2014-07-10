import numpy as np
from sklearn.naive_bayes import GaussianNB


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

    yPredicted_s = []
    yProba_s = []

    for n in range(8):
        # Training:
        gnb = classifier(xsTrain_s[n], yTrain_s[n])

        #Prediction:
        label_predicted, proba_predicted = prediction(gnb, xsValidation_s[n])

        yPredicted_s.append(label_predicted)
        yProba_s.append(proba_predicted)

    return yPredicted_s, yProba_s

