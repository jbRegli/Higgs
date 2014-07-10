import numpy as np
from sklearn.ensemble import RandomForestClassifier

"""
Random forest classifier

Meta-parameters:
    - n_trees: int
    Number of trees in the forest

"""

def classifier(xTrain, yTrain, n_trees=10):
    """
    Train a naive baise classifier on xTrain and yTrain and return the trained
    classifier
    """

    rdf = RandomForestClassifier(n_estimators= n_trees)
    rdf.fit(xTrain, yTrain)

    return rdf


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


def get_yPredicted_s(xsTrain_s, yTrain_s, xsValidation_s, n_trees= 10):
    """
    Perform the training and the prediction on the 8 sub-sets
    """

    yPredicted_s = []
    yProba_s = []

    for n in range(8):
        # Training:
        rdf = classifier(xsTrain_s[n], yTrain_s[n], n_trees= n_trees)

        #Prediction:
        label_predicted, proba_predicted = prediction(rdf, xsValidation_s[n])

        yPredicted_s.append(label_predicted)
        yProba_s.append(proba_predicted)

    return yPredicted_s, yProba_s

