import numpy as np
from sklearn import svm

"""
Support Vector Machine

Meta-parameters:
    NONE
"""

def classifier(xTrain, yTrain):
    """
    Train a svm classifier on xTrain and yTrain and return the trained
    classifier
    The parameter 'probability' must be set to True if we want the classifier to return a probability vector
    """
    clf= svm.SVC(probability = True)
    clf.fit(xTrain, yTrain)

    return clf


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
            clf = classifier(xsTrain_s[n], yTrain_s[n])

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
