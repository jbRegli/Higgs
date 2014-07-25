#-*- coding: utf-8 -*-

"""
Systeme de vote entre classifier
"""
import sys
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

sys.path.append('../Analyses/')
import analyse





#####################################
### LEARNING FROM THE CLASSIFIERS ###
#####################################

def create_inputs(dMethods, valid_s, ignore= []):
    """"
    Given the dictionary dMethods, compute the lists used to learn an 'on-top'
    classifier
    """

    # List of the different predictor used:
    first_layer_predictors = []
    for key in dMethods:
        if key not in ignore:
            first_layer_predictors.append(dMethods[key]['predictor_s'])

    # List of the first layer of prediction:
    first_layer_data = []
    for key in dMethods:
        if key not in ignore:
            first_layer_data.append((valid_s[0], dMethods[key]['yProba_s'], \
                            dMethods[key]['yPredicted_s'],\
                            dMethods[key]['classif_succ']))

    return first_layer_predictors, first_layer_data


def plot_first_layer_data(first_layer_data, valid_s):
    """
    Plo
    """
    return 0

def get_prediction_FL(first_layer_predictors, xTrain_s):
    """
    Given a list of trained first level classifiers and data, realize the
    output prediction for each classifier
    Return: List of lenght 'n_subset'
            Each element of this list is a list of lenght 'n_classifiers' with the
            prediction (0 or 1) for each classifier
    """
    print("Predict the train set...")

    # If we work with the splitted dataset:
    if type(xTrain_s) == list:

        first_layer_predictions = []
        # Prediction for each subset:

        for i in range(len(xTrain_s)):
            pred_s = []

            # Prediction for each classifier on the subset:
            for j in range(len(first_layer_predictors)):

                prediction = first_layer_predictors[j][i].predict(xTrain_s[i])
                pred_s.append(prediction)

            first_layer_predictions.append(zip(*pred_s))

    else:
        pred_train_s = []
        for elmt in first_layer_predictors:
            pred_s = elmt.predict(xTrain_s)
            first_layer_predictions.append(pred_s)

    return first_layer_predictions


def train_SL(first_layer_predictions, yTrain_s, method= 'tree', parameters= {}):
    """
    Given a list of prediction from the first layer and a list of effective
    labels, train a second layer classifier to learn the correct label from the
    predictions.
    """

    assert method in set(['tree', 'logisticReg', 'svm'])

    # If we work with the splitted dataset:
    if type(yTrain_s) == list:
        second_layer_predictors = []
        print ("Training an 'on-top' classifier...")
        for i in range(len(yTrain_s)):
            if method == 'tree':
                clf = DecisionTreeClassifier(**parameters)
            elif method == 'logisticReg':
                clf = LogisticRegression(**parameters)
            elif method == 'svm':
                clf = svm.SVC(**parameters)
            else:
                raise NotImplementedError("The classifier %s is not implemented"\
                        %(method))

            # Training the "on-top" classifier for the subset i:
            clf.fit(first_layer_predictions[i], yTrain_s[i])

            second_layer_predictors.append(clf)

    else:
        if method == 'tree':
            second_layer_predictors = DecisionTreeClassifier()#parameters)
        elif method == 'logisticReg':
            second_layer_predictors = LogisticRegression()#parameters)
        elif method == 'svm':
            clf = svm.SVC()#parameters)
        else:
            raise NotImplementedError("The classifier %s is not implemented"\
                        %(method))

        print ("Training an 'on-top' classifier for the dataset...")
        second_layer_predictors.fit(first_layer_predictions, yTrain_s)

    return second_layer_predictors


def predict_SL(second_layer_predictors, first_layer_data):
    """
    Given a list of trained "on-top" classifiers and a list of prediction, predict
    the labels and the probabilities
    """

    # If we work with the splitted dataset:

    if type(second_layer_predictors) == list:

        ID_s = first_layer_data[0][0]

        first_layer_predicted_label = []

        # Creation of a list: n_subset x n_ex/subset x n_classifer
        for j in range(len(first_layer_data[0][2])):
            predictions = []
            for clfier in first_layer_data:
                predictions.append(clfier[2][j])
            first_layer_predicted_label.append(zip(*predictions))

        # Predictions:
        final_prediction_s= []
        for i in range(len(second_layer_predictors)):
            # Predict the label of a subset:
            final_label_s = second_layer_predictors[i].predict(
                                                        first_layer_predicted_label[i])

            # Predict the proba of being a signal of a subset:
            final_proba_s = second_layer_predictors[i].predict_proba(
                                                first_layer_predicted_label[i])[:,1]

            final_prediction_s.append([ID_s[i], final_proba_s, final_label_s])

    else:
        ID_s = first_layer_data[0]

        first_layer_predicted_label = []

        # Creation of a list: n_ex/subset x n_classifer
        predictions = []
        for clfier in first_layer_data:
            first_layer_predicted_label.append(clfier[2])
        first_layer_predicted_label = zip(*first_layer_predicted_label)

        # Predictions:
        final_prediction_s= []

        final_label_s = second_layer_predictors.predict(
                                                      first_layer_predicted_label)
        final_proba_s = second_layer_predictors[i].predict_proba(
                                                    first_layer_predicted_label)[:,1]

        final_prediction_s = [ID_s, final_proba_s, final_label_s]

    return final_prediction_s


def classif_classifiers_error(final_prediction, y_true_s):
    """
    Compute the error made by the "on-top" classifier
    """
    y_predicted_s = zip(*final_prediction)[2]

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

def evaluate_AMS(final_prediction, valid_s):

    # Get s and b for each group (s_s, b_s) and the final final_s and
    # final_b:

    y_predicted_s = zip(*final_prediction)[2]

    final_s, final_b, s_s, b_s = submission.get_s_b_8(y_predicted_s, valid_s[2],
                                                  valid_s[3])

    # Balance the s and b
    final_s *= 250000/25000
    final_b *= 250000/25000

    # AMS final:
    AMS = hbc.AMS(final_s , final_b)
    print ("Expected AMS score for the 'on-top' classifier : %f") %AMS

    #AMS by group
    AMS_s = []
    for i, (s,b) in enumerate(zip(s_s, b_s)):
        s *= 250000/y_predicted_s[i].shape[0]
        b *= 250000/y_predicted_s[i].shape[0]
        score = hbc.AMS(s,b)
        AMS_s.append(score)
        print("Expected AMS score for the 'on-top' classifer: group %i : %f" \
                %(i, score))
    print(" ")

    return final_s, final_b, AMS, AMS_s


def SL_classification(dMethods, valid_s, train_s, ignore= [], method='tree',
                      parameters = {}):

    # Create the necessary inputs from the dictionnary of methods:
    first_layer_predictors, first_layer_data = create_inputs(dMethods, valid_s,
                                                             ignore= ignore)

    # Predict the outputs given by the first layer of classifiers on the train set
    first_layer_predictions = get_prediction_FL(first_layer_predictors,
                                                train_s[1])
    # Train the 'on-top' predictors:
    second_layer_predictors = train_SL(first_layer_predictions, train_s[2],
                                       method= method, parameters= parameters)

    # Get the prediction done by the second layer on the valid set:
    final_prediction_s = predict_SL(second_layer_predictors, first_layer_data)

    # Compute AMS
    final_s, final_b, AMS, AMS_s =evaluate_AMS (final_prediction_s, valid_s)

    # Classification error:
    classif_succ = []
    for i in range(len(final_prediction_s)):
        ratio = accuracy_score(final_prediction_s[i][2],valid_s[2][i],
                               normalize= True)
        print("On the subset %i - correct prediction = %f") %(i, ratio)
        classif_succ.append(ratio)

    print (" ")
    # Numerical score:
    if type(final_prediction_s) == list:
        for i in range(len(final_prediction_s)):

            sum_s, sum_b = submission.get_numerical_score(
                                                        final_prediction_s[i][2],
                                                        valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, final_prediction_s[i][2].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(final_prediction_s[2],
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(final_prediction_s[2].shape[0], sum_s, sum_b)

    print(" ")

    d = {'predictor_s': second_layer_predictors,
         'yPredicted_s': np.asarray(zip(*final_prediction_s)[2]),
         'yProba_s': np.asarray(zip(*final_prediction_s)[1]),
         'final_s': final_s, 'final_b': final_b,
         'sum_s': sum_s, 'sum_b': sum_b,
         'AMS': AMS, 'AMS_s': AMS_s,
         'classif_succ': classif_succ,
         'method': method}

    return final_prediction_s, d


def get_SL_test_prediction(dMethods, dSl, xsTest_s):
    """
    Predict the output of the 'on-top' classifier on the test set
    """
    # Compute the output of the various first layer classifier on the testset:
    first_layer_test_predictions = []
    for key in dMethods:
        test_prediction_s = analyse.get_test_prediction(
                                                dMethods[key]['method'],
                                                dMethods[key]['predictor_s'],
                                                xsTest_s)[0]
        first_layer_test_predictions.append(test_prediction_s)

    first_layer_test_predictions = zip(*first_layer_test_predictions)

    # If we work with the splitted dataset:
    if type(xsTest_s) == list:
        test_prediction_s = []
        test_proba_s = []

        for n in range(len(first_layer_test_predictions)):
            label_predicted = dSl['predictor_s'][n].predict(
                                    np.asarray(first_layer_test_predictions[n]).T)
            proba_predicted = dSl['predictor_s'][n].predict_proba(
                                    np.asarray(first_layer_test_predictions[n]).T)

            test_prediction_s.append(label_predicted)

            test_proba_s.append(proba_predicted[:,1])

    else:
        test_prediction_s = dSl['predictor_s'].predict(
                                       np.asarray(first_layer_test_predictions).T)
        test_proba_s = dSl['predictor_s'].predict_proba(
                                       np.asarray(first_layer_test_predictions).T)

    return test_prediction_s, test_proba_s


