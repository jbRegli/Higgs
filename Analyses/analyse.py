import numpy as np
import imp
import sys

import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda
import gradientBoosting

sys.path.append('../')
import HiggsBosonCompetition_AMSMetric_rev1 as hbc
import preTreatment
import submission
import xgBoost

sys.path.append('../PostTreatment')
import tresholding


def analyse(train_s, train2_s, valid_s, method_name, kwargs={}):
    """
    methode name = string, name of the method (eg :"naiveBayes")
    kwargs = dictionnary of the paraters of the method
    train_s = training set for the classifier(s)
    train2_s = training set for the meta parameters (eg : the best treshold)
    valid_s : validation set
    None of the set must be empty !
    """
    # Prediction on the validation set:
    print("------------------- Analyse: %s -----------------------") \
                        %(method_name)

    classifier_s = eval(method_name).train_classifier(train_s[1], train_s[2], kwargs)

    yProbaValid_s = eval(method_name).predict_proba(classifier_s, valid_s[1])
    yProbaTrain2_s = eval(method_name).predict_proba(classifier_s, train2_s[1])

    # Convert the validations vectors four 's' classes into one single s
    # classe
    if type(valid_s[2]) == list:
        for i in range(len(valid_s[2])):
            for j in range(valid_s[2][i].shape[0]):
                if valid_s[2][i][j] >=1:
                    valid_s[2][i][j] = 1

    # Convert the train2 vectors four 's' classes into one single s
    # classe
    if type(train2_s[2]) == list:
        for i in range(len(train2_s[2])):
            for j in range(train2_s[2][i].shape[0]):
                if train2_s[2][i][j] >=1:
                    train2_s[2][i][j] = 1

    # Let's define the vectors of probabilities of being 's'
    # Train2 set
    if type(yProbaTrain2_s) == list:
        yProbaTrain2Binary_s = []
        for i in range(8):
            yProbaTrain2Binary_s.append(np.zeros(len(yProbaTrain2_s[i][:,1])))
        for i in range(8):
            for j in range(len(yProbaTrain2_s[i][:,1])):
                yProbaTrain2Binary_s[i][j] = 1 - yProbaTrain2_s[i][j][0]
    else:
        yProbaTrain2Binary_s = np.zeros(len(yProbaTrain2_s[i][:,1]))
        for j in range(len(yProbaTrain2_s[i][:,1])):
            yProbaTrain2Binary_s[j] = 1 - yProbaTrain2_s[j][0]

    # Validation set
    if type(yProbaValid_s) == list:
        yProbaValidBinary_s = []
        for i in range(8):
            yProbaValidBinary_s.append(np.zeros(len(yProbaValid_s[i][:,1])))
        for i in range(8):
            for j in range(len(yProbaValid_s[i][:,1])):
                yProbaValidBinary_s[i][j] = 1 - yProbaValid_s[i][j][0]
    else:
        yProbaValidBinary_s = np.zeros(len(yProbaValid_s[i][:,1]))
        for j in range(len(yProbaValid_s[i][:,1])):
            yProbaValidBinary_s[j] = 1 - yProbaValid_s[j][0]

    # If we work with lists, let's get the concatenated vectors:
    # TRAIN SET
    if type(train_s[3]) ==list:
        weightsTrain_conca = preTreatment.concatenate_vectors(train_s[3])
    else:
        weightsTrain_conca = train_s[3]
    # VALID SET
    # Validation Vectors
    if type(valid_s[2]) == list:
        yValid_conca = preTreatment.concatenate_vectors(valid_s[2])
    else:
        yValid_conca = valid_s[2]
    # Weights Vectors
    if type(valid_s[3]) == list:
        weightsValid_conca = preTreatment.concatenate_vectors(valid_s[3])
    else:
        weightsValid_conca = valid_s[3]
    # Binary Proba Vectors
    if type(yProbaValidBinary_s) == list:
        yProbaValidBinary_conca = preTreatment.concatenate_vectors(
                                                              yProbaValidBinary_s)
    else:
        yProbaValidBinary_conca = yProbaValidBinary_s
    # All Proba Vectors
    if type(yProbaValid_s) == list:
        yProbaValid_conca = preTreatment.concatenate_vectors(yProbaValid_s)
    else:
        yProbaValid_conca = yProbaValid_s

    #TRAIN2 SET
    # Validation Vectors
    if type(train2_s[2]) == list:
        yTrain2_conca = preTreatment.concatenate_vectors(train2_s[2])
    else:
        yTrain2_conca = train2_s[2]
    # Weights Vectors
    if type(train2_s[3]) == list:
        weightsTrain2_conca = preTreatment.concatenate_vectors(train2_s[3])
    else:
        weightsTrain2_conca = train2_s[3]
    # Binary Proba Vectors
    if type(yProbaTrain2Binary_s) == list:
        yProbaTrain2Binary_conca = preTreatment.concatenate_vectors(
                                                            yProbaTrain2Binary_s)
    else:
        yProbaTrain2Binary_conca = yProbaTrain2Binary_s
    # All Proba Vectors
    if type(yProbaTrain2_s) == list:
        yProbaTrain2_conca = preTreatment.concatenate_vectors(yProbaTrain2_s)
    else:
        yProbaTrain2_conca = yProbaTrain2_s

    # Let's rebalance the weight so their sum is equal to the total sum
    # of the train set
    sumWeightsTotal = sum(weightsTrain_conca)+sum(weightsTrain2_conca)+sum(weightsValid_conca)
    weightsTrain2_conca *= sumWeightsTotal/sum(weightsTrain2_conca)
    weightsValid_conca *= sumWeightsTotal/sum(weightsValid_conca)
    for i in range(8):
        train2_s[3][i] *= sumWeightsTotal/sum(weightsTrain2_conca)
        valid_s[3][i] *= sumWeightsTotal/sum(weightsValid_conca)

    # Let's get the best global treshold on the train2 set
    AMS_treshold_train2, best_treshold_global = tresholding.best_treshold(yProbaTrain2Binary_conca,
                                                     yTrain2_conca, weightsTrain2_conca)
    yPredictedValid_conca_treshold = tresholding.get_yPredicted_treshold(
                                                        yProbaValidBinary_conca,
                                                        best_treshold_global)
    # Let's get the best ratio treshold on the train2 set
    AMS_ratio_global_train2, best_ratio_global = tresholding.best_ratio(yProbaTrain2Binary_conca,
                                               yTrain2_conca,
                                               weightsTrain2_conca)

    yPredictedValid_conca_ratio_global = tresholding.get_yPredicted_ratio(
                                                        yProbaValidBinary_conca,
                                                        best_ratio_global)
    # Let's get the best ratios combinaison
    if type(train_s[2]) == list:
        AMS_ratio_combinaison_train2, best_ratio_combinaison = tresholding.best_ratio_combinaison_global(
                                                                            yProbaTrain2Binary_s,
                                                                            train2_s[2],
                                                                            train2_s[3],
                                                                            30)
        yPredictedValid_ratio_comb_s, yPredictedValid_conca_ratio_combinaison = tresholding.get_yPredicted_ratio_8(
                                                                                    yProbaValidBinary_s,
                                                                                    best_ratio_combinaison)

    # Let's compute the final s and b for each method
    s_treshold, b_treshold = submission.get_s_b(
                                                yPredictedValid_conca_treshold,
                                                yValid_conca,
                                                weightsValid_conca)
    s_ratio_global, b_ratio_global = submission.get_s_b(
                                            yPredictedValid_conca_ratio_global,
                                            yValid_conca,
                                            weightsValid_conca)
    if type(train_s[2]) == list:
        s_ratio_combinaison, b_ratio_combinaison = submission.get_s_b(
                                            yPredictedValid_conca_ratio_combinaison,
                                            yValid_conca,
                                            weightsValid_conca)

    # AMS final:
    AMS_treshold_valid = hbc.AMS(s_treshold, b_treshold)
    AMS_ratio_global_valid = hbc.AMS(s_ratio_global, b_ratio_global)
    if type(train_s[2]) == list:
        AMS_ratio_combinaison_valid = hbc.AMS(s_ratio_combinaison, b_ratio_combinaison)
    """
    #AMS by group:
    if type(train_s[2]) == list:
        AMS_s = []
        for i, (s,b) in enumerate(zip(s_s, b_s)):
            s *= 250000/yPredictedValid_s[i].shape[0]
            b *= 250000/yPredictedValid_s[i].shape[0]
            score = hbc.AMS(s,b)
            AMS_s.append(score)
    """
    # Classification error:
    classif_succ_treshold = eval(method_name).get_classification_error(yPredictedValid_conca_treshold,
                                                       yValid_conca,
                                                       normalize= True)
    classif_succ_ratio_global = eval(method_name).get_classification_error(yPredictedValid_conca_ratio_global,
                                                       yValid_conca,
                                                       normalize= True)
    classif_succ_ratio_combinaison = eval(method_name).get_classification_error(yPredictedValid_conca_ratio_combinaison,
                                                       yValid_conca,
                                                       normalize= True)



    # Numerical score:
    """
    if type(yProbaValid_s) == list:
        sum_s_treshold_s = []
        sum_b_treshold_s = []
        sum_s_ratio_global_s = []
        sum_b_ratio_global_s = []
        sum_s_ratio_combinaison_s = []
        sum_b_ratio_combinaison_s = []
        for i in range(len(yPredictedValid_s)):
            # treshold
            sum_s_treshold, sum_b_treshold = submission.get_numerical_score(yPredictedValid_conca_treshold_s[i],
                                                          valid_s[2][i])
            sum_s_treshold_s.append(sum_s)
            sum_b_treshold_s.append(sum_b)
            # ratio global
            sum_s_ratio_global, sum_b_ratio_global = submission.get_numerical_score(yPredictedValid_conca_ratio_global_s[i],
                                                          valid_s[2][i])
            sum_s_ratio_global_s.append(sum_s_ratio_global)
            sum_b_ratio_global_s.append(sum_b_ratio_global)
            # ratio combinaison
            sum_s_ratio_combinaison, sum_b_ratio_combinaison = submission.get_numerical_score(yPredictedValid_conca_ratio_combinaison_s[i],
                                                          valid_s[2][i])
            sum_s_ratio_combinaison_s.append(sum_s_ratio_combinaison)
            sum_b_ratio_combinaison_s.append(sum_b_ratio_combinaison)




    else:
        sum_s, sum_b = submission.get_numerical_score(yPredictedValid_s,
                                                           valid_s[2])
    """
    d = {'classifier_s':classifier_s,
         'yPredictedValid_conca_treshold': yPredictedValid_conca_treshold,
         'yPredictedValid_conca_ratio_global' : \
                 yPredictedValid_conca_ratio_global,
         'yProbaTrain2_s': yProbaTrain2_s,
         'yProbaTrain2Binary_s': yProbaTrain2Binary_s,
         'yProbaTrain2_conca': yProbaTrain2_conca,
         'yProbaTrain2Binary_conca': yProbaTrain2Binary_conca,
         'yProbaValid_s':yProbaValid_s,
         'yProbaValidBinary_s':yProbaValidBinary_s,
         'yProbaValid_conca':yProbaValid_conca,
         'yProbaValidBinary_conca': yProbaValidBinary_conca,
         'AMS_treshold_train2':AMS_treshold_train2,
         'AMS_ratio_global_train2':AMS_ratio_global_train2,
         'AMS_treshold_valid':AMS_treshold_valid,
         'AMS_ratio_global_valid':AMS_ratio_global_valid,
         'best_treshold_global' : best_treshold_global,
         'best_ratio_global':best_ratio_global,
         'classif_succ_treshold': classif_succ_treshold,
         'classif_succ_ratio_global': classif_succ_ratio_global,
         'method': method_name,
         'parameters': kwargs}

    if type(train_s[2])==list:
        d['yPredictedValid_conca_ratio_combinaison'] = yPredictedValid_conca_ratio_combinaison
        d['AMS_ratio_combinaison_train2'] = AMS_ratio_combinaison_train2
        d['AMS_ratio_combinaison_valid'] = AMS_ratio_combinaison_valid,
        d['best_ratio_combinaison'] = best_ratio_combinaison,
        d['classif_succ_ratio_combinaison'] = classif_succ_ratio_combinaison

    return d


def get_test_prediction(method_name, classifier_s, xsTest_s):
    return eval(method_name).get_test_prediction(classifier_s, xsTest_s)



