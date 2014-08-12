# -*- coding: utf-8 -*-
"""
test the effect of the tresholding of label vector on the AMS score performance
Choose the range of the events labels you wanna keep in the ratio_s array.
returns one txt file by method, with the AMS for each group and each ratio.
TODO : add a visualisation function to see the AMS = f(ratio) for each group
"""

import sys

import numpy as np
import time
from sklearn.metrics import accuracy_score

import tokenizer
import preTreatment

sys.path.append('PostTreatment/')
import preTreatment
import tresholding
import combineClassifiers
import mergeClassifiers

import tresholding

import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc


sys.path.append('Analyses/')
import analyse # Function computing an analyse for any method in the good format
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda
import gradientBoosting
import xgBoost

def train(max_depth, n_rounds):

    ###############
    ### IMPORT ####
    ###############
    # Importation parameters:
    split= True
    normalize = True
    noise_var = 0.
    train_size = 200000
    train_size2 = 25000
    valid_size = 25000
    remove_999 = False

    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, train2_s, valid_s,  test_s = tokenizer.extract_data(split= split, \
                                             normalize= normalize, \
                                             remove_999 = remove_999, \
                                             noise_variance= noise_var, \
                                             n_classes = "multiclass", \
                                             train_size = train_size, \
                                             train_size2 = train_size2, \
                                             valid_size = valid_size)

    
    #RANDOM FOREST:
    #kwargs_grad = {}
    #kwargs_rdf = {'n_estimators': 100}
    print "Training on the train set ..."
    #predictor_s = randomForest.train_classifier(train_s[1], train_s[2], kwargs_rdf)

    #XGBOOST
    kwargs_xgb = {'bst_parameters': \
                {'booster_type': 0, \
                     #'objective': 'binary:logitraw',
                     'objective': 'multi:softprob', 'num_class': 5,
                     'bst:eta': 0.1, # the bigger the more conservative
                     'bst:subsample': 1, # prevent over fitting if <1
                     'bst:max_depth': max_depth, 'eval_metric': 'auc', 'silent': 1,
                     'nthread': 8 }, \
                'n_rounds': n_rounds}

    predictor_s = xgBoost.train_classifier(train_s[1], train_s[2], train_s[3], 550000, kwargs_xgb)
    
    #TEST / SUBMISSION
    """
    yProbaTest_s = []
    yProbaTestBinary_s = []

    print "Classifying the test set..."
    for i in range(8):
        yProbaTest = xgBoost.predict_proba(predictor_s[i], test_s[1][i])
        yProbaTest_s.append(yProbaTest)
    print "Making the binary proba vector..."
    for i in range(8):
        yProbaTestBinary_s.append(np.zeros(yProbaTest_s[i].shape[0]))
    for i in range(8):
        for j in range(yProbaTest_s[i].shape[0]):
            yProbaTestBinary_s[i][j] = 1 - yProbaTest_s[i][j][0]

    print "Concatenating the vectors..."
    yProbaTestBinary = preTreatment.concatenate_vectors(yProbaTestBinary_s)
    IDs = preTreatment.concatenate_vectors(test_s[0])


    yProbaTestBinaryRanked = submission.rank_signals(yProbaTestBinary)
    
    yPredictedTest = tresholding.get_yPredicted_ratio(yProbaTestBinary, 0.15)

    s = submission.print_submission(IDs, yProbaTestBinaryRanked, yPredictedTest, "newAMSmesure") 

    
    """
    # TRAIN AND VALID
    
    yPredictedTrain2_s = []
    yProbaTrain2_s = []
    yProbaTrain2Binary_s = []
    yPredictedValid_s = []
    yProbaValid_s = []
    yProbaValidBinary_s = []

    print "Classifying the train2 set..."
    for i in range(8):
        yProbaTrain2 = xgBoost.predict_proba(predictor_s[i], train2_s[1][i])
        yProbaTrain2_s.append(yProbaTrain2)
    print "Classifying the valid set..."
    for i in range(8):
        yProbaValid = xgBoost.predict_proba(predictor_s[i], valid_s[1][i])
        yProbaValid_s.append(yProbaValid)

    print "Making the binary proba vector..."
    for i in range(8):
        yProbaTrain2Binary_s.append(np.zeros(yProbaTrain2_s[i].shape[0]))
        yProbaValidBinary_s.append(np.zeros(yProbaValid_s[i].shape[0]))
    for i in range(8):
        for j in range(yProbaTrain2_s[i].shape[0]):
            yProbaTrain2Binary_s[i][j] = 1 - yProbaTrain2_s[i][j][0]
        for j in range(yProbaValid_s[i].shape[0]):
            yProbaValidBinary_s[i][j] = 1 - yProbaValid_s[i][j][0]

    print "Concatenating the vectors..."
    yProbaTrain2Binary = preTreatment.concatenate_vectors(yProbaTrain2Binary_s)
    yProbaValidBinary = preTreatment.concatenate_vectors(yProbaValidBinary_s)
    yTrain2 = preTreatment.concatenate_vectors(train2_s[2])
    yValid = preTreatment.concatenate_vectors(valid_s[2])
    weightsTrain2 = preTreatment.concatenate_vectors(train2_s[3])
    weightsValid = preTreatment.concatenate_vectors(valid_s[3])

    print "Putting all the real labels to 1"
    yTrain2 = preTreatment.multiclass2binary(yTrain2)
    yValid = preTreatment.multiclass2binary(yValid)

    print "Getting the best ratios..."
    best_ams_train2_global, best_ratio_global = tresholding.best_ratio(yProbaTrain2Binary, yTrain2, weightsTrain2)
    #best_ams_train2_combinaison, best_ratio_combinaison = tresholding.best_ratio_combinaison_global(yProbaTrain2Binary_s, train2_s[2], train2_s[3], 1)

    yPredictedValid = tresholding.get_yPredicted_ratio(yProbaValidBinary, 0.15)
    yPredictedValid_best_ratio_global = tresholding.get_yPredicted_ratio(yProbaValidBinary, best_ratio_global)
    #yPredictedValid_best_ratio_combinaison_s, yPredictedValid_best_ratio_combinaison = tresholding.get_yPredicted_ratio_8(yProbaTrain2Binary_s, best_ratio_combinaison)

    #Let's compute the predicted AMS
    s, b = submission.get_s_b(yPredictedValid, yValid, weightsValid)
    AMS = hbc.AMS(s,b)
    #s_best_ratio_combinaison, b_best_ratio_combinaison = submission.get_s_b(yPredictedValid_best_ratio_combinaison, yValid, weightsValid)
    #AMS_best_ratio_combinaison = hbc.AMS(s_best_ratio_combinaison, b_best_ratio_combinaison)
    s_best_ratio_global, b_best_ratio_global = submission.get_s_b(yPredictedValid_best_ratio_global, yValid, weightsValid)
    AMS_best_ratio_global = hbc.AMS(s_best_ratio_global, b_best_ratio_global)

    print "AMS 0.15 = %f" %AMS
    print " "
    #print "AMS best ratio combi= %f" %AMS_best_ratio_combinaison
    #print "best AMS train2 ratio combinaison= %f" %best_ams_train2_combinaison
    #print "best ratio combinaison train 2 = %s" %str(best_ratio_combinaison)
    print " "
    print "best AMS valid ratio global= %f" %AMS_best_ratio_global
    print "best AMS train2 ratio global= %f" %best_ams_train2_global
    print "best ratio global train2 = %f" %best_ratio_global
 

    return AMS
    
def main():
    AMS_max = 0.
    for i in range(4):
        for j in range(3):
            print "i, j", i, j
            AMS = train(2 + 2*i, 10 +j*20)
            if AMS_max < AMS:
                AMS_max = AMS
                best_params = (4 +2*i, 10 + j*10)
    print "AMS_max : %f" %AMS_max
    print "best params : %i, %i" %best_params
    return AMS_max, best_params


if __name__ == '__main__':
    train(6, 150)
    #main()
