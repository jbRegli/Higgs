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

def main():

    ###############
    ### IMPORT ####
    ###############
    # Importation parameters:
    split= True
    normalize = True
    noise_var = 0.
    train_size = 250000
    train_size2 = 0
    valid_size = 0

    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, test_s = tokenizer.extract_data(split= split, \
                                                      normalize= normalize, \
                                                      noise_variance= noise_var, \
                                                     n_classes = "multiclass", \
                                                     train_size = train_size, \
                                                     train_size2 = train_size2, \
                                                     valid_size = valid_size)


    #RANDOM FOREST:
    kwargs_grad = {}
    #kwargs_rdf = {'n_estimators': 100}
    print "Training on the train set ..."
    predictor_s = gradientBoosting.get_predictors(train_s[1], train_s[2], **kwargs_grad)


    yPredictedTest = []
    yProbaTest = []

    print "Classifying the test set..."
    for i in range(8):
        yPredicted, yProba = gradientBoosting.prediction(predictor_s[i], test_s[1][i])
        yPredictedTest.append(yPredicted)
        yProbaTest.append(yProba)

    
    print "Finalizing the vectors for the submission..."

    yProbaTestFinal = []
    
    for i in range(8):
        yProbaTestFinal.append(np.zeros(yPredictedTest[i].shape[0]))
    for i in range(8):
        for j in range(yPredictedTest[i].shape[0]):
            yProbaTestFinal[i][j] = 1 - yProbaTest[i][j][0]

    yPredictedTest_conca = preTreatment.concatenate_vectors(yPredictedTest)
    yProbaTestFinal_conca = preTreatment.concatenate_vectors(yProbaTestFinal)
    IDTest_conca = preTreatment.concatenate_vectors(test_s[0])

    #Let's make all the 1,2,3,4 signal to 1
    for i in range(yPredictedTest_conca.shape[0]):
        if yPredictedTest_conca[i] >=1:
            yPredictedTest_conca[i] = 1
    
    # Let's treshold
    yPredictedTest_conca_treshold = tresholding.get_yPredicted_ratio(yProbaTestFinal_conca, 0.16)
    #let's rank the proba
    yProbaTestFinal_conca_ranked = submission.rank_signals(yProbaTestFinal_conca)
    # let's make the ID int
    for i in IDTest_conca:
        IDTest_conca_int = IDTest_conca.astype(np.int64)

    sub = submission.print_submission(IDTest_conca_int, yProbaTestFinal_conca_ranked, yPredictedTest_conca_treshold, name = 'submission_gradB_ratio15')

    return sub

if __name__ == '__main__':
    main()
  
