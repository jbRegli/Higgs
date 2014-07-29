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
import onTopClassifier
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
    ratio_train = 0.9

    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split= split, \
                                                      normalize= normalize, \
                                                      noise_variance= noise_var, \
                                                     ratio_train= ratio_train,
                                                     n_classes = "multiclass")


    # RANDOM FOREST:
    #kwargs_rdf = {'n_estimators': 50}

    #dRandomForest = analyse.analyse(train_s, valid_s, "randomForest", kwargs_rdf)
    # GRADIENT BOOSTING
    d = analyse.analyse(train_s, valid_s, "xgBoost")

    yPredictedTest = []
    yProbaTest = []

    print "Classifying the test set..."
    for i in range(8):
        yPredicted, yProba = gradientBoosting.prediction(d['predictor_s'][i], test_s[1][i])
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
    yPredictedTest_conca_treshold = tresholding.get_yPredicted_treshold(yProbaTestFinal_conca, d['best_treshold_global'])
    #let's rank the proba
    temp = yProbaTestFinal_conca.argsort()
    yProbaTestFinal_conca_ranked = np.arange(len(yProbaTestFinal_conca))[temp.argsort()]
    for i in range(yProbaTestFinal_conca_ranked.shape[0]):
        yProbaTestFinal_conca_ranked[i] = yProbaTestFinal_conca_ranked[i] + 1
    # let's make the ID int
    for i in IDTest_conca:
        IDTest_conca_int = IDTest_conca.astype(np.int64)

    sub = submission.print_submission(IDTest_conca_int, yProbaTestFinal_conca_ranked, yPredictedTest_conca_treshold, name = 'submission_gradB_treshold')

    AMS_validation = d['AMS']

    return d 
   
    

if __name__ == '__main__':
    main()
  
