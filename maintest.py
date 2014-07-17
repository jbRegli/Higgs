# -*- coding: utf-8 -*-
"""
Perform a full analysis of the dataset
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score

import tokenizer
import preTreatment
import postTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc


import sys
sys.path.append('Analyses/')
import analyse # Function computing an analyse for any method in the good format
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda

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
    train_s, valid_s, test_s = tokenizer.extract_data(split= split,
                                                      normalize= normalize,
                                                      noise_variance= noise_var,
                                                      ratio_train= ratio_train)
    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)

    print(" ")
    print(" ")

    ######################
    ### PRE-TREATMENT ####
    ######################
    print("------------------------- Pre-treatment --------------------------")
    ### Average number of signal per subset:
    print("Train subsets signal average:")
    train_s_average = preTreatment.ratio_sig_per_dataset(train_s[2])
    print(" ")
    print("Valid subsets signal average:")
    valid_s_average = preTreatment.ratio_sig_per_dataset(valid_s[2])

    print(" ")
    print(" ")

    ############
    # ANALYSES #
    ############

    # Dictionnary that will contain all the data for each methods. In the end
    # we'll have a dict of dict
    # Keys of the methods : {naiveBayes, svm, kNeighbors, lda, qda, adaBoost,
    #                       randomForest}
    dMethods ={}
    # RANDOM FOREST:
    kwargs_rdf= {'n_trees': 50}
    dMethods['randomForest'] = analyse.analyse(train_s, valid_s, 'randomForest',
                                               kwargs_rdf)

    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    print("post treatment")
    yProba_s = dMethods['randomForest']['yProba_s']
    yPredicted_s = dMethods['randomForest']['yPredicted_s']

    for n in range(8):
        L = []
        for i in range(yPredicted_s[n].shape[0]):
            if yPredicted_s[n][i] == 1:
                L.append(yProba_s[n][i][1])

        L.sort(reverse = True)
        prob_limit = L[int(len(L)*0.45)]


        for i in range(yPredicted_s[n].shape[0]):
            if yProba_s[n][i][1] < prob_limit:
                yPredicted_s[n][i] = 0
            else:
                yPredicted_s[n][i] = 1

    # Numerical score:
    if type(yPredicted_s) == list:
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    
    # Get s and b for each group (s_s, b_s) and the final final_s and
    # final_b:
    final_s, final_b, s_s, b_s = submission.get_s_b_8(yPredicted_s, valid_s[2],
                                                  valid_s[3])

    # Balance the s and b
    final_s *= 250000/25000
    final_b *= 250000/25000
    # AMS final:
    AMS = hbc.AMS(final_s , final_b)
    print ("Expected AMS score for randomforest : %f") %AMS
    #AMS by group
    AMS_s = []
    for i, (s,b) in enumerate(zip(s_s, b_s)):
        s *= 250000/yPredicted_s[i].shape[0]
        b *= 250000/yPredicted_s[i].shape[0]
        score = hbc.AMS(s,b)
        AMS_s.append(score)
        print("Expected AMS score for randomforest :  for group %i is : %f" %(i, score))
    print(" ")

    
    ##############
    # SUBMISSION #
    ##############
    print("-------------------------- Submission ---------------------------")

    # Prediction on the test set:
    # method used for the submission
    # TODO : Verifier que le nom de la method a bien la bonne forme(
    # creer une liste de noms de methodes)

    #method = "randomForest"

    #test_prediction_s, test_proba_s = eval(method).get_test_prediction(
    #                                            dMethods[method]['predictor_s'],
    #                                            test_s[1])

    test_prediction_s, test_proba_s = postTreatment.get_SL_test_prediction(
                                                dMethods, dSl, test_s[1])


    print("Test subsets signal average:")
    test_s_average = preTreatment.ratio_sig_per_dataset(test_prediction_s)
    print(" ")

    #RankOrder = np.arange(1,550001)

    if type(test_prediction_s) == list:
        test_prediction_s = np.concatenate(test_prediction_s)
        test_proba_s = np.concatenate(test_proba_s)
        RankOrder = postTreatment.rank_signals(test_proba_s)
        ID = np.concatenate(test_s[0])
    else:
        ID = test_s[0]

    # Create a submission file:
    sub = submission.print_submission(ID, RankOrder , test_prediction_s)

    return sub

if __name__ == '__main__':
    main()



