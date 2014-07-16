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
import HiggsBosonCompetition_AMSMetric_rev1 as ams



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

    #dictionnary that will contain all the data for each methods. In the end we'll have a dict of dict
    # keys of the methods : {naiveBayes, svm, kNeighbors, lda, qda, adaBoost,
    #                       randomForest}
    dMethods ={}

    # NAIVE BAYES:
    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s, valid_s, 'naiveBayes', kwargs_bayes)
    # SVM
    """
    kwargs_svm ={}
    dMethods['svm'] = analyse.analyse(train_s, valid_s,'svm', kwargs_svm)
    """
    # K NEIGHBORS
    kwargs_kn = {'n_neighbors':50}
    dMethods['kNeighbors'] = analyse.analyse(train_s, valid_s, 'kNeighbors', kwargs_kn)
    # LDA
    kwargs_lda = {}
    dMethods['lda'] = analyse.analyse(train_s, valid_s, 'lda', kwargs_lda)
    # QDA
    kwargs_qda= {}
    dMethods['qda'] = analyse.analyse(train_s, valid_s, 'qda', kwargs_qda)
    # ADABOOST
    kwargs_ada= {   'base_estimators': None, 
                    'n_estimators': 50,
                    'learning_rate': 1.,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
    dMethods['adaBoost'] = analyse.analyse(train_s, valid_s, 'adaBoost', kwargs_ada)
####### RANDOM FOREST:
    kwargs_rdf= {'n_trees': 100}
    dMethods['randomForest'] = analyse.analyse(train_s, valid_s, 'randomForest', kwargs_rdf)

    ##################
    # POST-TREATMENT #
    ##################
    print("------------------------ Merged predictor -----------------------")
    """
    prediction_list =[(valid_s[0],rdf_yProba_s,rdf_yPredicted_s,\
                            rdf_classif_succ),\
                       (valid_s[0],rdf2_yProba_s,rdf2_yPredicted_s,\
                            rdf2_classif_succ),\
                       (valid_s[0],rdf3_yProba_s,rdf3_yPredicted_s,\
                            rdf3_classif_succ)]

    final_pred_s = postTreatment.merge_classifiers(prediction_list)

    # Classification error:
    for i in range(len(final_pred_s)):
        ratio = accuracy_score(final_pred_s[i][2],valid_s[2][i], normalize= True)
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(final_pred_s) == list:
        for i in range(len(final_pred_s)):
            sum_s, sum_b = submission.get_numerical_score(final_pred_s[i][2],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, final_pred_s[i][2].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(final_pred_s[2],
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(final_pred_s[2].shape[0], sum_s, sum_b)

    print(" ")

    # Transform the probabilities in rank:
    #final_pred = postTreatment.rank_signals(final_pred)
    """

    ##############
    # SUBMISSION #
    ##############
    print("-------------------------- Submission ---------------------------")

    # Prediction on the test set:
    # method used for the submission
    # TODO : Verifier que le nom de la method a bien la bonne forme(
    # creer une liste de noms de methodes)

    method = "randomForest" 

    test_prediction_s, test_proba_s = eval(method).get_test_prediction(
                                                           dMethods[method]['predictor_s'],
                                                           test_s[1])

    print("Test subsets signal average:")
    test_s_average = preTreatment.ratio_sig_per_dataset(test_prediction_s)
    print(" ")

    RankOrder = np.arange(1,550001)

    if type(test_prediction_s) == list:
        test_prediction_s = np.concatenate(test_prediction_s)
        test_proba_s = np.concatenate(test_proba_s)
        ID = np.concatenate(test_s[0])
    else:
        ID = test_s[0]

    # Create a submission file:
    sub = submission.print_submission(ID, RankOrder , test_prediction_s)

    return sub

if __name__ == '__main__':
    main()



