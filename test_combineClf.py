# -*- coding: utf-8 -*-
"""
Perform a full analysis of the dataset
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score

import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

import sys
sys.path.append('Analyses/')
import analyse # Function computing an analyse for any method in the good format
import tuningModel
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda

sys.path.append('PostTreatment')
import onTopClassifier
import mergeClassifiers
import combineClassifiers


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

    # NAIVE BAYES:

    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s, valid_s, 'naiveBayes',
                                              kwargs_bayes)

    # SVM
    kwargs_tuning_svm ={'kernel': ["rbf", "poly"], 'C' : [0.025],
                        'probability': [True]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid_s, 'svm',
                                             kwargs_tuning_svm)

    #dMethods['svm'] = combineClassifiers.select_best_classifiers(dTuning,
    #                                                                valid_s)


    # K NEIGHBORS
    kwargs_tuning_kn = {'n_neighbors': [10,20]}
    dTuning = tuningModel.parameters_grid_search(train_s, valid_s, 'kNeighbors',
                                             kwargs_tuning_kn)

    dMethods['kNeighbors'] = combineClassifiers.select_best_classifiers(dTuning,
                                                                        valid_s)

    # LDA
    kwargs_lda = {}
    dMethods['lda'] = analyse.analyse(train_s, valid_s, 'lda', kwargs_lda)
    # QDA
    kwargs_qda= {}
    dMethods['qda'] = analyse.analyse(train_s, valid_s, 'qda', kwargs_qda)


    # ADABOOST
    kwargs_ada= {'n_estimators': 50,
                 'learning_rate': 1.0, 'algorithm': 'SAMME.R',
                 'random_state': None}
    #kwargs_ada = {}

    dMethods['adaBoost'] = analyse.analyse(train_s, valid_s, 'adaBoost',
                                            kwargs_ada)

    # GRADIENT BOOSTING:
    kwargs_tuning_gradB = {'loss': 'deviance', 'learning_rate': 0.1,
                    'n_estimators': [100,200], 'subsample': 1.0,
                    'min_samples_split': 2, 'min_samples_leaf': 1,
                    'max_depth': [3,5,7], 'init': None, 'random_state': None,
                    'max_features': None, 'verbose': 0}

    dTuning = tuningModel.parameters_grid_search(train_s, valid_s,
                                                'gradientBoosting',
                                                kwargs_tuning_gradB)

    dMethods['gradientBoosting'] = combineClassifiers.select_best_classifiers(
                                                                dTuning,
                                                                valid_s)

    # RANDOM FOREST:
    kwargs_tuning_rdf = {'n_estimators': [10,20,50,100]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid_s, 'randomForest',
                                             kwargs_tuning_rdf)

    dMethods['randomForest'] = combineClassifiers.select_best_classifiers(dTuning,
                                                                          valid_s)


    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    print("------------------------ Post Treatment -----------------------")

    d = combineClassifiers.select_best_classifiers(dMethods, valid_s)

    print (" ")
    for i in range(len(d['parameters'])):
        print "Best classifier for subset %i : " %i
        if type(d['method'][i]) == list:
            print d['method'][i][i], ": ", d['parameters'][i]
        else:
            print d['method'][i], ": ", d['parameters'][i]

    """
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

    test_prediction_s, test_proba_s = onTopClassifier.get_SL_test_prediction(
                                                dMethods, dSl, test_s[1])


    print("Test subsets signal average:")
    test_s_average = preTreatment.ratio_sig_per_dataset(test_prediction_s)
    print(" ")

    #RankOrder = np.arange(1,550001)

    if type(test_prediction_s) == list:
        test_prediction_s = np.concatenate(test_prediction_s)
        test_proba_s = np.concatenate(test_proba_s)
        RankOrder = onTopClassifier.rank_signals(test_proba_s)
        ID = np.concatenate(test_s[0])
    else:
        ID = test_s[0]

    # Create a submission file:
    sub = submission.print_submission(ID, RankOrder , test_prediction_s)
    """
    return d

if __name__ == '__main__':
    main()

