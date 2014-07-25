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
import tuningModel
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
    ratio_train = 0.5

    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split= split, \
                                                      normalize= normalize, \
                                                      noise_variance= noise_var, \
                                                     ratio_train= ratio_train)
    #Let's make two valid and weights set
    L1 =[]
    L2 = []
    for j in range(4):
        L11 = []
        L22 = []
        for i in range(8):
            L11.append(valid_s[j][i][:int(0.8*valid_s[j][i].shape[0])])
            L22.append(valid_s[j][i][int(0.8*valid_s[j][i].shape[0]):])
        L1.append(L11)
        L2.append(L22)
    valid1_s = (L1[0], L1[1], L1[2], L1[3])
    valid2_s = (L2[0], L2[1], L2[2], L2[3])
    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)

    print(" ")
    print(" ")

    # Create the elected vectors for each group (best AMS score)
    best_yPredicted_s = [np.zeros(valid1_s[2][i].shape[0]) for i in range(8)]
    best_yProba_s = [np.zeros(valid1_s[2][i].shape[0]) for i in range(8)]
    best_AMS_s = [0. for i in range(8)]
    best_method_s = [0 for i in range(8)]
    best_ratio_s = [0 for i in range(8)]
    best_AMS_1_method = 0.
    best_AMS = 0.
    best_method = "methode"
    best_ratio = "0."
    Es_s = [0. for i in range(8)]
    Eb_s = [0. for i in range(8)]


    ######################
    ### PRE-TREATMENT ####
    ######################
    print("------------------------- Pre-treatment --------------------------")
    ### Average number of signal per subset:
    print("Train subsets signal average:")
    train_s_average = preTreatment.ratio_sig_per_dataset(train_s[2])
    print(" ")
    print("Valid subsets signal average:")
    valid1_s_average = preTreatment.ratio_sig_per_dataset(valid1_s[2])

    print(" ")
    print(" ")

    ############
    # ANALYSES #
    ############

    # Dictionnary that will contain all the data for each methods. In the end
    # we'll have a dict of dict
    # Keys of the methods : {naiveBayes, svm, kNeighbors, lda, qda, adaBoost,
    #                       randomForest, gradientBoosting}
    dMethods ={}

    # NAIVE BAYES:

    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s, valid1_s, 'naiveBayes',
                                              kwargs_bayes)


    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s, valid1_s, 'naiveBayes',
                                              kwargs_bayes)

    # SVM
    """
    kwargs_svm ={}
    dMethods['svm'] = analyse.analyse(train_s, valid1_s,'svm', kwargs_svm)
    """

    # K NEIGHBORS
    kwargs_tuning_kn = {'n_neighbors': [20,50]}
    dTuning = tuningModel.parameters_grid_search(train_s, valid1_s, 'kNeighbors',
                                             kwargs_tuning_kn)
    
    dMethods['kNeighbors'] = combineClassifiers.select_best_classifiers(dTuning, valid1_s)
    
    # LDA
    kwargs_lda = {}
    dMethods['lda'] = analyse.analyse(train_s, valid1_s, 'lda', kwargs_lda)
    # QDA
    kwargs_qda= {}
    dMethods['qda'] = analyse.analyse(train_s, valid1_s, 'qda', kwargs_qda)

    # ADABOOST
    kwargs_ada= {   'n_estimators': 50,
                    'learning_rate': 1.,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
    dMethods['adaBoost'] = analyse.analyse(train_s, valid1_s, 'adaBoost',
                                           kwargs_ada)

    # RANDOM FOREST:
    kwargs_tuning_rdf = {'n_estimators': [10,50,100]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid1_s, 'randomForest',
                                             kwargs_tuning_rdf)

    dMethods['randomForest'] = combineClassifiers.select_best_classifiers(dTuning,
                                                                valid1_s)
    """
    # GRADIENT BOOSTING
    
    kwargs_gradB = {}

    dMethods['gradientBoosting'] = analyse.analyse(train_s, valid1_s, 'gradientBoosting', kwargs_gradB)


    kwargs_tuning_gradB = {'loss': ['deviance'], 'learning_rate': [0.1],
                    'n_estimators': [100], 'subsample': [1.0],
                    'min_samples_split': [2], 'min_samples_leaf': [1],
                    'max_depth': [10], 'init': [None], 'random_state': [None],
                    'max_features': [None], 'verbose': [0]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid1_s,
                                                'gradientBoosting',
                                                kwargs_tuning_gradB)

    dMethods['gradientBoosting'] = combineClassifiers.select_best_classifiers(
                                                                dTuning,
                                                         valid1_s)
    """
    
    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    print("-------------------- Best overall combination --------------------")

    dCombine = combineClassifiers.select_best_classifiers(dMethods, valid1_s)

    print("-------------------------- Thresholding --------------------------")

    # FOR EACH METHOD:
    for method in dMethods:

        yProba_s = dMethods[method]['yProba_s']
        yPredicted_s = dMethods[method]['yPredicted_s']

        # Best treshold group by group
        for i in range(8):
            best_treshold = tresholding.best_treshold(yProba_s[i], valid1_s[2][i],
                                                      valid1_s[3][i])
            yPredicted_s[i] = tresholding.get_yPredicted_treshold(yProba_s[i],
                                                                  best_treshold)
            s, b = submission.get_s_b(yPredicted_s[i], valid1_s[2][i],
                                      valid1_s[3][i])
            s *= 250000/yPredicted_s[i].shape[0]
            b *= 250000/yPredicted_s[i].shape[0]
            ams = hbc.AMS(s,b)
            if ams > best_AMS_s[i]:
                best_yPredicted_s[i] = yPredicted_s[i]
                best_yProba_s[i] = yProba_s[i]
                best_AMS_s[i] = ams
                best_method_s[i] = str(method)
                best_ratio_s[i] = best_treshold
                Es_s[i] = s*yPredicted_s[i].shape[0]/250000
                Eb_s[i] = b*yPredicted_s[i].shape[0]/250000


    # MARKOV CHAIN AMS GLOBAL OBJECTIVE
    ams_markov_s = []
    n_steps_s = np.arange(10,100,10)
    for n_steps in range(10):
        print "best AMS : %f" %best_AMS
        for n in range(10):
            for i in range(8):
                for method in dMethods:

                    yProba_s = dMethods[method]['yProba_s']
                    yPredicted_s = dMethods[method]['yPredicted_s']

                    treshold_s = np.arange(0., 1., 0.1)

                    for treshold in treshold_s:
                        yPredicted_s[i] = tresholding.get_yPredicted_treshold(yProba_s[i],
                                                                  treshold)
                        s, b = submission.get_s_b(yPredicted_s[i], valid1_s[2][i],
                                      valid1_s[3][i])
                        total_s = sum(Es_s) - Es_s[i] +s #somme des s avec la nouvelle valeur de s pour le groupe i et les espérances pour les autres groupes
                        total_b = sum(Eb_s) - Eb_s[i] +b
                        yValid_conca = preTreatment.concatenate_vectors(valid1_s[2])
                        total_s *= 250000/yValid_conca.shape[0]
                        total_b *= 250000/yValid_conca.shape[0]
                        ams = hbc.AMS(total_s,total_b)
                        ams_markov_s.append(ams)
                        if ams > best_AMS:
                            best_yPredicted_s[i] = yPredicted_s[i]
                            best_yProba_s[i] = yProba_s[i]
                            best_AMS = ams
                            best_method_s[i] = str(method)
                            best_ratio_s[i] = best_treshold
                            Es_s[i] = s
                            Eb_s[i] = b
                            best_n = n

        yPredicted2_s = []
        for i in range(8):
            yPredicted, yProba = eval(best_method_s[i]).prediction(dMethods[best_method_s[i]]['predictor_s'][i], valid2_s[1][i])
            yPredicted = tresholding.get_yPredicted_treshold(yProba, best_ratio_s[i])
            yPredicted2_s.append(yPredicted)

        finals, finalb ,s_s, b_s = submission.get_s_b_8(yPredicted2_s, valid2_s[2], valid2_s[3])
        yValid_conca = preTreatment.concatenate_vectors(valid2_s[2])
        finals *= 250000/yValid_conca.shape[0]
        finalb *= 250000/yValid_conca.shape[0]

        AMS2 = hbc.AMS(finals, finalb)
        print "AMS après %i samples : %f" %(n_steps*10, AMS2)
    
    print " "
    print "Best AMS final sur le second test d'entrainement  : %f" %best_AMS
    print "----parametres pour obtenir le meilleur score--------------"

    for n in range(8):
        print "group %i:  -method %s - ratio %f" \
                %(n, best_method_s[n], best_ratio_s[n])




if __name__ == '__main__':
    main()




