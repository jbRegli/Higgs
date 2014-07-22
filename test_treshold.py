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

    # Create the elected vectors for each group (best AMS score)
    best_yPredicted_s = [np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    best_yProba_s = [ np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    best_AMS_s = [0. for i in range(8)]
    best_method_s = [0 for i in range(8)]
    best_ratio_s = [0 for i in range(8)]
    best_AMS_1_method = 0.
    best_method = "methode"
    best_ratio = "0."

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
    #                       randomForest, gradientBoosting}
    dMethods ={}

    # NAIVE BAYES:
    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s, valid_s, 'naiveBayes',
                                              kwargs_bayes)

    """
    # SVM

    kwargs_svm ={}
    dMethods['svm'] = analyse.analyse(train_s, valid_s,'svm', kwargs_svm)
    """

    # K NEIGHBORS
    kwargs_tuning_kn = {'n_neighbors': [10,20]}
    dTuning = tuningModel.parameters_grid_search(train_s, valid_s, 'kNeighbors',
                                             kwargs_tuning_kn)

    dMethods['kNeighbors'] = combineClassifiers.select_best_classifiers(dTuning, valid_s)

    # LDA
    kwargs_lda = {}
    dMethods['lda'] = analyse.analyse(train_s, valid_s, 'lda', kwargs_lda)
    # QDA
    kwargs_qda= {}
    dMethods['qda'] = analyse.analyse(train_s, valid_s, 'qda', kwargs_qda)

    # ADABOOST
    kwargs_ada= {   'n_estimators': 50,
                    'learning_rate': 1.,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
    dMethods['adaBoost'] = analyse.analyse(train_s, valid_s, 'adaBoost',
                                           kwargs_ada)

    # RANDOM FOREST:
    kwargs_tuning_rdf = {'n_estimators': [10,50,100]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid_s, 'randomForest',
                                             kwargs_tuning_rdf)

    dMethods['randomForest'] = combineClassifiers.select_best_classifiers(dTuning,
                                                                valid_s)
    """
    # GRADIENT BOOSTING
    kwargs_tuning_gradB = {'loss': ['deviance'], 'learning_rate': [0.1],
                    'n_estimators': [100,200], 'subsample': [1.0],
                    'min_samples_split': [2], 'min_samples_leaf': [1],
                    'max_depth': [3,5,7], 'init': [None], 'random_state': [None],
                    'max_features': [None], 'verbose': [0]}

    dTuning = tuningModel.parameters_grid_search(train_s, valid_s,
                                                'gradientBoosting',
                                                kwargs_tuning_gradB)

    dMethods['gradientBoosting'] = combineClassifiers.select_best_classifiers(
                                                                dTuning,
                                                                valid_s)
    """
    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    print("-------------------- Best overall combination --------------------")

    dCombine = combineClassifiers.select_best_classifiers(dMethods, valid_s)

    print("-------------------------- Thresholding --------------------------")

     # COMBINED CLASSIFIERS:
    f = open("Tests/test_treshold_combined.txt","w")

    yProba_s = dCombine['yProba_s']
    yPredicted_s = dCombine['yPredicted_s']

    ratio_s = np.arange(0.05,1.0,0.05)

    for ratio in ratio_s:
        f.write("-----ratio = "+str(ratio)+"-----\n")
        f.write("\n")

        yPredicted_treshold_s = tresholding.proba_treshold(yPredicted_s,
                                                           yProba_s, ratio)
        # Numerical score:
        if type(yPredicted_treshold_s) == list:
            for i in range(len(yPredicted_treshold_s)):
                print "valid_s[2][%i] shape = %i" %(i, valid_s[2][i].shape[0])
                print "yPredictedtreshold_s[%i] shape = %i" \
                        %(i, yPredicted_treshold_s[i].shape[0])

                sum_s, sum_b = submission.get_numerical_score(
                                                        yPredicted_treshold_s[i],
                                                        valid_s[2][i])

                print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, yPredicted_treshold_s[i].shape[0], i, sum_s, i, sum_b)

        # Get s and b for each group (s_s, b_s) and the final final_s and
        # final_b:
        final_s, final_b, s_s, b_s = submission.get_s_b_8(yPredicted_treshold_s,
                                                          valid_s[2],
                                                          valid_s[3])

        # Balance the s and b
        final_s *= 250000/25000
        final_b *= 250000/25000

        # AMS final:
        AMS = hbc.AMS(final_s , final_b)
        if AMS > best_AMS_1_method:
            best_AMS_1_method = AMS
            best_method = "combined"
            best_ratio = ratio
        f.write("AMS total = " + str(AMS )+ "\n")
        print ("Expected AMS score for combined methods : %f") %AMS

        #AMS by group
        AMS_s = []
        for i, (s,b) in enumerate(zip(s_s, b_s)):
            s *= 250000/yPredicted_treshold_s[i].shape[0]
            b *= 250000/yPredicted_treshold_s[i].shape[0]
            score = hbc.AMS(s,b)
            if score > best_AMS_s[i]:
                best_yPredicted_s[i] = yPredicted_treshold_s[i]
                best_yProba_s[i] = yProba_s[i]
                best_AMS_s[i] = score
                best_method_s[i] = "combined"
                best_ratio_s[i] = ratio
            AMS_s.append(score)
            f.write("AMS for group %i is %f" %(i, score))
            f.write("\n")
            print("Expected AMS score for randomforest :  for group %i is : %f"\
                    %(i, score))
        print(" ")

        f.write("\n")
        f.write("\n")


    """
    # FOR EACH METHOD:
    for method in dMethods:
        print method
        f = open("Tests/test_treshold_"+str(method)+".txt","w")

        yProba_s = dMethods[method]['yProba_s']
        yPredicted_s = dMethods[method]['yPredicted_s']


        ratio_s = np.arange(0.01,1.0,0.01)

        f.write("-----"+str(method)+"-----\n")

        for ratio in ratio_s:
            f.write("-----ratio = "+str(ratio)+"-----\n")
            f.write("\n")

            yPredicted_treshold_s = tresholding.proba_treshold(yPredicted_s, yProba_s, ratio)

            # Numerical score:
            if type(yPredicted_treshold_s) == list:
                for i in range(len(yPredicted_treshold_s)):
                    sum_s, sum_b = submission.get_numerical_score(
                                                        yPredicted_treshold_s[i],
                                                        valid_s[2][i])
                    print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                            %(i, yPredicted_treshold_s[i].shape[0], i, sum_s, i, sum_b)

            # Get s and b for each group (s_s, b_s) and the final final_s and
            # final_b:
            final_s, final_b, s_s, b_s = submission.get_s_b_8(
                                                yPredicted_treshold_s, valid_s[2],
                                                valid_s[3])

            # Balance the s and b
            final_s *= 250000/25000
            final_b *= 250000/25000

            # AMS final:
            AMS = hbc.AMS(final_s , final_b)
            if AMS > best_AMS_1_method:
                best_AMS_1_method = AMS
                best_method = str(method)
                best_ratio = ratio
            f.write("AMS total = "+str(AMS)+"\n")
            print "Expected AMS score for %s : %f" %(str(method),AMS)

            #AMS by group
            AMS_s = []
            for i, (s,b) in enumerate(zip(s_s, b_s)):
                s *= 250000/yPredicted_treshold_s[i].shape[0]
                b *= 250000/yPredicted_treshold_s[i].shape[0]
                score = hbc.AMS(s,b)
                if score > best_AMS_s[i]:
                    best_yPredicted_s[i] = yPredicted_treshold_s[i]
                    best_yProba_s[i] = yProba_s[i]
                    best_AMS_s[i] = score
                    best_method_s[i] = str(method)
                    best_ratio_s[i] = ratio

                AMS_s.append(score)
                f.write("AMS for group %i is %f" %(i, score))
                f.write("\n")
                print("Expected AMS score for randomforest :  for group %i is : %f" %(i, score))
            print(" ")

            f.write("\n")
            f.write("\n")

    best_final_s, best_final_b, best_s_s, best_b_s = submission.get_s_b_8(best_yPredicted_s, valid_s[2], valid_s[3])

    best_final_s *= 250000/25000
    best_final_b  *= 250000/25000
    best_AMS = hbc.AMS(best_final_s, best_final_b)


    print "Best AMS using one of the methods : %f" %best_AMS_1_method
    print "method : %s" %(str(method))
    print " ratio : %f" %(ratio)
    print " "
    print "Best AMS final : %f" %best_AMS
    print " "

    for n in range(8):
        print "Best AMS for group %i is: %f and is  obtained with method %s and ratio %f" %(n, best_AMS_s[n], best_method_s[n], best_ratio_s[n])
"""
if __name__ == '__main__':
    main()




