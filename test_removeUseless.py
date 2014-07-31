# -*- coding: utf-8 -*-
"""
Perform a full analysis of the dataset
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score
import copy

import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

import sys
sys.path.append('Analyses/')
import analyse # Function computing an analyse for any method in the good format
import tuningModel

sys.path.append('PostTreatment/')
import onTopClassifier
import mergeClassifiers
import tresholding



def main():
    ###############
    ### IMPORT ####
    ###############
    # Importation parameters:
    split= True
    normalize = True
    noise_var = 0.
    n_classes = "multiclass"
    train_size = 200000
    train_size2 = 25000
    valid_size = 25000


    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, train2_s, valid_s, test_s = tokenizer.extract_data(split= split,
                                                      normalize= normalize,
                                                      noise_variance= noise_var,
                                                      n_classes = n_classes,
                                                      train_size = train_size,
                                                      train_size2 = train_size2,
                                                      valid_size = valid_size)

    # Remerging the y and weights of the validation if necessary:
    if type(valid_s[2]) == list:
        yValid_conca = preTreatment.concatenate_vectors(valid_s[2])
        weights_conca = preTreatment.concatenate_vectors(valid_s[3])

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
    """
    kwargs_bayes = {}
    dMethods['naiveBayes'] =  analyse.analyse(train_s= train_s, train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name = 'naiveBayes',
                                              kwargs = kwargs_bayes)
    """
    # SVM
    """
    kwargs_svm ={}
    dMethods['svm'] = analyse.analyse(train_s, valid_s,'svm', kwargs_svm)
    """

    # K NEIGHBORS
    kwargs_kn = {'n_neighbors': 100}
    dMethods['kNeighbors'] = analyse.analyse(train_s= train_s,
                                              train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name= 'kNeighbors',
                                              kwargs= kwargs_kn)

    """
    # LDA
    kwargs_lda = {}
    dMethods['lda'] = analyse.analyse(train_s= train_s, train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name = 'lda',
                                              kwargs = kwargs_lda)

    # QDA
    kwargs_qda= {}
    dMethods['qda'] = analyse.analyse(train_s= train_s, train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name = 'qda',
                                              kwargs = kwargs_qda)
    # ADABOOST
    kwargs_ada= {   'n_estimators': 50,
                    'learning_rate': 1.,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
    dMethods['adaBoost'] = analyse.analyse(train_s, valid_s, 'adaBoost',
                                           kwargs_ada)
    """
    # RANDOM FOREST:
    kwargs_randomForest= {'n_estimators': 100}
    dMethods['randomForest'] = analyse.analyse(train_s= train_s,
                                              train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name = 'randomForest',
                                              kwargs = kwargs_randomForest)
    """
    # ADABOOST2
    kwargs_ada= {   'n_estimators': 100,
                    'learning_rate': .5,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
    dMethods['adaBoost2'] = analyse.analyse(train_s= train_s,
                                            train2_s= train2_s,
                                            valid_s= valid_s,
                                            method_name= 'adaBoost',
                                            kwargs= kwargs_ada)

    # RANDOM FOREST 3:
    kwargs_randomForest= {'n_estimators': 100}
    dMethods['randomForest3'] = analyse.analyse(train_s= train_s,
                                                train2_s= train2_s,
                                                valid_s= valid_s,
                                                method_name= 'randomForest',
                                                kwargs= kwargs_randomForest)

    # RANDOM FOREST 4:
    kwargs_randomForest= {'n_estimators': 100}
    dMethods['randomForest4'] = analyse.analyse(train_s= train_s,
                                                train2_s= train2_s,
                                                valid_s= valid_s,
                                                method_name = 'randomForest',
                                                kwargs = kwargs_randomForest)

    # RANDOM FOREST 5:
    kwargs_randomForest= {'n_estimators': 100}
    dMethods['randomForest5'] = analyse.analyse(train_s= train_s,
                                                train2_s= train2_s,
                                                valid_s= valid_s,
                                                method_name = 'randomForest',
                                                kwargs = kwargs_randomForest)
    """
    """
    # GRADIENT BOOSTING:
    kwargs_gradB = {'loss': 'deviance', 'learning_rate': 0.1,
                    'n_estimators': 100, 'subsample': 1.0,
                    'min_samples_split': 2, 'min_samples_leaf': 200,
                    'max_depth': 10, 'init': None, 'random_state': None,
                    'max_features': None, 'verbose': 0}

    dMethods['gradientBoosting'] = analyse.analyse(train_s= train_s,
                                              train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name= 'gradientBoosting'
                                              kwargs= kwargs_gradB)
    """
    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    print("---------------------- Feature importance: -----------------------")

    importance_lim = 0.03

    if type(dMethods['randomForest']['predictor_s']) == list:

        # Create a copy of the dataset to be modified:
        train_RM_s= copy.deepcopy(train_s)
        train2_RM_s= copy.deepcopy(train2_s)
        valid_RM_s= copy.deepcopy(valid_s)

        for i,predictor_s in enumerate(dMethods['randomForest']['predictor_s']):
            toBeRemove = []

            #importance_lim = np.mean(np.asarray(
            #                            predictor_s.feature_importances_))\
            #               - np.var(np.asarray(
            #                           predictor_s.feature_importances_))/5\

            for j,importance in  enumerate(predictor_s.feature_importances_):
                # Remove this feature for the dataset if its infuence is lower
                # than importance_lim
                if importance < importance_lim:
                    toBeRemove.append(j)

            train_RM_s[1][i] = np.delete(train_RM_s[1][i],toBeRemove,axis=1)
            train_RM_s[4][i] = np.delete(train_RM_s[4][i],toBeRemove)

            train2_RM_s[1][i] = np.delete(train2_RM_s[1][i],toBeRemove,axis=1)
            train2_RM_s[4][i] = np.delete(train2_RM_s[4][i],toBeRemove)

            valid_RM_s[1][i] = np.delete(valid_RM_s[1][i],toBeRemove,axis=1)
            valid_RM_s[4][i] = np.delete(valid_RM_s[4][i],toBeRemove)

            print ("Subset %i: %i features removed out of %i" \
                    %(i, len(toBeRemove), len(predictor_s.feature_importances_)) )

    else:
        print "Dataset: "
        print dMethods['randomForest']['predictor_s'].feature_importances_

    dMethods_RM ={}

    print train_RM_s[1][0].shape
    print train_s[1][0].shape


    # GRADIENT BOOSTING on the modified dataset:
    dMethods_RM['kNeighbors'] = analyse.analyse(train_s= train_RM_s,
                                              train2_s= train2_RM_s,
                                              valid_s= valid_RM_s,
                                              method_name= 'kNeighbors',
                                              kwargs= kwargs_kn)

    dMethods_RM['randomForest'] = analyse.analyse(train_s= train_s,
                                              train2_s= train2_s,
                                              valid_s= valid_s,
                                              method_name = 'randomForest',
                                              kwargs = kwargs_randomForest)

    """
    dMethods_RM['gradientBoosting'] = analyse.analyse(train_s= train_RM_s,
                                              train2_s= train2_RM_s,
                                              valid_s= valid_RM_s,
                                              method_name= 'gradientBoosting',
                                              kwargs= kwargs_gradB)
    """

    # Compare the 2 gradient boosting methods:
    print (" AMS kNeighbors =       ", dMethods['kNeighbors']['AMS'])
    print (" AMS kNeighbors RM =    ", dMethods_RM['kNeighbors']['AMS'])

    print(" ")
    print (" AMS randomForest =       ", dMethods['randomForest']['AMS'])
    print (" AMS randomForest RM =    ", dMethods_RM['randomForest']['AMS'])



    """
    print (" AMS gradientBoosting =     ", dMethods['gradientBoosting']['AMS'])
    print (" AMS gradientBoosting RM =  ", dMethods_RM['gradientBoosting']['AMS'])
    """

    print("------------------------ On-top predictor -----------------------")
    # Classifiers to be ignored:
    #ignore = ['randomForest2', 'randomForest']
    ignore = []
    clf_onTop = 'randomForest'
    parameters = {}#{'C': 0.5, 'kernel': 'rbf', 'degree': 3, 'gamma': 0.0,
                 # 'coef0': 0.0, 'shrinking':True, 'probability':True,
                 # 'tol': 0.001, 'cache_size': 200, 'class_weight': None}

    print ("We will use an 'on-top' predictor on %i classifiers using a %s.") \
            %(len(dMethods.keys())-len(ignore), clf_onTop)

    final_prediction_s, dOnTop = onTopClassifier.SL_classification(dMethods,
                                        valid_s, train_s,
                                        ignore = ignore,
                                        method= clf_onTop, parameters= parameters)

    print("-------------------------- Tresholding -------------------------")
    ### ON THE 'ON-TOP' CLASSIFIER:
    # Create the elected vectors for each group (best AMS score)
    OT_best_yPredicted_s = [np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    OT_best_yProba_s = [np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    OT_best_AMS_s = [0. for i in range(8)]
    OT_best_method_s = [0 for i in range(8)]
    OT_best_ratio_s = [0 for i in range(8)]
    OT_best_sum_s_s = [0 for i in range(8)]
    OT_best_sum_b_s =  [0 for i in range(8)]
    OT_best_method = "On-top"

    OT_yProba_s = dOnTop['yProba_s']
    OT_yPredicted_s = dOnTop['yPredicted_s']

    #Let's concatenate the vectors
    OT_yProba_conca = preTreatment.concatenate_vectors(OT_yProba_s)
    OT_yPredicted_conca = preTreatment.concatenate_vectors(OT_yPredicted_s)

    # Best treshold global
    OT_best_ratio = tresholding.best_treshold(OT_yProba_conca, yValid_conca,
                                                 weights_conca)
    OT_yPredicted_treshold = tresholding.get_yPredicted_treshold(OT_yProba_conca,
                                                                 OT_best_ratio)

    OT_s, OT_b = submission.get_s_b(OT_yPredicted_treshold, yValid_conca,
                                    weights_conca)
    OT_s *= 10
    OT_b *= 10
    OT_best_AMS = hbc.AMS(OT_s,OT_b)


    # COMPARISON BEST TRESHOLD IN DMETHOD
    # FOR EACH METHOD:
    best_yPredicted_s = [np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    best_yProba_s = [np.zeros(valid_s[2][i].shape[0]) for i in range(8)]
    best_AMS_s = [0. for i in range(8)]
    best_method_s = [0 for i in range(8)]
    best_ratio_s = [0 for i in range(8)]
    best_AMS_1_method = 0.
    best_method = "methode"
    best_ratio = "0."


    for method in dMethods:

        yProba_s = dMethods[method]['yProba_s']
        yPredicted_s = dMethods[method]['yPredicted_s']

        #Let's concatenate the vectors
        yProba_conca = preTreatment.concatenate_vectors(yProba_s)
        yPredicted_conca = preTreatment.concatenate_vectors(yPredicted_s)

        # Best treshold global
        best_treshold = tresholding.best_treshold(yProba_conca, yValid_conca, weights_conca)
        yPredicted_treshold = tresholding.get_yPredicted_treshold(yProba_conca, best_treshold)

        s, b = submission.get_s_b(yPredicted_treshold, yValid_conca, weights_conca)
        s *= 10
        b *= 10
        ams = hbc.AMS(s,b)
        if ams > best_AMS_1_method:
            best_AMS_1_method = ams
            best_method = str(method)
            best_ratio = best_treshold


    # Let's concatenate the 8 vectors which performs the best on each on
    # each of the sub group and tresholding it
    best_yPredicted_conca = preTreatment.concatenate_vectors(best_yPredicted_s)
    best_treshold_conca = tresholding.best_treshold(best_yPredicted_conca, yValid_conca, weights_conca)
    best_yPredicted_conca_treshold = tresholding.get_yPredicted_treshold(best_yPredicted_conca, best_treshold_conca)

    best_final_s, best_final_b, best_s_s, best_b_s = submission.get_s_b_8(best_yPredicted_s, valid_s[2], valid_s[3])
    best_s_treshold, best_b_treshold = submission.get_s_b(best_yPredicted_conca_treshold, yValid_conca, weights_conca)

    best_final_s *= 10
    best_final_b *= 10
    best_s_treshold *= 10
    best_b_treshold *= 10
    best_AMS = hbc.AMS(best_final_s, best_final_b)
    best_AMS_treshold = hbc.AMS(best_s_treshold, best_b_treshold)


    print "Best AMS using one of the methods : %f" %best_AMS_1_method
    print "    method : %s" %(str(method))
    print "    ratio : %f" %(best_ratio)
    print " "
    print "Best AMS concatenate: %f" %best_AMS
    print "Best AMS concatenate  after final tresholding : %f" %best_AMS_treshold
    print "best ratio on the concatenated vector : %f" %best_treshold_conca
    print " "
    print "Best AMS on-top : %f" %OT_best_AMS
    print "Best ratio on the concatenated vector : %f" %OT_best_ratio
    print " "



    """
    # Best treshold group by group
    for i in range(8):
        OT_best_treshold_s = tresholding.best_treshold(OT_yProba_s[i],
                                                       valid_s[2][i],
                                                       valid_s[3][i])

        OT_yPredicted_s[i] = tresholding.get_yPredicted_treshold(OT_yProba_s[i],
                                                              OT_best_treshold_s)

        s, b = submission.get_s_b(OT_yPredicted_s[i], valid_s[2][i],
                                  valid_s[3][i])

        s *= 250000/yPredicted_s[i].shape[0]
        b *= 250000/yPredicted_s[i].shape[0]

        ams = hbc.AMS(s,b)
        if ams > best_AMS_s[i]:
            best_yPredicted_s[i] = yPredicted_s[i]
            best_yProba_s[i] = yProba_s[i]
            best_AMS_s[i] = ams
            best_method_s[i] = dOnTop['method']
            best_ratio_s[i] = best_treshold
            best_sum_s_s[i] = s
            best_sum_b_s[i] =  b

    for n in range(8):
        print "Best AMS group %i: %f - method %s - ratio %f" \
                %(n, best_AMS_s[n], best_method_s[n], best_ratio_s[n])

    print "Best AMS : %f" %best_AMS_1_method
    print "    ratio : %f" %(best_ratio)
    print " "
    """



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
                                                dMethods, dOnTop, test_s[1])


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

    return sub
    """
    return 0
if __name__ == '__main__':
    main()



