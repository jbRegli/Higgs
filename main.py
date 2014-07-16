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
#import pca # example but empty


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

    ### Test
    #analyse.analyse('naiveBayses')


####### NAIVE BAYSE:
    # Prediction on the vaidation set:
    print("------------------- Naive bayse prediction -----------------------")
    nb_predictor_s, nb_yPredicted_s, nb_yProba_s =  naiveBayes.get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1])
    # Get s and b:
    nb_final_s, nb_final_b = submission.get_s_b_8(nb_yPredicted_s, valid_s[2],
                                                  valid_s[3])

    # AMS:
    #AMS = ams.AMS(nb_final_s * 550000 /25000, nb_final_b* 550000 /25000)
    #print ("The expected score for naive bayse is %f") %AMS

    # Classification error:
    nb_classif_succ = naiveBayes.get_classification_error(nb_yPredicted_s,
                                                             valid_s[2],
                                                             normalize= True)
    for i, ratio in enumerate(nb_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(nb_yPredicted_s) == list:
        for i in range(len(nb_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(nb_yPredicted_s[i],
                                                            valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, nb_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(nb_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(nb_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")
####### SVM
    """
    # SVM parameters

    # Prediction on the vaidation set:
    print("------------------- SVM prediction ---------------------")
    svm_predictor_s, svm_yPredicted_s, svm_yProba_s = svm.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1]
                                                                )
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    svm_final_s, svm_final_b = submission.get_s_b_8(svm_yPredicted_s, valid_s[2],
                                                    valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    svm_classif_succ = randomForest.get_classification_error(svm_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(svm_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")
    """
####### K Neighbors
    # K neighbors paramaters
    n_neighbors = 1000 

    # Prediction on the vaidation set:
    print("------------------- K Neighbors prediction ---------------------")
    neigh_predictor_s, neigh_yPredicted_s, neigh_yProba_s = kNeighbors.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                n_neighbors = n_neighbors)
    # Get s and b:
    print("Computing final_s and final_b for KN...")
    neigh_final_s, neigh_final_b = submission.get_s_b_8(neigh_yPredicted_s, valid_s[2],
                                                    valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    neigh_classif_succ = randomForest.get_classification_error(neigh_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(neigh_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(neigh_yPredicted_s) == list:
        for i in range(len(neigh_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(neigh_yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, neigh_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(neigh_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(neigh_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

####### Ada
    # Ada Boost parameters
    base_estimators = None
    n_estimators = 50
    learning_rate = 1.
    algorithm = 'SAMME.R'
    random_state = None

    # Prediction on the vaidation set:
    print("------------------- Ada Boost prediction ---------------------")
    ada_predictor_s, ada_yPredicted_s, ada_yProba_s = adaBoost.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                base_estimators = base_estimators,
                                                                n_estimators = n_estimators,
                                                                learning_rate = learning_rate,
                                                                algorithm = algorithm,
                                                                random_state = random_state)
    # Get s and b:
    print("Computing final_s and final_b for KN...")
    ada_final_s, ada_final_b = submission.get_s_b_8(ada_yPredicted_s, valid_s[2],
                                                    valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    ada_classif_succ = randomForest.get_classification_error(ada_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(ada_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(ada_yPredicted_s) == list:
        for i in range(len(ada_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(ada_yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, ada_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(ada_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(ada_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")
####### RANDOM FOREST:
    # Random forest parameters:
    n_trees = 10

    # Prediction on the vaidation set:
    print("------------------- Random forest prediction ---------------------")
    rdf_predictor_s, rdf_yPredicted_s, rdf_yProba_s = randomForest.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                n_trees = n_trees)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    rdf_final_s, rdf_final_b = submission.get_s_b_8(rdf_yPredicted_s, valid_s[2],
                                                    valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    rdf_classif_succ = randomForest.get_classification_error(rdf_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(rdf_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(rdf_yPredicted_s) == list:
        for i in range(len(rdf_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(rdf_yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, rdf_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(rdf_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(rdf_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

####### RANDOM FOREST 2:
    # Random forest parameters:
    n_trees = 10

    # Prediction on the vaidation set:
    print("------------------ Random forest prediction 2 --------------------")
    rdf2_predictor_s, rdf2_yPredicted_s, rdf2_yProba_s = randomForest.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                n_trees = n_trees)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    rdf2_final_s, rdf2_final_b = submission.get_s_b_8(rdf2_yPredicted_s,
                                                      valid_s[2],
                                                      valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    rdf2_classif_succ = randomForest.get_classification_error(rdf2_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(rdf2_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(rdf2_yPredicted_s) == list:
        for i in range(len(rdf2_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(rdf2_yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, rdf2_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(rdf2_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(rdf2_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

####### RANDOM FOREST 3:
    # Random forest parameters:
    n_trees = 10

    # Prediction on the vaidation set:
    print("------------------ Random forest prediction 3 --------------------")
    rdf3_predictor_s, rdf3_yPredicted_s, rdf3_yProba_s = randomForest.\
                                                get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                n_trees = n_trees)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    rdf3_final_s, rdf3_final_b = submission.get_s_b_8(rdf3_yPredicted_s,
                                                      valid_s[2],
                                                      valid_s[3])

    # AMS:
    #AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    #print ("The expected score for random forest is %f") %AMS
    #print(" ")

    # Classification error:
    rdf3_classif_succ = randomForest.get_classification_error(rdf3_yPredicted_s,
                                                              valid_s[2],
                                                              normalize= True)
    for i, ratio in enumerate(rdf3_classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")

    # Numerical score:
    if type(rdf3_yPredicted_s) == list:
        for i in range(len(rdf3_yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(rdf3_yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, rdf3_yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(rdf3_yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(rdf3_yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")



    ##################
    # POST-TREATMENT #
    ##################
    print("------------------------ Merged predictor -----------------------")

    prediction_list =[(valid_s[0],rdf_yProba_s,rdf_yPredicted_s,\
                            rdf_classif_succ),\
                       (valid_s[0],rdf2_yProba_s,rdf2_yPredicted_s,\
                            rdf2_classif_succ),\
                       (valid_s[0],rdf3_yProba_s,rdf3_yPredicted_s,\
                            rdf3_classif_succ)]

    final_pred_s = postTreatment.merge_classifier(prediction_list)

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


    ##############
    # SUBMISSION #
    ##############
    print("-------------------------- Submission ---------------------------")

    # Prediction on the test set:
    test_prediction_s, test_proba_s = randomForest.get_test_prediction(
                                                           rdf_predictor_s,
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



