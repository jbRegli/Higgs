# -*- coding: utf-8 -*-
"""
Perform a full analysis of the dataset
"""

import numpy as np
import time

import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as ams

import sys
sys.path.append('Analyses/')
import naiveBayes
import randomForest
#import pca # example but empty


def main():
    ###############
    ### IMPORT ####
    ###############
    # Importation parameters:
    split= False
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
    ### Average number of signal per subset:
    print("Train subsets signal average:")
    train_s_average = preTreatment.ratio_sig_per_dataset(train_s[3])
    print(" ")
    print("Valid subsets signal average:")
    valid_s_average = preTreatment.ratio_sig_per_dataset(valid_s[3])

    print(" ")
    print(" ")

    ############
    # ANALYSES #
    ############
####### NAIVE BAYSE:
    # Prediction on the vaidation set:
    print("Naive bayse prediction...")
    yPredicted_s, yProba_s =  naiveBayes.get_yPredicted_s(train_s[1],
                                                            train_s[2],
                                                            valid_s[1])
    # Get s and b:
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[2], valid_s[3])

    # AMS:

    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for naive bayse is %f") %AMS

    # Numerical score:
    if type(yPredicted_s) == list:
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                            valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i]= %i - sum_b[%i]= %i" \
                    %(i, yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(yPredicted_s,
                                                            valid_s[2])
             print "%i elements - sum_s= %i - sum_b= %i" \
                    %(yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

####### RANDOM FOREST:
    # Random forest parameters:
    n_trees = 10

    # Prediction on the vaidation set:
    print("Random forest prediction...")
    predictor_s, yPredicted_s, yProba_s = randomForest.get_yPredicted_s(
                                                            train_s[1],
                                                            train_s[2],
                                                            valid_s[1],
                                                            n_trees = n_trees)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[2], valid_s[3])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for random forest is %f") %AMS
    print(" ")

    # Numerical score:
    if type(yPredicted_s) == list:
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                       valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i]= %i - sum_b[%i]= %i" \
                    %(i, yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(yPredicted_s,
                                                       valid_s[2])
             print "%i elements - sum_s= %i - sum_b= %i" \
                    %(yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    # TBD

    ##############
    # SUBMISSION #
    ##############
    test_prediction_s, test_proba_s = randomForest.get_test_prediction(
                                                           predictor_s, test_s[1])

    print("Test subsets signal average:")
    test_s_average = preTreatment.ratio_sig_per_dataset(test_prediction_s)
    print(" ")
    if type(test_prediction_s) == list:
        test_prediction_s = np.concatenate(test_prediction_s)
        test_proba_s = np.concatenate(test_proba_s)
        ID = np.concatenate(test_s[0])

    else:
        ID = test_s[0]
    # Create a submission file:
    sub = submission.print_submission(ID, test_proba_s , test_prediction_s)

    return sub

if __name__ == '__main__':
    main()



