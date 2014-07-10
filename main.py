"""
Perform a full analysis of the dataset
"""

import numpy as np
import time

import tokenizer
import preTreatment as PT
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
    # Import the training data:
    print("Extracting the training set...")
    start = time.clock()
    train_s, valid_s = tokenizer.get_8_bins(normalize= True, noise_variance= 0.)
    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)

    # Import the test data:
    print("Extracting the test set...")
    test_s = tokenizer.get_test_data()

    ######################
    ### PRE-TREATMENT ####
    ######################
    # TBD

    ############
    # ANALYSES #
    ############
    ### Naive Bayse:
    # Prediction on the vaidation set:
    print("Naive bayse prediction...")
    yPredicted_s, yProba_s =  naiveBayes.get_yPredicted_s(train_s[0],
                                                            train_s[1],
                                                            valid_s[0])
    # Get s and b:
    print("Computing final_s and final_b for NB...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[1], valid_s[2])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for naive bayse is %f") %AMS

    ### Random forest:
    # Prediction on the vaidation set:
    print("Naive bayse prediction...")
    yPredicted_s, yProba_s =  randomForest.get_yPredicted_s(train_s[0],
                                                            train_s[1],
                                                            valid_s[0],
                                                            n_trees = 100)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[1], valid_s[2])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for random forest is %f") %AMS


    ##################
    # POST-TREATMENT #
    ##################
    # TBD

    ##############
    # SUBMISSION #
    ##############
    # Create a submission file:
    #sub = submission.print_submission(testIDs, RankOrder, yLabels)

    return yProba_s

if __name__ == '__main__':
    main()



