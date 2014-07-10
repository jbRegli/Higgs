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


def importation():
    ###############
    ### IMPORT ####
    ###############
    # Import the training data:
    print("Extracting the data sets...")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.get_8_bins(normalize= True,
                                                     noise_variance= 0.)
    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)

    return train_s, valid_s, test_s


def save_importation()

    return 0


def main(train_s, valid_s, test_s):
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
    yPredicted_s, yProba_s =  naiveBayes.get_yPredicted_s(train_s[1],
                                                            train_s[2],
                                                            valid_s[1])
    # Get s and b:
    print("Computing final_s and final_b for NB...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[2], valid_s[3])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for naive bayse is %f") %AMS
    print(" ")

    ### Random forest:
    # Prediction on the vaidation set:
    print("Random forest prediction...")
    predictor_s, yPredicted_s, yProba_s =  randomForest.get_yPredicted_s(train_s[1],
                                                            train_s[2],
                                                            valid_s[1],
                                                            n_trees = 10)
    # Get s and b:
    print("Computing final_s and final_b for RDF...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[2], valid_s[3])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print ("The expected score for random forest is %f") %AMS
    print(" ")

    ##################
    # POST-TREATMENT #
    ##################
    # TBD

    ##############
    # SUBMISSION #
    ##############
    test_prediction, test_proba = get_test_prediction(predictor_s, test_s[1])

    # Create a submission file:
    sub = submission.print_submission(np.concatenate(test_s[0]), test_proba ,
                                                    test_prediction)

    return sub

if __name__ == '__main__':
    main()



