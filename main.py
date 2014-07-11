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

    return train_s, valid_s, test_s


def save_importation(train_s, valid_s, test_s):

    np.savetxt("Reshape_dataset/train_ID.csv", train_s[0], delimiter=",")
    np.savetxt("Reshape_dataset/train_data.csv", train_s[1], delimiter=",")
    np.savetxt("Reshape_dataset/train_label.csv", train_s[2], delimiter=",")
    np.savetxt("Reshape_dataset/train_weights.csv", train_s[3], delimiter=",")

    np.savetxt("Reshape_dataset/valid_ID.csv", train_s[0], delimiter=",")
    np.savetxt("Reshape_dataset/valid_data.csv", train_s[1], delimiter=",")
    np.savetxt("Reshape_dataset/valid_label.csv", train_s[2], delimiter=",")
    np.savetxt("Reshape_dataset/valid_weights.csv", train_s[3], delimiter=",")

    np.savetxt("Reshape_dataset/test_ID.csv", train_s[0], delimiter=",")
    np.savetxt("Reshape_dataset/test_data.csv", train_s[1], delimiter=",")

    return 0


def main():
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

    # Numerical score:
    for i in range(8):
        sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                       valid_s[2][i])
        print "We have: sum_s[%i]= %i and sum_b[%i]= %i" %(i, sum_s, i, sum_b)
        print yPredicted_s[i].shape
    print(" ")

    ### Random forest:
    # Prediction on the vaidation set:
    print("Random forest prediction...")
    predictor_s, yPredicted_s, yProba_s = randomForest.get_yPredicted_s(
                                                            train_s[1],
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

    # Numerical score:
    for i in range(8):
        sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                       valid_s[2][i])
        print "We have: sum_s[%i]= %i and sum_b[%i]= %i" %(i, sum_s, i, sum_b)


    ##################
    # POST-TREATMENT #
    ##################
    # TBD

    ##############
    # SUBMISSION #
    ##############
    test_prediction, test_proba = randomForest.get_test_prediction(predictor_s, test_s[1])

    # Create a submission file:
    sub = submission.print_submission(np.concatenate(test_s[0]), test_proba ,
                                                    test_prediction)

    return sub

if __name__ == '__main__':
    main()



