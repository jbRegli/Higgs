"""
Perform a full analysis of the dataset
"""

import numpy as np
<<<<<<< HEAD
import time
=======
>>>>>>> FETCH_HEAD

import tokenizer
import preTreatment as PT
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as ams

import sys
sys.path.append('/Analyses')
#import pca # example but empty

def main():
    # Import the training data:
    print("Extracting the training set...")
<<<<<<< HEAD
    start = time.clock()
    train_s, valid_s = tokenizer.get_8_bins(normalize=True, noise_variance= 0.)
    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)
=======
    train_s, valid_s = tokenizer.get_8_bins(normalize= False, noise_variance= 0.)
>>>>>>> FETCH_HEAD

    # Import the test data:
    print("Extracting the test set...")
    test_s = tokenizer.get_test_data()

    # Prediction on the vaidation set:
    print("Naive bayse prediction...")
<<<<<<< HEAD
    yPredicted_s, yProba_s =  submission.get_yPredicted_s(train_s[0],
                                                            train_s[1],
                                                            valid_s[0])
=======
    yPredicted_s =  submission.get_yPredicted_s(train_s[0],
                                                train_s[1],
                                                valid_s[0])

>>>>>>> FETCH_HEAD
    # Get s and b:
    print("Computing final_s and final_b...")
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[1], valid_s[2])

    # AMS:
    AMS = ams.AMS(final_s * 550000 /25000, final_b* 550000 /25000)
    print AMS

    # Create a submission file:
    #sub = submission.print_submission(testIDs, RankOrder, yLabels)

<<<<<<< HEAD
    return yProba_s
=======
    return 0
>>>>>>> FETCH_HEAD

if __name__ == '__main__':
    main()



