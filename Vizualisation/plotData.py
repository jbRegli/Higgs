import numpy as np
import time
from sklearn.metrics import accuracy_score
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as ams


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

    ################
    ### PLOTING ####
    ################
    print valid_s[0][0].shape
    #plt.scatter(valid_s[0]

    return 0


if __name__ == '__main__':
    main()



