import numpy as np
import time
from sklearn.metrics import accuracy_score
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append('../')
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
    xValid_s = valid_s[1]
    yValid_s = valid_s[2]
    nameValid_s = valid_s[4]

    for j in range(len(valid_s[1])):
        print "subset %i" %j
        for i in range(valid_s[1][j].shape[1]):
            plt.scatter(xValid_s[j][:,0],xValid_s[j][:,i], c= yValid_s[j][:])
            plt.xlabel(nameValid_s[j][0])
            plt.ylabel(nameValid_s[j][i])

            plt.show()

    return 0


if __name__ == '__main__':
    main()



