import numpy as np
import time
from sklearn.metrics import accuracy_score
import sys



import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append('./..')
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
    train_s, valid_s, test_s = tokenizer.extract_data(split = split,
                                                      normalize = normalize,
                                                      noise_variance = 0.,
                                                      #n_classes = "multiclass",
                                                      n_classes = "binary",
                                                      train_size = 200000,
                                                      train_size2 = 0,
                                                      valid_size = 50000)

    stop = time.clock()
    print ("Extraction time: %i s") %(stop-start)


    print(" ")
    print(" ")

    ################
    ### PLOTING ####
    ################
    for
    plt.scatter(valid_s[0]

    return 0


if __name__ == '__main__':
    main()



