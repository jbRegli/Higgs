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
    train_size = 230000
    valid_size = 20000
    """
    ##############
    ### Binary ###
    ##############
    split= False
    normalize = True
    noise_var = 0.

    print("Binary non splitted ")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split = split,
                                                      normalize = normalize,
                                                      noise_variance = 0.,
                                                      #n_classes = "multiclass",
                                                      n_classes = "binary",
                                                      train_size = train_size,
                                                      train_size2 = 0,
                                                      valid_size = valid_size)

    # Ploting
    colors = ['c', 'r']
    names = ['b', 's']

    for i in range(valid_s[1].shape[1]):
        for j in range(i):
            xValue = np.asarray(zip(*valid_s[1]))[i,:]
            yValue = np.asarray(zip(*valid_s[1]))[j,:]
            cValue = valid_s[2]
            axName = valid_s[4]

            b_index = cValue == 0
            s_index = cValue == 1

            b = plt.scatter(xValue[b_index], yValue[b_index],
                            c = cValue, color= colors[0])
            s = plt.scatter(xValue[s_index], yValue[s_index],
                             c = cValue, color= colors[1])

            plt.legend((b,s), names)

            plt.xlabel(axName[i])
            plt.ylabel(axName[j])

            title = str(axName[j]) + " vs " + str(axName[i])

            plt.title(title)

            savepath = "./Vizualisation/Binary/All/"

            plt.savefig( savepath + title + ".png", bbox_inches='tight')


    # Importation parameters:
    split= True
    normalize = True
    noise_var = 0.

    print("Binary splited ")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split = split,
                                                      normalize = normalize,
                                                      noise_variance = 0.,
                                                      #n_classes = "multiclass",
                                                      n_classes = "binary",
                                                      train_size = train_size,
                                                      train_size2 = 0,
                                                      valid_size = valid_size)

    # Ploting
    colors = ['c', 'r']
    names = ['b', 's']

    for subset in range(len(valid_s[0])):
        for i in range(valid_s[1][subset].shape[1]):
            for j in range(i):
                xValue = np.asarray(zip(*valid_s[1][subset]))[i,:]
                yValue = np.asarray(zip(*valid_s[1][subset]))[j,:]
                cValue = valid_s[2][subset]
                axName = valid_s[4][subset]

                b_index = cValue == 0
                s2_index = cValue == 1

                b = plt.scatter(xValue[b_index], yValue[b_index],
                                c = cValue, color= colors[0])
                s = plt.scatter(xValue[s_index], yValue[s_index],
                                 c = cValue, color= colors[1])

                plt.legend((b,s), names)

                plt.xlabel(axName[i])
                plt.ylabel(axName[j])

                title = str(axName[j]) + " vs " + str(axName[i])

                plt.title(title)

                savepath = "./Vizualisation/Binary/Subset_" + str(subset) \
                                + "/"

                plt.savefig( savepath + title + ".png", bbox_inches='tight')

    """

    ##################
    ### Multiclass ###
    ##################
    """
    split= False
    normalize = True
    noise_var = 0.

    print("Mutliclass non splitted ")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split = split,
                                                      normalize = normalize,
                                                      noise_variance = 0.,
                                                      n_classes = "multiclass",
                                                      #n_classes = "binary",
                                                      train_size = train_size,
                                                      train_size2 = 0,
                                                      valid_size = valid_size)

    # Ploting
    colors = ['c', 'g', 'y', 'm', 'r']
    names = ['b', 's1', 's2', 's3', 's4']

    for i in range(valid_s[1].shape[1]):
        for j in range(i):
            xValue = np.asarray(zip(*valid_s[1]))[i,:]
            yValue = np.asarray(zip(*valid_s[1]))[j,:]
            cValue = valid_s[2]
            axName = valid_s[4]

            b_index = cValue == 0
            s1_index = cValue == 1
            s2_index = cValue == 2
            s3_index = cValue == 3
            s4_index = cValue == 4

            b = plt.scatter(xValue[b_index], yValue[b_index],
                            c = cValue, color= colors[0])
            s1 = plt.scatter(xValue[s1_index], yValue[s1_index],
                             c = cValue, color= colors[1])
            s2 = plt.scatter(xValue[s2_index], yValue[s2_index],
                             c = cValue, color= colors[2])
            s3 = plt.scatter(xValue[s3_index], yValue[s3_index],
                             c = cValue, color= colors[3])
            s4 = plt.scatter(xValue[s4_index], yValue[s4_index],
                             c = cValue, color= colors[4])

            plt.legend((b,s1,s2,s3,s4), names)

            plt.xlabel(axName[i])
            plt.ylabel(axName[j])

            title = str(axName[j]) + " vs " + str(axName[i])

            plt.title(title)

            savepath = "./Vizualisation/MultiClass/All/"

            plt.savefig( savepath + title + ".png", bbox_inches='tight')

    """
    # Importation parameters:
    split= True
    normalize = True
    noise_var = 0.

    print("Mutliclass splited ")
    start = time.clock()
    train_s, valid_s, test_s = tokenizer.extract_data(split = split,
                                                      normalize = normalize,
                                                      noise_variance = 0.,
                                                      n_classes = "multiclass",
                                                      #n_classes = "binary",
                                                      train_size = train_size,
                                                      train_size2 = 0,
                                                      valid_size = valid_size)

    # Ploting
    colors = ['c', 'g', 'y', 'm', 'r']
    names = ['b', 's1', 's2', 's3', 's4']

    for subset in range(len(valid_s[0])):
        for i in range(valid_s[1][subset].shape[1]):
            for j in range(i):
                xValue = np.asarray(zip(*valid_s[1][subset]))[i,:]
                yValue = np.asarray(zip(*valid_s[1][subset]))[j,:]
                cValue = valid_s[2][subset]
                axName = valid_s[4][subset]

                b_index = cValue == 0
                s1_index = cValue == 1
                s2_index = cValue == 2
                s3_index = cValue == 3
                s4_index = cValue == 4

                b = plt.scatter(xValue[b_index], yValue[b_index],
                                c = cValue[b_index], color= 'c')
                s1 = plt.scatter(xValue[s1_index], yValue[s1_index],
                                 c = cValue[s1_index], color= 'g')
                s2 = plt.scatter(xValue[s2_index], yValue[s2_index],
                                 c = cValue[s2_index], color= 'y')
                s3 = plt.scatter(xValue[s3_index], yValue[s3_index],
                                 c = cValue[s3_index], color= 'm')
                s4 = plt.scatter(xValue[s4_index], yValue[s4_index],
                                c = cValue[s4_index], color= 'r')

                plt.legend((b,s1,s2,s3,s4), names)

                plt.xlabel(axName[i])
                plt.ylabel(axName[j])

                title = str(axName[j]) + " vs " + str(axName[i])

                plt.title(title)

                savepath = "./Vizualisation/MultiClass/Subset_" + str(subset) \
                                + "/"

                plt.savefig( savepath + title + ".png", bbox_inches='tight')

    return 0


if __name__ == '__main__':
    main()



