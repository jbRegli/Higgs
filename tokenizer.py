# -*- coding: utf-8 -*-
"""
Load the dataset
"""


# functions to extract the data

import numpy as np
import random,string,math,csv

import preTreatment


def get_all_data(normalize = True, noise_variance = 0., ratio_train = 0.9):
    """
    normalize : binary
    if True normalize all the data

    noise_variance : float
    variance of the gaussian noise added to the signal.
    if noise_variance == 0. we don't add noise

    ratio_train = float
    percentage of the data base used for training

    Outputs :

    (xsTrain, yTrain, weightsTrain),
    (xsValidation, yValidation, weightsValidation)
    xsTrain : np array of size((training_ratio*250000, 30)) of float representing the features
    yTrain : np array of size(training_ratio*250000) of int representing the label of the data (1 if boson, 0 else)
    weightsTrain : np array of size(training_ratio*250000) of float representing the weights

    xsValidation : np array of size(((1-training_ratio)*250000, 30)) of float representing the features
    yValidation : np array of size((1-training_ratio)*250000) of int representing the label of the data (1 if boson, 0 else)
    weightsValidation : np array of size((1-training_ratio)*250000) of float representing the weights
    """
    # Extracting training.csv:
    all = list(csv.reader(open("training.csv","rb"), delimiter=','))

    xs = np.array([map(float, row[1:-2]) for row in all[1:]])
    (numPoints,numFeatures) = xs.shape
    eventID = np.array([int(row[0]) for row in all[1:]])

    # Extracting test.csv
    test = list(csv.reader(open("test.csv", "rb"),delimiter=','))

    xsTest = np.array([map(float, row[1:]) for row in test[1:]])
    eventID_test = np.array([int(row[0]) for row in test[1:]])

    # Normalize
    if normalize == True:
        print("    Normalizing...")
        xs, xsTest = preTreatment.normalize(xs, xsTest)

    # Add gaussian noise
    if noise_variance != 0.:
        print("    Noising:")
        xs = preTreatment.add_noise(xs)

    # Select label
    print("    Preparing the training set and the validation set...")
    sSelector = np.array([row[-1] == 's' for row in all[1:]])
    bSelector = np.array([row[-1] == 'b' for row in all[1:]])

    # Select weights
    weights = np.array([float(row[-2]) for row in all[1:]])
    sumWeights = np.sum(weights)
    sumSWeights = np.sum(weights[sSelector])
    sumBWeights = np.sum(weights[bSelector])

    randomPermutation = random.sample(range(len(xs)), len(xs))
    numPointsTrain = int(numPoints*0.9)
    numPointsValidation = numPoints - numPointsTrain

    # Spliting trainset and validation set:
    eventID_train = eventID[randomPermutation[:numPointsTrain]]
    eventID_valid = eventID[randomPermutation[numPointsTrain:]]

    xsTrain = xs[randomPermutation[:numPointsTrain]]
    xsValidation = xs[randomPermutation[numPointsTrain:]]

    sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
    bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
    sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
    bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

    # create vector of 0(b) and 1(s) for the label
    yTrain = np.zeros(numPointsTrain)
    yValidation = np.zeros(numPointsValidation)

    for n in xrange(numPointsTrain):
        if sSelectorTrain[n]:
            yTrain[n] = 1

    for n in xrange(numPointsValidation):
        if sSelectorValidation[n]:
            yValidation[n] = 1

    weightsTrain = weights[randomPermutation[:numPointsTrain]]
    weightsValidation = weights[randomPermutation[numPointsTrain:]]

    sumWeightsTrain = np.sum(weightsTrain)
    sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
    sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])

    return ((eventID_train, xsTrain, yTrain, weightsTrain), \
           (eventID_valid, xsValidation, yValidation, weightsValidation), \
           (eventID_test, xsTest))


def get_8_bins(normalize = True, noise_variance = 0., ratio_train= 0.9):
    """
    returns (xsTrain_s, yTrain_s, weightsTrain_s), (xsValidation_s, yValidation_s, weightsValidation_s)
    list of the data containing the eight different groups
    """

    # Extracting the train set, the validation set and the test set:
    Train, Validation, Test = get_all_data(normalize = normalize,
                                     noise_variance = noise_variance,
                                     ratio_train= ratio_train)

    ID_train, xsTrain, yTrain, weightsTrain  = Train[0], Train[1], Train[2], \
                                               Train[3]
    ID_valid, xsValid, yValid, weightsValid = Validation[0], Validation[1], \
                                              Validation[2], Validation[3]

    ID_test, xsTest = Test[0], Test[1]

    # Splitting the data into sub-groups:
    ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s = preTreatment.\
                           split_8_matrix(ID_train, xsTrain, yTrain, weightsTrain)



    print("    Splitting the valid set")
    ID_valid_s, xsValid_s,yValid_s, weightsValid_s = preTreatment.\
                           split_8_matrix(ID_valid, xsValid, yValid, weightsValid)


    print("    Splitting the test set")
    ID_test_s, xsTest_s = preTreatment.split_8_matrix(ID_test, xsTest)

    # Delete the columns full of -999
    print("    Deleting the invalid inputs")
    print("    WARNING: I'm not really sure of what is done here. To be checked.")
    # (if u see any suspicious looking person, or article ...)
    for i in range(8):
        for index_column in range(xsTrain.shape[1]):
            if xsTrain_s[i].shape[1] > index_column:
                # Train set:
                if xsTrain_s[i][0,index_column] == -999:
                    xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[index_column],1)
            #else:
            #    print ("Something weird happened for train at %i") %index_column
            #    # Validation set:
            #if xsValid_s[i].shape[1] > index_column:
                #if xsValid_s[i][0,index_column] == -999:
                    xsValid_s[i] = np.delete(xsValid_s[i], np.s_[index_column],1)
            #else:
            #    print ("Something weird happened for valid at %i") %index_column

            #if xsTest_s[i].shape[1] > index_column:
                # Test set:
                #if xsTest_s[i][0,index_column] == -999:
                    xsTest_s[i] = np.delete(xsTest_s[i], np.s_[index_column],1)
            #else:
            #    print ("Something weird happened for test at %i") %index_column

        # Deleting the feature identical within each group:
        xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[22],1)
        xsValid_s[i] = np.delete(xsValid_s[i], np.s_[22],1)
        xsTest_s[i]  = np.delete(xsTest_s[i],  np.s_[22],1)

    return (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s), \
           (ID_valid_s, xsValid_s, yValid_s, weightsValid_s), \
           (ID_test_s, xsTest_s)


def extract_data(split= True, normalize= True, noise_variance= 0.,
                 ratio_train= 0.9):
    """
    Function wrapping the extraction of the data for any of the possible cases.
    """
    if split == True:
        # Split the data into 8 sub-datasets:
        return get_8_bins(normalize= normalize, noise_variance= noise_variance,
                          ratio_train= ratio_train)
    else:
        # Extract the data as a unique dataset:
        return get_all_data(normalize= normalize, noise_variance= noise_variance,
                            ratio_train = ratio_train)


