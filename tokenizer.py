# -*- coding: utf-8 -*-
"""
Load the dataset
"""


# functions to extract the data

import numpy as np
import random,string,math,csv

import preTreatment


def get_all_data(normalize = True, noise_variance = 0., n_classes = "binary",
                 train_size = 200000, train_size2 = 25000, valid_size = 25000,
                 datapath = "", translate= False):
    """
    normalize : binary
    if True normalize all the data

    noise_variance : float
    variance of the gaussian noise added to the signal.
    if noise_variance == 0. we don't add noise

    ratio_train = float
    percentage of the data base used for training

    n_classes = string in {"binary", multiclass"}
    if multiclasses 4 classes for the events {1,2,3,4}, and 1 for the non event : {0}
    default = "binary"

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
    all = list(csv.reader(open(datapath + "training.csv","rb"), delimiter=','))

    names_train = np.array([row for row in all[0][1:-2]])

    xs = np.array([map(float, row[1:-2]) for row in all[1:]])
    (numPoints,numFeatures) = xs.shape
    eventID = np.array([int(row[0]) for row in all[1:]])

    # Extracting test.csv
    test = list(csv.reader(open(datapath + "test.csv", "rb"),delimiter=','))

    names_test = np.array([row for row in test[0][1:-2]])

    xsTest = np.array([map(float, row[1:]) for row in test[1:]])
    eventID_test = np.array([int(row[0]) for row in test[1:]])

    # Normalize
    if normalize == True:
        print("    Normalizing...")
        xs, xsTest = preTreatment.normalize(xs, xsTest)

    # translate
    if translate == True:
        print("    Translating in [0,1]...")
        xs, xsTest = preTreatment.translate_01(xs, xsTest)

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


    # Spliting trainsets and validation set:
    eventID_train = eventID[randomPermutation[:train_size]]
    if train_size2 !=0:
        eventID_train2 = eventID[randomPermutation[train_size:train_size+train_size2]]
    if valid_size != 0:
        eventID_valid = eventID[randomPermutation[(train_size + train_size2):]]

    xsTrain = xs[randomPermutation[:train_size]]
    if train_size2 != 0:
        xsTrain2 = xs[randomPermutation[train_size: train_size + train_size2]]
    if valid_size != 0:
        xsValidation = xs[randomPermutation[train_size + train_size2:]]

    sSelectorTrain = sSelector[randomPermutation[:train_size]]
    bSelectorTrain = bSelector[randomPermutation[:train_size]]
    if train_size2!=0:
        sSelectorTrain2 = sSelector[randomPermutation[train_size:train_size + train_size2]]
        bSelectorTrain2 = bSelector[randomPermutation[train_size:train_size + train_size2]]
    if valid_size !=0:
        sSelectorValidation = sSelector[randomPermutation[train_size + train_size2:]]
        bSelectorValidation = bSelector[randomPermutation[train_size + train_size2:]]

    # create vector of 0(b) and 1(s) for the label
    yTrain = np.zeros(train_size)
    if train_size2 !=0:
        yTrain2 = np.zeros(train_size2)
    if valid_size !=0:
        yValidation = np.zeros(valid_size)

    for n in xrange(train_size):
        if sSelectorTrain[n]:
            yTrain[n] = 1
    if train_size2 !=0:
        for n in xrange(train_size2):
            if sSelectorTrain2[n]:
                yTrain2[n] = 1
    if valid_size !=0:
        for n in xrange(valid_size):
            if sSelectorValidation[n]:
                yValidation[n] = 1

    weightsTrain = weights[randomPermutation[:train_size]]
    if train_size2 !=0:
        weightsTrain2 = weights[randomPermutation[train_size:train_size + train_size2]]
    if valid_size != 0:
        weightsValidation = weights[randomPermutation[train_size + train_size2:]]

    if n_classes == "multiclass":
        yTrain = preTreatment.binary2multiclass(yTrain, weightsTrain)
        if train_size2 !=0:
            yTrain2 = preTreatment.binary2multiclass(yTrain2, weightsTrain2)
        if valid_size !=0:
            yValidation = preTreatment.binary2multiclass(yValidation, weightsValidation)

    sumWeightsTrain = np.sum(weightsTrain)
    sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
    sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])

    if train_size2 != 0:
        sumWeightsTrain2 = np.sum(weightsTrain2)
        sumSWeightsTrain2 = np.sum(weightsTrain[sSelectorTrain2])
        sumBWeightsTrain2 = np.sum(weightsTrain[bSelectorTrain2])


    if train_size2 !=0 and valid_size !=0:
        return ((eventID_train, xsTrain, yTrain, weightsTrain, names_train), \
           (eventID_train2, xsTrain2, yTrain2, weightsTrain2, names_train), \
           (eventID_valid, xsValidation, yValidation, weightsValidation, names_train), \
           (eventID_test, xsTest, names_train))
    if train_size2 !=0 and valid_size ==0:
        return ((eventID_train, xsTrain, yTrain, weightsTrain, names_train), \
           (eventID_train2, xsTrain2, yTrain2, weightsTrain2, names_train), \
           (eventID_test, xsTest, names_train))
    if train_size2 ==0 and valid_size !=0:
        return ((eventID_train, xsTrain, yTrain, weightsTrain, names_train), \
           (eventID_valid, xsValidation, yValidation, weightsValidation, names_train), \
           (eventID_test, xsTest, names_train))
    if train_size2 ==0 and valid_size ==0:
        return ((eventID_train, xsTrain, yTrain, weightsTrain, names_train), \
           (eventID_test, xsTest, names_train))





def get_8_bins(normalize = True, noise_variance = 0., remove_999= True,
                n_classes = "binary",
                train_size = 200000, train_size2 = 25000, valid_size = 25000,
                datapath = "", translate= False):
    """
    returns (xsTrain_s, yTrain_s, weightsTrain_s), (xsValidation_s, yValidation_s, weightsValidation_s)
    list of the data containing the eight different groups
    """

    # Extracting the train set, the validation set and the test set:
    if train_size2 !=0 and valid_size !=0:
        Train, Train2, Validation, Test = get_all_data(normalize = normalize,
                                     noise_variance = noise_variance,
                                     n_classes = n_classes,
                                     train_size = train_size,
                                     train_size2 = train_size2,
                                     valid_size = valid_size,
                                     datapath= datapath,
                                     translate = translate)
    if train_size2 !=0 and valid_size ==0:
        Train, Train2, Test = get_all_data(normalize = normalize,
                                     noise_variance = noise_variance,
                                     n_classes = n_classes,
                                     train_size = train_size,
                                     train_size2 = train_size2,
                                     valid_size = valid_size,
                                     datapath= datapath,
                                     translate = translate)

    if train_size2 ==0 and valid_size !=0:
        Train, Validation, Test = get_all_data(normalize = normalize,
                                     noise_variance = noise_variance,
                                     n_classes = n_classes,
                                     train_size = train_size,
                                     train_size2 = train_size2,
                                     valid_size = valid_size,
                                    datapath= datapath,
                                    translate = translate)

    if train_size2==0 and valid_size ==0:
        Train, Test = get_all_data(normalize = normalize,
                                     noise_variance = noise_variance,
                                     n_classes = n_classes,
                                     train_size = train_size,
                                     train_size2 = train_size2,
                                     valid_size = valid_size,
                                     datapath= datapath,
                                     translate = translate)


    ID_train, xsTrain, yTrain, weightsTrain, nameTrain  = Train[0], Train[1], \
                                                Train[2], Train[3], Train[4]
    if train_size2 !=0:
        ID_train2, xsTrain2, yTrain2, weightsTrain2, nameTrain2  = Train2[0], \
                                        Train2[1], Train2[2], Train2[3], Train2[4]
    if valid_size !=0:
        ID_valid, xsValid, yValid, weightsValid, nameValid = Validation[0], \
                                                    Validation[1], Validation[2],\
                                                    Validation[3], Validation[4]

    ID_test, xsTest, nameTest = Test[0], Test[1], Test[2]

    # Splitting the data into sub-groups:
    print("    Splitting the train sets")
    ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s = \
            preTreatment.split_8_matrix(ID_train, xsTrain, yTrain, weightsTrain)
    if train_size2 !=0:
        ID_train2_s, xsTrain2_s, yTrain2_s, weightsTrain2_s = \
            preTreatment.split_8_matrix(ID_train2, xsTrain2, yTrain2,
                                            weightsTrain2)
    if valid_size !=0:
        print("    Splitting the valid set")
        ID_valid_s, xsValid_s,yValid_s, weightsValid_s = \
            preTreatment.split_8_matrix(ID_valid, xsValid, yValid, weightsValid)

    print("    Splitting the test set")
    ID_test_s, xsTest_s = preTreatment.split_8_matrix(ID_test, xsTest)

    nameTrain_s = [nameTrain] * 8
    if train_size2 !=0:
        nameTrain2_s = [nameTrain2] * 8
    if valid_size !=0:
        nameValid_s = [nameValid] * 8
    nameTest_s = [nameTest] * 8

    # Delete the columns full of -999
    if remove_999 == True:
        print("    Deleting the invalid inputs")
        # (if u see any suspicious looking person, or article ...)
        for i in range(8):
            # Deleting the feature identical within each group:
            xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[22],1)
            nameTrain_s[i] = np.delete(nameTrain_s[i], 22)

            if train_size2 !=0:
                xsTrain2_s[i] = np.delete(xsTrain2_s[i], np.s_[22],1)
                nameTrain2_s[i] = np.delete(nameTrain2_s[i], 22)

            if valid_size !=0:
                xsValid_s[i] = np.delete(xsValid_s[i], np.s_[22],1)
                nameValid_s[i] = np.delete(nameValid_s[i], 22)

            xsTest_s[i]  = np.delete(xsTest_s[i],  np.s_[22],1)
            nameTest_s[i] = np.delete(nameTest_s[i], 22)

            # List of columns to be removed: if there is a -999
            toBeRemoved = np.sum(xsTrain_s[i]==-999., axis=0) >= 1
            tBR_ind = []
            for j,elmt in enumerate(toBeRemoved):
                if elmt == True:
                    tBR_ind.append(j)

            # Remove the colums:
            xsTrain_s[i] = np.delete(xsTrain_s[i], tBR_ind, axis=1)
            nameTrain_s[i] = np.delete(nameTrain_s[i], tBR_ind)

            if train_size2 !=0:
                xsTrain2_s[i] = np.delete(xsTrain2_s[i], tBR_ind, axis=1)
                nameTrain2_s[i] = np.delete(nameTrain2_s[i], tBR_ind)

            if valid_size !=0:
                xsValid_s[i] = np.delete(xsValid_s[i], tBR_ind, axis=1)
                nameValid_s[i] = np.delete(nameValid_s[i], tBR_ind)

            xsTest_s[i] = np.delete(xsTest_s[i], tBR_ind, axis=1)
            nameTest_s[i] = np.delete(nameTest_s[i], tBR_ind)

    if train_size2 !=0 and valid_size !=0:
        return (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s, nameTrain_s), \
           (ID_train2_s, xsTrain2_s, yTrain2_s, weightsTrain2_s, nameTrain2_s), \
           (ID_valid_s, xsValid_s, yValid_s, weightsValid_s, nameValid_s), \
           (ID_test_s, xsTest_s, nameTest_s)

    if train_size2 != 0 and valid_size ==0:
        return (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s, nameTrain_s), \
           (ID_train2_s, xsTrain2_s, yTrain2_s, weightsTrain2_s, nameTrain2_s), \
           (ID_test_s, xsTest_s, nameTest_s)

    if train_size2 == 0 and valid_size !=0:
         return (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s, nameTrain_s), \
           (ID_valid_s, xsValid_s, yValid_s, weightsValid_s, nameValid_s), \
           (ID_test_s, xsTest_s, nameTest_s)

    if train_size2 ==0 and valid_size ==0:
         return (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s, nameTrain_s), \
           (ID_test_s, xsTest_s, nameTest_s)


def extract_data(split= True, normalize= True,
                 noise_variance= 0., remove_999= True,
                 n_classes = "binary", train_size = 200000, train_size2 = 25000,
                 valid_size = 25000,
                 datapath= "", translate = False):
    """
    Function wrapping the extraction of the data for any of the possible cases.
    """
    if split == True:
        # Split the data into 8 sub-datasets:
        return get_8_bins(normalize= normalize,
                          noise_variance= noise_variance,
                          remove_999 = remove_999, n_classes = n_classes,
                          train_size = train_size, train_size2 = train_size2,
                          valid_size = valid_size,
                          datapath= datapath, translate = translate)
    else:
        # Extract the data as a unique dataset:
        return get_all_data(normalize= normalize,
                            noise_variance= noise_variance,
                            n_classes = n_classes,
                            train_size = train_size,
                            train_size2 = train_size2,
                            valid_size = valid_size,
                            datapath= datapath, translate = translate)


