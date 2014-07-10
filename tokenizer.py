"""
Load the dataset
"""

# -*- coding: utf-8 -*-

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

    (xsTrain, yTrain, weightsTrain), (xsValidation, yValidation, weightsValidation)
    xsTrain : np array of size((training_ratio*250000, 30)) of float representing the features
    yTrain : np array of size(training_ratio*250000) of int representing the label of the data (1 if boson, 0 else)
    weightsTrain : np array of size(training_ratio*250000) of float representing the weights

    xsValidation : np array of size(((1-training_ratio)*250000, 30)) of float representing the features
    yValidation : np array of size((1-training_ratio)*250000) of int representing the label of the data (1 if boson, 0 else)
    weightsValidation : np array of size((1-training_ratio)*250000) of float representing the weights


    """
    all = list(csv.reader(open("training.csv","rb"), delimiter=','))

    xs = np.array([map(float, row[1:-2]) for row in all[1:]])
    (numPoints,numFeatures) = xs.shape

    #normalize
    if normalize == True:
        xs = preTreatment.normalize(xs)

    #add gaussian noise
    if noise_variance != 0.:
        xs = preTreatment.add_noise(xs)


    #select label
    sSelector = np.array([row[-1] == 's' for row in all[1:]])
    bSelector = np.array([row[-1] == 'b' for row in all[1:]])

    #select weights
    weights = np.array([float(row[-2]) for row in all[1:]])
    sumWeights = np.sum(weights)
    sumSWeights = np.sum(weights[sSelector])
    sumBWeights = np.sum(weights[bSelector])

    randomPermutation = random.sample(range(len(xs)), len(xs))
    numPointsTrain = int(numPoints*0.9)
    numPointsValidation = numPoints - numPointsTrain

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

    return (xsTrain, yTrain, weightsTrain), (xsValidation, yValidation, weightsValidation)


def get_8_bins(normalize = True, noise_variance = 0.):
    """
    returns (xsTrain_s, yTrain_s, weightsTrain_s), (xsValidation_s, yValidation_s, weightsValidation_s)
    list of the data containing the eight different groups
    """

    # Extracting the data:
    Train, Validation = get_all_data(normalize = normalize, noise_variance = noise_variance)

    xsTrain, yTrain, weightsTrain  = Train[0], Train[1], Train[2]
    xsValidation, yValidation, weightsValidation = Validation[0], Validation[1], Validation[2]

    # Splitting them into sub-groups:
    xsTrain_s = []
    yTrain_s = []
    weightsTrain_s =[]
    xsValidation_s = []
    yValidation_s = []
    weightsValidation_s = []

    for n in range(8):
        xsTrain_s.append(np.zeros((0,xsTrain.shape[1])))
        yTrain_s.append(np.zeros(0))
        weightsTrain_s.append(np.zeros(0))
        xsValidation_s.append(np.zeros((0,xsValidation.shape[1])))
        yValidation_s.append(np.zeros(0))
        weightsValidation_s.append(np.zeros(0))


    for i in range(xsTrain.shape[0]):
        if xsTrain[i,0] != -999:
            if xsTrain[i,22] == 0:
                xsTrain_s[0] = np.vstack([xsTrain_s[0], xsTrain[i,:]])
                yTrain_s[0] = np.append(yTrain_s[0], yTrain[i])
                weightsTrain_s[0] = np.append(weightsTrain_s[0], weightsTrain[i])

            if xsTrain[i,22] == 1:
                xsTrain_s[1] = np.vstack([xsTrain_s[1], xsTrain[i,:]])
                yTrain_s[1] = np.append(yTrain_s[1], yTrain[i])
                weightsTrain_s[1] = np.append(weightsTrain_s[1], weightsTrain[i])

            if xsTrain[i,22] == 2:
                xsTrain_s[2] = np.vstack([xsTrain_s[2], xsTrain[i,:]])
                yTrain_s[2] = np.append(yTrain_s[2], yTrain[i])
                weightsTrain_s[2] = np.append(weightsTrain_s[2], weightsTrain[i])

            if xsTrain[i,22] == 3:
                xsTrain_s[3] = np.vstack([xsTrain_s[3], xsTrain[i,:]])
                yTrain_s[3] = np.append(yTrain_s[3], yTrain[i])
                weightsTrain_s[3] = np.append(weightsTrain_s[3], weightsTrain[i])
        else:
            if xsTrain[i,22] == 0:
                xsTrain_s[4] = np.vstack([xsTrain_s[4], xsTrain[i,:]])
                yTrain_s[4] = np.append(yTrain_s[4], yTrain[i])
                weightsTrain_s[4] = np.append(weightsTrain_s[4], weightsTrain[i])

            if xsTrain[i,22] == 1:
                xsTrain_s[5] = np.vstack([xsTrain_s[5], xsTrain[i,:]])
                yTrain_s[5] = np.append(yTrain_s[5], yTrain[i])
                weightsTrain_s[5] = np.append(weightsTrain_s[5], weightsTrain[i])

            if xsTrain[i,22] == 2:
                xsTrain_s[6] = np.vstack([xsTrain_s[6], xsTrain[i,:]])
                yTrain_s[6] = np.append(yTrain_s[6], yTrain[i])
                weightsTrain_s[6] = np.append(weightsTrain_s[6], weightsTrain[i])

            if xsTrain[i,22] == 3:
                xsTrain_s[7] = np.vstack([xsTrain_s[7], xsTrain[i,:]])
                yTrain_s[7] = np.append(yTrain_s[7], yTrain[i])
                weightsTrain_s[7] = np.append(weightsTrain_s[7], weightsTrain[i])

    for i in range(xsValidation.shape[0]):

        if xsValidation[i,0] != -999:
            if xsValidation[i,22] == 0:
                xsValidation_s[0] = np.vstack([xsValidation_s[0], xsValidation[i,:]])
                yValidation_s[0] = np.append(yValidation_s[0], yValidation[i])
                weightsValidation_s[0] = np.append(weightsValidation_s[0], weightsValidation[i])

            if xsValidation[i,22] == 1:
                xsValidation_s[1] = np.vstack([xsValidation_s[1], xsValidation[i,:]])
                yValidation_s[1] = np.append(yValidation_s[1], yValidation[i])
                weightsValidation_s[1] = np.append(weightsValidation_s[1], weightsValidation[i])

            if xsValidation[i,22] == 2:
                xsValidation_s[2] = np.vstack([xsValidation_s[2], xsValidation[i,:]])
                yValidation_s[2] = np.append(yValidation_s[2], yValidation[i])
                weightsValidation_s[2] = np.append(weightsValidation_s[2], weightsValidation[i])

            if xsValidation[i,22] == 3:
                xsValidation_s[3] = np.vstack([xsValidation_s[3], xsValidation[i,:]])
                yValidation_s[3] = np.append(yValidation_s[3], yValidation[i])
                weightsValidation_s[3] = np.append(weightsValidation_s[3], weightsValidation[i])
        else:
            if xsValidation[i,22] == 0:
                xsValidation_s[4] = np.vstack([xsValidation_s[4], xsValidation[i,:]])
                yValidation_s[4] = np.append(yValidation_s[4], yValidation[i])
                weightsValidation_s[4] = np.append(weightsValidation_s[4], weightsValidation[i])

            if xsValidation[i,22] == 1:
                xsValidation_s[5] = np.vstack([xsValidation_s[5], xsValidation[i,:]])
                yValidation_s[5] = np.append(yValidation_s[5], yValidation[i])
                weightsValidation_s[5] = np.append(weightsValidation_s[5], weightsValidation[i])

            if xsValidation[i,22] == 2:
                xsValidation_s[6] = np.vstack([xsValidation_s[6], xsValidation[i,:]])
                yValidation_s[6] = np.append(yValidation_s[6], yValidation[i])
                weightsValidation_s[6] = np.append(weightsValidation_s[6], weightsValidation[i])

            if xsValidation[i,22] == 3:
                xsValidation_s[7] = np.vstack([xsValidation_s[7], xsValidation[i,:]])
                yValidation_s[7] = np.append(yValidation_s[7], yValidation[i])
                weightsValidation_s[7] = np.append(weightsValidation_s[7], weightsValidation[i])

    # Delete the columns full of -999
    # (if u see any suspicious looking person, or article ...)
    for i in range(8):
        for index_column in range(xsTrain.shape[1]):
            if xsTrain_s[i].shape[1] > index_column:
                if xsTrain_s[i][0,index_column] == -999:
                    xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[index_column],1)
                if xsValidation_s[i][0,index_column] == -999:
                    xsValidation_s[i] = np.delete(xsValidation_s[i], np.s_[index_column],1)

        # Deleting the feature identical within each group:
        xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[22],1)
        xsValidation_s[i] = np.delete(xsValidation_s[i], np.s_[22],1)

    return (xsTrain_s, yTrain_s, weightsTrain_s), (xsValidation_s, yValidation_s, weightsValidation_s)


def get_test_data():

    test = list(csv.reader(open("test.csv", "rb"),delimiter=','))
    xsTest = np.array([map(float, row[1:]) for row in test[1:]])

    testIds = np.array([int(row[0]) for row in test[1:]])

    return xsTest, testIds
