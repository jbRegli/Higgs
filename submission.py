"""
Given a prediction compute the grading and a submission file
"""

import numpy as np
import time
import csv
import scipy.stats as ss

def print_submission(testIDs, RankOrder, yLabels,
    name = 'submission_'+time.strftime("%H:%M:%S")):
    """
    Creates the csv submission file
    ----------
    Inputs:
    ----------
        testIDs: np.array(int, size :Test DataSet Size)
        vecors of the IDs of the test examples

        RankOrder: np.array(int, size :Test DataSet Size)
        vecors of the ranks : the higher is the rank, the most likely the label is to
        be 's'

        yLabels: np.array(binary (1 or 0),size:Test DataSet Size)
        vecors of the labels of the test examples (1 for label 's', 0 for label 'b')

    ----------
    Outputs:
    ----------
        [0] : np.array(string, size:(Test DataSet Size, 3))
        np array in the submission format (includes the header : EventId,RankOrder,
        Class)

    """
    labels = []
    for i in range(len(testIDs)):
        if yLabels[i] == 1:
            labels.append('s')
        else:
            labels.append('b')

    testIDs = testIDs.astype(np.int64)
    labels = np.array(labels)

    sub = np.array([[str(testIDs[i]), str(RankOrder[i]), labels[i] ] for i in range(len(testIDs))])
    sub = sub[sub[:,0].argsort()]
    sub = np.append([['EventID', 'RankOrder', 'Class']], sub, axis = 0)

    with open(name + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(sub.shape[0]):
            writer.writerow(sub[i])

    return sub


def get_s_b(yPredicted, yValidation, weightsValidation):
    """
    Takes in input the vector of predicted labels on the validation set,
    the vector of real label of the validation set and the vector of weights on the
    validation set, and returns the weighted sum of the real positive (s) and the the
    weighted sum of the real negative (b)
    """
    print ("+++ get_s_b: type(yPredicted)= ", type(yPredicted))

    if type(yPredicted) != list:
        print ("+++ get_s_b: yPredicted.shape= ", yPredicted.shape)
        print ("+++ get_s_b: yValidation.shape= ", yValidation.shape)
        print ("+++ get_s_b: weightsValidation.shape= ", weightsValidation.shape)


        if yPredicted.shape[0] != yValidation.shape[0] or \
                yValidation.shape[0] != weightsValidation.shape[0]:
            print "submission.get_s_b: "
            print "Bad inputs shapes. Inputs must be the same size"
            if yPredicted.shape[0] != yValidation.shape[0]:
                print "yPredicted.shape= ", yPredicted.shape
                print "yValidation.shape= ", yValidation.shape
            else:
                print "yValidation.shape= ", yValidation.shape
                print "weightsValidation.shape= ", weightsValidation.shape
            exit()
        #Balance the weights
        sumW_total = 411691.836 #sum of the weights on all the training set (250 000)
        sumW = sum(weightsValidation)
        weightsValidationBalanced = weightsValidation * sumW_total/sumW

<<<<<<< HEAD
        s = np.dot(yPredicted*yValidation, weightsValidation)
        #yPredictedComp = np.ones(yPredicted.shape) - yPredicted
        #vector with label 0 for event and label 1 for non event
        yValidationComp = np.ones(yValidation.shape[0]) - yValidation
        #vector with label 0 for event and label 1 for non event
        b = np.dot(yPredicted*yValidationComp, weightsValidation)
=======
        s = np.dot(yPredicted*yValidation, weightsValidationBalanced)
        #yPredictedComp = np.ones(yPredicted.shape) - yPredicted #vector with label 0 for event and label 1 for non event
        yValidationComp = np.ones(yValidation.shape[0]) - yValidation #vector with label 0 for event and label 1 for non event
        b = np.dot(yPredicted*yValidationComp, weightsValidationBalanced)
>>>>>>> 0f6d96e59d08b83d891c061998d4d6a1f66148b5

        return s, b

    else:
        final_s, final_b, s_s, b_s = get_s_b_8(yPredicted, yValidation, weightsValidation)

        return final_s, final_b, s_s, b_s


def get_numerical_score(yPredicted, yValidation):

    if yPredicted.shape != yValidation.shape:
        print "submission.get_numerical_score: "
        print "Bad inputs shapes. Inputs must be the same size"
        exit()

    sum_s = np.sum(yPredicted*yValidation)
    #yPredictedComp = np.ones(yPredicted.shape) - yPredicted #vector with label 0 for event and label 1 for non event
    yValidationComp = np.ones(yValidation.shape) - yValidation #vector with label 0 for event and label 1 for non event
    sum_b = np.sum(yPredicted*yValidationComp)

    return sum_s, sum_b


def get_s_b_8(yPredicted_s, yValidation_s, weightsValidation_s):
    final_s = 0.
    final_b =0.
    s_s = []
    b_s = []
    for n in range(8):
        s, b = get_s_b(yPredicted_s[n], yValidation_s[n], weightsValidation_s[n])
        s_s.append(s)
        b_s.append(b)
        final_s +=s
        final_b +=b

    return final_s, final_b, s_s, b_s


###############
### RANKING ###
###############

def rank_signals(proba_prediction):
    """
    Given a list of probability of being a signal, return the rank prediction
    """
    temp = proba_prediction.argsort()
    rank_prediction = np.arange(len(proba_prediction))[temp.argsort()]
    rank_prediction += np.ones(len(rank_prediction))
    #rank_prediction = ss.rankdata(proba_prediction,method = 'ordinal')

    return rank_prediction
