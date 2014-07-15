"""
Given a prediction compute the grading and a submission file
"""

import numpy as np
import time
import csv

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

    if yPredicted.shape != yValidation.shape or \
            yValidation.shape != weightsValidation.shape:
        print "Bad inputs shapes. Inputs must be the same size"
        exit()

    s = np.dot(yPredicted*yValidation, weightsValidation)
    #yPredictedComp = np.ones(yPredicted.shape) - yPredicted #vector with label 0 for event and label 1 for non event
    yValidationComp = np.ones(yValidation.shape) - yValidation #vector with label 0 for event and label 1 for non event
    b = np.dot(yPredicted*yValidationComp, weightsValidation)

    return s, b


def get_numerical_score(yPredicted, yValidation):

    if yPredicted.shape != yValidation.shape:
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
    for n in range(8):
        s, b = get_s_b(yPredicted_s[n], yValidation_s[n], weightsValidation_s[n])
        final_s +=s
        final_b +=b

    return final_s, final_b
