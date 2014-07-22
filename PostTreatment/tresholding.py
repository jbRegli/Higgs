#-*- coding: utf-8 -*-

"""
Systeme de vote entre classifier
"""
import sys
import numpy as np
import scipy.stats as ss

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

sys.path.append('Analyses/')
import analyse


def proba_treshold(yPredicted_s, yProba_s, ratio):
    """
    return a vector or a list of vector which keeps only the ratio% of events with the highest proba
    y = list of vectors of label or vector of label (if list, returns a list, if vector, returns a vector)
    yProba = vector of vectors of proba or vector of label
    ratio : the percentage of events we want to keep
    """
    
    if type(yPredicted_s) == list:
        yPredicted_thd_s = []
        L=[]
        for yPredicted, yProba in zip(yPredicted_s, yProba_s):
            yPredicted_thd = np.zeros_like(yPredicted)

            for i in range(yPredicted.shape[0]):
                if yPredicted[i] ==1:
                    L.append(yProba[i][1])

            L.sort(reverse = True)
            if len(L) != 0:
                prob_limit = L[int(len(L)*ratio)]
            else:
                prob_limit = 0.

            for i in range(yPredicted.shape[0]):
                if yProba[i][1] < prob_limit:
                    yPredicted_thd[i] = 0
                else:
                    yPredicted_thd[i] = 1
            
            yPredicted_thd_s.append(yPredicted_thd)
        
        return yPredicted_thd_s

    else:
        yPredicted_thd = np.zeros_like(yPredicted)

        for i in range(yPredicted.shape[0]):
            if yPredicted[i] ==1:
                L.append(yProba[i][1])

        L.sort(reverse = True)
        prob_limit = L[int(len(L)*ratio)]

        for i in range(yPredicted.shape[0]):
            if yProba[i][1] < prob_limit:
                yPredicted_thd[i] = 0
            else:
                yPredicted_thd[i] = 1

        return yPredicted_thd

def get_yPredicted_treshold(yProba, treshold):
    """
    return vector of predicted label with a confidence treshold of treshold
    yProba : vectors of the probabilities computed with a classifier
    yValid : vectors of the true label of the data
    yWeights : vectors of the weights
    pas : size of the interval between two probabilities tested
    """
    yPredicted = np.zeros_like(yProba)
    for i in range(yPredicted.shape[0]):
        if yProba[i] > treshold:
            yPredicted[i] = 1.
    return yPredicted

def best_treshold(yProba, yValidation, weightsValidation, pas = 0.01):
    """
    Returns the treshold that maximises the AMS for the vectors of proba given
    yProba : vectors of the probabilities computed with a classifier
    yValid : vectors of the true label of the data
    yWeights : vectors of the weights
    pas : size of the interval between two probabilities tested
    """
    treshold_s = np.arange(0., 1.0, pas)
    best_ams = 0.

    for treshold in treshold_s:
        yPredicted = get_yPredicted_treshold(yProba, treshold)
        s, b = submission.get_s_b(yPredicted, yValidation, weightsValidation)
        s *= 250000/yPredicted.shape[0]
        b *= 250000/yPredicted.shape[0]
        ams = hbc.AMS(s,b)
        if ams > best_ams:
            best_treshold = treshold
            best_ams = ams

    return best_treshold







