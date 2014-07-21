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




