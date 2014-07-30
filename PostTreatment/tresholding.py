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

sys.path.append('../')
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc
import preTreatment

sys.path.append('../Analyses/')



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

def get_yPredicted_ratio(yProba, ratio):
    """
    Returns vector of yPredicted keeping the ratio% highest percentages
    ratio : float
    yProba : vector of proba
    """
    print "get_yPredicted_ratio"
    yPredicted = np.zeros_like(yProba)
    yProbaSorted = yProba[yProba.argsort()]
    print "len yProba : %i" %len(yProba)
    print "len yProbaSorted : %i" %len(yProbaSorted)
    print "ratio : %f" %float(ratio)
    
    treshold = yProbaSorted[int((1. - float(ratio))*len(yProba))]
    yPredicted = get_yPredicted_treshold(yProba, treshold)

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
        if ams >= best_ams:
            best_treshold = treshold
            best_ams = ams

    return best_treshold

def best_ratio(yProba, yValidation, weightsValidation, pas = 0.01):
    ratio_s = np.arange(0.01, 0.99, pas)
    best_ams = 0.

    for ratio in ratio_s:
        yPredicted = get_yPredicted_ratio(yProba, ratio)
        s, b = submission.get_s_b(yPredicted, yValidation, weightsValidation)
        s *= 250000/yPredicted.shape[0]
        b *= 250000/yPredicted.shape[0]
        ams = hbc.AMS(s,b)
        if ams >= best_ams:
            best_ratio = ratio
            best_ams = ams

    return best_ratio

def get_yPredicted_ratio_8(yProba_s, ratio_s):
    """
    returns a list of predicted y for each group associated with each ratio
    """
    print "get_yPredicted_ratio_8"
    print "type ratio_s %s" %type(ratio_s)
    yPredicted_s = []
    for i, ratio in enumerate(ratio_s):
        yPredicted_s.append(get_yPredicted_ratio(yProba_s[i], ratio))
    yPredicted_conca = preTreatment.concatenate_vectors(yPredicted_s)

    return yPredicted_s, yPredicted_conca

def best_ratio_combinaison(yProba_s, yValidation_s, weightsValidation_s):
    best_ratio_comb = [0.,0.,0.,0.,0.,0.,0.,0.]
    AMS_max = 0.
    ratio_s = np.arange(0.04,0.2,0.04)
    for a in ratio_s:
        for b in ratio_s:
            for c in ratio_s:
                for d in ratio_s:
                    for e in ratio_s:
                        for f in ratio_s:
                            for g in ratio_s:
                                for h in ratio_s:
                                    print "best_ratio_combinaison"
                                    yPredicted_s = get_yPredicted_ratio_8(yProba_s,
                                    ["%.2f" %a,"%.2f" %b,"%.2f" %c,"%.2f" %d,"%.2f" %e,"%.2f" %f,"%.2f" %g,"%.2f" %h])[0]
                                    fs, fb, s, b = submission.get_s_b(yPredicted_s, yValidation_s, weightsValidation_s)
                                    fs *=10
                                    fb *=10
                                    AMS = hbc.AMS(fs, fb)
                                    if AMS > AMS_max:
                                        AMS_max = AMS
                                        best_ratio_comb = [a,b,c,d,e,f,g,h]
    return AMS_max, best_ratio_comb


