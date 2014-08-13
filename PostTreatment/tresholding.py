#-*- coding: utf-8 -*-

"""
Systeme de vote entre classifier
"""
import sys
import numpy as np
import scipy.stats as ss
import itertools

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

    yPredicted[yProba > treshold] =1
    """
    for i in range(yPredicted.shape[0]):
        if yProba[i] > treshold:
            yPredicted[i] = 1.
    """
    return yPredicted

def get_yPredicted_ratio(yProba, ratio):
    """
    Returns vector of yPredicted keeping the ratio% highest percentages
    ratio : float
    yProba : vector of proba
    """
    yPredicted = np.zeros_like(yProba)

    if ratio !=0:
            yProbaSorted = yProba[yProba.argsort()]
            treshold = np.max(yProbaSorted[int((1. - float(ratio))*len(yProba))])
            yPredicted = get_yPredicted_treshold(yProba, treshold)

    return yPredicted

def best_treshold(yProba, yValidation, weightsValidation, pas = 0.01):
    """
    Returns the treshold that maximises the AMS for the vectors of proba given
    yProba : vectors of the probabilities computed with a classifier
    yValid : vectors of the true label of the data
    yWeights : vectors of the weights
    pas : size of the interval between two probabilities tested
    The weights must be balanced !
    """

    treshold_s = np.arange(0., 1.0, pas)
    best_ams = 0.

    for treshold in treshold_s:
        yPredicted_prov = get_yPredicted_treshold(yProba, treshold)

        # if we work with multi-class:
        if len(yPredicted_prov.shape) == 2:
            if yPredicted_prov.shape[1] == 5:
                # Reduce multiclass to binary
                yPredicted = np.ones(yPredicted_prov.shape[0])
                yPredicted[yPredicted_prov[:,4] == 0] = 0
            else:
                print "Error: in best_treshold() the shape of the input isn't correct"
        else:
            yPredicted = yPredicted_prov

        s, b = submission.get_s_b(yPredicted, yValidation, weightsValidation)
        ams = hbc.AMS(s,b)
        if ams >= best_ams:
            best_treshold = treshold
            best_ams = ams

    return best_ams, best_treshold


def best_ratio(yProba, yValidation, weightsValidation, pas = 0.01):
    ratio_s = np.arange(0., 0.99, pas)
    best_ams = 0.

    for ratio in ratio_s:


        yPredicted = get_yPredicted_ratio(yProba, ratio)

        s, b = submission.get_s_b(yPredicted, yValidation, weightsValidation)

        if b >= 0. and s >= 0.:
            ams = hbc.AMS(s,b)
            if ams >= best_ams:
                best_ratio = ratio
                best_ams = ams
        else:
            if b < 0.:
                print ("WARNING: For a ratio of %f, b < 0 (b= %f).") %(ratio, b)
                print ("This ratio has been ignored.")
            else:
                print ("WARNING: For a ratio of %f, s < 0 (s= %f).") %(ratio, s)
                print ("This ratio has been ignored.")
        ams = hbc.AMS(s,b)
        if ams >= best_ams:
            best_ratio = ratio
            best_ams = ams

    return best_ams, best_ratio

def get_yPredicted_ratio_8(yProba_s, ratio_s):
    """
    returns a list of predicted y for each group associated with each ratio
    """
    yPredicted_s = []
    for i, ratio in enumerate(ratio_s):
        yPredicted_s.append(get_yPredicted_ratio(yProba_s[i], ratio))
    yPredicted_conca = preTreatment.concatenate_vectors(yPredicted_s)

    return yPredicted_s, yPredicted_conca

def best_ratio_combinaison(yProba_s, yValidation_s, weightsValidation_s, ratio_s):
    """
    returns the best ratio combinaison with the ratios specified in ratio_s for each
    group
    ratio_s : List of the list of the ratios to test for each group
    the size of each list should not exceed 4 for computationnal time issues
    """
    best_ratio_comb = [0.,0.,0.,0.,0.,0.,0.,0.]
    AMS_max = 0.
    """
    ratio_1_s = [0.06, 0.08,0.10,0.12]
    ratio_2_s = [0.15,0.16,0.17,0.18]
    ratio_3_s = [0.36,0.38,0.40,0.42]
    ratio_4_s = [0.16,0.18,0.2,0.22]
    ratio_5_s = [0.007,0.008,0.009,0.01]
    ratio_6_s = [0.003,0.004,0.005,0.006]
    ratio_7_s = [0.003,0.004,0.005,0.006]
    ratio_8_s = [0.007,0.008,0.009,0.01]
    """
    g_combinaisons = itertools.product(ratio_s[0], ratio_s[1],
                                       ratio_s[2], ratio_s[3],
                                       ratio_s[4], ratio_s[5],
                                       ratio_s[6], ratio_s[7])

    # if we work with multi-class:
    if len(yProba_s[0].shape) == 2:
            if yProba_s[0].shape[1] == 5:
                for i,subset in enumerate(yProba_s):
                    yProba_s[i] =  preTreatment.multiclass2binary(subset)

    compteur = 0

    for combinaison in g_combinaisons:
        #if compteur%10000==0:
            # print "number of iterations : %i" %compteur
        compteur +=1

        L = list(combinaison)

        yPredicted_s, yPredicted_conca = get_yPredicted_ratio_8(yProba_s, L)

        finals, finalb, s_s, b_s = submission.get_s_b(yPredicted_s,
                                                      yValidation_s,
                                                      weightsValidation_s)

        AMS = hbc.AMS(finals, finalb)
        if AMS > AMS_max:
            AMS_max = AMS
            best_ratio_comb = L

    return AMS_max, best_ratio_comb

def best_ratio_combinaison_global(yProba_s, yValidation_s, weightsValidation_s,
                                  max_iters):
    """
    returns the best ratio combinaison global after n iterations
    """
    AMS_max = 0.
    best_ratio_comb = [0.,0.,0.,0.,0.,0.,0.,0.]
    ratio_s = []
    for i in range(8):
        ratio_s.append([0, 0.25,0.5])

    for n in range(max_iters):
        # print "iteration globale : %i" %n
        AMS_new, ratio_comb = best_ratio_combinaison(yProba_s, yValidation_s,
                                                     weightsValidation_s, ratio_s)
        for i in range(8):
            # Case 1 : minimum
            if ratio_comb[i] == ratio_s[i][0]:
                if ratio_s[i][0] !=0:
                    ratio_s[i] = [0, ratio_s[i][0], ratio_s[i][1]]
                else:
                    ratio_s[i] = [0, ratio_s[i][1]/2, ratio_s[i][1]]

            # Case 2 : maximum
            if ratio_comb[i] == ratio_s[i][2]:
                ratio_s[i] = [ratio_s[i][1], ratio_s[i][2],
                                    2*ratio_s[i][2] -  ratio_s[i][1]]
            else:
                ratio_s[i] = [(ratio_s[i][1]+ratio_s[i][0])/2,
                                ratio_s[i][1], (ratio_s[i][2] + ratio_s[i][1])/2]

        if AMS_new > AMS_max:
            AMS_max = AMS_new
            best_ratio_comb = ratio_comb
            #else:
             #   equilibre = True

    return AMS_max, best_ratio_comb
