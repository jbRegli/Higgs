import numpy as np
import submission
import imp
import sys

from itertools import product

import analyse # Function computing an analyse for any method in the good format
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda

sys.path.append('../')
import HiggsBosonCompetition_AMSMetric_rev1 as hbc


def parameters_grid_search(train_s, valid_s, method_name, kwargs):
    """
    methode name = string, name of the method (eg :"naiveBayes")
    kwargs = dictionnary of the parameters of the method: range to be tested
    """
    exp = 0
    kwargs_test = {}
    dTuning = {}
    for items in product(*kwargs.values()):
        for i, key in enumerate(kwargs.keys()):
            kwargs_test[key] = items[i]

        d = analyse.analyse(train_s, valid_s, method_name, kwargs_test)

        dTuning[exp]= d
        exp += 1

    return dTuning






