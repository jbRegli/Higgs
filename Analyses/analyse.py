import numpy as np
import imp
import sys

sys.path.append('Analyses/')
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda
import gradientBoosting

sys.path.append('../')
import HiggsBosonCompetition_AMSMetric_rev1 as hbc
import preTreatment
import submission


def analyse(train_s, valid_s, method_name, kwargs):
    """
    methode name = string, name of the method (eg :"naiveBayes")
    kwargs = dictionnary of the paraters of the method
    """


    # Prediction on the validation set:
    print("------------------- Analyse: %s -----------------------") \
        %(method_name)

    predictor_s, yPredicted_s, yProba_s = eval(method_name).get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                **kwargs)
    # Let's convert the four 's' classes in s
    # TODO: Option 4 's' scenario?
    if type(yPredicted_s) == list:
        for i in range(len(yPredicted_s)):
            for j in range(yPredicted_s[i].shape[0]):
                if yPredicted_s[i][j] >=1:
                    yPredicted_s[i][j] =1
    else:
        for j in range(yPredicted_s.shape[0]):
                if yPredicted_s[j] >=1:
                    yPredicted_s[j] =1

    # Let's define the vector of binary probabilities b or s, 0 or 1
    if type(yProba_s) == list:
        yProbaBinary_s = []
        for i in range(8):
            yProbaBinary_s.append(np.zeros(yPredicted_s[i].shape[0])
        for i in range(8):
            for j in range(yPredicted_s[i].shape[0]):
                yProbaBinary_s[i][j] = 1 - yProba_s[i][j][0]
    else:
        yProbaBinary_s = np.zeros(yPredicted_s.shape[0])
        for j in range(yPredicted_s.shape[0]):
            yProbaBinary_s[j] = 1 - yProba_s[j][0]

    # Get s and b for each group (s_s, b_s) and the final final_s and
    # final_b:

    final_s, final_b, s_s, b_s = submission.get_s_b_8(yPredicted_s, valid_s[2],
                                                  valid_s[3])

    # Balance the s and b
    if type(valid_s[2]) == list:
        yValid_conca = preTreatment.concatenate_vectors(valid_s[2])
    else:
        yValid_conca = valid_s[2]

    best_treshold_global = postTreatment.best_treshold(y

    final_s *= 250000/yValid_conca.shape[0]
    final_b *= 250000/yValid_conca.shape[0]
    # AMS final:
    AMS = hbc.AMS(final_s , final_b)

    #AMS by group:
    if type(valid_s[2]) == list:
        AMS_s = []
        for i, (s,b) in enumerate(zip(s_s, b_s)):
            s *= 250000/yPredicted_s[i].shape[0]
            b *= 250000/yPredicted_s[i].shape[0]
            score = hbc.AMS(s,b)
            AMS_s.append(score)
    else:
        AMS_s = AMS

    # Classification error:
    classif_succ = eval(method_name).get_classification_error(yPredicted_s,
                                                       valid_s[2],
                                                       normalize= True)

    # Numerical score:
    if type(yPredicted_s) == list:
        sum_s_s = []
        sum_b_s = []
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                          valid_s[2][i])
            sum_s_s.append(sum_s)
            sum_b_s.append(sum_b)

    else:
        sum_s_s, sum_b_s = submission.get_numerical_score(yPredicted_s,
                                                           valid_s[2])

    d = {'predictor_s':predictor_s, 'yPredicted_s': yPredicted_s,
         'yProba_s': yProba_s,
         'final_s':final_s, 'final_b':final_b,
         'sum_s':sum_s_s, 'sum_b': sum_b_s,
         'AMS':AMS, 'AMS_s': AMS_s,
         'classif_succ': classif_succ,
         'method': method_name,
         'parameters': kwargs}


    return d


def get_test_prediction(method_name, predictor_s, xsTest_s):
    return eval(method_name).get_test_prediction(predictor_s, xsTest_s)



