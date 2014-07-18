import numpy as np
import submission
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

sys.path.append('../')
import HiggsBosonCompetition_AMSMetric_rev1 as hbc


def analyse(train_s, valid_s, method_name, kwargs):
    """
    methode name = string, name of the method (eg :"naiveBayes")
    kwargs = dictionnary of the paraters of the method
    """


    # Prediction on the validation set:
    print("------------------- Analyse: %s -----------------------") \
        %(method_name)

    #import sys

    #sys.path.append('Analyses/')

    #method = imp.load_source(method_script,
                             #str("./Analyses/" + method_script + ".py"))

    predictor_s, yPredicted_s, yProba_s = eval(method_name).get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1],
                                                                **kwargs)
    # Get s and b for each group (s_s, b_s) and the final final_s and
    # final_b:
    final_s, final_b, s_s, b_s = submission.get_s_b_8(yPredicted_s, valid_s[2],
                                                  valid_s[3])

    # Balance the s and b
    final_s *= 250000/25000
    final_b *= 250000/25000
    # AMS final:
    AMS = hbc.AMS(final_s , final_b)
    print ("Expected AMS score for "+method_name+" : %f") %AMS
    #AMS by group
    AMS_s = []
    for i, (s,b) in enumerate(zip(s_s, b_s)):
        s *= 250000/yPredicted_s[i].shape[0]
        b *= 250000/yPredicted_s[i].shape[0]
        score = hbc.AMS(s,b)
        AMS_s.append(score)
        print("Expected AMS score for "+method_name+" :  for group %i is : %f" %(i, score))
    print(" ")

    # Classification error:
    classif_succ = eval(method_name).get_classification_error(yPredicted_s,
                                                       valid_s[2],
                                                       normalize= True)

    for i, ratio in enumerate(classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")
    # Numerical score:
    if type(yPredicted_s) == list:
        sum_s_s = []
        sum_b_s = []
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                          valid_s[2][i])
            sum_s_s.append(sum_s)
            sum_b_s.append(sum_b)

            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, yPredicted_s[i].shape[0], i, sum_s, i, sum_b)

    else:
             sum_s_s, sum_b_s = submission.get_numerical_score(yPredicted_s,
                                                           valid_s[2])

             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(yPredicted_s.shape[0], sum_s_s, sum_b_s)



    d = {'predictor_s':predictor_s, 'yPredicted_s': yPredicted_s,
         'yProba_s': yProba_s,
         'final_s':final_s, 'final_b':final_b,
         'sum_s':sum_s_s, 'sum_b': sum_b_s,
         'AMS':AMS, 'AMS_s': AMS_s,
         'classif_succ': classif_succ,
         'method': method_name,
         'parameters': kwargs}

    print(" ")

    return d


def get_test_prediction(method_name, predictor_s, xsTest_s):
    return eval(method_name).get_test_prediction(predictor_s, xsTest_s)



