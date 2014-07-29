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
import xgBoost

sys.path.append('../PostTreatment')
import tresholding


def analyse(train_s, train2_s=None, valid_s=None, method_name=None, kwargs=None):
    """
    methode name = string, name of the method (eg :"naiveBayes")
    kwargs = dictionnary of the paraters of the method
    """


    # Prediction on the validation set:
    print("------------------- Analyse: %s -----------------------") \
        %(method_name)

    predictor_s, yPredictedTrain2_s, yProbaTrain2_s = eval(method_name).get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                train2_s[1],
                                                                **kwargs)

    yPredictedValid_s, yProbaValid_s = eval(method_name).get_test_prediction(predictor_s, valid_s[1])
                                                            


    # Let's convert the four 's' classes in s
    # TODO: Option 4 's' scenario?
    if type(yPredictedTrain2_s) == list:
        for i in range(len(yPredictedTrain2_s)):
            for j in range(yPredictedTrain2_s[i].shape[0]):
                if yPredictedTrain2_s[i][j] >=1:
                    yPredictedTrain2_s[i][j] =1
    else:
        for j in range(yPredictedTrain2_s.shape[0]):
                if yPredictedTrain2_s[j] >=1:
                    yPredictedTrain2_s[j] =1

    if type(yPredictedValid_s) == list:
        for i in range(len(yPredictedValid_s)):
            for j in range(yPredictedValid_s[i].shape[0]):
                if yPredictedValid_s[i][j] >=1:
                    yPredictedValid_s[i][j] =1
    else:
        for j in range(yPredictedValid_s.shape[0]):
                if yPredictedValid_s[j] >=1:
                    yPredictedValid_s[j] =1

    #convert the validations vectors four s into s
    if type(valid_s[2]) == list:
        for i in range(len(valid_s[2])):
            for j in range(valid_s[2][i].shape[0]):
                if valid_s[2][i][j] >=1:
                    valid_s[2][i][j] = 1

    #convert the validations vectors four s into s
    if type(train2_s[2]) == list:
        for i in range(len(train2_s[2])):
            for j in range(train2_s[2][i].shape[0]):
                if train2_s[2][i][j] >=1:
                    train2_s[2][i][j] = 1


    # Let's define the vector of probabilities of 's'
    if type(yProbaTrain2_s) == list:
        yProbaTrain2Binary_s = []
        for i in range(8):
            yProbaTrain2Binary_s.append(np.zeros(yPredictedTrain2_s[i].shape[0]))
        for i in range(8):
            for j in range(yPredictedTrain2_s[i].shape[0]):
                yProbaTrain2Binary_s[i][j] = 1 - yProbaTrain2_s[i][j][0]
    else:
        yProbaTrain2Binary_s = np.zeros(yPredictedTrain2_s.shape[0])
        for j in range(yPredictedTrain2_s.shape[0]):
            yProbaTrain2Binary_s[j] = 1 - yProbaTrain2_s[j][0]

    if type(yPredictedValid_s) == list:
        for i in range(len(yPredictedValid_s)):
            for j in range(yPredictedValid_s[i].shape[0]):
                if yPredictedValid_s[i][j] >=1:
                    yPredictedValid_s[i][j] =1
    else:
        yProbaValidBinary_s = np.zeros(yPredictedValid_s.shape[0])
        for j in range(yPredictedValid_s.shape[0]):
            yProbaValidBinary_s[j] = 1 - yProbaValid_s[j][0]

    # Let's define the vector of probabilities of 's'
    if type(yProbaValid_s) == list:
        yProbaValidBinary_s = []
        for i in range(8):
            yProbaValidBinary_s.append(np.zeros(yPredictedValid_s[i].shape[0]))
        for i in range(8):
            for j in range(yPredictedValid_s[i].shape[0]):
                yProbaValidBinary_s[i][j] = 1 - yProbaValid_s[i][j][0]
    else:
        yProbaValidBinary_s = np.zeros(yPredictedValid_s.shape[0])
        for j in range(yPredictedValid_s.shape[0]):
            yProbaValidBinary_s[j] = 1 - yProbaValid_s[j][0]

    # If we work with lists, let's get the concatenated vectors:
    # VALID SET
    # Validation Vectors
    if type(valid_s[2]) == list:
        yValid_conca = preTreatment.concatenate_vectors(valid_s[2])
    else:
        yValid_conca = valid_s[2]
    # Weights Vectors
    if type(valid_s[3]) == list:
        weightsValid_conca = preTreatment.concatenate_vectors(valid_s[3])
    else:
        weights_conca = valid_s[3]
    # Binary Proba Vectors
    if type(yProbaValidBinary_s) == list:
        yProbaValidBinary_conca = preTreatment.concatenate_vectors(yProbaValidBinary_s)
    else:
        yProbaValidBinary_conca = yProbaValidBinary_s
    # All Proba Vectors
    if type(yProbaValid_s) == list:
        yProbaValid_conca = preTreatment.concatenate_vectors(yProbaValid_s)
    else:
        yProbaValid_conca = yProbaValid_s
    # Predicted Valid Vectors
    if type(yPredictedValid_s) == list:
        yPredictedValid_conca = preTreatment.concatenate_vectors(yPredictedValid_s)
    else:
        yPredictedValid_conca = yPredictedValid_s

    #TRAIN2 SET
    # Validation Vectors
    if type(train2_s[2]) == list:
        yTrain2_conca = preTreatment.concatenate_vectors(train2_s[2])
    else:
        yTrain2_conca = train2_s[2]
    # Weights Vectors
    if type(train2_s[3]) == list:
        weightsTrain2_conca = preTreatment.concatenate_vectors(train2_s[3])
    else:
        weightsTrain2_conca = train2_s[3]
    # Binary Proba Vectors
    if type(yProbaTrain2Binary_s) == list:
        yProbaTrain2Binary_conca = preTreatment.concatenate_vectors(yProbaTrain2Binary_s)
    else:
        yProbaTrain2Binary_conca = yProbaTrain2Binary_s
    # All Proba Vectors
    if type(yProbaTrain2_s) == list:
        yProbaTrain2_conca = preTreatment.concatenate_vectors(yProbaTrain2_s)
    else:
        yProbaTrain2_conca = yProbaTrain2_s
    # Predicted Valid Vectors
    if type(yPredictedTrain2_s) == list:
        yPredictedTrain2_conca = preTreatment.concatenate_vectors(yPredictedTrain2_s)
    else:
        yPredictedTrain2_conca = yPredictedTrain2_s


    # Get s and b for each group (s_s, b_s) and the final final_s and
    # final_b:
    if type(yPredictedValid_s) == list:
        final_s, final_b, s_s, b_s = submission.get_s_b(yPredictedValid_s, valid_s[2],
                                                          valid_s[3])
    else:
        final_s, final_b = submission.get_s_b(yPredictedValid_s, valid_s[2], valid_s[3])

    # Let's get the best global treshold and ratio on the train2 set and
    # estimate
    best_treshold_global = tresholding.best_treshold(yProbaTrain2Binary_conca, yTrain2_conca,
                                                                        weightsTrain2_conca)
    yPredictedValid_conca_treshold = tresholding.get_yPredicted_treshold(yProbaValidBinary_conca,
                                                                  best_treshold_global)

    best_ratio_global = tresholding.best_ratio(yProbaTrain2Binary_conca, yTrain2_conca,
                                                                        weightsTrain2_conca)
    print "yPredictedValid_conca_treshold type : %s" %yPredictedValid_conca_treshold.shape[0]
    print "best_ratio_global : %f" %best_ratio_global
    yPredictedValid_conca_ratio = tresholding.get_yPredicted_ratio(yProbaValidBinary_conca,
                                                                  best_ratio_global)

    final_s_treshold, final_b_treshold = submission.get_s_b(yPredictedValid_conca_treshold,
                                                        yValid_conca, weightsValid_conca)
    final_s_ratio, final_b_ratio = submission.get_s_b(yPredictedValid_conca_ratio,
                                                        yValid_conca, weightsValid_conca)




    #Balance the s and b

    final_s *= 250000/yValid_conca.shape[0]
    final_b *= 250000/yValid_conca.shape[0]
    final_s_treshold *= 250000/yValid_conca.shape[0]
    final_b_treshold *= 250000/yValid_conca.shape[0]
    final_s_ratio *= 250000/yValid_conca.shape[0]
    final_b_ratio *= 250000/yValid_conca.shape[0]

    # AMS final:
    AMS = hbc.AMS(final_s , final_b)
    AMS_treshold = hbc.AMS(final_s_treshold, final_b_treshold)
    AMS_ratio = hbc.AMS(final_s_ratio, final_b_ratio)
    
    #AMS by group:
    if type(valid_s[2]) == list:
        AMS_s = []
        for i in range(8):
            print s_s[i]
            print b_s[i]
        for i, (s,b) in enumerate(zip(s_s, b_s)):
            s *= 250000/yPredictedValid_s[i].shape[0]
            b *= 250000/yPredictedValid_s[i].shape[0]
            score = hbc.AMS(s,b)
            AMS_s.append(score)
    else:
        AMS_s = AMS
   
    AMS_s = AMS
    # Classification error:
    classif_succ = eval(method_name).get_classification_error(yPredictedValid_s,
                                                       valid_s[2],
                                                       normalize= True)

    # Numerical score:
    if type(yPredictedValid_s) == list:
        sum_s_s = []
        sum_b_s = []
        for i in range(len(yPredictedValid_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredictedValid_s[i],
                                                          valid_s[2][i])
            sum_s_s.append(sum_s)
            sum_b_s.append(sum_b)

    else:
        sum_s_s, sum_b_s = submission.get_numerical_score(yPredictedValid_s,
                                                           valid_s[2])

    d = {'predictor_s':predictor_s,
         'yPredicted_s': yPredictedValid_s, 'yPredicted_conca': yPredictedValid_conca, 'yPredicted_conca_treshold':yPredictedValid_conca_treshold,
         'yProba_s': yProbaValid_s, 'yProba_conca': yProbaValid_conca,
         'yProbaBinary_s': yProbaValidBinary_s, 'yProbaBinary_conca': yProbaValidBinary_conca,
         'final_s':final_s, 'final_b':final_b,
         'sum_s':sum_s_s, 'sum_b': sum_b_s,
         'AMS':AMS, 'AMS_s': AMS_s, 'AMS_treshold': AMS_treshold, 'best_treshold_global' : best_treshold_global,
         'AMS_ratio': AMS_ratio, 'best_ratio_global':best_ratio_global,
         'classif_succ': classif_succ,
         'method': method_name,
         'parameters': kwargs}


    return d


def get_test_prediction(method_name, predictor_s, xsTest_s):
    return eval(method_name).get_test_prediction(predictor_s, xsTest_s)



