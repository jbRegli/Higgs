# -*- coding: utf-8 -*-

# this is the example script to use xgboost to train
import inspect
import os
import sys
import numpy as np
import time
from sklearn.metrics import accuracy_score

import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

# TODO: add path of xgboost python module
code_path_jb = '/home/regli/Applications/Python/xgboost/python'
sys.path.append(code_path_jb)
code_path_nico = '../xgboost/python'
sys.path.append(code_path_nico)

import xgboost as xgb


def preparation(yTrain, wTrain, test_size= 550000):

    wRebalance = wTrain * float(test_size) / wTrain.shape[0]

    sum_wpos = sum(wRebalance[i] for i in range(len(yTrain)) if yTrain[i] == 1.0)
    sum_wneg = sum(wRebalance[i] for i in range(len(yTrain)) if yTrain[i] == 0.0)

    return wRebalance, sum_wpos, sum_wneg


def classifier(xTrain, yTrain, wTrain, test_size, **kwargs):

    wRebalance, sum_wpos, sum_wneg  = preparation(yTrain, wTrain, test_size)

    xgmat = xgb.DMatrix(xTrain, label= yTrain, missing = -999.0,
                        weight= wRebalance)

    # scale weight of positive examples
    kwargs['scale_pos_weight'] = sum_wneg/sum_wpos

    print kwargs.keys()

    # You can directly throw param in, though we want to watch multiple metrics
    # here
    plst = list(kwargs.items())+[('eval_metric', 'ams@0.15 ')]

    watchlist = [ (xgmat,'train') ]
    # boost 120 tres
    num_round = 200

    bst = xgb.train( plst, xgmat, num_round, watchlist );

    # save out model
    #bst.save_model('higgs.model')

    return bst


def prediction(predictor, testset):

    xgmat = xgb.DMatrix(testset, missing = -999.0 )

    # Label prediction:
    # NA
    # Probability of being???
    proba_predicted = predictor.predict(xgmat)

    return proba_predicted


def get_yPredicted_s(xsTrain_s, yTrain_s, wTrain_s, xsValid_s, test_size,
                     **kwargs):
    """
    Perform the training and the prediction on the 8 sub-sets
    """
    # If we work with the splitted dataset:
    if type(xsTrain_s) == list:
        predictor_s = []
        yPredicted_s = []
        yProba_s = []

        for n in range(len(xsTrain_s)):
            # Training:
            xgb = classifier(xsTrain_s[n], yTrain_s[n], wTrain_s[n], test_size,
                             **kwargs)

            # Prediction:
            proba_predicted = prediction(xgb, xsValid_s[n])

            predictor_s.append(xgb)
            yProba_s.append(proba_predicted)
    else:
        # Training:
        predictor_s = classifier(xsTrain_s, yTrain_s, wTrain_s, test_size)

        #Prediction:
        yProba_s = prediction(predictor_s, xsValid_s)

    return predictor_s, yProba_s


def get_test_prediction(predictor_s, xsTest_s):
    """
    Predict the output of this classifier on the test set
    """

    # If we work with the splitted dataset:
    if type(xsTest_s) == list:
        test_proba_s = []

        for n in range(len(xsTest_s)):
            proba_predicted = prediction(predictor_s[n], xsTest_s[n])

            test_proba_s.append(proba_predicted)

    else:
        test_proba_s = prediction(predictor_s, xsTest_s)

    return test_proba_s



###############
### IMPORT ####
###############
"""
# Importation parameters:
split= False
normalize = True
noise_var = 0.
ratio_train = 0.9

# Import the training data:
print("Extracting the data sets...")
start = time.clock()
train_s, valid_s, test_s = tokenizer.extract_data(split= split,
                                                      normalize= normalize,
                                                      noise_variance= noise_var,
                                                      ratio_train= ratio_train)
stop = time.clock()
print ("Extraction time: %i s") %(stop-start)

# Create the predictor
predictor_s, yProba_s = get_yPredicted_s(train_s[1], train_s[2], train_s[3],
                                         valid_s[1])

print np.amax(yProba_s)
print np.amin(yProba_s)



################
# Evaluation of the AMS on the validation set:
if type(yProba_s) == list:
    yprob_valid = np.concatenate(yProba_s)
    wValid = np.concatenate(valid_s[3])
    idValid = np.concatenate(valid_s[0])
    yValid = np.concatenate(valid_s[2])
else:
    yprob_valid = yProba_s
    wValid = valid_s[3]
    idValid = valid_s[0]
    yValid = valid_s[2]

# Labeling:
threshold_ratio = 0.15
ypred_valid = np.ones(yprob_valid.shape[0])

info = np.array([[float(idValid[i]), float(yprob_valid[i]),
                  float(ypred_valid[i]), float(yValid[i]),
                  float(wValid[i]) ]
                        for i in range(idValid.shape[0])])

info = info[info[:,1].argsort()].T

for i in range(int(info[1].shape[0]* threshold_ratio)):
    info[2][i] = 1

final_s, final_b = submission.get_s_b(info[2], info[3], info[4])

print final_s
print final_b

# Balance the s and b
final_s *= 250000/25000
final_b *= 250000/25000
# AMS final:
AMS = hbc.AMS(final_s , final_b)
print ("Expected AMS score for xgb %f") %AMS





##############################################################################
# Prediction on the test set
test_proba_s = get_test_prediction(predictor_s, test_s[1])

# Submission
if type(test_s[1]) == list:
    idx = np.concatenate(test_s[0])
    ypred = np.concatenate(test_proba_s)
else:
    idx = test_s[0]
    ypred = test_proba_s

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ]

rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1

modelfile = 'higgs.model'
outfile = 'tentative.csv'
# make top 15% as positive
threshold_ratio = 0.15

# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()

print ('finished writing into prediction file')
"""
