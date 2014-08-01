# -*- coding: utf-8 -*-

# this is the example script to use xgboost to train
import inspect
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.append('./../')
import tokenizer
import preTreatment
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

# TODO: add path of xgboost python module
code_path_jb = '/home/regli/Applications/Python/xgboost/python'
sys.path.append(code_path_jb)
code_path_nico = '/home/nicolas/xgboost/python'
sys.path.append(code_path_nico)
import xgboost as xgb

"""
Gradient boosting machine (xbBoost version)

Meta-parameters:
    -objective: str
        What is the objective function. Define the task perform by the machine
        (binary classification, multi-class...)
    -num_class: (optional) int
        Number of classes
        To be used only for a 'multi:softmax' objective
    -'bst:eta': float
        Step size shrinkage used in update to prevents overfitting
    -bst:max_depth: int
        Maximum depth of the classifier
    -eval_metric: str
        evaluation metrics for validation data, a default metric will be assigned
        according to objective (rmse for regression, and error for classification,
        mean average precision for ranking)
"""


def preparation(yTrain, wTrain, test_size= 550000):

    wRebalance = wTrain * float(test_size) / wTrain.shape[0]

    sum_wpos = sum(wRebalance[i] for i in range(len(yTrain)) if yTrain[i] == 1.0)
    sum_wneg = sum(wRebalance[i] for i in range(len(yTrain)) if yTrain[i] == 0.0)

    return wRebalance, sum_wpos, sum_wneg


def train_classifier(xTrain_s, yTrain_s, wTrain_s, test_size, kwargs):
    """
    Train a naive baise classifier on xTrain and yTrain and return the trained
    classifier
    """
    if type(xTrain_s) != list:
        # Resclale the weights:
        wRebalance, sum_wpos, sum_wneg  = preparation(yTrain_s, wTrain_s, test_size)

        xgmat = xgb.DMatrix(xTrain_s, label= yTrain_s, missing = -999.0,
                            weight= wRebalance)

        # Scale weight of positive examples
        kwargs['bst_parameters']['scale_pos_weight'] = sum_wneg/sum_wpos

        # You can directly throw param in, though we want to watch multiple
        # metrics here
        plst = list(kwargs['bst_parameters'].items())+[('eval_metric',
                                                                'ams@0.15 ')]

        watchlist = [ (xgmat,'train') ]

        # Boost n_round trees
        n_round = kwargs['n_rounds']

        classifier_s = xgb.train( plst, xgmat, n_round, watchlist );

    else:
        classifier_s = train_classifier_8(xTrain_s, yTrain_s, wTrain_s,
                                                            test_size, **kwargs)
    return classifier_s


def train_classifier_8(xTrain_s, yTrain_s, wTrain_s, test_size, **kwargs):
    """
    performs the training and returns the predictors
    """
    # If we work with the splitted dataset:

    classifier_s = []

    for n in range(len(xTrain_s)):
        classifier = train_classifier(xTrain_s[n], yTrain_s[n], wTrain_s[n],
                                      test_size, kwargs)
        classifier_s.append(classifier)

    return classifier_s


def predict_proba(classifier_s, dataset_s):
    """
    Given a dataset and a classifier, compute the proba prediction
    This function can be use for validation as well as for the test.
    """
    if type(classifier_s) != list:
        # Probability of being in each label
        xgmat = xgb.DMatrix(dataset_s, missing = -999.0 )
        proba_predicted_s = classifier_s.predict(xgmat)

    else:
        proba_predicted_s = predict_proba_8(classifier_s, dataset_s)

    return proba_predicted_s


def predict_proba_8(classifier_s, dataset_s):
    """
    Predict the output of this classifier on the the dataset divided in 8 groups
    """
    # If we work with the splitted dataset:
    proba_predicted_s = []

    for n in range(len(dataset_s)):
        proba_predicted = predict_proba(classifier_s[n], dataset_s[n])
        proba_predicted_s.append(proba_predicted)

    return proba_predicted_s


def get_classification_error(y_predicted_s, y_true_s, normalize= True):

    if type(y_predicted_s) == list:
        prediction_error_s = []

        for n in range(len(y_predicted_s)):
            prediction_error_s.append(accuracy_score(y_true_s[n],
                                                     y_predicted_s[n],
                                                     normalize=normalize))
    else:
        prediction_error_s = accuracy_score(y_true_s, y_predicted_s,
                                            normalize=normalize)

    return prediction_error_s



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
