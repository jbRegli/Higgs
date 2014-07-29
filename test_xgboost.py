import sys
import numpy as np
import time
from sklearn.metrics import accuracy_score
import tokenizer
import preTreatment

sys.path.append('PostTreatment/')
import preTreatment
import tresholding
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc

sys.path.append('Analyses/')
import xgBoost

train_s, train2_s, valid_s, test_s = tokenizer.extract_data(split = True, normalize = True, noise_variance = 0., n_classes = "multiclass", train_size = 200000, train_size2 = 25000, valid_size = 25000)

kwargs_xgb = {'objective': 'multi:softmax', 'num_class': 5, 'bst:eta': 0.1,
              'bst:max_depth': 10, 'eval_metric': 'auc', 'silent': 1, 'nthread': 16 }


predictor_s, yTrain2ProbaBinary_s = xgBoost.get_yPredicted_s(train_s[1], train_s[2], train_s[3], train2_s[1], 550000, kwargs_xgb)


#predict the second test set and the validation set
yTrain2ProbaBinary_s = xgBoost.get_test_prediction(predictor_s, train2_s[1])
yValidProbaBinary_s = xgBoost.get_test_prediction(predictor_s, valid_s[1])
"""
#let's create the binary vector proba
yTrain2ProbaBinary_s = []
yValidProbaBinary_s = []
for i in range(8):
    yTrain2ProbaBinary_s.append(np.zeros(yTrain2Proba_s[i].shape[0]))
    yValidProbaBinary_s.append(np.zeros(yValidProba_s[i].shape[0]))
for i in range(8):
    for j in range(yTrain2ProbaBinary_s[i].shape[0]):
        yTrain2ProbaBinary_s[i][j] = 1 - yTrain2Proba_s[i][j][0]
    for j in range(yValidProbaBinary_s[i].shape[0]):
        yValidProbaBinary_s[i][j] = 1 - yValidProba_s[i][j][0]
"""
#let's concatenate
yTrain2ProbaBinary_conca = preTreatment.concatenate_vectors(yTrain2ProbaBinary_s)
yValidProbaBinary_conca = preTreatment.concatenate_vectors(yValidProbaBinary_s)
yTrain2Label_conca = preTreatment.concatenate_vectors(train2_s[2])
yValidLabel_conca = preTreatment.concatenate_vectors(valid_s[2])
yTrain2Weights_conca = preTreatment.concatenate_vectors(train2_s[3])
yValidWeights_conca = preTreatment.concatenate_vectors(valid_s[3])

#compute the best treshold
best_treshold_global = tresholding.best_treshold(yTrain2ProbaBinary_conca, yTrain2Label_conca, yTrain2Weights_conca, pas = 0.01)

yValidPredicted_conca = tresholding.get_yPredicted_treshold(yValidProbaBinary_conca, best_treshold_global)
yTrain2Predicted_conca = tresholding.get_yPredicted_treshold(yTrain2ProbaBinary_conca, best_treshold_global)

svalid, bvalid = submission.get_s_b(yValidPredicted_conca, yValidLabel_conca, yValidWeights_conca)
strain2, btrain2 = submission.get_s_b(yTrain2Predicted_conca, yTrain2Label_conca, yTrain2Weights_conca)

svalid *= 250000/yValidPredicted_conca.shape[0]
bvalid *= 250000/yValidPredicted_conca.shape[0]
strain2 *= 250000/yTrain2Predicted_conca.shape[0]
btrain2 *= 250000/yTrain2Predicted_conca.shape[0]


AMS_valid = hbc.AMS(svalid,bvalid)
AMS_train2 = hbc.AMS(strain2, btrain2)

print "AMS valid = %f --- AMS train2 : %f" %(AMS_valid, AMS_train2)


