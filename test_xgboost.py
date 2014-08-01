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

train_s, test_s = tokenizer.extract_data(split = True, normalize = True, noise_variance = 0., n_classes = "multiclass", train_size = 250000, train_size2 = 0, valid_size = 0)

train_s_2, valid_s_2, test_s_2 = tokenizer.extract_data(split = True, normalize = True, noise_variance = 0., n_classes = "multiclass", train_size = 200000, train_size2 = 0, valid_size = 50000)

kwargs_xgb = {'bst_parameters': \
                    {'objective': 'multi:softmax', 'num_class': 5, 'bst:eta': 0.1,
                     'bst:max_depth': 10, 'eval_metric': 'auc', 'silent': 0,
                     'nthread': 8 }, \
              'n_rounds': 200}

print "Getting the classifiers..."


predictor_s = xgBoost.train_classifier(train_s[1], train_s[2], train_s[3],
                                        550000, kwargs_xgb)



"""
print "Making predictions on the train2 test..."
for i in range(100):
    yPredictedTrain2_s_s = []
    yPredictedTrain2_s = xgBoost.predict_proba(predictor_s, train2_s[1])
    yPredictedTrain2_s_s.append(yPredictedTrain2_s)

# Vector to count the label
yTrain2Label_s = []
for i in range(8):
    yTrain2Label = np.zeros((yPredictedTrain2_s_s[0][i].shape[0], 5))
    yTrain2Label_s.append(yTrain2Label)

# Monte Carlo to estimate the proba
# Compting the occurence of each label
print "Estimating the probabilities ..."
for n in range(100):
    for i in range(8):
        for j in range(yPredictedTrain2_s_s[0][i].shape[0]):
            label = int(yPredictedTrain2_s_s[0][i][j])
            yTrain2Label_s[i][j, label] +=1

yTrain2Proba_s =[]
for i in range(8):
    yTrain2Proba = np.zeros(yPredictedTrain2_s_s[0][i].shape[0])
    for j in range(yTrain2Proba.shape[0]):
        #yTrain2Proba[j] = 1 - (yTrain2Label_s[i][j,0] - sum(yTrain2Label_s[i][j,k] for k in [1,2,3,4]))/sum(yTrain2Label_s[i][j,k] for k in range(5))
        yTrain2Proba[j] = 1 - (yTrain2Label_s[i][j,0]/sum(yTrain2Label_s[i][j,k] for k in range(5)))
    yTrain2Proba_s.append(yTrain2Proba)

yTrain2Proba = preTreatment.concatenate_vectors(yTrain2Proba_s)
yTrain2 = preTreatment.concatenate_vectors(train2_s[2])
weightsTrain2 = preTreatment.concatenate_vectors(train2_s[3])

best_ams_train2, best_ratio = tresholding.best_ratio(yTrain2Proba, yTrain2, weightsTrain2)
print "Train2 - best ratio : %f - best ams : %f" %(best_ratio, best_ams_train2)

print "Making predictions on the validation test..."

for i in range(100):
    yPredictedValid_s_s = []
    yPredictedValid_s = xgBoost.predict_proba(predictor_s, valid_s[1])
    yPredictedValid_s_s.append(yPredictedValid_s)
# vector to count the label
yValidLabel_s = []
for i in range(8):
    yValidLabel = np.zeros((yPredictedValid_s_s[0][i].shape[0], 5))
    yValidLabel_s.append(yValidLabel)
# Monte Carlo to estimate the proba
# compting the occurence of each label
print "Estimating the probabilities ..."
for n in range(100):
    for i in range(8):
        for j in range(yPredictedValid_s_s[0][i].shape[0]):
            label = int(yPredictedValid_s_s[0][i][j])
            yValidLabel_s[i][j, label] +=1

yValidProba_s =[]
for i in range(8):
    yValidProba = np.zeros(yPredictedValid_s_s[0][i].shape[0])
    for j in range(yValidProba.shape[0]):
        #yValidProba[j] = 1 - (yValidLabel_s[i][j,0] - sum(yValidLabel_s[i][j,k] for k in [1,2,3,4]))/sum(yValidLabel_s[i][j,k] for k in range(5))
        yValidProba[j] = 1 - (yValidLabel_s[i][j,0]/sum(yValidLabel_s[i][j,k] for k in range(5)))
    yValidProba_s.append(yValidProba)


yValidProba = preTreatment.concatenate_vectors(yValidProba_s)
yValid = preTreatment.concatenate_vectors(valid_s[2])
weightsValid = preTreatment.concatenate_vectors(valid_s[3])

yValidPredicted = tresholding.get_yPredicted_ratio(yValidProba, best_ratio)
"""
print "Making predictions on the test set..."

for i in range(100):
    yPredictedTest_s_s = []
    yPredictedTest_s = xgBoost.get_test_prediction(predictor_s, test_s[1])
    yPredictedTest_s_s.append(yPredictedTest_s)
# vector to count the label
yTestLabel_s = []
for i in range(8):
    yTestLabel = np.zeros((yPredictedTest_s_s[0][i].shape[0], 5))
    yTestLabel_s.append(yTestLabel)
# Monte Carlo to estimate the proba
# compting the occurence of each label
print "Estimating the probabilities ..."
for n in range(100):
    for i in range(8):
        for j in range(yPredictedTest_s_s[0][i].shape[0]):
            label = int(yPredictedTest_s_s[0][i][j])
            yTestLabel_s[i][j, label] +=1


yTestProba_s =[]
for i in range(8):
    yTestProba = np.zeros(yPredictedTest_s_s[0][i].shape[0])
    for j in range(yTestProba.shape[0]):
        #yTestProba[j] = 1 - (yTestLabel_s[i][j,0] - sum(yTestLabel_s[i][j,k] for k in [1,2,3,4]))/sum(yTestLabel_s[i][j,k] for k in range(5))
        yTestProba[j] = 1 - (yTestLabel_s[i][j,0]/sum(yTestLabel_s[i][j,k] for k in range(5)))
    yTestProba_s.append(yTestProba)


IDTest = preTreatment.concatenate_vectors(test_s[0])
yTestProba = preTreatment.concatenate_vectors(yTestProba_s)
yTestPredicted = tresholding.get_yPredicted_ratio(yTestProba, 0.98)
yTestProbaRanked = submission.rank_signals(yTestProba)


IDTest = IDTest.astype(np.int64)

sub = submission.print_submission(IDTest, yTestProbaRanked, yTestPredicted, "submssion_xgboost_5class_bis")

"""
s, b = submission.get_s_b(yValidPredicted, yValid, weightsValid)

s *=10
b*=10

AMS = hbc.AMS(s, b)


print "AMS valid = %f :" %AMS
"""

print "AMS valid = %f" %AMS



