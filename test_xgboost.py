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

#################
# PRE-TREATMENT #
#################
print("---------------------- Feature importance: -----------------------")

importance_lim = 0.03

featureImportance = preTreatment.featureUsage(train_s)

train_RM_s, train_RM_s_2, valid_RM_s_2, test_RM_2 = preTreatment.\
            removeUnusedFeature(train_s, train_s_2, valid_s_2, test_s,
                                featureImportance,
                                importance_lim = importance_lim)


kwargs_xgb = {'bst_parameters': \
                {'booster_type': 0,
                     'objective': 'multi:softprob', 'num_class': 5,
                     'bst:eta': 0.3,
                     'bst:subsample': 0.5,
                     'bst:max_depth': 10, 'eval_metric': 'auc', 'silent': 1,
                     'nthread': 8 }, \
                'n_rounds': 200}

############
# ANALYSES #
############
print "Getting the classifiers..."
predictor_s = xgBoost.train_classifier(train_RM_s[1], train_RM_s[2],
                                       train_RM_s[3],
                                       550000, kwargs_xgb)

print(" ")
print "Making predictions on the train2 test..."
yTrain2Proba_s = xgBoost.predict_proba(predictor_s, train_RM_s_2[1])

yTrain2Proba = preTreatment.concatenate_vectors(yTrain2Proba_s)

yTrain2 = preTreatment.concatenate_vectors(train_RM_s_2[2])
weightsTrain2 = preTreatment.concatenate_vectors(train_RM_s_2[3])

best_ams_train2, best_ratio = tresholding.best_ratio(yTrain2Proba, yTrain2, weightsTrain2)
print "Train2 - best ratio : %f - best ams : %f" %(best_ratio, best_ams_train2)
print(" ")


print "Making predictions on the validation set..."
yValid2Proba_s = xgBoost.predict_proba(predictor_s, valid_RM_s_2[1])

yValid2Proba = preTreatment.concatenate_vectors(yValid2Proba_s)
yValidPredicted = tresholding.get_yPredicted_ratio(yValid2Proba, best_ratio)
yPredictedValid = preTreatment.multiclass2binary(yValidPredicted)

yValid2 = preTreatment.concatenate_vectors(valid_RM_s_2[2])
weightsValidation = preTreatment.concatenate_vectors(valid_RM_s_2[3])

s, b = submission.get_s_b(yPredictedValid, yValid2, weightsValidation)
s *= 250000/yPredictedValid.shape[0]
b *= 250000/yPredictedValid.shape[0]
ams = hbc.AMS(s,b)

print "Valid_RM_2 - ratio : %f - best ams : %f" %(best_ratio, ams)
print(" ")


print "Making predictions on the test set..."
yPredictedTest_s = xgBoost.predict_proba(predictor_s, test_RM_s[1])

yPredictedTest = preTreatment.concatenate_vectors(yPredictedTest_s)
yTestPredicted = tresholding.get_yPredicted_ratio(yPredictedTest, best_ratio)
yTestPredicted = preTreatment.multiclass2binary(yTestPredicted)


IDTest = preTreatment.concatenate_vectors(test_RM_s[0])
yTestProbaRanked = np.arange(550000) + 1

IDTest = IDTest.astype(np.int64)

sub = submission.print_submission(IDTest, yTestProbaRanked, yTestPredicted, "submssion_xgb_5c_PandT")


print "finish!!!"



