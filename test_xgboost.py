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

##########
# IMPORT #
##########
print(" ")
print("---------------------------- Import: ---------------------------")

train_s, test_s = tokenizer.extract_data(split = False, normalize = True, noise_variance = 0., n_classes = "multiclass", train_size = 250000, train_size2 = 0, valid_size = 0)

print(" ")

train_s_2, valid_s_2, test_s_2 = tokenizer.extract_data(split = False, normalize = True, noise_variance = 0., n_classes = "multiclass", train_size = 200000, train_size2 = 0, valid_size = 50000)

print(" ")

#################
# PRE-TREATMENT #
#################
print("---------------------- Feature importance: ----------------------")

# Compute the feature usage:
featureImportance = preTreatment.featureUsage(train_s)

# Remove the least used feature from each subset

for importance_lim in np.arrange(0.005, 0.05 , 0.005)
    train_RM_s, train_RM_s_2, valid_RM_s_2, test_RM_s = preTreatment.\
            removeUnusedFeature(train_s, train_s_2, valid_s_2, test_s,
                                featureImportance,
                                importance_lim = importance_lim)

print(" ")

############
# ANALYSES #
############
print("-------------------------- XgBoost: ----------------------------")

# XgBoost parameters:
kwargs_xgb = {'bst_parameters': \
                {'booster_type': 0,
                     'objective': 'multi:softprob', 'num_class': 5,
                     'bst:eta': 0.1, # the bigger the more conservative
                     'bst:subsample': 1, # prevent over fitting if <1
                     'bst:max_depth': 10, 'eval_metric': 'auc', 'silent': 1,
                     'nthread': 8 }, \
                'n_rounds': 100}

print "Getting the classifiers..."
# Training:
predictor_s = xgBoost.train_classifier(train_RM_s[1], train_RM_s[2],
                                       train_RM_s[3],
                                       550000, kwargs_xgb)

print(" ")
print "Making predictions on the train2 test..."
# Prediction of the train set 2:
predProba_Train2_s = xgBoost.predict_proba(predictor_s, train_RM_s_2[1])

# Concatenate results & data:
predProba_Train2 = preTreatment.concatenate_vectors(predProba_Train2_s)
yTrain2 = preTreatment.concatenate_vectors(train_RM_s_2[2])
weightsTrain2 = preTreatment.concatenate_vectors(train_RM_s_2[3])

# Looking for the best threshold:
best_ams_train2, best_ratio = tresholding.best_ratio(predProba_Train2, yTrain2, weightsTrain2)
print "Train2 - best ratio : %f - best ams : %f" %(best_ratio, best_ams_train2)
print(" ")


print "Making predictions on the validation set..."
# Prediction of the validation set 2:
predProba_Valid2_s = xgBoost.predict_proba(predictor_s, valid_RM_s_2[1])

# Thresholding the predictions:
predProba_Valid2 = preTreatment.concatenate_vectors(predProba_Valid2_s)
predLabel5_Valid2 = tresholding.get_yPredicted_ratio(predProba_Valid2, best_ratio)

# Binarize the prediction:
predLabel_Valid2 = preTreatment.multiclass2binary(predLabel5_Valid2)

# Concatenate data:
yValid2 = preTreatment.concatenate_vectors(valid_RM_s_2[2])
weightsValidation = preTreatment.concatenate_vectors(valid_RM_s_2[3])

# Estimation the AMS:
s, b = submission.get_s_b(predLabel_Valid2, yValid2, weightsValidation)
s *= 250000/predLabel_Valid2.shape[0]
b *= 250000/predLabel_Valid2.shape[0]
ams = hbc.AMS(s,b)

print "Valid_RM_2 - ratio : %f - best ams : %f" %(best_ratio, ams)
print(" ")


print "Making predictions on the test set..."
# Prediction of the test set:
predProba_Test_s = xgBoost.predict_proba(predictor_s, test_RM_s[1])

# Thresholding the predictions:
predProba_Test = preTreatment.concatenate_vectors(predProba_Test_s)
predLabel5_Test = tresholding.get_yPredicted_ratio(predProba_Test, best_ratio)

# Binarize the prediction:
predLabel_Test = preTreatment.multiclass2binary(predLabel5_Test)

# Concatenate data:
IDTest = preTreatment.concatenate_vectors(test_RM_s[0])
IDTest = IDTest.astype(np.int64)

# Rank the prediction:
yTestProbaRanked = submission.rank_signals()

# Create the submission file:
submission_name = "submssion_xgb_5c_PandT"

print ("Generating a submsission file named %s" %submission_name)

sub = submission.print_submission(IDTest, yTestProbaRanked, yTestPredicted, "submssion_xgb_5c_PandT")

# Finish...
print "finish!!!"



