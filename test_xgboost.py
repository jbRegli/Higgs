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

split = True
norm = True
remove_999 = False
n_classes = "multiclass" #"binary"

train_s, train_s_2, valid_s_2, test_s = tokenizer.extract_data(
                                                        split = split,
                                                        normalize = norm,
                                                        remove_999 = remove_999,
                                                        noise_variance = 0.,
                                                        n_classes = n_classes,
                                                        train_size = 180000,
                                                        train_size2 = 35000,
                                                        valid_size = 35000)

print(" ")

#################
# PRE-TREATMENT #
#################
print("---------------------- Feature importance: ----------------------")

# Compute the feature usage:
featureImportance = preTreatment.featureUsage(train_s, n_estimators= 10)

# Number of features (sum if splited dataset)
if type(train_s[1]) == list:
    n_total_feature = 0
    for elmt in featureImportance:
        n_total_feature += len(elmt)
else:
    n_total_feature = len(featureImportance)

# Remove the least used feature from each subset
n_removeFeatures_old = 0

best_ams = 0.
best_imp_lim = 0.
best_best_ratio = 0.
best_n_removeFeatures = 0.


print(" Looping over importance_limit")
for importance_lim in np.arange(0.0, 0.1 , 0.001):
    train_RM_s, train_RM_s_2, valid_RM_s_2, test_RM_s, n_removeFeatures = \
            preTreatment.removeUnusedFeature(train_s, train_s_2, valid_s_2,
                                             test_s,
                                             featureImportance,
                                             importance_lim = importance_lim)

    if (n_removeFeatures != n_removeFeatures_old or best_ams == 0) \
            and n_removeFeatures <= n_total_feature -1:
        print(" ")
        print("Testing importance_limit= %f" %importance_lim)
        n_removeFeatures_old = n_removeFeatures
        ############
        # ANALYSES #
        ############
        print("-------------------------- XgBoost: ----------------------------")

        # XgBoost parameters:
        kwargs_xgb = {'bst_parameters': \
                {'booster_type': 0,
                     #'objective': 'binary:logitraw',
                     'objective': 'multi:softprob', 'num_class': 5,
                     'bst:eta': 0.1, # the bigger the more conservative
                     'bst:subsample': 1, # prevent over fitting if <1
                     'bst:max_depth': 15, 'eval_metric': 'auc', 'silent': 1,
                     'nthread': 8 }, \
                'n_rounds': 10}

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
        if type(train_s[1]) == list:

            print ("+++ predProba_Train2_s[0][1] =",  predProba_Train2_s[0][1])
            best_ams_train2, best_ratio = tresholding.\
                                    best_ratio_combinaison_global(
                                                        predProba_Train2_s,
                                                        train_RM_s_2[2],
                                                        train_RM_s_2[3],
                                                        20)
        else:
            best_ams_train2, best_ratio = tresholding.best_ratio(
                                                                predProba_Train2,
                                                                yTrain2,
                                                                weightsTrain2)


        print "Train2 - best ratio : %f - best ams : %f" \
                %(best_ratio, best_ams_train2)
        print(" ")


        print "Making predictions on the validation set..."
        # Prediction of the validation set 2:
        predProba_Valid2_s = xgBoost.predict_proba(predictor_s, valid_RM_s_2[1])

        # Thresholding the predictions:
        predProba_Valid2 = preTreatment.concatenate_vectors(predProba_Valid2_s)
        predLabel5_Valid2 = tresholding.get_yPredicted_ratio(predProba_Valid2,
                                                             best_ratio)

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

        # Saving the model if it's better:
        if ams > best_ams:
            best_ams = ams
            best_imp_lim = importance_lim
            best_best_ratio = best_ratio
            best_n_removeFeatures = n_removeFeatures
            best_predictor_s = predictor_s


print "Best Valid_RM_2 - ratio: %f - best ams : %f - importance_lim: %f - n_removeFeature: %i" \
        %(best_best_ratio, best_ams, best_imp_lim, best_n_removeFeatures)
print(" ")

print "Making predictions on the test set..."
train_RM_s, train_RM_s_2, valid_RM_s_2, test_RM_s, n_removeFeatures = \
        preTreatment.removeUnusedFeature(train_s, train_s_2, valid_s_2, test_s,
                                         featureImportance,
                                         importance_lim = best_imp_lim)

# Prediction of the test set:
predProba_Test_s = xgBoost.predict_proba(best_predictor_s, test_RM_s[1])

# Thresholding the predictions:
predProba_Test = preTreatment.concatenate_vectors(predProba_Test_s)
predLabel5_Test = tresholding.get_yPredicted_ratio(predProba_Test, best_ratio)

# Binarize the prediction:
predLabel_Test = preTreatment.multiclass2binary(predLabel5_Test)

# Concatenate data:
IDTest = preTreatment.concatenate_vectors(test_RM_s[0])
IDTest = IDTest.astype(np.int64)

# Rank the prediction:
predProbaRank_Test = np.zeros(predProba_Test.shape[0])
if len(predProba_Test.shape) == 2:
    if predProba_Test.shape[1] == 5:
        predProbaRank_Test[:] = np.max(predProba_Test[:,1:4],axis = 1)
    else:
        print "Error!!!!!"
else:
    predProbaRank_Test = predProba_Test

yTestProbaRanked = submission.rank_signals(predProbaRank_Test)


# Create the submission file:
submission_name = "submssion_xgb_5c_SPT_r120_ams" + str(best_ams)[0:6]

print ("Generating a submsission file named %s" %submission_name)

sub = submission.print_submission(IDTest, yTestProbaRanked, predLabel_Test,
                                  submission_name)

# Finish...
print "finish!!!"



