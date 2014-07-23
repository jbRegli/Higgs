import tokenizer
import preTreatment
import numpy as np

import sys
sys.path.append('Analyses/')
import analyse
import naiveBayes
import randomForest
import svm
import kNeighbors
import adaBoost
import lda
import qda

print "Extracting the data..."

Train, Validation, Test = tokenizer.get_all_data(normalize = True,
                                     noise_variance = 0.,
                                     ratio_train= 0.9)

ID_train, xsTrain, yTrain, weightsTrain  = Train[0], Train[1], Train[2], \
                                               Train[3]
ID_valid, xsValid, yValid, weightsValid = Validation[0], Validation[1], \
                                              Validation[2], Validation[3]

ID_test, xsTest = Test[0], Test[1]

print "Splitting the data into sub-groups..."

ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s = preTreatment.\
                           split_8_matrix(ID_train, xsTrain, yTrain, weightsTrain)
train_s = (ID_train_s, xsTrain_s, yTrain_s, weightsTrain_s)

ID_valid_s, xsValid_s,yValid_s, weightsValid_s = preTreatment.\
                           split_8_matrix(ID_valid, xsValid, yValid, weightsValid)
valid_s = (ID_valid_s, xsValid_s,yValid_s, weightsValid_s)

ID_test_s, xsTest_s = preTreatment.split_8_matrix(ID_test, xsTest)

#Liste des indices des colonnes a suppprimer
# supprimer toutes les features pouvant ne pas etre definies
#L_delete = [28,27,26,25,24,23,22,6,5,4] 

# supprimer toutes les features derivees + les features ne pouvant pas
# etre definies 
L_delete = [28,27,26,25,24,23,22,12,11,10,9,8,7,6,5,4,3,2,1,0]


print "Deleting the column..."

for i in range(8):
    for index_column in L_delete:
        xsTrain_s[i] = np.delete(xsTrain_s[i], np.s_[index_column],1)
        xsValid_s[i] = np.delete(xsValid_s[i], np.s_[index_column],1)
        xsTest_s[i] = np.delete(xsTest_s[i], np.s_[index_column],1)

print "Training each groups"

dMethods ={}

# NAIVE BAYES:
kwargs_bayes = {}
dMethods['naiveBayes'] =  analyse.analyse(train_s, valid_s, 'naiveBayes', kwargs_bayes)
# SVM
"""
kwargs_svm ={}
dMethods['svm'] = analyse.analyse(train_s, valid_s,'svm', kwargs_svm)
"""
# K NEIGHBORS
kwargs_kn = {'n_neighbors':50}
dMethods['kNeighbors'] = analyse.analyse(train_s, valid_s, 'kNeighbors', kwargs_kn)
# LDA
kwargs_lda = {}
dMethods['lda'] = analyse.analyse(train_s, valid_s, 'lda', kwargs_lda)
# QDA
kwargs_qda= {}
dMethods['qda'] = analyse.analyse(train_s, valid_s, 'qda', kwargs_qda)
# ADABOOST
kwargs_ada= {   'base_estimators': None, 
                    'n_estimators': 50,
                    'learning_rate': 1.,
                    'algorithm': 'SAMME.R',
                    'random_state':None}
dMethods['adaBoost'] = analyse.analyse(train_s, valid_s, 'adaBoost', kwargs_ada)
####### RANDOM FOREST:
kwargs_rdf= {'n_trees': 100}
dMethods['randomForest'] = analyse.analyse(train_s, valid_s, 'randomForest', kwargs_rdf)




