"""
Systeme de vote entre classifier
...
"""
import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def merge_classifiers(prediction_list, method='weighted_avg'):
    """
    Given a list of predictions [(ID_s, proba_s, label_s, performance_s)] merge them into a single prediction
    """
    assert method in set(['average', 'weighted_avg' , 'lg_regression', None])

    # If we have only one classifier:
    if len(prediction_list) == 1:
        return prediction_list[0]

    else:
        # If we work with the splitted dataset:
        if type(prediction_list[0][0]) == list:
            final_prediction_s = []

            for n in range(len(prediction_list[0][0])):
                # Merge the prediction: Loop over the events of the subset
                f_prediction = []
                # EventID list
                f_prediction.append([])
                # Proba list
                f_prediction.append([])
                # Class list
                f_prediction.append([])

                # Sum of the prediction qualities for this subset:
                sum_qual = 0.
                for j in range(len(prediction_list)):
                    sum_qual += prediction_list[j][3][n]

                # Create a dictionary of proba, label and prediction quality per
                # event in the subset:
                # Rk: we add the quality of the classifier each example to
                #     simplify the ponderate average computation
                predic_dic = {}
                for elmt in prediction_list:
                    for i in range(elmt[0][n].shape[0]):
                        if elmt[0][n][i] in predic_dic:
                            predic_dic[elmt[0][n][i]][0].append(elmt[1][n][i])
                            predic_dic[elmt[0][n][i]][1].append(elmt[2][n][i])
                            predic_dic[elmt[0][n][i]][2].append(elmt[3][n] /\
                                                                    sum_qual)
                        else:
                            predic_dic[elmt[0][n][i]]=[[elmt[1][n][i]], \
                                                       [elmt[2][n][i]], \
                                                       [elmt[3][n]/sum_qual]]

                for key in predic_dic.keys():
                    # Label:
                    if method == 'average':
                        label = np.mean(np.asarray(predic_dic[key][1]))
                    elif method == 'weighted_avg':
                        label = np.mean(np.multiply(
                                            np.asarray(predic_dic[key][1]),
                                            np.asarray(predic_dic[key][2])))
                    else:
                        print("Not implemented averaging method...")
                        exit()

                    if label > 0.5:
                        label = 1
                    else:
                        label = 0

                    # Proba:
                    proba = 0
                    for i in range(len(predic_dic[key][0])):
                        if predic_dic[key][1][i] == 1:
                            proba += predic_dic[key][0][i] * predic_dic[key][2][i]
                        else:
                            proba -= predic_dic[key][0][i] * predic_dic[key][2][i]

                    f_prediction[0].append(key)
                    f_prediction[1].append(proba)
                    f_prediction[2].append(label)

                f_prediction[0] = np.asarray(f_prediction[0])
                f_prediction[1] = np.asarray(f_prediction[1])
                f_prediction[2] = np.asarray(f_prediction[2])

                final_prediction_s.append(f_prediction)

        else:
            # Check if all the prediction have the same shape:
            n_elmt = 0
            for elmt in prediction_list:
                if elmt[0].all() == prediction_list[0][0].all():
                    n_elmt = len(elmt[0])
                else:
                    if len(elmt) != n_elmt:
                        print("Error: Predictions don't have the same shape.")
                        exit()

            # Merge the prediction: Loop over the events
            final_prediction_s = []
            # EventID list
            final_prediction_s.append([])
            # Proba list
            final_prediction_s.append([])
            # Class list
            final_prediction_s.append([])

            # Create a dictionary of proba and label per event:
            predic_dic = {}
            for elmt in prediction_list:
                for i in range(elmt[0].shape[0]):
                    if elmt[0][i] in predic_dic:
                        predic_dic[elmt[0][i]][0].append(elmt[1][i])
                        predic_dic[elmt[0][i]][1].append(elmt[2][i])
                    else:
                        predic_dic[elmt[0][i]]=[[elmt[1][i]],[elmt[2][i]]]

            for key in predic_dic.keys():
                # Label:
                label = np.mean(np.asarray(predic_dic[key][1]))
                if label > 0.5:
                    label = 1
                else:
                    label = 0

                # Proba:
                proba = np.sum(np.asarray(predic_dic[key][0]))

                final_prediction_s[0].append(key)
                final_prediction_s[1].append(proba)
                final_prediction_s[2].append(label)

    return final_prediction_s


def classif_clasifiers_learn(list_pred_train_s, yTrain_s):

    # If we work with the splitted dataset:
    if type(ylabel_s) == list:

        classif_classifiers_s = []

        for i in range(ylabel_s):
            clf = LogisticRegression(C=1e5)

            clf.fit(zip(*list_pred_train_s)[i], yTrain_s[i])

            classif_classifiers_s.append(clf)

    else:
        classif_classifiers_s = LogisticRegression(C=1e5)
        classif_classifiers_s.fit(list_pred_train_s, yTrain_s)

    return classif_classifiers_s


def classif_classifiers_predict(classif_classifiers_s, prediction_list):


    # If we work with the splitted dataset:
    if type(classif_classifiers_s) == list:

        ID_s = zip(*prediction_list)[0]
        pred_proba_s = zip(*prediction_list)[1]
        pred_label_s = zip(*prediction_list)[2]

        for i in range(len(classif_classifiers_s)):
            # Predict the label of a subset:
            final_label_s = classif_classifiers_s[i].predict(pred_label_s[i])
            # Predict the proba of being a signal of a subset:
            final_proba_s = classif_classifiers_s[i].predict_proba(
                                                            pred_lable_s[i])[1]

            final_prediction_s.append(ID[i], final_proba_s, final_label_s)

    else:
        ID_s = prediction_list[0]
        pred_proba_s = prediction_list[1]
        pred_label_s = prediction_list[2]


        final_label_s = classif_classifiers_s.predict(pred_label_s)
        final_proba_s = classif_classifiers_s[i].predict_proba(pred_lable_s)[1]

        final_prediction_s = [ID, final_proba_s, final_label_s]

    return final_prediction_s

def classif_classifiers_error(final_prediction, y_true_s):

    y_predicted_s = final_prediction[2]

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







def rank_signals(prediction):
    prediction[1] = ss.rankdata(prediction[1],method = 'ordinal')

    return prediction
