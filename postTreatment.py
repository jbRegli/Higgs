"""
Systeme de vote entre classifier
...
"""
import numpy as np
import scipy.stats as ss


def merge_classifier(prediction_list):
    """
    Given a list of predictions [(ID, proba, label)] merge them into a single
    prediction
    """

    if len(prediction_list) == 1:
        return prediction_list[0]

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
        final_prediction = []
        # EventID list
        final_prediction.append([])
        # Proba list
        final_prediction.append([])
        # Class list
        final_prediction.append([])

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
            proba = 0
            for i in range(len(predic_dic[key][0])):
                if predic_dic[key][1][i] == 1:
                    proba += predic_dic[key][0][i]
                else:
                    proba -= predic_dic[key][0][i]

            final_prediction[0].append(key)
            final_prediction[1].append(proba)
            final_prediction[2].append(label)

        return final_prediction

def rank_signals(prediction):
    prediction[1] = ss.rankdata(prediction[1],method = 'ordinal')

    return prediction
