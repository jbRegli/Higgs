"""
Given a set of classifiers (or several time the same with different parameters) produce the highest global AMS by picking the most efficient on each subset
"""

import sys
sys.path.append('../')
import submission
import HiggsBosonCompetition_AMSMetric_rev1 as hbc


def select_best_classifiers(dTuning, valid_s, criteria= 'ams'):

    # If we work with a splitted dataset:
    if type(dTuning[dTuning.keys()[0]]['predictor_s']) == list:

        first_key = dTuning.keys()[0]
        best_parameters = [None] * len(dTuning[first_key]['sum_s'])

        # Initialize best_parameters:
        for i in range(len(best_parameters)):
            if criteria == 'ams':
                best_parameters[i] = {'experience': 0,
                                      'score': dTuning[first_key]['AMS_s'][i]}
            elif criteria == 'sum_s':
                best_parameters[i] = {'experience': 0,
                                      'score': dTuning[first_key]['sum_s'][i]}
            elif criteria == 'sum_b':
                best_parameters[i] = {'experience': 0,
                                      'score': dTuning[first_key]['sum_b'][i]}
            else:
                print "tuningModel.select_best_parameters: not implemented criteria"
                exit()

        # Looking for the best parameters for each subset:
        for exp in dTuning:
            for i in range(len(best_parameters)):
                if criteria == 'ams':
                    if dTuning[exp]['AMS_s'][i] > best_parameters[i]['score']:
                        best_parameters[i]['experience'] = exp
                        best_parameters[i]['score'] = dTuning[exp]['AMS_s'][i]

        # Build the new dictionnary of methods:
        predictor_s = [None] * len(best_parameters)
        yPredicted_s = [None] * len(best_parameters)
        yProba_s = [None] * len(best_parameters)
        sum_s_s = [None] * len(best_parameters)
        sum_b_s = [None] * len(best_parameters)
        AMS_s = [None] * len(best_parameters)
        classif_succ_s = [None] * len(best_parameters)
        method_s = [None] * len(best_parameters)
        parameters_s = [None] * len(best_parameters)

        for i in range(len(best_parameters)):
            # Best experience for this subset:
            exp = best_parameters[i]['experience']

            print type(dTuning[exp]['predictor_s'])

            # Fill the parameters:
            predictor_s[i] = dTuning[exp]['predictor_s'][i]
            yPredicted_s[i] = dTuning[exp]['yPredicted_s'][i]
            yProba_s[i] = dTuning[exp]['yProba_s'][i]
            sum_s_s[i] = dTuning[exp]['sum_s'][i]
            sum_b_s[i] = dTuning[exp]['sum_b'][i]
            AMS_s[i] = dTuning[exp]['AMS_s'][i]
            classif_succ_s[i] = dTuning[exp]['classif_succ'][i]
            method_s[i] = dTuning[exp]['method']
            if type(dTuning[exp]['parameters']) == list:
                parameters_s[i] = dTuning[exp]['parameters'][i]
            else:
                parameters_s[i] = dTuning[exp]['parameters']

        # Get s and b for each group (s_s, b_s) and the final final_s and
        # final_b:
        final_s, final_b, s_s, b_s = submission.get_s_b_8(yPredicted_s,
                                                          valid_s[2],
                                                          valid_s[3])
        # Balance the s and b
        final_s *= 250000/25000
        final_b *= 250000/25000
        # AMS final:
        AMS = hbc.AMS(final_s , final_b)
        print ("Expected AMS score for the combined classifiers : %f") %AMS

        for i in range(len(AMS_s)):
        # AMS by group
            print("Expected AMS score for  : for group %i is : %f" %(i, AMS_s[i]))


        d = {'predictor_s': predictor_s, 'yPredicted_s': yPredicted_s,
             'yProba_s': yProba_s,
             'final_s':final_s, 'final_b':final_b,
             'sum_s':sum_s_s, 'sum_b': sum_b_s,
             'AMS':AMS, 'AMS_s': AMS_s,
             'classif_succ': classif_succ_s,
             'method': method_s,
             'parameters': parameters_s}

    return d



