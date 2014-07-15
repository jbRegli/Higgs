import numpy as np
import submission
import imp

def analyse(method_script):

####### NAIVE BAYSE:
    # Prediction on the vaidation set:
    print("------------------- Analyse: %s -----------------------") \
        %(method_script)

    #import sys

    #sys.path.append('Analyses/')

    method = imp.load_source(method_script,
                             str("./Analyses/" + method_script + ".py"))

    predictor_s, yPredicted_s, yProba_s = method.get_yPredicted_s(
                                                                train_s[1],
                                                                train_s[2],
                                                                valid_s[1])
    # Get s and b:
    final_s, final_b = submission.get_s_b_8(yPredicted_s, valid_s[2],
                                                  valid_s[3])

    # AMS:
    #AMS = ams.AMS(nb_final_s * 550000 /25000, nb_final_b* 550000 /25000)
    #print ("The expected score for naive bayse is %f") %AMS

    # Classification error:
    classif_succ = method.get_classification_error(yPredicted_s,
                                                       valid_s[2],
                                                       normalize= True)

    for i, ratio in enumerate(classif_succ):
        print("On the subset %i - correct prediction = %f") %(i, ratio)

    print (" ")
    # Numerical score:
    if type(yPredicted_s) == list:
        for i in range(len(yPredicted_s)):
            sum_s, sum_b = submission.get_numerical_score(yPredicted_s[i],
                                                          valid_s[2][i])
            print "Subset %i: %i elements - sum_s[%i] = %i - sum_b[%i] = %i" \
                    %(i, yPredicted_s[i].shape[0], i, sum_s, i, sum_b)
    else:
             sum_s, sum_b = submission.get_numerical_score(yPredicted_s,
                                                           valid_s[2])
             print "%i elements - sum_s = %i - sum_b = %i" \
                    %(yPredicted_s.shape[0], sum_s, sum_b)

    print(" ")

