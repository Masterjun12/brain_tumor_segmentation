import numpy as np

def sensitivity(cm, n):  # recall, True Positive Rate
    sen = []
    for i in range(n):
        tp = cm[i][i]
        sen1 = round((tp / np.sum(cm[i, :])), 4)
        sen.append(sen1)
    return sen

class_name = ["0", "1", "2", "3"]

def specificity(cm, n): # True_negative_rate
    TNR = []
    all = np.sum(cm)
    for i in range(n):
        TP = cm[i][i]
        FP = np.sum(cm[:,i]) - TP
        FN = np.sum(cm[i,:]) - TP
        TN = all - FP - FN - TP
        tnr = round((TN/(FP+TN)),4)
        TNR.append(tnr)
    return TNR

def F1_score(cm, n):
    f1_score = []

    for i in range(n):
        TP = cm[i][i]
        FP = np.sum(cm[:,i]) - TP
        FN = np.sum(cm[i,:]) - TP
        f1score = round(((2*TP) / (2*TP+FP+FN)),4)
        f1_score.append(f1score)
    return f1_score

def precision(cm, n): # Ppv
    positive_predictive_value = []
    
    for i in range(n):
        TP = cm[i][i]
        TP_FP = np.sum(cm[:,i])
        ppv = round((TP/TP_FP), 4)
        positive_predictive_value.append(ppv)
    return positive_predictive_value
