import numpy as np
import pandas as pd


def first_valid_idx(x):
    # Return the index of the first valid value along an axis in a Pandas DataFrame or Series
    if x.first_valid_index() is None:
        return None
    else:
        return x[x.first_valid_index()]

def calc_metrics(predictions, labels):
    # Calculate True positives, false positives, etc.

    TP_ = np.logical_and(predictions, labels)
    FP_ = np.logical_and(predictions, np.logical_not(labels))
    TN_ = np.logical_and(np.logical_not(predictions), np.logical_not(labels))
    FN_ = np.logical_and(np.logical_not(predictions), labels)

    TP=sum(TP_)
    FP=sum(FP_)
    TN=sum(TN_)
    FN=sum(FN_)
    
    return TP,FP,TN,FN

def calc_error_rates(TP, FP, TN, FN):
    # Calculate precision, recall, accuracy, TPR, TNR and BER
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    BER = 1.0 - (TPR+TNR)/2
    return precision, recall, accuracy, TPR, TNR, BER