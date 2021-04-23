# PA1 Code // Derek Cheng CS 165B

import numpy as np

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    #Initialize data and assign data values in first row to their respectful values
    sizeA = training_input[0][1]
    sizeB = training_input[0][2]
    sizeC = training_input[0][3]
    del training_input[0]
    

    
    #Centroid calculations
    start = 0
    end = sizeA
    centroidA = np.array(compute_centroid(training_input, start, end))
    start = end
    end = start + sizeB
    centroidB = np.array(compute_centroid(training_input, start, end))
    start = end
    end = start + sizeC
    centroidC = np.array(compute_centroid(training_input, start, end))

    
    
    #Discriminant calculations
    discriminant_array = []
    w_array = []
    t_array = []
    
    compute_discriminant_train(centroidA, centroidB, discriminant_array, w_array, t_array)
    compute_discriminant_train(centroidA, centroidC, discriminant_array, w_array, t_array)
    compute_discriminant_train(centroidB, centroidC, discriminant_array, w_array, t_array)


    #Testing parameters
    sizeTestA = testing_input[0][1]
    sizeTestB = testing_input[0][2]
    sizeTestC = testing_input[0][3]
    del testing_input[0]

    testA = testing_input[: sizeTestA]
    testB = testing_input[sizeTestA : (sizeTestA + sizeTestB)]
    testC = testing_input[ (sizeTestA + sizeTestB) : ]

    TP_A, FP_A , FN_A, TN_A = 0 , 0, 0, 0
    TP_B, FP_B, FN_B, TN_B = 0, 0, 0, 0
    TP_C, FP_C, FN_C, TN_C = 0, 0, 0, 0

    #Classification
    for data_value in testA:
        if ((np.array(data_value).dot(w_array[0]) >= t_array[0])):
            if ((np.array(data_value).dot(w_array[1]) >= t_array[1])):
                TP_A += 1
                TN_B += 1
                TN_C += 1
            else:
                FN_A += 1
                TN_B += 1
                FP_C += 1
        else:
            if ((np.array(data_value).dot(w_array[2]) > t_array[2])):
                FN_A += 1
                FP_B += 1
                TN_C += 1
            else:
                FN_A += 1
                TN_B += 1
                FP_C += 1

    for data_value in testB:
        if ((np.array(data_value).dot(w_array[0]) >= t_array[0])):
            if ((np.array(data_value).dot(w_array[1]) >= t_array[1])):
                FP_A += 1
                FN_B += 1
                TN_C += 1
            else:
                TN_A += 1
                FN_B += 1
                FP_C += 1               
        else:
            if ((np.array(data_value).dot(w_array[2]) > t_array[2])):
                TN_A += 1
                TP_B += 1
                TN_C += 1
            else:
                TN_A += 1
                FN_B += 1
                FP_C += 1 
                
    for data_value in testC:
        if ((np.array(data_value).dot(w_array[0]) >= t_array[0])):
            if ((np.array(data_value).dot(w_array[1]) >= t_array[1])):
                FP_A += 1
                TN_B += 1
                FN_C += 1
            else:
                TN_A += 1
                TN_B += 1
                TP_C += 1          
        else:
            if ((np.array(data_value).dot(w_array[2]) > t_array[2])):
                TN_A += 1
                FP_B += 1
                FN_C += 1
            else:
                TN_A += 1
                TN_B += 1
                TP_C += 1 

    #Return averages of TPR, FPR, Error Rate, Accuracy, Precision
    TPR = (TP_A + TP_B + TP_C) / (TP_A + TP_B + TP_C + FN_A + FN_B + FN_C)
    FPR = (FP_A + FP_B + FP_C) / (FP_A + FP_B + FP_C + TN_A + TN_B + TN_C)
    error_rate = (FP_A + FP_B + FP_C + FN_A +FN_B + FN_C) / (TP_A + TP_B + TP_C + FN_A + FN_B + FN_C + FP_A + FP_B + FP_C + TN_A + TN_B + TN_C)
    accuracy = (TP_A + TP_B + TP_C + TN_A + TN_B + TN_C) / (TP_A + TP_B + TP_C + FN_A + FN_B + FN_C + FP_A + FP_B + FP_C + TN_A + TN_B + TN_C)
    precision = (TP_A + TP_B + TP_C) / (TP_A + TP_B + TP_C + FP_A + FP_B + FP_C)

    solution = {
        "tpr" : TPR,
        "fpr" : FPR,
        "error_rate" : error_rate,
        "accuracy" : accuracy,
        "precision" : precision
    }
    
    return solution


#Compute the centroids for each of the classes (A,B,C)
"""
Input parameters:
    training_input = training input
    start = start index
    end = end index
"""
def compute_centroid(training_input, start, end):
    arr = training_input[start:end]
    return np.mean(arr, axis=0)

#Function for discriminant calculations
"""
Input parameters:
    centroid1, centroid2 = centroids
    arr = array in which the determinant matrices will be stored
"""
def compute_discriminant_train(centroid1,centroid2, arr, w, t):
    w.append(centroid1 - centroid2)
    t.append(np.dot((centroid1 - centroid2),((centroid1 + centroid2)/2)))
    arr.append(np.dot((centroid1 - centroid2),((centroid1 + centroid2)/2)))









"""
class DataValues:
    def __init__(self, features, sizeA, sizeB, sizeC):
        self._features = 0
        self._sizeA = 0
        self._sizeB = 0
        self._sizeC = 0
    def set_features(self, value):
        self._features = value
    def set_sizeA(self, value):
        self._sizeA = value
    def set_sizeB(self, value):
        self._sizeB = value
    def set_sizeC(self, value):
        self._sizeC = value

    def get_features(self):
        return self._features
    def get_sizeA(self):
        return self._sizeA
    def get_sizeB(self):
        return self._sizeB
    def get_sizeC(self):
        return self._sizeC
"""