import statistics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

from collections import Counter, defaultdict

import warnings
warnings.filterwarnings('ignore')


def euc_distance(x_train, x_test_point):
    """
    Input:
    - x_train: corresponding to the training data
    - x_test_point: corresponding to the test point
    Output:
    -distances: The distances between the test point and each point in the training data.
    """

    distances = []                              # create empty list called distances
    for row in range(len(x_train)):             # Loop over the rows of x_train
        current_train_point = x_train[row]      # Get them point by point
        current_distance = 0                    # initialize the distance by zero

        for col in range(len(current_train_point)):   # Loop over the columns of the row
            current_distance += (current_train_point[col] - x_test_point[col]) **2
        current_distance = np.sqrt(current_distance)

        distances.append(current_distance)            # Append the distances

    # Store distances in a dataframe
    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


def find_k(distance_point, y_train):
    """
    Input:
        -distance_point: the distances between the test point and each point in the training data.
        -K             : the number of neighbors

    Output:
        -df_nearest: the nearest K neighbors between the test point and the training data.

    """

    k_dci_scores = defaultdict(int)

    for test_k in range(3, 20, 2):

        df_nearest = distance_point.sort_values(by=['dist'], axis=0)
        df_nearest = df_nearest[:test_k]

        counter_vote = Counter(y_train[df_nearest.index])
        # print(counter_vote)

        max_counter_vote = max(counter_vote, key=counter_vote.get)
        # print(max_counter_vote)

        max_classification_score = counter_vote[max_counter_vote]

        sum_classification_score = 0
        for i in set(y_train):
            sum_classification_score += counter_vote[i]

        dci = max_classification_score / sum_classification_score
        # print(dci)
        k_dci_scores[test_k] = dci

    # print(k_dci_scores)
    selected_k = max(k_dci_scores, key=k_dci_scores.get)
    # print("K Selected: ", selected_k)
    return selected_k


def prediction(distance_point, y_train, K):
    """
    Input:
        -df_nearest: dataframe contains the nearest K neighbors between the full training dataset and the test point.
        -y_train: the labels of the training dataset.

    Output:
        -y_pred: the prediction based on Majority Voting

    """

    # Sort values using the sort_values function
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    classes = np.unique(y_train)
    class_values = []

    for cl in classes:
        class_values.append([])
    dist_harm_mean_class = defaultdict(int)

    # print(class_values)
    # print(df_nearest)

    for neigh in df_nearest.index:
        class_values[y_train[neigh]].append(neigh)

    for cl in classes:
        # print("Class: ", cl)
        one_class_values = class_values[cl][:K]
        # print(one_class_values)

        class_dist = []
        for val in one_class_values:
            class_dist.append(df_nearest["dist"].loc[val])

        harm_mean = statistics.harmonic_mean(class_dist)
        dist_harm_mean_class[cl] = harm_mean

    selected_class = min(dist_harm_mean_class, key=dist_harm_mean_class.get)
    return selected_class


def class_based_knn(x_train, y_train, x_test):
    """
    Input:
    -x_train: the full training dataset
    -y_train: the labels of the training dataset
    -x_test: the full test dataset

    Output:
    -y_pred: the prediction for the whole test set based on Majority Voting.
    """

    y_pred = []

    # Loop over all the test set and perform the three steps
    for x_test_point in x_test:
        distance_point = euc_distance(x_train, x_test_point)
        k = find_k(distance_point, y_train)
        y_pred_point = prediction(distance_point, y_train, k)
        y_pred.append(y_pred_point)

    return y_pred


if __name__ == '__main__':

    bupa = pd.read_csv('datasets/bupa-liver.csv')
    pima = pd.read_csv('datasets/pima-indians.csv')
    canc = pd.read_csv('datasets/breast-cancer-w.csv')
    heart = pd.read_csv('datasets/heart-disease.csv')
    veh = pd.read_csv('datasets/vehicle.csv')
    hous = pd.read_csv('datasets/boston-housing.csv')

    '''
    Bupa Liver
    '''




    '''
    Pima Indians
    '''

    # feature_columns_pima = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
    #                         'DiabetesPedigreeFunction', 'Age']
    # x_pima = pima[feature_columns_pima]
    # y_pima = pima['Outcome'].values
    #
    # x_train_pima, x_test_pima, y_train_pima, y_test_pima = train_test_split(x_pima, y_pima, test_size=0.2, random_state=0)
    #
    # scaler = Normalizer().fit(x_train_pima)              # the scaler is fitted to the training set
    # normalized_x_train_pima = scaler.transform(x_train_pima)  # the scaler is applied to the training set
    # normalized_x_test_pima = scaler.transform(x_test_pima)    # the scaler is applied to the test set
    #
    # y_pred_pima = class_based_knn(normalized_x_train_pima, y_train_pima, normalized_x_test_pima)
    # accuracy_pima = accuracy_score(y_test_pima, y_pred_pima)
    # print("Pima Indians accuracy: %.2f" % accuracy_pima)

    '''
    Breast Cancer
    '''

    # feature_columns_canc = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    #                         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    # x_canc = canc[feature_columns_canc]
    # canc['Class'] = canc['Class'].replace([2, 4], [0, 1])
    # y_canc = canc['Class'].values
    #
    # x_train_canc, x_test_canc, y_train_canc, y_test_canc = train_test_split(x_canc, y_canc, test_size=0.2, random_state=0)
    #
    # scaler = Normalizer().fit(x_train_canc)
    # normalized_x_train_canc = scaler.transform(x_train_canc)
    # normalized_x_test_canc = scaler.transform(x_test_canc)
    #
    # y_pred_canc = class_based_knn(normalized_x_train_canc, y_train_canc, normalized_x_test_canc)
    # accuracy_canc = accuracy_score(y_test_canc, y_pred_canc)
    # print("Breast Cancer accuracy: %.2f" % accuracy_canc)

    '''
    Heart Disease
    '''

    # heart.drop(heart[heart['ca'] == "?"].index, inplace=True)
    # heart.drop(heart[heart['thal'] == "?"].index, inplace=True)
    #
    # feature_columns_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    #                          'exang', 'oldpeak', 'slope', 'ca', 'thal']
    # x_heart = heart[feature_columns_heart]
    # heart['num'] = heart['num'].replace([1, 2, 3, 4], 1)
    # y_heart = heart['num'].values
    #
    # x_train_heart, x_test_heart, y_train_heart, y_test_heart = train_test_split(x_heart, y_heart, test_size=0.2, random_state=0)
    #
    # scaler = Normalizer().fit(x_train_heart)
    # normalized_x_train_heart = scaler.transform(x_train_heart)
    # normalized_x_test_heart = scaler.transform(x_test_heart)
    #
    # y_pred_heart = class_based_knn(normalized_x_train_heart, y_train_heart, normalized_x_test_heart)
    # accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
    # print("Heart Disease accuracy: %.2f" % accuracy_heart)


