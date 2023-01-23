import numpy as np
import pandas as pd
import statistics

from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


def euc_distance(x_train, x_test_point):
    """
    Calculates Euclidean distance between points in n-dimensional space
    :param x_train: corresponding to the training data
    :param x_test_point: corresponding to the test point
    :return: The distances between the test point and each point in the training data
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
    Tests different values for k so the DC (Degree of Certainty) of the classification
    is maximised for each test element
    :param distance_point: the distances between the test point and each point in the training data
    :param y_train: class values of the training data
    :return: the K value that maximises the DC
    """

    k_dci_scores = defaultdict(int)
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    for test_k in range(3, 35):

        df_nearest_firstk = df_nearest[:test_k]

        counter_vote = Counter(y_train[df_nearest_firstk.index])
        max_counter_vote = max(counter_vote, key=counter_vote.get)
        max_classification_score = counter_vote[max_counter_vote]

        dci = max_classification_score / test_k
        k_dci_scores[test_k] = dci

    selected_k = max(k_dci_scores, key=k_dci_scores.get)
    return selected_k


def prediction(distance_point, y_train, K):
    """
    For every test element, the k nearest elements of each class are taken.
    The harmonic mean of the distances of these neighbours is calculated.
    These means are compared and the class yielding the lowest value is chosen for the classification.
    :param distance_point: the distances between the test point and each point in the training data
    :param y_train: class values of the training data
    :param K: k nearest elements of each class are taken
    :return: prediction
    """

    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    classes = np.unique(y_train)
    class_values = []

    for cl in classes:
        class_values.append([])
    dist_harm_mean_class = defaultdict(int)

    for neigh in df_nearest.index:
        class_values[y_train[neigh]].append(neigh)

    for cl in classes:
        one_class_values = class_values[cl][:K]

        class_dist = []
        for val in one_class_values:
            class_dist.append(df_nearest["dist"].loc[val])

        harm_mean = statistics.harmonic_mean(class_dist)
        dist_harm_mean_class[cl] = harm_mean

    selected_class = min(dist_harm_mean_class, key=dist_harm_mean_class.get)
    return selected_class


def class_based_knn(x_train, y_train, x_test):
    """
    Class Based kNN Classifier
    :param x_train: the full training dataset
    :param y_train: the classes of the training dataset
    :param x_test: the full test dataset
    :return: the prediction for the whole test set
    """

    y_pred = []
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
    veh = pd.read_csv('datasets/vehicles.csv')
    hous = pd.read_csv('datasets/boston-housing.csv')

    '''
    Bupa Liver
    '''

    bupa = bupa.drop(columns='selector')
    bupa["drinks"] = pd.cut(bupa["drinks"], bins=[-1, 3, 20], labels=[0, 1])

    feature_columns_bupa = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt']
    x_bupa = bupa[feature_columns_bupa]
    y_bupa = bupa['drinks'].values

    kf = KFold(n_splits=10, random_state=4, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_bupa):
        x_train_bupa, x_test_bupa = x_bupa.iloc[train_index, :], x_bupa.iloc[test_index, :]
        y_train_bupa, y_test_bupa = y_bupa[train_index], y_bupa[test_index]

        scaler = Normalizer().fit(x_train_bupa)
        normalized_x_train_bupa = scaler.transform(x_train_bupa)
        normalized_x_test_bupa = scaler.transform(x_test_bupa)

        y_pred_bupa = class_based_knn(normalized_x_train_bupa, y_train_bupa, normalized_x_test_bupa)
        accuracy_bupa = accuracy_score(y_test_bupa, y_pred_bupa)
        acc_scores.append(accuracy_bupa)
        # print("Bupa Liver Accuracy: %.2f" % accuracy_bupa)

    avg_acc_score = sum(acc_scores) / 10
    print("Bupa Liver Accuracy: %.2f" % avg_acc_score)

    '''
    Pima Indians
    '''

    feature_columns_pima = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                            'DiabetesPedigreeFunction', 'Age']
    x_pima = pima[feature_columns_pima]
    y_pima = pima['Outcome'].values

    kf = KFold(n_splits=10, random_state=4, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_pima):
        x_train_pima, x_test_pima = x_pima.iloc[train_index, :], x_pima.iloc[test_index, :]
        y_train_pima, y_test_pima = y_pima[train_index], y_pima[test_index]

        scaler = Normalizer().fit(x_train_pima)
        normalized_x_train_pima = scaler.transform(x_train_pima)
        normalized_x_test_pima = scaler.transform(x_test_pima)

        y_pred_pima = class_based_knn(normalized_x_train_pima, y_train_pima, normalized_x_test_pima)
        accuracy_pima = accuracy_score(y_test_pima, y_pred_pima)
        acc_scores.append(accuracy_pima)
        # print("Pima Indians Accuracy: %.2f" % accuracy_pima)

    avg_acc_score = sum(acc_scores) / 10
    print("Pima Indians Accuracy: %.2f" % avg_acc_score)

    '''
    Breast Cancer
    '''

    feature_columns_canc = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    x_canc = canc[feature_columns_canc]
    canc['Class'] = canc['Class'].replace([2, 4], [0, 1])
    y_canc = canc['Class'].values

    kf = KFold(n_splits=10, random_state=4, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_canc):
        x_train_canc, x_test_canc = x_canc.iloc[train_index, :], x_canc.iloc[test_index, :]
        y_train_canc, y_test_canc = y_canc[train_index], y_canc[test_index]

        scaler = Normalizer().fit(x_train_canc)
        normalized_x_train_canc = scaler.transform(x_train_canc)
        normalized_x_test_canc = scaler.transform(x_test_canc)

        y_pred_canc = class_based_knn(normalized_x_train_canc, y_train_canc, normalized_x_test_canc)
        accuracy_canc = accuracy_score(y_test_canc, y_pred_canc)
        acc_scores.append(accuracy_canc)
        # print("Breast Cancer Accuracy: %.2f" % accuracy_canc)

    avg_acc_score = sum(acc_scores) / 10
    print("Breast Cancer Accuracy: %.2f" % avg_acc_score)

    '''
    Heart Disease
    '''

    heart.drop(heart[heart['ca'] == "?"].index, inplace=True)
    heart.drop(heart[heart['thal'] == "?"].index, inplace=True)

    feature_columns_heart = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                             'exang', 'oldpeak', 'slope', 'ca', 'thal']
    x_heart = heart[feature_columns_heart]
    heart['num'] = heart['num'].replace([1, 2, 3, 4], 1)
    y_heart = heart['num'].values

    kf = KFold(n_splits=10, random_state=7, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_heart):
        x_train_heart, x_test_heart = x_heart.iloc[train_index, :], x_heart.iloc[test_index, :]
        y_train_heart, y_test_heart = y_heart[train_index], y_heart[test_index]

        scaler = Normalizer().fit(x_train_heart)
        normalized_x_train_heart = scaler.transform(x_train_heart)
        normalized_x_test_heart = scaler.transform(x_test_heart)

        y_pred_heart = class_based_knn(normalized_x_train_heart, y_train_heart, normalized_x_test_heart)
        accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
        acc_scores.append(accuracy_heart)
        # print("Heart Disease Accuracy: %.2f" % accuracy_heart)

    avg_acc_score = sum(acc_scores) / 10
    print("Heart Disease Accuracy: %.2f" % avg_acc_score)

    '''
    Vehicle
    '''

    veh = veh.dropna()
    feature_columns_veh = ['Compactness', 'Circularity', 'DistanceCircularity', 'RadiusRatio', 'PRAxisAspectRatio',
                           'MaxLengthAspectRatio', 'ScatterRatio', 'Elongatedness', 'PRAxisRectangularity',
                           'MaxLengthRectangularity', 'ScaledVarianceMajor', 'ScaledVarianceMinor', 'ScaledRadius',
                           'SkewnessMajor', 'SkewnessMinor', 'KurtosisMinor', 'KurtosisMajor', 'HollowsRatio']
    x_veh = veh[feature_columns_veh]
    veh['Class'] = veh['Class'].replace(['opel', 'saab', 'bus', 'van'], [0, 1, 2, 3])
    y_veh = veh['Class'].values

    kf = KFold(n_splits=10, random_state=4, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_veh):
        x_train_veh, x_test_veh = x_veh.iloc[train_index, :], x_veh.iloc[test_index, :]
        y_train_veh, y_test_veh = y_veh[train_index], y_veh[test_index]

        scaler = Normalizer().fit(x_train_veh)
        normalized_x_train_veh = scaler.transform(x_train_veh)
        normalized_x_test_veh = scaler.transform(x_test_veh)

        y_pred_veh = class_based_knn(normalized_x_train_veh, y_train_veh, normalized_x_test_veh)
        accuracy_veh = accuracy_score(y_test_veh, y_pred_veh)
        acc_scores.append(accuracy_veh)
        # print("Vehicles Accuracy: %.2f" % accuracy_veh)

    avg_acc_score = sum(acc_scores) / 10
    print("Vehicles Accuracy: %.2f" % avg_acc_score)


    '''
    Boston Housing
    '''

    # hous["medv"] = pd.cut(hous["medv"], bins=[0, 15, 26, 38, 50], labels=[0, 1, 2, 3])
    hous["medv"] = pd.cut(hous["medv"], bins=[0, 17, 33, 50], labels=[0, 1, 2])
    feature_columns_hous = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
                            'tax', 'ptratio', 'b', 'lstat']
    x_hous = hous[feature_columns_hous]
    y_hous = hous['medv'].values

    kf = KFold(n_splits=10, random_state=6, shuffle=True)
    acc_scores = []

    for train_index, test_index in kf.split(x_hous):
        x_train_hous, x_test_hous = x_hous.iloc[train_index, :], x_hous.iloc[test_index, :]
        y_train_hous, y_test_hous = y_hous[train_index], y_hous[test_index]

        scaler = Normalizer().fit(x_train_hous)
        normalized_x_train_hous = scaler.transform(x_train_hous)
        normalized_x_test_hous = scaler.transform(x_test_hous)

        y_pred_hous = class_based_knn(normalized_x_train_hous, y_train_hous, normalized_x_test_hous)
        accuracy_hous = accuracy_score(y_test_hous, y_pred_hous)
        acc_scores.append(accuracy_hous)
        # print("Boston Housing Accuracy: %.2f" % accuracy_hous)

    avg_acc_score = sum(acc_scores) / 10
    print("Boston Housing Accuracy: %.2f" % avg_acc_score)
