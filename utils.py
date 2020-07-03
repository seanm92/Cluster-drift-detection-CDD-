import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

class load_db:
    @staticmethod
    def load_intrusion(working_path, is_hst=False):
        KDD = pd.read_csv(working_path+'\\KDD_train.csv', header=None)
        KDD = KDD.drop_duplicates()
        KDD_test = pd.read_csv(working_path+'\\KDD_test.csv', header=None)
        KDD_test = KDD_test.drop_duplicates()
        
        del KDD[KDD.columns[20]]
        del KDD_test[KDD_test.columns[20]]
        
        y_train = KDD.values[:, 41]
        del KDD[KDD.columns[40]]
        del KDD_test[KDD_test.columns[40]]
        
        ### icmp category did not observed in trainig set, and consist two sample only in test set
        KDD_test = KDD_test[KDD_test.iloc[:, 2] != 'icmp']
        y_test = KDD_test.values[:, 40]
        for i in [1, 2, 3]:
            le = LabelEncoder()
            KDD.loc[:, i] = le.fit_transform(KDD.loc[:, i])
            KDD_test.loc[:, i] = le.transform(KDD_test.loc[:, i])
        
        onehotencoder = OneHotEncoder(categorical_features=[1, 2, 3])
        KDD = onehotencoder.fit_transform(KDD).toarray()
        KDD_test = onehotencoder.transform(KDD_test).toarray()

        X_train = np.delete(KDD, [117], 1)
        X_train_normal = X_train[y_train == 0, :]
        X_test = np.delete(KDD_test, [117], 1)

        categorical_col = []
        numeric_col = []
        for i in range(X_train.shape[1]):
            if len(np.unique(X_train[:, i])) == 2:
                categorical_col.append(i)
            else:
                numeric_col.append(i)
        if not is_hst:
            X_train_numeric = X_train_normal[:, numeric_col]
            StSc = StandardScaler()
            X_train_numeric_scaled = StSc.fit_transform(X_train_numeric)
            X_train_cat = X_train_normal[:, categorical_col]
            X_train_all = np.concatenate((X_train_numeric_scaled, X_train_cat), axis=1)

            X_test_numeric = X_test[:, numeric_col]
            X_test_numeric_scaled = StSc.transform(X_test_numeric)
            X_test_cat = X_test[:, categorical_col]
            X_test_all = np.concatenate((X_test_numeric_scaled, X_test_cat), axis=1)

        else:
            X_train_numeric = X_train_normal[:, numeric_col]
            MM = MinMaxScaler()
            X_train_numeric_scaled = MM.fit_transform(X_train_numeric)
            X_train_cat = X_train_normal[:, categorical_col]
            X_train_all = np.concatenate((X_train_numeric_scaled, X_train_cat), axis=1)

            X_test_numeric = X_test[:, numeric_col]
            MM = MinMaxScaler()
            X_test_numeric_scaled = MM.fit_transform(X_test_numeric)
            X_test_cat = X_test[:, categorical_col]
            X_test_all = np.concatenate((X_test_numeric_scaled, X_test_cat), axis=1)

        return X_train_all, X_test_all, y_test

    @staticmethod
    def load_fraud(working_path, is_hst=False):
        Fraud = pd.read_csv(working_path+'\\Fraud.csv')
        del Fraud['isFlaggedFraud']
        del Fraud['nameOrig']
        del Fraud['nameDest']
        
        le = LabelEncoder()
        Fraud.iloc[:, 1] = le.fit_transform(Fraud.iloc[:, 1])
        onehotencoder = OneHotEncoder(categorical_features=[1])
        Fraud = onehotencoder.fit_transform(Fraud).toarray()
        
        y = Fraud[:, 11]
        step = Fraud[:, 5]
        X = np.delete(Fraud, [5, 11], 1)
        X_train = X[step < 2, :]
        y_train = y[step < 2]
        X_train_normal = X_train[y_train == 0, :]
        y_train = y[step < 2]
        X_test = X[step >= 2, :]
        y_test = y[step >= 2]

        if not is_hst:
            X_train_numeric = X_train[:, 5:]
            StSc = StandardScaler()
            X_train_numeric_scaled = StSc.fit_transform(X_train_numeric)
            X_train_all = np.concatenate((X_train_numeric_scaled, X_train[:, :5]), axis=1)
            X_test_numeric = X_test[:, 5:]
            X_test_numeric_scaled = StSc.transform(X_test_numeric)
            X_test_all = np.concatenate((X_test_numeric_scaled, X_test[:, :5]), axis=1)
            X_train_all = X_train_all[y_train == 0, :]
        else:
             X_train_numeric = X_train_normal[:, 5:]
             MM = MinMaxScaler()
             X_train_numeric_scaled = MM.fit_transform(X_train_numeric)
             X_train_cat = X_train_normal[:, :5]
             X_train_all = np.concatenate((X_train_numeric_scaled, X_train_cat), axis=1)

             X_test_numeric = X_test[:, 5:]
             MM = MinMaxScaler()
             X_test_numeric_scaled = MM.fit_transform(X_test_numeric)
             X_test_cat = X_test[:, :5]
             X_test_all = np.concatenate((X_test_numeric_scaled, X_test_cat), axis=1)
        return X_train_all, X_test_all, y_test

def Hopkins(X):
    n, d = X.shape
    #d = len(vars) # columns
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
     
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        H = 0 
    return H 

class evaluate:
    @staticmethod
    def evaluate_intrusion(y_pred, y_test):
        normal_detect = ((y_test == 0) & (y_pred == 1)).sum()/(sum(y_test == 0))
        dos_detect = ((y_test == 1) & (y_pred == -1)).sum()/(sum(y_test == 1))
        U2R_detect = ((y_test == 2) & (y_pred == -1)).sum()/(sum(y_test == 2))
        R2L_detect = ((y_test == 3) & (y_pred == -1)).sum()/(sum(y_test == 3))
        probe_detect = ((y_test == 4) & (y_pred == -1)).sum()/(sum(y_test == 4))
        return [normal_detect, dos_detect, U2R_detect, R2L_detect, probe_detect]
    @staticmethod
    def evaluate_binary(y_pred, y_test):
        ND = ((y_test == 0) & (y_pred == 1)).sum()/(sum(y_test == 0))
        FD = ((y_test == 1) & (y_pred == -1)).sum()/(sum(y_test == 1))
        return ND, FD

    def AUC(y_pred_proba, y_test):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        return auc(fpr, tpr)

    



