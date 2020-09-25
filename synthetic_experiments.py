from utils import evaluate
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import svm
from tqdm import tqdm
import numpy as np
from CDD import CDD
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import inv, matrix_rank, pinv
from scipy.stats import f
from pyod.models.auto_encoder import AutoEncoder
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from pyod.models.loda import LODA
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.anomaly_detection import HalfSpaceTrees
import pandas as pd


def Hotelling(x, mean, S):
    P = mean.shape[0]
    a = x-mean
    # check rank to decide the type of inverse(regular or adjused to non inverses matrix)
    if matrix_rank(S) == P:
        b = inv(np.array(S))
    else:
        b = pinv(np.array(S))
    c = np.transpose(x-mean)
    T2 = np.matmul(a, b)
    T2 = np.matmul(T2, c)
    return T2


def Hoteliing_SPC_proba(normal_data, new_data):
    normal_mean = np.mean(normal_data, axis=0)
    normal_cov = np.cov(np.transpose(normal_data))
    normal_size = normal_data.shape[0]
    M,P = new_data.shape
    anomalies_scores = np.zeros(M)
    for i in range(M):
        obs = new_data[i, :]
        T2 = Hotelling(obs, normal_mean, normal_cov)
        Fstat = T2 * ((normal_size-P)*normal_size)/(P*(normal_size-1)*(normal_size+1))
        anomalies_scores[i] = f.cdf(Fstat, P, normal_size -P)
    return anomalies_scores


# ------------- diffrent number of clusters ------------- #
def number_of_clusters_examine(n_clusters):
    num_of_experiments = 10
    training_size = 1000
    total_test_size = 10000
    anomaly_fraction = 0.01
    normal_test_size = int(total_test_size*(1-anomaly_fraction))
    anomaly_size = total_test_size - normal_test_size 
    
    centers = [[2, 2], [-2, -2], [2, -2], [-2, 2], [6, 6], [-6, -6], [6, -6], [-6, 6]]
    current_centers = centers[:n_clusters]
    X_train = make_blobs(n_samples=training_size, centers=current_centers, cluster_std=0.5, random_state=0)[0]
    X_test_normal = make_blobs(n_samples=normal_test_size, centers=current_centers, cluster_std=0.5, random_state=0)[0]
    X_test_noval = np.random.RandomState(0).uniform(low=-8, high=8, size=(anomaly_size, 2))
    X_test = np.concatenate((X_test_normal, X_test_noval), axis=0)
    y_test = np.array([0 if x < normal_test_size else 1 for x in range(total_test_size)])
    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    mm = MinMaxScaler()
    X_train_hst = mm.fit_transform(X_train.copy())
    mm = MinMaxScaler()
    X_test_hst = mm.fit_transform(X_test.copy())

    # --- CDD --- #
    auc_cdd_ls = []
    for r in range(num_of_experiments):
        cdd = CDD(random=r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test)
        auc_cdd_ls.append(evaluate.AUC(y_pred_proba, y_test))
    auc_CDD = np.mean(auc_cdd_ls)
    # ---AE --- #
    auc_ae_ls = []
    for r in range(num_of_experiments):
        AE = AutoEncoder(hidden_neurons=[64, 2, 2, 64], random_state=r)
        AE.fit(X_train)
        ae_pred_proba = AE.predict_proba(X_test)[:, 1]
        auc_ae_ls.append(evaluate.AUC(ae_pred_proba, y_test))
    auc_AE = np.mean(auc_ae_ls)
    
    # --- Iforest --- #
    auc_if_ls = []
    for r in range(num_of_experiments):
        IF = IsolationForest(random_state = r)
        IF.fit(X_train)  
        sklearn_score_anomalies = IF.decision_function(X_test)
        original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
        auc_if_ls.append(evaluate.AUC(original_paper_score,y_test))
    auc_IF = np.mean(auc_if_ls)
    
    # --- one-class-SVM --- #
    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(X_train)
    sklearn_score_anomalies = clf.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_SVM = evaluate.AUC(original_paper_score, y_test)
    
    # --- LOF --- #
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    sklearn_score_anomalies = lof.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_LOF = evaluate.AUC(original_paper_score, y_test)
    
    # --- T2 --- #
    y_pred_hot = Hoteliing_SPC_proba(X_train, X_test)
    auc_hot = evaluate.AUC(y_pred_hot, y_test)

    # --- HST --- #
    aucs_hst = []
    for r in tqdm(range(num_of_experiments)):
        hst = HalfSpaceTrees(n_features=X_train.shape[1], random_state=r)
        hst.fit(X_train_hst, np.zeros(X_test_hst.shape[0]))
        pred_hst = np.zeros(X_test_hst.shape[0])
        for i in tqdm(range(X_test_hst.shape[0])):
            hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
            pred_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
        aucs_hst.append(evaluate.AUC(pred_hst, y_test))
    auc_hst = np.mean(aucs_hst)
    # --- LODA --- #
    aucs_loda = []
    for r in tqdm(range(num_of_experiments)):
        loda = LODA()
        loda.fit(X_train)
        pred_loda = np.zeros(X_test.shape[0])
        for i in tqdm(range(X_test.shape[0])):
            loda.fit(X_test[i, :].reshape(1, -1))
            pred_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
        aucs_loda.append(evaluate.AUC(1-pred_loda, y_test))
    auc_loda = np.mean(aucs_loda)
    return [auc_CDD, auc_AE, auc_IF, auc_SVM, auc_LOF, auc_hot, auc_hst, auc_loda]


Nclusters = range(1, 9)
cdd_results = np.zeros(len(Nclusters))
ae_results = np.zeros(len(Nclusters))
iforest_results = np.zeros(len(Nclusters))
svm_results = np.zeros(len(Nclusters))
lof_results = np.zeros(len(Nclusters))
hot_results = np.zeros(len(Nclusters))
hst_results = np.zeros(len(Nclusters))
loda_results = np.zeros(len(Nclusters))

for ind, n_clusters in enumerate(tqdm(Nclusters)):
    results = number_of_clusters_examine(n_clusters=n_clusters)
    cdd_results[ind] = results[0]
    ae_results[ind] = results[1]
    iforest_results[ind] = results[2]
    svm_results[ind] = results[3]
    lof_results[ind] = results[4]
    hot_results[ind] = results[5]
    hst_results[ind] = results[6]
    loda_results[ind] = results[7]

plt.figure(figsize=(10, 7))
plt.plot(Nclusters, cdd_results, marker='.')
plt.plot(Nclusters, ae_results, marker='*')
plt.plot(Nclusters, iforest_results, marker='P')
plt.plot(Nclusters, svm_results, marker='^')
plt.plot(Nclusters, lof_results, marker='D')
plt.plot(Nclusters, hot_results, marker='p')
plt.plot(Nclusters, hst_results, marker=',')
plt.plot(Nclusters, loda_results, marker='1')
plt.legend(['CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'], loc='lower left')
plt.xlabel('Number of sources', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.grid(alpha=0.5)

df_nsources = pd.DataFrame(columns=['Nclusters','CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'])
df_nsources['Nclusters'] = Nclusters
df_nsources['CDD'] = cdd_results
df_nsources['AE'] = ae_results
df_nsources['IForest'] = iforest_results
df_nsources['OCSVM'] = svm_results
df_nsources['LOF'] = lof_results
df_nsources['$T^2$ SPC'] = hot_results
df_nsources['HST'] = hst_results
df_nsources['LODA'] = loda_results
df_nsources.to_csv('df_nsources.csv')


# ------------- concept drift - abrupt ------------- #
def concept_drift_comparison(drift_level=0):
    num_of_experiments = 10
    training_size = 1000
    total_test_size = 10000
    anomaly_fraction = 0.01
    normal_test_size = int(total_test_size*(1-anomaly_fraction))
    anomaly_size = total_test_size - normal_test_size 
    
    centers = [[2, 2], [-2, -2]]
    test_centers = centers.copy()
    test_centers[0] = [x+drift_level for x in test_centers[0]] 
    
    X_train = make_blobs(n_samples=training_size, centers=centers, cluster_std=0.5,random_state=0)[0]
    X_test_normal = make_blobs(n_samples=normal_test_size, centers=test_centers, cluster_std=0.5,random_state=0)[0]
    X_test_noval = np.random.RandomState(0).uniform(low=-8, high=8, size=(anomaly_size, 2))
    X_test = np.concatenate((X_test_normal, X_test_noval), axis=0)
    y_test = np.array([0 if x < normal_test_size else 1 for x in range(total_test_size)])
    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    mm = MinMaxScaler()
    X_train_hst = mm.fit_transform(X_train)
    mm = MinMaxScaler()
    X_test_hst = mm.fit_transform(X_test)

    # --- CDD --- #
    auc_cdd_ls = []
    for r in range(num_of_experiments):  
        cdd = CDD(random=r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test)
        auc_cdd_ls.append(evaluate.AUC(y_pred_proba, y_test))
    auc_CDD = np.mean(auc_cdd_ls) 
        
    # --- AE --- #
    auc_ae_ls = []
    for r in range(num_of_experiments):
        AE = AutoEncoder(hidden_neurons=[64, 2, 2, 64], random_state=r)
        AE.fit(X_train)
        ae_pred_proba = AE.predict_proba(X_test)[:, 1]
        auc_ae_ls.append(evaluate.AUC(ae_pred_proba, y_test))
    auc_AE = np.mean(auc_ae_ls)
    
    # --- Iforest --- #
    auc_if_ls = []
    for r in range(num_of_experiments):
        IF = IsolationForest(random_state=r)
        IF.fit(X_train)  
        sklearn_score_anomalies = IF.decision_function(X_test)
        original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
        auc_if_ls.append(evaluate.AUC(original_paper_score, y_test))
    auc_IF = np.mean(auc_if_ls)
    
    # --- one-class-SVM --- #
    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(X_train)
    sklearn_score_anomalies = clf.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_SVM = evaluate.AUC(original_paper_score, y_test)
    
    # --- LOF --- #
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    sklearn_score_anomalies = lof.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_LOF = evaluate.AUC(original_paper_score, y_test)
    
    # --- T2 --- #
    y_pred_hot = Hoteliing_SPC_proba(X_train, X_test)
    auc_hot = evaluate.AUC(y_pred_hot, y_test)

    # --- HST ---#
    aucs_hst = []
    for r in tqdm(range(num_of_experiments)):
        hst = HalfSpaceTrees(n_features=X_train.shape[1], random_state=r)
        hst.fit(X_train_hst, np.zeros(X_test_hst.shape[0]))
        pred_hst = np.zeros(X_test_hst.shape[0])
        for i in tqdm(range(X_test_hst.shape[0])):
            hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
            pred_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
        aucs_hst.append(evaluate.AUC(pred_hst, y_test))
    auc_hst = np.mean(aucs_hst)
    # --- LODA --- #
    aucs_loda = []
    for r in tqdm(range(num_of_experiments)):
        loda = LODA()
        loda.fit(X_train)
        pred_loda = np.zeros(X_test.shape[0])
        for i in tqdm(range(X_test.shape[0])):
            loda.fit(X_test[i, :].reshape(1, -1))
            pred_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
        aucs_loda.append(evaluate.AUC(1-pred_loda, y_test))
    auc_loda = np.mean(aucs_loda)
    return [auc_CDD, auc_AE, auc_IF, auc_SVM, auc_LOF, auc_hot, auc_hst, auc_loda]


drift_levels = np.arange(start=0, stop=1.1, step=0.1)
cdd_results1 = np.zeros(len(drift_levels))
ae_results1 = np.zeros(len(drift_levels))
iforest_results1 = np.zeros(len(drift_levels))
svm_results1 = np.zeros(len(drift_levels))
lof_results1 = np.zeros(len(drift_levels))
hot_results1 = np.zeros(len(drift_levels))
hst_results1 = np.zeros(len(drift_levels))
loda_results1 = np.zeros(len(drift_levels))

for ind, drift_level in enumerate(tqdm(drift_levels)):
    results = concept_drift_comparison(drift_level=drift_level)
    cdd_results1[ind] = results[0]
    ae_results1[ind] = results[1]
    iforest_results1[ind] = results[2]
    svm_results1[ind] = results[3]
    lof_results1[ind] = results[4]
    hot_results1[ind] = results[5]
    hst_results1[ind] = results[6]
    loda_results1[ind] = results[7]

plt.figure(figsize=(10, 7))
plt.plot(drift_levels, cdd_results1, marker='.')
plt.plot(drift_levels, ae_results1, marker='*')
plt.plot(drift_levels, iforest_results1, marker='P')
plt.plot(drift_levels, svm_results1, marker='^')
plt.plot(drift_levels, lof_results1, marker='D')
plt.plot(drift_levels, hot_results1, marker='p')
plt.plot(drift_levels, hst_results1, marker=',')
plt.plot(drift_levels, loda_results1, marker='1')
plt.legend(['CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'], loc='lower left')
plt.xlabel('Drift level', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.grid(alpha=0.5)

df_drift = pd.DataFrame(columns=['Drift level', 'CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'])
df_drift['Drift level'] = drift_levels
df_drift['CDD'] = cdd_results1
df_drift['AE'] = ae_results1
df_drift['IForest'] = iforest_results1
df_drift['OCSVM'] = svm_results1
df_drift['LOF'] = lof_results1
df_drift['$T^2$ SPC'] = hot_results1
df_drift['HST'] = hst_results1
df_drift['LODA'] = loda_results1
df_drift.to_csv('df_drift.csv')


def dimension_effect_comparison(dim=1):
    num_of_experiments = 10
    training_size = 1000
    total_test_size = 10000
    anomaly_fraction = 0.01
    normal_test_size = int(total_test_size*(1-anomaly_fraction))
    anomaly_size = total_test_size - normal_test_size 
    
    centers = [[2, 2], [-2, -2]]
    test_centers = centers.copy()

    X_train = make_blobs(n_samples=training_size, centers=centers, cluster_std=0.5, random_state=0)[0]
    X_train = np.concatenate((X_train.copy(), np.random.RandomState(0).uniform(low=-1, high=1, size=(X_train.shape[0], dim))), axis=1)
    
    X_test_normal = make_blobs(n_samples=normal_test_size, centers=test_centers, cluster_std=0.5,random_state=0)[0]
    X_test_noval = np.random.RandomState(0).uniform(low=-8, high=8, size=(anomaly_size, 2))
    X_test = np.concatenate((X_test_normal, X_test_noval), axis=0)
    y_test = np.array([0 if x < normal_test_size else 1 for x in range(total_test_size)])
    X_test,y_test = shuffle(X_test, y_test, random_state=0)
    
    X_test = np.concatenate((X_test.copy(), np.random.RandomState(0).uniform(low=-1, high=1, size=(X_test.shape[0], dim))), axis=1)

    mm = MinMaxScaler()
    X_train_hst = mm.fit_transform(X_train)
    mm = MinMaxScaler()
    X_test_hst = mm.fit_transform(X_test)
    # --- CDD --- #
    auc_cdd_ls = []
    for r in range(num_of_experiments):  
        cdd = CDD(random=r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test)
        auc_cdd_ls.append(evaluate.AUC(y_pred_proba, y_test))
    auc_CDD = np.mean(auc_cdd_ls) 
        
    # --- AE --- #
    auc_ae_ls = []
    for r in range(num_of_experiments):
        AE = AutoEncoder(hidden_neurons=[64, 2, 2, 64], random_state=r)
        AE.fit(X_train)
        ae_pred_proba = AE.predict_proba(X_test)[:, 1]
        auc_ae_ls.append(evaluate.AUC(ae_pred_proba, y_test))
    auc_AE = np.mean(auc_ae_ls)
    
    # --- Iforest ---#
    auc_if_ls = []
    for r in range(num_of_experiments):
        IF = IsolationForest(random_state = r)
        IF.fit(X_train)  
        sklearn_score_anomalies = IF.decision_function(X_test)
        original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
        auc_if_ls.append(evaluate.AUC(original_paper_score,y_test))
    auc_IF = np.mean(auc_if_ls)
    
    # --- one-class-SVM  --- #
    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(X_train)
    sklearn_score_anomalies = clf.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_SVM = evaluate.AUC(original_paper_score, y_test)
    
    # --- LOF --- #
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    sklearn_score_anomalies = lof.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_LOF = evaluate.AUC(original_paper_score, y_test)
    
    # --- T2 --- #
    y_pred_hot = Hoteliing_SPC_proba(X_train, X_test)
    auc_hot = evaluate.AUC(y_pred_hot, y_test)

    # --- HST --- #
    aucs_hst = []
    for r in tqdm(range(num_of_experiments)):
        hst = HalfSpaceTrees(n_features=X_train.shape[1], random_state=r)
        hst.fit(X_train_hst, np.zeros(X_test_hst.shape[0]))
        pred_hst = np.zeros(X_test_hst.shape[0])
        for i in tqdm(range(X_test_hst.shape[0])):
            hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
            pred_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
        aucs_hst.append(evaluate.AUC(pred_hst, y_test))
    auc_hst = np.mean(aucs_hst)
    # --- LODA --- #
    aucs_loda = []
    for r in tqdm(range(num_of_experiments)):
        loda = LODA()
        loda.fit(X_train)
        pred_loda = np.zeros(X_test.shape[0])
        for i in tqdm(range(X_test.shape[0])):
            loda.fit(X_test[i, :].reshape(1, -1))
            pred_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
        aucs_loda.append(evaluate.AUC(1-pred_loda, y_test))
    auc_loda = np.mean(aucs_loda)
    return [auc_CDD, auc_AE, auc_IF, auc_SVM, auc_LOF, auc_hot, auc_hst, auc_loda]


dimensions = range(1, 11)
cdd_results2 = np.zeros(len(dimensions))
ae_results2 = np.zeros(len(dimensions))
iforest_results2 = np.zeros(len(dimensions))
svm_results2 = np.zeros(len(dimensions))
lof_results2 = np.zeros(len(dimensions))
hot_results2 = np.zeros(len(dimensions))
hst_results2 = np.zeros(len(dimensions))
loda_results2 = np.zeros(len(dimensions))

for ind, dim in enumerate(tqdm(dimensions)):
    results = dimension_effect_comparison(dim=dim)
    cdd_results2[ind] = results[0]
    ae_results2[ind] = results[1]
    iforest_results2[ind] = results[2]
    svm_results2[ind] = results[3]
    lof_results2[ind] = results[4]
    hot_results2[ind] = results[5]
    hst_results2[ind] = results[6]
    loda_results2[ind] = results[7]

plt.figure(figsize=(10, 7))
plt.plot(dimensions, cdd_results2, marker='.')
plt.plot(dimensions, ae_results2, marker='*')
plt.plot(dimensions, iforest_results2, marker='P')
plt.plot(dimensions, svm_results2, marker='^')
plt.plot(dimensions, lof_results2, marker='D')
plt.plot(dimensions, hot_results2, marker='p')
plt.plot(dimensions, hst_results2, marker='h')
plt.plot(dimensions, loda_results2, marker='1')
plt.legend(['CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'], loc='lower left')
plt.xlabel('Irrelevant dimension', fontsize=14)
plt.ylabel('AUC', fontsize=14)
plt.grid(alpha=0.5)

df_dimension = pd.DataFrame(columns=['Irrelevant dimension', 'CDD', 'AE', 'IForest', 'OCSVM', 'LOF', '$T^2$ SPC', 'HST', 'LODA'])
df_dimension['Irrelevant dimension'] = dimensions
df_dimension['CDD'] = cdd_results2
df_dimension['AE'] = ae_results2
df_dimension['IForest'] = iforest_results2
df_dimension['OCSVM'] = svm_results2
df_dimension['LOF'] = lof_results2
df_dimension['$T^2$ SPC'] = hot_results2
df_dimension['HST'] = hst_results2
df_dimension['LODA'] = loda_results2
df_dimension.to_csv('df_dimension.csv')

