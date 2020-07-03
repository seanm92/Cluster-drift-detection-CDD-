import sys
path = r'C:\Users\97254\Desktop\CDD\CDD_code'
sys.path.append(path)
from utils import load_db, Hopkins, evaluate
from sklearn.metrics import auc
from sklearn.ensemble import IsolationForest
from sklearn import svm
from tqdm import tqdm
import numpy as np
from CDD import CDD
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import inv, matrix_rank, pinv
from scipy.stats import f
from pyod.models.auto_encoder import AutoEncoder
import scipy
from skmultiflow.anomaly_detection import HalfSpaceTrees
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.loda import LODA
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pandas as pd

def Hotelling(x,mean,S):
    P = mean.shape[0]
    a = x-mean
    ##check rank to decide the type of inverse(regular or adjused to non inverses matrix)
    if matrix_rank(S) == P:
        b = inv(np.matrix(S))
    else:
        b = pinv(np.matrix(S))
    c = np.transpose(x-mean)
    T2 = np.matmul(a,b)
    T2 = np.matmul(T2,c)
    return T2
def Hoteliing_SPC_proba(normal_data,new_data,alpha =0.05):
    normal_mean = np.mean(normal_data,axis = 0)
    normal_cov = np.cov(np.transpose(normal_data))
    normal_size = normal_data.shape[0]
    M,P = new_data.shape
    anomalie_scores = np.zeros(M)
    for i in range(M):
        obs = new_data[i,:]
        T2 = Hotelling(obs,normal_mean,normal_cov)
        Fstat = T2 * ((normal_size-P)*normal_size)/(P*(normal_size-1)*(normal_size+1))
        anomalie_scores[i] = f.cdf(Fstat, P, normal_size -P)
    return anomalie_scores
def auc_over_time(y_test, y_pred_proba ,interval):
    break_points = np.arange(0,y_test.shape[0], interval)
    aucs = []
    for b in range(len(break_points)-1):
        if b!=len(break_points):
            y_test_temp = y_test[break_points[b]:break_points[b+1]]
            y_pred_proba_temp = y_pred_proba[break_points[b]:break_points[b+1]]
        else:
            y_test_temp = y_test[break_points[b]:y_test.shape[0]]
            y_pred_proba_temp = y_pred_proba[break_points[b]:y_test.shape[0]]
        aucs.append(evaluate.AUC(y_pred_proba_temp, y_test_temp))
    return aucs

########################## Intrusion detection #######################
X_train,X_test,y_test = load_db.load_intrusion(path)
num_of_experiments = 10
y_test_binary = np.array([0 if x == 0 else 1 for x in y_test.tolist()])

##### CDD #####
hopkins = Hopkins(X_train)
aucs_cdd_intrusion = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random=r)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    aucs_cdd_intrusion[r] = evaluate.AUC(y_pred_proba, y_test_binary)
aucs_cdd_intrusion = np.mean(aucs_cdd_intrusion)
##### AutoEncoder #####
aucs_ae_intrusion = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    AE = AutoEncoder(random_state = r)
    AE.fit(X_train)
    ae_pred_proba = AE.predict_proba(X_test)[:,1]
    aucs_ae_intrusion[r] = evaluate.AUC(ae_pred_proba,y_test_binary)
auc_ae_intrusion = np.mean(aucs_ae_intrusion)
##### Isolation forest #####
aucs_if_intrusiom = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    IF = IsolationForest(random_state=r)
    IF.fit(X_train)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    aucs_if_intrusiom[r] = evaluate.AUC(original_paper_score,y_test_binary)
auc_if_intrusion = np.mean(aucs_if_intrusiom)
#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm_intrusion = evaluate.AUC(original_paper_score,y_test_binary)

#### LOF ####
lof = LocalOutlierFactor(novelty=True, n_neighbors=200)
lof.fit(X_train)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof_intrusion = evaluate.AUC(original_paper_score,y_test_binary)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train,X_test)
auc_hot_intrusion = evaluate.AUC(y_pred_proba_hot,y_test_binary)

##### HalfSpaceTrees #####
X_train_hst, X_test_hst, _ = load_db.load_intrusion(path, is_hst=True)
aucs_hst_intrusion = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    hst = HalfSpaceTrees(n_features=X_train.shape[1],window_size=20000, n_estimators=200, depth=15, size_limit=50)
    hst.fit(X_train_hst, np.zeros(X_train_hst.shape[0]))
    y_pred_proba_hst = np.zeros(X_test_hst.shape[0])
    for i in tqdm(range(X_test_hst.shape[0])):
        hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
        y_pred_proba_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
    auc_hst_intrusion = evaluate.AUC(y_pred_proba_hst, y_test_binary)
    aucs_hst_intrusion[r] = auc_hst_intrusion
auc_hst_intrusion = np.mean(aucs_hst_intrusion)

##### LODA #####
aucs_loda_intrusion = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    loda = LODA(n_bins=100, n_random_cuts=500, contamination=0.3)
    loda.fit(X_train)
    y_pred_proba_loda = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        loda.fit(X_test[i, :].reshape(1, -1))
        y_pred_proba_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
    aucs_loda_intrusion[r] = evaluate.AUC(1-y_pred_proba_loda, y_test_binary)
auc_loda_intrusion = np.mean(aucs_loda_intrusion)

########################## Fraud detection #######################
X_train, X_test, y_test = load_db.load_fraud(path)
##### CDD #####
hopkins = Hopkins(X_train)
aucs_cdd_fraud = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random=r)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    aucs_cdd_fraud[r] = evaluate.AUC(y_pred_proba, y_test)
auc_cdd_fraud = np.mean(aucs_cdd_fraud)

##### AutoEncoder #####
aucs_ae_fraud = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    AE = AutoEncoder(random_state=r)
    AE.fit(X_train)
    ae_pred_proba = AE.predict_proba(X_test)[:,1]
    aucs_ae_fraud[r] = evaluate.AUC(ae_pred_proba, y_test)
auc_ae_fraud = np.mean(aucs_ae_fraud)

##### Isolation forest #####
aucs_if_fraud = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    IF = IsolationForest(random_state=r)
    IF.fit(X_train)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    aucs_if_fraud[r] = evaluate.AUC(original_paper_score,y_test)
auc_if_fraud = np.mean(aucs_if_fraud)

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm_fraud = evaluate.AUC(original_paper_score, y_test)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof_fraud = evaluate.AUC(original_paper_score, y_test)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train,X_test)
auc_hot_fraud = evaluate.AUC(y_pred_proba_hot,y_test)

##### HalfSpaceTrees #####
X_train_hst, X_test_hst, y_test = load_db.load_fraud(path, is_hst=True)
aucs_hst_fraud = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    y_pred_proba_hst = np.zeros(X_test_hst.shape[0])
    hst = HalfSpaceTrees(n_features=X_train_hst.shape[1], random_state=r, depth=10, window_size=2000)
    hst.fit(X_train_hst, np.zeros(X_train_hst.shape[0]))
    for i in tqdm(range(X_test_hst.shape[0])):
        hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
        y_pred_proba_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:,1]
    aucs_hst_fraud[r] = evaluate.AUC(y_pred_proba_hst, y_test)
auc_hst_fraud = np.mean(aucs_hst_fraud)

##### LODA #####
aucs_loda_fraud = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    loda = LODA(contamination=0.3)
    loda.fit(X_train)
    y_pred_proba_loda = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        loda.fit(X_test[i, :].reshape(1, -1))
        y_pred_proba_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
    aucs_loda_fraud[r] = evaluate.AUC(1-y_pred_proba_loda, y_test_binary)
auc_loda_fraud = np.mean(aucs_loda_fraud)

########################## Page-blocks #######################
pb = pd.read_csv(r'C:\Users\97254\Desktop\page-blocks.csv')
X = np.array(pb.drop('class', axis=1))
y = np.array([1 if a != 1 else 0 for a in pb['class'].tolist()])
X, y = shuffle(X, y, random_state=0)
train_inds = random.sample(np.where(y == 0)[0].tolist(), 200)
test_inds = [ind for ind in range(y.shape[0]) if ind not in train_inds]
X_train = X[train_inds, :]
X_test = X[test_inds, :]
y_test = y[test_inds]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.copy())
X_test = scaler.transform(X_test.copy())

##### CDD #####
hopkins = Hopkins(X_train)
aucs_cdd_pb = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    cdd = CDD(random=r)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    aucs_cdd_pb[r] = evaluate.AUC(y_pred_proba, y_test)
auc_cdd_pb = np.mean(aucs_cdd_pb)

##### AutoEncoder #####
aucs_ae_pb = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    AE = AutoEncoder(random_state=r, hidden_neurons=[64, 10, 10, 64])
    AE.fit(X_train)
    ae_pred_proba = AE.predict_proba(X_test)[:, 1]
    aucs_ae_pb[r] = evaluate.AUC(ae_pred_proba, y_test)
auc_ae_pb = np.mean(aucs_ae_pb)

##### Isolation forest #####
aucs_if_pb = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    IF = IsolationForest(random_state=r)
    IF.fit(X_train)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    aucs_if_pb[r] = evaluate.AUC(original_paper_score,y_test)
auc_if_pb = np.mean(aucs_if_pb)

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm_pb = evaluate.AUC(original_paper_score, y_test)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof_pb = evaluate.AUC(original_paper_score,y_test)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train, X_test)
auc_hot = evaluate.AUC(y_pred_proba_hot, y_test)

##### HalfSpaceTrees #####
mm = MinMaxScaler()
X_train_hst = mm.fit_transform(X_train)
mm = MinMaxScaler()
X_test_hst = mm.fit_transform(X_test)
aucs_hst = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    hst = HalfSpaceTrees(n_features=X_train_hst.shape[1], random_state=r)
    hst.fit(X_train_hst, np.zeros(X_train_hst.shape[0]))
    y_pred_proba_hst = np.zeros(X_test_hst.shape[0])
    for i in tqdm(range(X_test_hst.shape[0])):
        hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
        y_pred_proba_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
    aucs_hst[r] = evaluate.AUC(y_pred_proba_hst, y_test)
auc_hst = np.mean(aucs_hst)

##### LODA #####
aucs_loda_pb = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    loda = LODA()
    loda.fit(X_train)
    y_pred_proba_loda = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        loda.fit(X_test[i, :].reshape(1, -1))
        y_pred_proba_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
    aucs_loda_pb[r] = evaluate.AUC(1 - y_pred_proba_loda, y_test)
auc_loda_pb = np.mean(aucs_loda_pb)

########################## setellite #######################
sett = pd.read_csv(r'C:\Users\97254\Desktop\setellite.csv')
X = np.array(sett.drop('Target', axis=1))
y = np.array([1 if a == "'Anomaly'" else 0 for a in sett['Target'].tolist()])
X, y = shuffle(X, y, random_state=0)
train_inds = random.sample(np.where(y == 0)[0].tolist(), 200)
test_inds = [ind for ind in range(y.shape[0]) if ind not in train_inds]
X_train = X[train_inds, :]
X_test = X[test_inds, :]
y_test = y[test_inds]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.copy())
X_test = scaler.transform(X_test.copy())

##### CDD #####
hopkins = Hopkins(X_train)
aucs_cdd_sett = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    cdd = CDD(random=r)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    aucs_cdd_sett[r] = evaluate.AUC(y_pred_proba, y_test)
auc_cdd_sett = np.mean(aucs_cdd_sett)

##### AutoEncoder #####
aucs_ae_sett = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    AE = AutoEncoder(random_state=r)
    AE.fit(X_train)
    ae_pred_proba = AE.predict_proba(X_test)[:, 1]
    aucs_ae_sett[r] = evaluate.AUC(ae_pred_proba, y_test)
auc_ae_sett = np.mean(aucs_ae_sett)

##### Isolation forest #####
aucs_if_sett = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    IF = IsolationForest(random_state=r)
    IF.fit(X_train)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    aucs_if_sett[r] = evaluate.AUC(original_paper_score,y_test)
auc_if_sett = np.mean(aucs_if_sett)

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm_sett = evaluate.AUC(original_paper_score,y_test)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof_sett = evaluate.AUC(original_paper_score,y_test)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train, X_test)
auc_hot = evaluate.AUC(y_pred_proba_hot, y_test)

##### HalfSpaceTrees #####
mm = MinMaxScaler()
X_train_hst = mm.fit_transform(X_train)
mm = MinMaxScaler()
X_test_hst = mm.fit_transform(X_test)
aucs_hst_sett = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    hst = HalfSpaceTrees(n_features=X_train_hst.shape[1], random_state=r)
    hst.fit(X_train_hst, np.zeros(X_train_hst.shape[0]))
    y_pred_proba_hst = np.zeros(X_test_hst.shape[0])
    for i in tqdm(range(X_test_hst.shape[0])):
        hst.fit(X_test_hst[i, :].reshape(1, -1), np.array(0).reshape(1, -1))
        y_pred_proba_hst[i] = hst.predict_proba(X_test_hst[i, :].reshape(1, -1))[:, 1]
    aucs_hst_sett[r] = evaluate.AUC(y_pred_proba_hst, y_test)
auc_hst_sett = np.mean(aucs_hst_sett)

##### LODA #####
aucs_loda_sett = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    loda = LODA()
    loda.fit(X_train)
    y_pred_proba_loda = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        loda.fit(X_test[i, :].reshape(1, -1))
        y_pred_proba_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
    aucs_loda_sett[r] = evaluate.AUC(1 - y_pred_proba_loda, y_test)
auc_loda = np.mean(aucs_loda_sett)













