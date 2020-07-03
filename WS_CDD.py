import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
path = r'C:\Users\97254\Desktop\CDD_code'
sys.path.append(path)
from utils import evaluate
from CDD import CDD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import inv,matrix_rank,pinv
from scipy.stats import f
from pyod.models.auto_encoder import AutoEncoder 
import scipy
from sklearn import svm
from tqdm import tqdm
from pyod.models.loda import LODA
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.anomaly_detection import HalfSpaceTrees

#### T2 ####
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
ws = pd.read_csv(r'C:\Users\97254\Desktop\Phenomix\DBs\WS.csv')
ws['size diff'] = 0
for d in range(1, 17):
    for i in range(120):
          ws.at[i+120*d,'size diff'] = ws['Size'][i+120*d] - ws['Size'][i+120*(d-1)]

ws['temp diff'] = ws['temp']-ws['AVGtemp']/100
ws['temp diff1'] = ws['temp']-ws['MAXtemp']/100

ws = ws[['Treatment', 'SPAD', 'WaterLeft', 'Ecleft', 'size diff', 'temp diff', 'temp diff1']]
X = np.array(ws.drop('Treatment', axis=1))
y = np.array(ws['Treatment'])

y = np.array([0 if a == 'A' else 1 for a in y.tolist()])
X_train = X[:360, :]
X_test = X[360:, :]
y_test = y[360:]

#### input for hst
mm = MinMaxScaler()
X_train_hst = mm.fit_transform(X_train)
mm = MinMaxScaler()
X_test_hst = mm.fit_transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.copy())
X_test = scaler.transform(X_test.copy()) 
num_of_experiments = 10
##### CDD #####
aucs_cdd_ws = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random=r)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    aucs_cdd_ws[r] = evaluate.AUC(y_pred_proba, y_test)
auc_cdd_ws = np.mean(aucs_cdd_ws)

##### Isolation forest #####
aucs_if_ws = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    IF = IsolationForest(random_state=r)
    IF.fit(X_train)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    aucs_if_ws[r] = evaluate.AUC(original_paper_score,y_test)
auc_if_ws = np.mean(aucs_if_ws)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train,X_test)
auc_hot_ws = evaluate.AUC(y_pred_proba_hot,y_test)

##### AutoEncoder #####
aucs_ae_ws = np.zeros(num_of_experiments)
for r in range(num_of_experiments):
    AE = AutoEncoder(hidden_neurons=[64, 6, 6, 64], random_state=r)
    AE.fit(X_train)
    ae_pred_proba = AE.predict_proba(X_test)[:,1]
    aucs_ae_ws[r] = evaluate.AUC(ae_pred_proba,y_test)
auc_ae_ws = np.mean(aucs_ae_ws)

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm_ws = evaluate.AUC(original_paper_score,y_test)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof_ws = evaluate.AUC(original_paper_score,y_test)

#### LODA ####
aucs_loda_ws = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    loda = LODA()
    loda.fit(X_train)
    y_pred_proba_loda = np.zeros(X_test.shape[0])
    for i in tqdm(range(X_test.shape[0])):
        loda.fit(X_test[i, :].reshape(1, -1))
        y_pred_proba_loda[i] = loda.decision_function(X_test[i, :].reshape(1, -1))
    aucs_loda_ws[r] = evaluate.AUC(1-y_pred_proba_loda, y_test)
auc_loda_ws = np.mean(aucs_loda_ws)

##### HalfSpaceTrees #####
aucs_hst_ws = np.zeros(num_of_experiments)
for r in tqdm(range(num_of_experiments)):
    hst = HalfSpaceTrees(n_features=X_train_hst.shape[1], n_estimators=100)
    hst.fit(X_train_hst, np.zeros(X_train_hst.shape[0]))
    y_pred_proba_hst = np.zeros(X_test_hst.shape[0])
    for i in tqdm(range(X_test_hst.shape[0])):
        hst.fit(X_test_hst[i,:].reshape(1,-1), np.array(0).reshape(1,-1))
        y_pred_proba_hst[i] = hst.predict_proba(X_test_hst[i,:].reshape(1,-1))[:,1]
    auc_hst_ws = evaluate.AUC(y_pred_proba_hst, y_test)
    aucs_hst_ws[r] = auc_hst_ws
auc_hst_ws = np.mean(aucs_hst_ws)














