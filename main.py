import sys
path = r'C:\Users\97254\Desktop\CDD_code'
sys.path.append(path)
from utils import load_db,Hopkins,evaluate
from sklearn.metrics import auc
from sklearn.ensemble import IsolationForest
from sklearn import svm
from tqdm import tqdm
import numpy as np
from CDD import CDD
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import inv,matrix_rank,pinv
from scipy.stats import f
from pyod.models.auto_encoder import AutoEncoder 
import scipy

########################## Intrusion detection #######################
X_train,X_test,y_test = load_db.load_intrusion(path)
num_of_experiments = 10
y_test_binary = np.array([0 if x == 0 else 1 for x in y_test.tolist()])

##### CDD #####
hopkins = Hopkins(X_train)
aucs = []
tnrs = []
drs = []

for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random = r)
    cdd.fit(X_train)
    y_pred = cdd.predict(X_test)
    y_pred_proba = cdd.predict_proba(X_test)
    tnr,dr =  evaluate.evaluate_binary(y_pred,y_test_binary)
    auc1 = evaluate.AUC(y_pred_proba,y_test_binary)
    tnrs.append(tnr)
    drs.append(dr)
    aucs.append(auc)

aucs_cdd = np.mean(aucs)
tnr_cdd = np.mean(tnrs)    
dr_cdd = np.mean(drs)    

##### AutoEncoder #####
aucs1 = []
tnrs1 = []
drs1 = []

for r in range(num_of_experiments):
    AE = AutoEncoder(random_state = r)
    AE.fit(X_train)
    ae_pred = AE.predict(X_test)
    ### assign normal to 1 and anomaly to -1, as all other algorithms
    ae_pred+=1
    ae_pred[ae_pred==2] = -1
    ae_pred_proba = AE.predict_proba(X_test)[:,1]
    tnr,dr =  evaluate.evaluate_binary(ae_pred,y_test_binary)
    auc1 = evaluate.AUC(ae_pred_proba,y_test_binary)
    tnrs1.append(tnr)
    drs1.append(dr)
    aucs1.append(auc)

aucs_ae = np.mean(aucs1)
tnr_ae = np.mean(tnrs1)    
dr_ae = np.mean(drs1) 

##### Isolation forest #####
aucs2 = []
tnrs2 = []
drs2 = []

for r in tqdm(range(num_of_experiments)):
    IF = IsolationForest(random_state = r)
    IF.fit(X_train)
    if_pred = IF.predict(X_test)
    sklearn_score_anomalies = IF.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    tnr,dr =  evaluate.evaluate_binary(if_pred,y_test_binary)
    auc1 = evaluate.AUC(original_paper_score,y_test_binary)
    tnrs2.append(tnr)
    drs2.append(dr)
    aucs2.append(auc)

aucs_if = np.mean(aucs2)
tnr_if = np.mean(tnrs2)    
dr_if = np.mean(drs2)     

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
svm_pred = clf.predict(X_test)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm = evaluate.AUC(original_paper_score,y_test_binary)
tnr_svm,dr_svm = evaluate.evaluate_binary(svm_pred,y_test_binary)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
lof_pred = lof.predict(X_test)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof = evaluate.AUC(original_paper_score,y_test_binary)
tnr_lof,dr_lof = evaluate.evaluate_binary(lof_pred,y_test_binary)

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
        Fstat = T2 * ((normal_size-P)*M)/(P*(normal_size-1)*(normal_size+1))
        anomalie_scores[i] = f.cdf(Fstat, P, normal_size -P)
    return anomalie_scores
def SPC(T2,M,N,alpha =0.05):
    F = scipy.stats.f.ppf(q=1-alpha, dfn=N, dfd=M-N)
    UCL = (N*(M-1)*(M+1)*F)/(M**2-M*N)
    return T2>UCL
def Hoteliing_SPC(normal_data,new_data,alpha =0.05):
    normal_mean = np.mean(normal_data,axis = 0)
    normal_cov = np.cov(np.transpose(normal_data))
    normal_size = normal_data.shape[0]
    M,P = new_data.shape
    anomalies = np.zeros(M)
    for i in range(M):
        obs = new_data[i,:]
        T2 = Hotelling(obs,normal_mean,normal_cov)
        anomalies[i] = SPC(T2,normal_size,P,alpha)
    return anomalies  

y_pred_proba_hot = Hoteliing_SPC_proba(X_train,X_test)
y_pred_hot = Hoteliing_SPC(X_train,X_test)
y_pred_hot+=1
y_pred_hot[y_pred_hot==2] = -1
auc_hot = evaluate.AUC(y_pred_proba_hot,y_test_binary)
tnr_hot,dr_hot = evaluate.evaluate_binary(y_pred_hot,y_test_binary)

########################## Fraud detection #######################
X_train1,X_test1,y_test1 = load_db.load_fraud(path)
num_of_experiments = 10

##### CDD #####
hopkins = Hopkins(X_train1)
aucs = []
tnrs = []
drs = []

for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random = r)
    cdd.fit(X_train1)
    y_pred = cdd.predict(X_test1)
    y_pred_proba = cdd.predict_proba(X_test1)
    tnr,dr =  evaluate.evaluate_binary(y_pred,y_test1)
    auc = evaluate.AUC(y_pred_proba,y_test1)
    tnrs.append(tnr)
    drs.append(dr)
    aucs.append(auc)

aucs_cdd = np.mean(aucs)
tnr_cdd = np.mean(tnrs)    
dr_cdd = np.mean(drs)    

##### AutoEncoder #####
aucs1 = []
tnrs1 = []
drs1 = []

for r in range(num_of_experiments):
    AE = AutoEncoder(random_state = r)
    AE.fit(X_train1)
    ae_pred = AE.predict(X_test1)
    ### assign normal to 1 and anomaly to -1, as all other algorithms
    ae_pred+=1
    ae_pred[ae_pred==2] = -1
    ae_pred_proba = AE.predict_proba(X_test1)[:,1]
    tnr,dr =  evaluate.evaluate_binary(ae_pred,y_test1)
    auc = evaluate.AUC(ae_pred_proba,y_test1)
    tnrs1.append(tnr)
    drs1.append(dr)
    aucs1.append(auc)

aucs_ae = np.mean(aucs1)
tnr_ae = np.mean(tnrs1)    
dr_ae = np.mean(drs1) 

##### Isolation forest #####
aucs2 = []
tnrs2 = []
drs2 = []

for r in tqdm(range(num_of_experiments)):
    IF = IsolationForest(random_state = r)
    IF.fit(X_train1)
    if_pred = IF.predict(X_test1)
    sklearn_score_anomalies = IF.decision_function(X_test1)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    tnr,dr =  evaluate.evaluate_binary(if_pred,y_test1)
    auc = evaluate.AUC(original_paper_score,y_test_binary)
    tnrs2.append(tnr)
    drs2.append(dr)
    aucs2.append(auc)

aucs_if = np.mean(aucs2)
tnr_if = np.mean(tnrs2)    
dr_if = np.mean(drs2)     

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train1)
svm_pred = clf.predict(X_test1)
sklearn_score_anomalies = clf.decision_function(X_test1)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm = evaluate.AUC(original_paper_score,y_test1)
tnr_svm,dr_svm = evaluate.evaluate_binary(svm_pred,y_test1)

#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train1)
lof_pred = lof.predict(X_test1)
sklearn_score_anomalies = lof.decision_function(X_test1)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof = evaluate.AUC(original_paper_score,y_test1)
tnr_lof,dr_lof = evaluate.evaluate_binary(lof_pred,y_test1)

#### T2 ####
y_pred_proba_hot = Hoteliing_SPC_proba(X_train1,X_test1)
y_pred_hot = Hoteliing_SPC(X_train1,X_test1)
y_pred_hot+=1
y_pred_hot[y_pred_hot==2] = -1
auc_hot = evaluate.AUC(y_pred_proba_hot,y_test1)
tnr_hot,dr_hot = evaluate.evaluate_binary(y_pred_hot,y_test1)








































