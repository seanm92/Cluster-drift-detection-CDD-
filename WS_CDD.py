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

ws = pd.read_csv(r'C:\Users\97254\Desktop\Phenomix\WS.csv')
ws['size diff'] = 0
for d in range(1,17):
    for i in range(120):
          ws['size diff'][i+120*d] = ws['Size'][i+120*d] - ws['Size'][i+120*(d-1)] 

ws['temp diff'] = ws['temp']-ws['AVGtemp']/100
ws['temp diff1'] = ws['temp']-ws['MAXtemp']/100


ws = ws[['Treatment','SPAD','WaterLeft', 'Ecleft','size diff', 'temp diff','temp diff1']]

X = np.array(ws[['SPAD','WaterLeft', 'Ecleft','size diff', 'temp diff','temp diff1']])
y = np.array(ws['Treatment'])

y = np.array([0 if a == 'A' else 1 for a in y.tolist()])
X_train = X[:360,:]
X_test = X[360:,:]
y_test = y[360:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.copy())
X_test = scaler.transform(X_test.copy()) 
num_of_experiments = 10
##### CDD #####
aucs = []
tnrs = []
drs = []
for r in tqdm(range(num_of_experiments)):
    cdd = CDD(random = r)
    cdd.fit(X_train)
    y_pred = cdd.predict(X_test)
    y_pred_proba = cdd.predict_proba(X_test)
    tnr,dr =  evaluate.evaluate_binary(y_pred,y_test)
    auc = evaluate.AUC(y_pred_proba,y_test)
    tnrs.append(tnr)
    drs.append(dr)
    aucs.append(auc)

aucs_cdd = np.mean(aucs)
tnr_cdd = np.mean(tnrs)    
dr_cdd = np.mean(drs)  

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
    tnr,dr =  evaluate.evaluate_binary(if_pred,y_test)
    auc = evaluate.AUC(original_paper_score,y_test)
    tnrs2.append(tnr)
    drs2.append(dr)
    aucs2.append(auc)

aucs_if = np.mean(aucs2)
tnr_if = np.mean(tnrs2)    
dr_if = np.mean(drs2)  

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
auc_hot = evaluate.AUC(y_pred_proba_hot,y_test)
tnr_hot,dr_hot = evaluate.evaluate_binary(y_pred_hot,y_test)

##### AutoEncoder #####
aucs1 = []
tnrs1 = []
drs1 = []

for r in range(num_of_experiments):
    AE = AutoEncoder(hidden_neurons = [64,6,6,64],random_state = r)
    AE.fit(X_train)
    ae_pred = AE.predict(X_test)
    ### assign normal to 1 and anomaly to -1, as all other algorithms
    ae_pred+=1
    ae_pred[ae_pred==2] = -1
    ae_pred_proba = AE.predict_proba(X_test)[:,1]
    tnr,dr =  evaluate.evaluate_binary(ae_pred,y_test)
    auc = evaluate.AUC(ae_pred_proba,y_test)
    tnrs1.append(tnr)
    drs1.append(dr)
    aucs1.append(auc)

aucs_ae = np.mean(aucs1)
tnr_ae = np.mean(tnrs1)    
dr_ae = np.mean(drs1) 

#### SVM ####
clf = svm.OneClassSVM(kernel="rbf")
clf.fit(X_train)
svm_pred = clf.predict(X_test)
sklearn_score_anomalies = clf.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_svm = evaluate.AUC(original_paper_score,y_test)
tnr_svm,dr_svm = evaluate.evaluate_binary(svm_pred,y_test)


#### LOF ####
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
lof_pred = lof.predict(X_test)
sklearn_score_anomalies = lof.decision_function(X_test)
original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
auc_lof = evaluate.AUC(original_paper_score,y_test)
tnr_lof,dr_lof = evaluate.evaluate_binary(lof_pred,y_test)


def prop_test(p1,p2,n):
    p = np.mean([p1,p2]) 
    return (p1-p2)/np.sqrt(2*p*(1-p)/n) > 1.645


from sklearn.metrics import silhouette_score
from itertools import repeat
from sklearn.mixture import GaussianMixture


max_k = 10
scores = np.zeros(max_k-1)
for k in range(2,max_k+1):
    gmm = GaussianMixture(n_components =k, random_state=0).fit(X_train)
    preds = gmm.predict(X_train)
    score = silhouette_score (X_train, preds, metric='euclidean')
    scores[k-2] = score

def get_clusters_data(data,K,random = 0):
    #this function estimates for each cluster its center,covarince matrix and number of exeampls
    #assume data is scaled
    gmm = GaussianMixture(n_components =K, random_state = random)
    labels = gmm.fit_predict(data)
    lst = [[] for i in repeat(None, K)]
    for k in range(K):
        temp_data = data[labels==k,:]
        mean = np.mean(temp_data,axis = 0)
        cov = np.cov(np.transpose(temp_data))
        m = temp_data.shape[0]
        lst[k].extend([mean,cov,m])
    return lst

tnr_by_k = np.zeros(max_k-1)
dr_by_k = np.zeros(max_k-1)
auc_by_k = np.zeros(max_k-1)
for k in range(2,max_k+1):
    aucs = []
    drs = []
    tnrs = []
    for r in tqdm(range(num_of_experiments)):
        cdd.Clusters_data = get_clusters_data(X_train,k,r)
        y_pred = cdd.predict(X_test)
        y_pred_proba = cdd.predict_proba(X_test)
        tnr,dr =  evaluate.evaluate_binary(y_pred,y_test)
        auc = evaluate.AUC(y_pred_proba,y_test)
        tnrs.append(tnr)
        drs.append(dr)
        aucs.append(auc)

    tnr_by_k[k-2] = np.mean(tnrs)
    dr_by_k[k-2] = np.mean(drs)    
    auc_by_k[k-2] = np.mean(aucs) 



updates = [0.01,0.05,0.1,0.2,0.5,0.8,0.9,1]
tnr_by_update = np.zeros(len(updates))
dr_by_update = np.zeros(len(updates))
auc_by_update = np.zeros(len(updates))

for ind,update in enumerate(updates):
    aucs = []
    drs = []
    tnrs = []
    for r in tqdm(range(num_of_experiments)):
        cdd = CDD(random = r)
        cdd.fit(X_train)
        y_pred = cdd.predict(X_test,update = update)
        cdd = CDD(random = r)
        y_pred_proba = cdd.predict_proba(X_test,update = update)
        tnr,dr =  evaluate.evaluate_binary(y_pred,y_test)
        auc = evaluate.AUC(y_pred_proba,y_test)
        tnrs.append(tnr)
        drs.append(dr)
        aucs.append(auc)
    tnr_by_update[ind] = np.mean(tnrs)
    dr_by_update[ind] = np.mean(drs)    
    auc_by_update[ind] = np.mean(aucs)     

alphas = [0.01,0.05,0.1,0.2,0.5,0.8]
tnr_by_alpha = np.zeros(len(alphas))
dr_by_alpha = np.zeros(len(alphas))
auc_by_alpha = np.zeros(len(alphas))

for ind,alpha in enumerate(alphas):
    aucs = []
    drs = []
    tnrs = []
    for r in tqdm(range(num_of_experiments)):
        cdd = CDD(random = r)
        cdd.fit(X_train)
        y_pred = cdd.predict(X_test,alpha = alpha)
        cdd = CDD(random = r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test,alpha = alpha)
        tnr,dr =  evaluate.evaluate_binary(y_pred,y_test)
        auc = evaluate.AUC(y_pred_proba,y_test)
        tnrs.append(tnr)
        drs.append(dr)
        aucs.append(auc)
    tnr_by_alpha[ind] = np.mean(tnrs)
    dr_by_alpha[ind] = np.mean(drs)    
    auc_by_alpha[ind] = np.mean(aucs)

tnr_by_alpha_hot = np.zeros(len(alphas))
dr_by_alpha_hot = np.zeros(len(alphas))
auc_by_alpha_hot = np.zeros(len(alphas))

for ind,alpha in enumerate(alphas):    
    y_pred_proba_hot = Hoteliing_SPC_proba(X_train,X_test,alpha = alpha)
    y_pred_hot = Hoteliing_SPC(X_train,X_test,alpha = alpha)
    y_pred_hot+=1
    y_pred_hot[y_pred_hot==2] = -1
    auc_hot = evaluate.AUC(y_pred_proba_hot,y_test)
    tnr_hot,dr_hot = evaluate.evaluate_binary(y_pred_hot,y_test)
    tnr_by_alpha_hot[ind] = tnr_hot
    dr_by_alpha_hot[ind] = dr_hot
    auc_by_alpha_hot[ind] = auc_hot

import matplotlib.pyplot as plt
plt.plot(alphas,tnr_by_alpha_hot)
plt.plot(alphas,dr_by_alpha_hot)
plt.plot(alphas,auc_by_alpha_hot)
plt.legend(['TNR','DR','AUC'],loc = 'lower left')
plt.xlabel('alpha',fontsize = 10)
plt.ylabel('Preformance ',fontsize = 10)
plt.title('Hotelling preformance')

plt.plot(alphas,tnr_by_alpha)
plt.plot(alphas,dr_by_alpha)
plt.plot(alphas,auc_by_alpha)
plt.legend(['TNR','DR','AUC'],loc = 'lower left')
plt.xlabel('alpha',fontsize = 10)
plt.ylabel('Preformance ',fontsize = 10)
plt.title('CDD preformance')


auc_cdd = auc_by_alpha[0]









