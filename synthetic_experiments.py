import sys
path = r'C:\Users\97254\Desktop\CDD_code'
sys.path.append(path)
from utils import evaluate
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import svm
from tqdm import tqdm
import numpy as np
from CDD import CDD
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import inv,matrix_rank,pinv
from scipy.stats import f
from pyod.models.auto_encoder import AutoEncoder 
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

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

########################### diffrent number of clusters ############################
def number_of_clusters_examine(n_clusters):
    num_of_experiments = 10
    training_size = 1000
    total_test_size = 10000
    anomaly_fraction = 0.01
    normal_test_size = int(total_test_size*(1-anomaly_fraction))
    anomaly_size = total_test_size - normal_test_size 
    
    centers = [[2, 2], [-2, -2], [2, -2],[-2,2],[6,6],[-6,-6],[6,-6],[-6,6]]
    current_centers = centers[:n_clusters]
    X_train = make_blobs(n_samples=training_size, centers=current_centers, cluster_std=0.5,random_state=0)[0]
    X_test_normal = make_blobs(n_samples=normal_test_size, centers=current_centers, cluster_std=0.5,random_state=0)[0]
    X_test_noval = np.random.RandomState(0).uniform(low = -8, high = 8, size=(anomaly_size, 2))
    X_test = np.concatenate((X_test_normal,X_test_noval),axis = 0)
    y_test = np.array([0 if x<normal_test_size else 1 for x in range(total_test_size)])
    X_test,y_test = shuffle(X_test,y_test,random_state =0)

     ############## CDD #################
    auc_cdd_ls = []
    for r in range(num_of_experiments):
        cdd = CDD(random = r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test)
        auc_cdd_ls.append(evaluate.AUC(y_pred_proba,y_test))
    auc_CDD = np.mean(auc_cdd_ls)
    ############# AE ##################
    auc_ae_ls = []
    for r in range(num_of_experiments):
        AE = AutoEncoder(hidden_neurons = [64,2,2,64],random_state = r)
        AE.fit(X_train)
        ae_pred_proba = AE.predict_proba(X_test)[:,1]
        auc_ae_ls.append(evaluate.AUC(ae_pred_proba,y_test))
    auc_AE = np.mean(auc_ae_ls)
    
    ############ Iforest ##############
    auc_if_ls = []
    for r in range(num_of_experiments):
        IF = IsolationForest(random_state = r)
        IF.fit(X_train)  
        sklearn_score_anomalies = IF.decision_function(X_test)
        original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
        auc_if_ls.append(evaluate.AUC(original_paper_score,y_test))
    auc_IF = np.mean(auc_if_ls)
    
    ########### OCSVM ###############
    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(X_train)
    sklearn_score_anomalies = clf.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_SVM = evaluate.AUC(original_paper_score,y_test)
    
    ########### LOF #############
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    sklearn_score_anomalies = lof.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_LOF = evaluate.AUC(original_paper_score,y_test)
    
    ########### T2 #############
    y_pred_hot = Hoteliing_SPC_proba(X_train,X_test)
    auc_hot = evaluate.AUC(y_pred_hot,y_test)
    return [auc_CDD,auc_AE,auc_IF,auc_SVM,auc_LOF,auc_hot]

Nclusters = range(1,9)
cdd_results = np.zeros(len(Nclusters))
ae_results = np.zeros(len(Nclusters))
iforest_results = np.zeros(len(Nclusters))
svm_results = np.zeros(len(Nclusters))
lof_results = np.zeros(len(Nclusters))
hot_results = np.zeros(len(Nclusters))
                       
for ind,n_clusters in enumerate(tqdm(Nclusters)):
    results = number_of_clusters_examine(n_clusters = n_clusters)
    cdd_results[ind] = results[0]
    ae_results[ind] = results[1]
    iforest_results[ind] = results[2]
    svm_results[ind] = results[3]
    lof_results[ind] = results[4]
    hot_results[ind] = results[5]

plt.plot(Nclusters,cdd_results,marker = '.')
plt.plot(Nclusters,ae_results,marker = '*')
plt.plot(Nclusters,iforest_results,marker = 'P')
plt.plot(Nclusters,svm_results,marker = '^')
plt.plot(Nclusters,lof_results,marker = 'D')
plt.plot(Nclusters,hot_results,marker = 'p')
plt.legend(['CDD','AE','Iforest','OCSVM','LOF','$T^2$ SPC'], loc = 'lower left')
plt.xlabel('Number of clusters',fontsize = 10)
plt.ylabel('AUC ',fontsize = 10)

##################################### concept drift #######################################
def concept_drift_comparison(drift_level = 0 ):
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
    X_test_noval = np.random.RandomState(0).uniform(low = -8, high = 8, size=(anomaly_size, 2))
    X_test = np.concatenate((X_test_normal,X_test_noval),axis = 0)
    y_test = np.array([0 if x<normal_test_size else 1 for x in range(total_test_size)])
    X_test,y_test = shuffle(X_test,y_test,random_state =0)

    ############## CDD #################
    auc_cdd_ls = []
    for r in range(num_of_experiments):  
        cdd = CDD(random = r)
        cdd.fit(X_train)
        y_pred_proba = cdd.predict_proba(X_test)
        auc_cdd_ls.append(evaluate.AUC(y_pred_proba,y_test))
    auc_CDD = np.mean(auc_cdd_ls) 
        
    ############# AE ##################
    auc_ae_ls = []
    for r in range(num_of_experiments):
        AE = AutoEncoder(hidden_neurons = [64,2,2,64],random_state = r)
        AE.fit(X_train)
        ae_pred_proba = AE.predict_proba(X_test)[:,1]
        auc_ae_ls.append(evaluate.AUC(ae_pred_proba,y_test))
    auc_AE = np.mean(auc_ae_ls)
    
    ############ Iforest ##############
    auc_if_ls = []
    for r in range(num_of_experiments):
        IF = IsolationForest(random_state = r)
        IF.fit(X_train)  
        sklearn_score_anomalies = IF.decision_function(X_test)
        original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
        auc_if_ls.append(evaluate.AUC(original_paper_score,y_test))
    auc_IF = np.mean(auc_if_ls)
    
    ########### OCSVM ###############
    clf = svm.OneClassSVM(kernel="rbf")
    clf.fit(X_train)
    sklearn_score_anomalies = clf.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_SVM = evaluate.AUC(original_paper_score,y_test)
    
    ########### LOF #############
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    sklearn_score_anomalies = lof.decision_function(X_test)
    original_paper_score = [-1*s + 0.5 for s in sklearn_score_anomalies]
    auc_LOF = evaluate.AUC(original_paper_score,y_test)
    
    ########### T2 #############
    y_pred_hot = Hoteliing_SPC_proba(X_train,X_test)
    auc_hot = evaluate.AUC(y_pred_hot,y_test)
    return [auc_CDD,auc_AE,auc_IF,auc_SVM,auc_LOF,auc_hot]
    
drift_levels = np.arange(start = 0,stop = 1.1,step = 0.1)
cdd_results = np.zeros(len(drift_levels))
ae_results = np.zeros(len(drift_levels))
iforest_results = np.zeros(len(drift_levels))
svm_results = np.zeros(len(drift_levels))
lof_results = np.zeros(len(drift_levels))
hot_results = np.zeros(len(drift_levels))
                       
for ind,drift_level in enumerate(tqdm(drift_levels)):
    results = concept_drift_comparison(drift_level = drift_level)
    cdd_results[ind] = results[0]
    ae_results[ind] = results[1]
    iforest_results[ind] = results[2]
    svm_results[ind] = results[3]
    lof_results[ind] = results[4]
    hot_results[ind] = results[5]

plt.plot(drift_levels,cdd_results,marker = '.')
plt.plot(drift_levels,ae_results,marker = '*')
plt.plot(drift_levels,iforest_results,marker = 'P')
plt.plot(drift_levels,svm_results,marker = '^')
plt.plot(drift_levels,lof_results,marker = 'D')
plt.plot(drift_levels,hot_results,marker = 'p')
plt.legend(['CDD','AE','Iforest','OCSVM','LOF','$T^2$ SPC'], loc = 'lower left')
plt.xlabel('Drift level',fontsize = 10)
plt.ylabel('AUC ',fontsize = 10)




                   


    
    


