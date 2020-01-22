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




X_train,X_test,y_test = load_db.load_intrusion(path)
y_test_binary = np.array([0 if x == 0 else 1 for x in y_test.tolist()])

###################### examine CDD results for diffrent number of clusters with the  intrusion detetction data set ########################
def get_clusters_data(data,K,random = 0):
    #this function estimates for each cluster its center,covarince matrix and number of exeampls
    #assume data is scaled
    gmm = GaussianMixture(n_components =K, random_state=random)
    labels = gmm.fit_predict(data)
    lst = [[] for i in repeat(None, K)]
    for k in range(K):
        temp_data = data[labels==k,:]
        mean = np.mean(temp_data,axis = 0)
        cov = np.cov(np.transpose(temp_data))
        m = temp_data.shape[0]
        lst[k].extend([mean,cov,m])
    return lst

max_k = 10
sil_scores = np.zeros(max_k-1)
aucs = np.zeros(max_k-1)
for k in range(2,max_k+1):
    gmm = GaussianMixture(n_components = k, random_state = 0)
    preds = gmm.fit_predict(X_train)
    score = silhouette_score(X_train, preds, metric='euclidean')
    sil_scores[k-2] = score
    fitted = get_clusters_data(X_train,k)
    y_pred_proba = CDD.predict_proba(fitted, X_test)
    fpr, tpr, thresholds = roc_curve(y_test_binary,y_pred_proba)
    auc_cdd = auc(fpr, tpr)
    aucs[k-2] = auc_cdd

K = list(range(2,max_k+1))
color = 'tab:red'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('AUC', color=color)
ax1.plot(K, aucs, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(axis='x', alpha=0.5)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Silhouette coefficient', color=color)  # we already handled the x-label with ax1
ax2.plot(K, sil_scores, color=color)
ax2.tick_params(axis='y', labelcolor=color)    
ax2.grid(axis='x', alpha=0.5)
 
########################### Hyper - parameters analysis with the fraud detection data set ########################
X_train1,X_test1,y_test1 = load_db.load_fraud(path)
###### Significant level effect #######
alphas = [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
scores_by_alpha_TN = np.zeros(len(alphas))
scores_by_alpha_DR = np.zeros(len(alphas))

cdd = CDD()
cdd.fit(X_train1)
for ind,alpha in enumerate(tqdm(alphas)):
    fraud_pred = cdd.predict(X_test1,update = 0.1,alpha = alpha)
    TN,DR = evaluate.evaluate_binary(fraud_pred, y_test1)
    scores_by_alpha_TN[ind] = TN
    scores_by_alpha_DR[ind] = DR
    
plt.plot(alphas,scores_by_alpha_DR)
plt.plot(alphas,scores_by_alpha_TN)  
plt.legend(['DR', 'TNR'], loc='lower left')
plt.xlabel(r'$\alpha$')
plt.ylabel('TNR\DR')
plt.title(r'$\alpha$ effect')
plt.grid(alpha = 0.5)

###### update parameter effect #######
updates = [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]

times_by_update = np.zeros(len(updates))
scores_by_update_auc = np.zeros(len(updates))
for ind,update in enumerate(tqdm(updates)):
    start = time.time()
    y_pred_proba = cdd.predict_proba(X_test1,update = update,alpha = 0.05)
    times_by_update[ind] = time.time()-start
    scores_by_update_auc[ind] = evaluate.AUC(y_pred_proba, y_test1)

plt.plot(updates,scores_by_update_auc)
plt.xlabel(r'$\pi$')
plt.ylabel('AUC')
plt.title('Update parameter effect')
plt.grid(alpha = 0.5)


    
    