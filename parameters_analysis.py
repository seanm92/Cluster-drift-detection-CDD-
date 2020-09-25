from utils import load_db, Hopkins, evaluate
from sklearn.metrics import auc
from sklearn.ensemble import IsolationForest
from sklearn import svm
from tqdm import tqdm
import numpy as np
from CDD import CDD
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


path = r'C:\Users\97254\Desktop\CDD\CDD_code'
X_train, X_test, y_test = load_db.load_intrusion(path)
y_test_binary = np.array([0 if x == 0 else 1 for x in y_test.tolist()])

# --- examine CDD results for diffrent number of clusters with the  intrusion detection data set --- #

max_k = 10
sil_scores = np.zeros(max_k-1)
aucs = np.zeros(max_k-1)
for k in range(2, max_k+1):
    gmm = GaussianMixture(n_components=k, random_state=0)
    preds = gmm.fit_predict(X_train)
    score = silhouette_score(X_train, preds, metric='euclidean')
    sil_scores[k-2] = score
    cdd = CDD(K=k)
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test)
    auc_cdd = evaluate.AUC(y_pred_proba, y_test_binary)
    aucs[k-2] = auc_cdd

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(111)
K = list(range(2, max_k+1))
color = 'tab:orange'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('AUC', color=color)
ax1.plot(K, aucs, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(axis='x', alpha=0.5)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Silhouette coefficient', color=color)  # we already handled the x-label with ax1
ax2.plot(K, sil_scores, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)    
ax2.grid(axis='x', alpha=0.5)
 
# ----- Hyper - parameters analysis with the fraud detection data set ----- #
X_train, X_test, y_test = load_db.load_fraud(path)
# --- Significant level effect --- #
alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
scores_by_alpha_TN = np.zeros(len(alphas))
scores_by_alpha_DR = np.zeros(len(alphas))
for ind, alpha in enumerate(tqdm(alphas)):
    cdd = CDD()
    cdd.fit(X_train)
    fraud_pred = cdd.predict(X_test, update=0.1, alpha=alpha)
    TN, DR = evaluate.evaluate_binary(fraud_pred, y_test)
    scores_by_alpha_TN[ind] = TN
    scores_by_alpha_DR[ind] = DR

plt.plot(alphas, scores_by_alpha_DR)
plt.plot(alphas, scores_by_alpha_TN)
plt.legend(['DR', 'TNR'], loc='lower left')
plt.xlabel(r'$\alpha$', fontsize=14)
plt.ylabel('TNR/DR', fontsize=12)
plt.title(r'$\alpha$ effect')
plt.grid(alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# --- update parameter effect --- #
updates = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
scores_by_update_auc = np.zeros(len(updates))
for ind, update in enumerate(tqdm(updates)):
    cdd = CDD()
    cdd.fit(X_train)
    y_pred_proba = cdd.predict_proba(X_test, update=update, alpha=0.05)
    scores_by_update_auc[ind] = evaluate.AUC(y_pred_proba, y_test)

plt.plot(updates, scores_by_update_auc)
plt.xlabel(r'$\pi$', fontsize=14)
plt.ylabel('AUC', fontsize=12)
plt.title('Update parameter effect')
plt.grid(alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
