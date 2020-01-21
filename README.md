# Cluster-drift-detection-CDD-

Cluster drift detection (CDD) is an online novelty detection algorithm. The CDD algorithm returns for each test example whether it comes from the normal (one-class) distribution or novel, and the probability that it is anomalous (novel). Parameters for the run: max_k - limit on the number of clusters; update - how frequently to update the normal profile clusters, and the update parameter, pi, which is the proportion of new examples added to a cluster that requires its update. For example, if update = 0.1, and an arbitrary cluster has 550 samples, we will update it statistics after 55 samples were associated to it. The default value is 0.1. alpha - the significant level for the statistical test to determine whether a sample statistically fits a cluster. The default value is 0.05.

## In Python:
``` python
import CDD
cdd = CDD(max_k = 10)
cdd.fit(X_train)
y_pred_proba = cdd.predict_proba(X_test,update = 0.1,alpha = 0.05)
y_pred = cdd.predict(X_test,update = 0.1,alpha = 0.05)
```




