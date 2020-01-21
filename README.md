# Cluster-drift-detection-CDD-

CDD is a streaming novelty detection algorithm.
The CDD algorithm returns for each test example whether it is normal (produced from the training set distrbution, or novel),
and the probabilty that the example is anomaly.
Parameters:
max_k:limitation for the number of clusters.
update:how frequently to update the normal profile clusters, the update parameter is the propotion of new examples added to a cluster that requier 
its update. For example, if update = 0.1 and an arbitrary cluster has 550 observations, we will update it statistics after 55 associated to it.
The default value is 0.1.
alpha:the significant level for the statistics test to detrmain whether an observation statistically fit a cluster.
The default value is 0.05.

##Code example with python:
```
import CDD
cdd = CDD(max_k = 10)
cdd.fit(X_train)
y_pred_proba = cdd.predict_proba(X_test,update = 0.1,alpha = 0.05)
y_pred = cdd.predict(X_test,update = 0.1,alpha = 0.05)
```


