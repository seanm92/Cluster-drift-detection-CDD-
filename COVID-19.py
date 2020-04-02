import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
path = r'C:\Users\97254\Desktop\CDD_code'
sys.path.append(path)
from utils import evaluate
from CDD import CDD

from numpy.linalg import inv,matrix_rank,pinv
from sklearn.model_selection import train_test_split

data_path = r'C:\Users\97254\Desktop\Covid_19'

covid = pd.read_csv(data_path+'\COVID19_open_line_list.csv')
covid = covid[['age', 'sex','outcome','symptoms','chronic_disease_binary','date_confirmation','date_onset_symptoms']]
covid['chronic_disease_binary'] = covid['chronic_disease_binary'].fillna(0) 

covid = covid[covid['age'].notna()]
covid = covid[covid['sex'].notna()]
covid = covid[covid.outcome != 'Symptoms only improved with cough. Currently hospitalized for follow-up.']
covid = covid[covid.outcome != '05.02.2020']


covid = covid[['age', 'sex','outcome','symptoms','chronic_disease_binary']]
#### process output ####     
covid['outcome'] = covid['outcome'].fillna(0) 
covid['outcome'] = covid['outcome'].replace(['died','death'],1)
covid['outcome'] = covid['outcome'].replace(['treated in an intensive care unit (14.02.2020)', 'severe'],2)
covid['outcome'] = covid['outcome'].replace(['discharged', 'discharge','Discharged','recovered'], 3)    
covid['outcome'] = covid['outcome'].replace(['stable'], 4)    
  
unique_words = {}
for i, val in covid.symptoms.iteritems():
    words = str(val).lower().split(" ")
    for w in words:
        if w not in unique_words:
            unique_words[w] =1
        else:
            unique_words[w] +=1

#### process symptoms ####       
covid['is_fever'] = covid.symptoms.str.contains('fever', na=False,case =False ).astype(int)
covid['is_cough'] = covid.symptoms.str.contains('cough', na=False,case =False ).astype(int)
covid['is_fatigue'] = covid.symptoms.str.contains('fatigue', na=False,case =False ).astype(int)
del covid['symptoms']
covid = covid[covid.age != 'Belgium']
covid = covid.reset_index(drop=True)
  
for i in range(covid.shape[0]):
    try:
        covid['age'][i] = float(covid['age'][i])
    except:
        covid['age'][i] = np.mean([float(x) for x in covid['age'][i].split('-')])   

le = LabelEncoder()
covid['sex'] = le.fit_transform(covid['sex'].str.lower())

X = np.float32(covid[['age', 'sex','chronic_disease_binary','is_fever','is_cough','is_fatigue']].values)
y = covid['outcome'].values
X = X[y!=2,:]
y = y[y!=2]
X = X[y!=4,:]
y = y[y!=4]


X_train = X[y==0,:]
X_test = X[y!=0,:]
y_test = y[y!=0]

scaler = StandardScaler()
X_train[:,0] = scaler.fit_transform(X_train[:,0].reshape((-1,1))).reshape(-1)           
X_test[:,0] = scaler.transform(X_test[:,0].reshape((-1,1))).reshape(-1)    

X_val, X_test1, y_val, y_test1 = train_test_split(X_test.copy(), y_test.copy(), test_size=0.5, random_state=0)

X_val_normal = X_val[y_val == 3,:]
X_val_novel = X_val[y_val == 1,:] 
y_val_normal = y_val[y_val == 3]
y_val_novel = y_val[y_val == 1]
y_test1 = np.array([0 if a == 3 else 1 for a in y_test.tolist()])

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
def compute_mean_dist_to_cluster(cluster_data,X):
    return np.mean([Hotelling(X[i,:],cluster_data[0],cluster_data[1]) for i in range(X.shape[0])])
def update_clusters_data(random = 0):
    cdd = CDD(random = random)
    cdd.fit(X_train)
    lst = cdd.Clusters_data
    normal_inds = []
    for c in range(len(lst)):
        dist_normal = compute_mean_dist_to_cluster(lst[c],X_val_normal)
        dist_novel = compute_mean_dist_to_cluster(lst[c],X_val_novel)
        if dist_normal<dist_novel:
            normal_inds.append(c)

    new_lst = []
    for i in range(len(lst)):
        if i in normal_inds:
           new_lst.append(lst[i]) 
    cdd.Clusters_data = new_lst
    return cdd       

cdd = update_clusters_data()
y_pred_proba = cdd.predict_proba(X_test,alpha = 0.8)
auc_cdd = evaluate.AUC(y_pred_proba,y_test1)
cdd = update_clusters_data(random = 0)
y_pred = cdd.predict(X_test,alpha = 0.8)
tnr_cdd,dr_cdd = evaluate.evaluate_binary(y_pred, y_test1)

    
































