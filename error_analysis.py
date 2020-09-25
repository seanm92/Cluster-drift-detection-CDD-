from utils import load_db
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


# ---------------- Intrusion detection ---------------- #
path = r'C:\Users\97254\Desktop\CDD\CDD_code'
X_train, X_test, y_test = load_db.load_intrusion(path)

numeric_cols = [col for col in range(X_train.shape[1]) if len(np.unique(X_train[:, col])) > 2]
X_train_numeric = X_train[:,numeric_cols]
pca = PCA(n_components=2)
X_train2d = pca.fit_transform(X_train_numeric)

n_components = 6  # --- best value by silhouette
gmm = GaussianMixture(n_components=n_components).fit(X_train)
prediction_gmm = gmm.predict(X_train)
X_testnumeric = X_test[:, numeric_cols]
X_test2d = pca.transform(X_testnumeric)
X_test2d_r2l = X_test2d[y_test == 3, :]
X_test2d_r2l_subset = X_test2d_r2l[np.random.randint(X_test2d_r2l.shape[0], size=200), :]
X_test2d_dos = X_test2d[y_test == 1, :]
X_test2d_dos_subset = X_test2d_dos[np.random.randint(X_test2d_dos.shape[0], size=200), :]

colormap = np.array(['olive', 'blue', 'green', 'lime', 'gray', 'brown'])
sc = plt.figure(figsize=(10, 7))
plt.scatter(X_train2d[:, 0], X_train2d[:, 1], c=colormap[prediction_gmm], s=10, cmap='viridis')
plt.scatter(X_test2d_dos_subset[:, 0], X_test2d_dos_subset[:, 1],c='red', s=10)
plt.scatter(X_test2d_r2l[:, 0], X_test2d_r2l[:, 1], c='orange',s=10)
plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC2', fontsize=14)
pop_a = mpatches.Patch(color='olive', label='Cluster 1')
pop_b = mpatches.Patch(color='blue', label='Cluster 2')
pop_c = mpatches.Patch(color='green', label='Cluster 3')
pop_d = mpatches.Patch(color='lime', label='Cluster 4')
pop_e = mpatches.Patch(color='gray', label='Cluster 5')
pop_f = mpatches.Patch(color='brown', label='Cluster 6')
pop_g = mpatches.Patch(color='red', label='DOS attack')
pop_h = mpatches.Patch(color='orange', label='R2L attack')
plt.legend(handles=[pop_a, pop_b, pop_c, pop_d, pop_e, pop_f, pop_g, pop_h])
