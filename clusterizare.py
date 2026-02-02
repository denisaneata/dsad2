import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust

from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster

def nan_replace(tabel):
    for col in tabel.columns:
        if tabel[col].isna().any():
            if is_numeric_dtype(tabel[col]):
                tabel[col] = tabel[col].fillna(tabel[col].mean())
            else:
                tabel[col] = tabel[col].fillna(tabel[col].mode()[0])


tabel = pd.read_csv("date.csv", index_col=0)   
nan_replace(tabel)

instante = list(tabel.index)
variabile = list(tabel.columns)[1:]           
X = tabel[variabile].values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

h = hclust.linkage(X_std, method="ward")

print("Matrice ierarhie :")
print(h[:10])

#partitie cu nr fix de clustere
k = 5                              
coduri = fcluster(h, k, criterion="maxclust")
partitie = np.array([f"C{c}" for c in coduri])

rez = pd.DataFrame({"Cluster": partitie}, index=instante)
rez.to_csv("p4.csv")              

#k optim
diferente = h[1:, 2] - h[:-1, 2]
k_diff_max = np.argmax(diferente)

n = len(instante)
p = n - 1
k_opt = p - k_diff_max

print("Numar optim de clustere:", k_opt)

#dendograma
prag = (h[p - k, 2] + h[p - k + 1, 2]) / 2

plt.figure(figsize=(10, 6))
plt.title(f"Dendrograma Ward (k={k})")
hclust.dendrogram(h, labels=instante, leaf_rotation=45,
                  color_threshold=prag)
plt.axhline(prag, linestyle="--")
plt.tight_layout()
plt.show()

#grafic partitie in primele 2 comp principale
pca = PCA(n_components=2)
Z = pca.fit_transform(X_std)

plt.figure(figsize=(8, 6))
plt.scatter(Z[:, 0], Z[:, 1], c=coduri)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Partitie în primele 2 componente principale")

for i, inst in enumerate(instante):
    plt.text(Z[i, 0], Z[i, 1], inst, fontsize=8)

plt.tight_layout()
plt.show()
