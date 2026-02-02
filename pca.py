#utils:
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.decomposition import PCA

def nan_replace(df):
    for col in df.columns:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                modul = df[col].mode()
                df[col].fillna(modul[0], inplace=True)


def to_dataframe(x, nume_randuri=None, nume_coloane=None, nume_fisier=None):
    df = pd.DataFrame(data=x, index=nume_randuri, columns=nume_coloane)
    if nume_fisier is not None:
        df.to_csv(nume_fisier)

    return df

#main

#scriem mereu
t = pd.read_csv("ceva.csv")
nan_replace(t)
variabile_observate = list(t.columns)[2:] #aici modificam in functie de cum avem variabilele
x_orig = t[variabile_observate]

x = (x_orig - np.mean(x_orig, axis=0)) / np.std(x_orig, axis=0)
n, m = x.shape

model_acp = PCA() 
model_acp.fit(x) 

alpha = model_acp.explained_variance_
print("alpha", alpha)

a = model_acp.components_

#componentele principale rezultate in urma acp
c = model_acp.transform(x)

# afisarea componentelor principale
labels = ["C" + str(i+1) for i in range(len(alpha))]
componente_df = to_dataframe(c, t.index, labels, "componente.csv")

#Kaiser
conditie = np.where(alpha > 1) 
print("conditie Kasier: ", conditie)
array_din_where = conditie[0]
nr_comp_s_kaiser = len(array_din_where)
print("Comp principale semnificative conform crit Kaiser: ", nr_comp_s_kaiser)

# Cattel
eps = alpha[0 : (m-1)] - alpha[1 : m]  
sigma = eps[0: (m-2)] - eps[1: len(eps)] 
indici_negativi = (sigma < 0)
print("Indici Cattel:", indici_negativi)

if any(indici_negativi):
    conditie = np.where(indici_negativi)
    array_din_where = conditie[0] 

    nr_comp_s_cattel = array_din_where[0] + 1  
else:
    nr_comp_s_cattel = None
print("Comp principale semnificative conform crit Cattel: ", nr_comp_s_cattel)

# procent de acoperire
ponderi = np.cumsum(alpha / sum(alpha))
conditie = np.where(ponderi > 0.8)
nr_comp_s_procent = conditie[0][0] + 1
print("Comp principale semnificative: ", nr_comp_s_procent)

# calcul corelatii intre variabilele initiale si componentele principale

corr = np.corrcoef(x, c, rowvar=False)
print(f"Corrcoef: {corr.shape}, {n}, {m}, \n, {corr}")
r_x_c = corr[:m, m:]
r_x_c_df = to_dataframe(r_x_c, variabile_observate, labels, "corelatii_factoriale.csv")

# comunalitati 
r_patrat = r_x_c * r_x_c
comunalitati = np.cumsum(r_patrat, axis=1)
comunalitati_df = to_dataframe(comunalitati, variabile_observate, labels, "comunalitati.csv")

# cosinusuri -
c_patrat = c ** 2
sume = c_patrat.sum(axis=1, keepdims=True)
cosin = c_patrat / sume
cosin_df = to_dataframe(cosin, t.index, labels, "cosinusuri.csv")









