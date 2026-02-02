import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sb

date_antrenament = pd.read_csv("park.csv")
date_test = pd.read_csv("park_test.csv")

variabila_tinta = date_antrenament.columns[-1]
variabile_predictor = date_antrenament.columns[:-1]

x_train = date_antrenament[variabile_predictor].values
y_train = date_antrenament[variabila_tinta].values

x_test = date_test[variabile_predictor].values
instante = date_antrenament.index

def calculeaza_acuratete(y_true, y_pred, clase):
    matrice_confuzie = confusion_matrix(y_true, y_pred, labels=clase)

    # global
    acc_global = np.round(np.diagonal(matrice_confuzie).sum() * 100 / matrice_confuzie.sum(), 3)

    # grup
    acc_grup = np.round(np.diagonal(matrice_confuzie) * 100 / np.sum(matrice_confuzie, axis=1), 3)

    # medie
    acc_mediu = np.mean(acc_grup)

    return matrice_confuzie, acc_global, acc_grup, acc_mediu
  
# Analiza Discriminanta Bayesiana (GaussianNB)
bda_model = GaussianNB()
bda_model.fit(x_train, y_train)
clase_bda = bda_model.classes_

# predictii
y_train_pred_bda = bda_model.predict(x_train)
y_test_pred_bda = bda_model.predict(x_test)

# calcul acuratete
mat_conf_bda, global_acc_bda, group_acc_bda, avg_acc_bda = calculeaza_acuratete(y_train, y_train_pred_bda, clase_bda)

# afisare rezulate
print("---BDA---")
print("Global Acc: ", global_acc_bda)
print("Group Acc: ", group_acc_bda)
print("Avg Acc: ", avg_acc_bda)
print("Cohen's kappa: ", cohen_kappa_score(y_train, y_train_pred_bda))

# salvare matrice de confuzie
tabel_mat_conf_bda = pd.DataFrame(mat_conf_bda, index=clase_bda, columns=clase_bda)
tabel_mat_conf_bda["Acuratete BDA"] = group_acc_bda
tabel_mat_conf_bda.to_csv("MatConfBDA.csv")

# salvare predictii
pd.DataFrame({variabila_tinta: y_train, "Predictie BDA": y_train_pred_bda}, index=instante).to_csv("ClasificareAntrenamentBDA.csv")
pd.DataFrame({"Predictie BDA": y_test_pred_bda}, index=date_test.index).to_csv("ClasificareTestBDA.csv")

# Analiza Discriminanta Liniara (LDA)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(x_train, y_train)
clase_lda = lda_model.classes_

# predictii
y_train_pred_lda = lda_model.predict(x_train)
y_test_pred_lda = lda_model.predict(x_test)

# calcul acuratete
mat_conf_lda, global_acc_lda, group_acc_lda, avg_acc_lda = calculeaza_acuratete(y_train, y_train_pred_lda, clase_lda)

# afisare rezulate
print("---LDA---")
print("Global Acc: ", global_acc_lda)
print("Group Acc: ", group_acc_lda)
print("Avg Acc: ", avg_acc_lda)
print("Cohen's kappa: ", cohen_kappa_score(y_train, y_train_pred_lda))

# salvare matrice de confuzie
tabel_mat_conf_lda = pd.DataFrame(mat_conf_lda, index=clase_lda, columns=clase_lda)
tabel_mat_conf_lda["Acuratete LDA"] = group_acc_lda
tabel_mat_conf_lda.to_csv("MatConfLDA.csv")

# salvare predictii
pd.DataFrame({variabila_tinta: y_train, "Predictie LDA": y_train_pred_lda}, index=instante).to_csv("ClasificareAntrenamentLDA.csv")
pd.DataFrame({"Predictie LDA": y_test_pred_lda}, index=date_test.index).to_csv("ClasificareTestLDA.csv")

#grafic
z_train = lda_model.transform(x_train)
means_lda = lda_model.means_
z_means = lda_model.transform(means_lda)

n_axe = min(len(variabile_predictor), len(clase_lda) - 1)

if n_axe > 1:
    plt.figure(figsize = (9, 9))
    sb.scatterplot(x=z_train[:, 0], y=z_train[:, 1], hue=y_train, hue_order=clase_lda)
    sb.scatterplot(x=z_means[:, 0], y=z_means[:, 1], hue=clase_lda, marker='s', s=255, legend=False)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title("LDA: instante si mediile claselor")
    plt.show()
else:
    print("Numar insuficient de axe pentru a reprezenta grafic instantele")

for i in range(n_axe):
    plt.figure(figsize=(9,9))
    for cls in clase_lda:
        sb.kdeplot(z_train[y_train == cls, i], fill=True, label=cls)
    plt.title(f"Distributie de-a lungul LD{i+1}")
    plt.show()

