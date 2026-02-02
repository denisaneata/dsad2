import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stts
from pandas.api.types import is_numeric_dtype

def nan_replace(t):
    nume_variabile = list(t.columns)
    for each in nume_variabile:
        if any(t[each].isna()):
            if is_numeric_dtype(t[each]):
                t[each].fillna(t[each].mean(), inplace=True)
            else:
                t[each].filnna(t[each].mode()[0], inplace=True)

def to_dataframe(x, nume_linii=None, nume_coloane=None, fisier=None):
    df = pd.DataFrame(x, nume_linii, nume_coloane)
    if fisier is not None:
        df.to_csv(fisier)

    return df

def test_bartlett(r2, n, p, q, m):
    v = 1 - r2

    chi2 = (-n + 1 + (p + q + 1) / 2) * np.log(np.flip(np.cumprod(np.flip(v))))
    nlib = [(p - k + 1) * (q - k + 1) for k in range(1, m + 1)]

    p_values = 1 - stts.chi2.cdf(chi2, nlib)
    return p_values

def main():
  
    df1 = pd.read_csv('mortalitate.csv', index_col=1)
    nan_replace(df1)

    df2 = pd.read_csv('Teritorial.csv', index_col=1)
    nan_replace(df2)

    # extragerea variabilelor si a observatiilor
    variabile_x = list(df1.columns)[1:]  
    variabile_y = list(df2.columns)[3:] 

    # restrangerea setului de date astfel incat sa pastram doar randurile comune
    df_merged = df1[variabile_x].join(df2[variabile_y], how='inner')

    nume_instante = df_merged.index
    x = df_merged[variabile_x].values
    y = df_merged[variabile_y].values

    # determinarea numarului maxim de perechi de componente canonice
    n, p = x.shape 
    _, q = y.shape  

    m = min(p, q)  # numarul maxim de dimensiuni canonice (perechi)

    # standardizarea datelor 
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    # etichete 
    etichete_z = ['z' + str(i+1) for i in range(m)]
    etichete_u = ['u' + str(i+1) for i in range(m)]
    etichete_radacini = ['rad' + str(i+1) for i in range(m)]

    # constructia si antrenareea modelului ACC
    model_cca = CCA(n_components=m)
    model_cca.fit(x, y)

    # determinarea componentelor canonice z si u 
    z, u = model_cca.transform(x, y)

    # corelatii canonice 
    r = np.array([np.corrcoef(z[:, i], u[:, i])[0, 1] for i in range(m)])
    r_squared = r ** 2

    # testul Bartlett - testarea perechilor de componente canonice semnificative
    p_values = test_bartlett(r_squared, n, p, q, m)
    df_radacini = pd.DataFrame({
        "R": np.round(r, 3),
        "R2": np.round(r_squared, 3),
        "p_value": np.round(p_values, 4)
    }, index=etichete_radacini)
    df_radacini.to_csv("radacini_canonice.csv")

    # numarul de perechi de componente canonice (zi, ui) semnificative
    nr_rad_semnificative = np.sum(p_values < 0.05)
    print(df_radacini)
    print("Numar dimensiuni canonice:", nr_rad_semnificative)

    # factori de incarcare canonici (canonical loadings)
    # corelatiile dintre variabilele initiale si componentele canonice
    z_std = np.std(z, axis=0, ddof=1)
    u_std = np.std(u, axis=0, ddof=1)

    r_xz = model_cca.x_loadings_ * z_std
    r_yu = model_cca.y_loadings_ * u_std

    df_r_xz = to_dataframe(r_xz, variabile_x, etichete_z, "r_xz.csv")
    df_r_yu = to_dataframe(r_yu, variabile_y, etichete_u, "r_yu.csv")

    # scoruri canonice
    df_scoruri_z = to_dataframe(z, nume_instante, etichete_z, "scoruri_z.csv")
    df_scoruri_u = to_dataframe(u, nume_instante, etichete_u, "scoruri_u.csv")

if __name__ == '__main__':
    main()


