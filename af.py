import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from pandas.core.dtypes.common import is_numeric_dtype

def nan_replace(df):
    for col in df.columns:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)


def to_dataframe(x, row_names, col_names, filename):
    df = pd.DataFrame(x, index=row_names, columns=col_names)
    df.to_csv(filename)
    return df
  
def main():
    df = pd.read_csv("ceva.csv")
    nan_replace(df)

    variable_names = list(df.columns)[2:]
    x = df[variable_names].values
  
    chi2, p_value = calculate_bartlett_sphericity(x)
    print(f"Bartlett: chi2 = {chi2}, p_value = {p_value}")

    if p_value > 0.05:
        print("Valoarea p_value este prea mare si nu putem aplica AF")
        return

    kmo_all, kmo_overall = calculate_kmo(x)
    print(f"KMO: kmo_overall = {kmo_overall}")

    if kmo_overall < 0.6:
        print("Valoarea kmo_overall este prea mica si nu putem aplica AF")
        return
    fa_n = FactorAnalyzer(rotation=None)
    fa_n.fit(x)

    valori_proprii, _  = fa_n.get_eigenvalues()
    print("Valori proprii: ", valori_proprii)

    n_factori = sum(valori_proprii > 1)
    print("Numar factori latenti semnificativi:", n_factori)

    fa = FactorAnalyzer(n_factors=n_factori, rotation='varimax')
    fa.fit(x)

    factor_labels = [f"F{i+1}" for i in range(n_factori)]

    # loadings =  importanta sau cat de relevant e fiecare factor pentru fiecare variabila initiala
    loadings = fa.loadings_
    loadings_df = to_dataframe(loadings, variable_names, factor_labels, "Loadings.csv")
    heatmap(loadings_df, vmin=-1, vmax=1, title="Factor loadings")

    # comunalitati = proportia dispersiei (variantei) explicate de catre toti factorii semnificativi
    comunalitati = fa.get_communalities()
    comunalitati_df = to_dataframe(comunalitati, variable_names, ["Comunalitati"], "Comunalitati.csv")
    heatmap(comunalitati_df, vmin=0, vmax=1, title="Comunalitati")

    # dispersia (varianta)
    dispersie, dispersie_prop, dispersie_cum = fa.get_factor_variance()
    dispersie_df = pd.DataFrame(
        data = {
            "Dispersie": dispersie,
            "Proportie": dispersie_prop,
            "Cumulat": dispersie_cum
        },
        index = factor_labels
    )
    dispersie_df.to_csv("Dispersie.csv")

    # scoruri factoriale = coordonatele fiecarei observatii (rand) in noul spatiu al factorilor
    scoruri = fa.transform(x)
    scoruri_df = to_dataframe(scoruri, df.index, factor_labels, "Scoruri.csv")
  
main()





