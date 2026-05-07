import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data\VEGA_dáta z 13-11-2024\spracovane_Pavol-Almasi\3. vlna všetko 28-11-2024.xlsx"
df = pd.read_excel(path)

# numerické stĺpce
num_df = df.select_dtypes(include=[np.number])

#  korelačna matica (absolútne hodnoty)
corr_matrix = num_df.corr().abs()

#prah pre vysokú koreláciu
threshold = 0.95

# páry s vysokou koreláciou
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={"level_0": "Atribút_1", "level_1": "Atribút_2", 0: "Korelácia"})
)

#  len tie, ktoré sú nad prahom
high_corr_pairs = corr_pairs[corr_pairs["Korelácia"] > threshold].sort_values(by="Korelácia", ascending=False)

#  top 30 párov
print(high_corr_pairs.head(30))

# ulož do Excelu
high_corr_pairs.to_excel("vysoko_korelovane_pary3.xlsx", index=False)