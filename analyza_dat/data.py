import pandas as pd
from sklearn.model_selection import train_test_split


#1.vlna
df = pd.read_excel("spracovane_Pavol-Almasi/1. vlna všetko 28-11-2024.xlsx")

# Prvých 5 riadkov
#print(df.head())

# Info o dátach
#print(df.info())

# Základné štatistiky
#print(df.describe(include='all'))

# Počet chýbajúcich hodnôt
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

#print(missing_df)

threshold = 40
missing_percent = df.isnull().mean() * 100
cols_to_keep = missing_percent[missing_percent < threshold].index
df = df[cols_to_keep]
print("Pôvodný počet stĺpcov:", len(missing_percent))
print("Počet ponechaných stĺpcov:", df.shape[1])
# kontorla, či niektorý zostávajúci stĺpec má viac ako 40 % chýbajucich hodnot
#missing_check = df.isnull().mean() * 100
#print(missing_check[missing_check > 40])
removed_cols = missing_percent[missing_percent >= 40].index
print("Odstránené stĺpce:")
print(removed_cols.tolist())

