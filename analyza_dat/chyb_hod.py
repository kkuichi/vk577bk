import pandas as pd

path = "../data/4_vlna.xlsx"
df = pd.read_excel(path)

# Výpočet chýbajúcich hodnôt
missing = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

missing_summary = pd.DataFrame({
    'Počet_chýbajúcich': missing,
    'Percento_chýbajúcich': missing_percent
})

# Zobrazenie top 20 atribútov s najviac chýbajúcimi hodnotami
print(missing_summary.head(20))
