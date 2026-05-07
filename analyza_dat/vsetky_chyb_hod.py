import pandas as pd

path = "../data/4_vlna.xlsx"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_excel(path)

missing = df.isnull().sum()
missing_percent = df.isnull().mean() * 100

missing_summary = pd.DataFrame({
    'Počet_chýbajúcich': missing,
    'Percento_chýbajúcich': missing_percent
})

# iba atribúty kde missing > 0
missing_filtered = missing_summary[missing_summary['Počet_chýbajúcich'] > 0]

# zoradiť zostupne
missing_filtered = missing_filtered.sort_values(by='Počet_chýbajúcich', ascending=False)

print(missing_filtered)
print("\n Celkový počet atribútov s missing hodnotami:", len(missing_filtered))
print(missing_filtered.index.tolist())

