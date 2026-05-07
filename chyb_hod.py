import pandas as pd

path =r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\1. vlna všetko 28-11-2024.xlsx"
# = r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\2. vlna všetko 28-11-2024.xlsx"
#path =r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\3. vlna všetko 28-11-2024.xlsx"
#path =r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\4. vlna všetko 28-11-2024.xlsx"
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
