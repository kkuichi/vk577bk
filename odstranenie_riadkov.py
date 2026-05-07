import pandas as pd

path = r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\4. vlna všetko 28-11-2024.xlsx"

df = pd.read_excel(path)

# Počet riadkov pred odstránením
before = df.shape[0]

# Odstránenie riadkov s chýbajúcou hodnotou v cieľovom atribúte
df = df.dropna(subset=["Závažnosť priebehu ochorenia"])

# Počet riadkov po odstránení
after = df.shape[0]

print(f"Odstránených riadkov: {before - after}")
print(f"Počet riadkov po čistení: {after}")

# Uloženie späť do pôvodného súboru
df.to_excel(path, index=False)
print("Dáta boli uložené spät do súboru.")
