import pandas as pd

path = "../data/4_vlna.xlsx"

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
