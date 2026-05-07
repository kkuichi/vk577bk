import pandas as pd

files = [
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\1. vlna všetko 28-11-2024.xlsx",
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\2. vlna všetko 28-11-2024.xlsx",
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\3. vlna všetko 28-11-2024.xlsx",
    r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\4. vlna všetko 28-11-2024.xlsx",
]
def convert_binary_and_gender(df, filename):
    df = df.copy()

    #TRUE/FALSE konverzia (stringy aj booleany)
    try:
        df = df.replace({
            True: 1, False: 0,
            "TRUE": 1, "FALSE": 0,
            "True": 1, "False": 0,
            "true": 1, "false": 0
        })
    except Exception as e:
        print(f"❌ Chyba pri konverzii TRUE/FALSE v {filename}: {e}")

    # Pohlavie
    try:
        if "Pohlavie" in df.columns:
            df["Pohlavie"] = (
                df["Pohlavie"]
                .astype(str)
                .replace({
                    "Žena": 0,
                    "Muž": 1
                })
            )

            # kontrola neznámych hodnôt
            mask = ~df["Pohlavie"].isin([0, 1])
            if mask.any():
                print(f"⚠️ Upozornenie: Neznáme hodnoty v 'Pohlavie' v {filename}:")
                print(df.loc[mask, "Pohlavie"].unique())
        else:
            print(f"⚠️ Upozornenie: Stĺpec 'Pohlavie' nie je v súbore {filename}")
    except Exception as e:
        print(f"❌ Chyba pri konverzii Pohlavie v {filename}: {e}")

    return df


for path in files:

    df = pd.read_excel(path)

    df = convert_binary_and_gender(df, path)

    df.to_excel(path, index=False)
