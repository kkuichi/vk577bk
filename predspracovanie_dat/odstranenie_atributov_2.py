import pandas as pd
DATA_DIR = Path("../data")

files = [
    DATA_DIR / "1_vlna.xlsx",
    DATA_DIR / "2_vlna.xlsx",
    DATA_DIR / "3_vlna.xlsx",
    DATA_DIR / "4_vlna.xlsx",
]

def drop_selected_columns(df, filename):
    df = df.copy()

    explicit_drop = ["Unnamed: 23", "Typ vakcíny"]

    prefixes = ["S-Chol", "S-IgG", "S-IgA", "S-Ig M"]

    cols_to_drop = []

    for col in explicit_drop:
        if col in df.columns:
            cols_to_drop.append(col)
        else:
            print(f"  Upozornenie: '{col}' sa nenašiel v {filename}")

    for col in df.columns:
        col_str = str(col)
        if any(col_str.startswith(p) for p in prefixes):
            cols_to_drop.append(col)

    cols_to_drop = list(set(cols_to_drop))

    print(f"  Odstraňujem {len(cols_to_drop)} stĺpcov v súbore {filename}:")
    for c in cols_to_drop:
        print("   -", c)

    df.drop(columns=cols_to_drop, inplace=True)

    print("  Počet zostávajúcich stĺpcov:", df.shape[1])
    print("  Zostávajúce stĺpce:")
    for c in df.columns:
        print("   •", c)

    return df


for path in files:
    df = pd.read_excel(path)

    df = drop_selected_columns(df, path)
    print("  Počet zostávajúcich stĺpcov:", df.shape[1])


    df.to_excel(path, index=False)
