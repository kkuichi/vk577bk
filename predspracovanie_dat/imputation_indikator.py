import pandas as pd
import os

DATA_DIR = Path("../data")

files = [
    DATA_DIR / "1_vlna.xlsx",
    DATA_DIR / "2_vlna.xlsx",
    DATA_DIR / "3_vlna.xlsx",
    DATA_DIR / "4_vlna.xlsx",
]

# Stĺpce nad hranicou 25 % missing budú prepísané na 0/1
cols_to_binary = [
    "S-LD first", "S-LD last", "S-LD min", "S-LD max",
    "S-AMS last", "S-AMS min", "S-AMS first", "S-AMS max",
    "S-KM min", "S-KM last", "S-KM max", "S-KM first",
    "S-FER first", "S-FER min", "S-FER max", "S-FER last",
    "S-CK-MB min", "S-CK-MB max", "S-CK-MB first", "S-CK-MB last",
    "S-CK min", "S-CK first", "S-CK max", "S-CK last",
    "S-VITD max", "S-VITD first", "S-VITD last", "S-VITD min",
    "S-CB last", "S-CB first", "S-CB max", "S-CB min",
]

for i, path in enumerate(files, start=1):
    print("\nSpracúvam súbor:", path)
    df = pd.read_excel(path)

    modified = []
    missing = []

    for col in cols_to_binary:
        if col in df.columns:
            # 1 = test bol robený (hodnota nie je NaN), 0 = test nebol robený (NaN)
            df[col] = df[col].notna().astype(int)
            modified.append(col)
        else:
            missing.append(col)

    print(f"  Prepísaných stĺpcov na binárne 0/1: {len(modified)}")
    if modified:
        print("  Prepísané stĺpce:")
        for c in modified:
            print("   -", c)

    if missing:
        print("  Nasledovné stĺpce sa v tomto datasete nenašli (neboli prepísané):")
        for c in missing:
            print("   -", c)

    # vytvorenie názvu nového súboru: "1. vlna imputation.xlsx", ...
    folder, _ = os.path.split(path)
    new_filename = f"{i}. vlna imputation.xlsx"
    new_path = os.path.join(folder, new_filename)

    df.to_excel(new_path, index=False)
    print("  Uložené ako:", new_path)

print("\nHotovo – hodnoty v zvolených stĺpcoch boli pre všetky 4 vlny prepísané na 0/1 a uložené do nových súborov.")
