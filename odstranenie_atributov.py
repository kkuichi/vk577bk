import pandas as pd

files = [
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\1. vlna všetko 28-11-2024.xlsx",
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\2. vlna všetko 28-11-2024.xlsx",
    #r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\3. vlna všetko 28-11-2024.xlsx",
    r"C:\Users\veron\Documents\tuke\Ing.studium\Diplomová práca\data pre programovanie\VEGA_dáta z 13-11-2024-prog\spracovane_Pavol-Almasi-prog\4. vlna všetko 28-11-2024.xlsx",
]

# Definícia stĺpcov na odstránenie

admin_cols = [
    "Poradie", "Meno", "Dátum príjmu", "Kód príjmu",
    "Dátum prepustenia", "Kód prepustenia",
]

text_cols = [
    "HLN Dg.", "Diagnózy", "DRG výkony", "Liečba",
    "SVLZ správy", "Mikrobiológia ", "Epikríza",
    "Terajšie ochorenie", "Dôvod hospitalizácie",
    "Objektívny nález", "Osobná anamnéza",
    "Lieková anamnéza", "Návyková anamnéza",
    "Epidemiologická anamnéza",
]

drug_keywords = [
    "FABIFLU", "IV-BECT", "VEKLURY", "PAXLOVID", "LAGEVRIO",
    "PYRIDOXIN", "ACIDUM", "CALCIFEROL", "MAGNESIUM",
    "EREVIT", "VITAMIN", "ALPHA", "PREDNISON", "DEXAMED",
    "DEXAMETAZÓN", "BIODEXONE", "HYDROCORTISONE",
    "OLUMIANT", "Anakinra", "RoActemra", "POLYOXIDONIUM",
    "IMUNOR", "IMMODIN", "Isoprinosine", "INOMED",
    "Azithromycin", "Ceftriaxon", "MOLOXIN", "MOXIFLOXACIN",
    "CIPROFLOXACIN", "OZZION", "OMEMYL", "NOLPAZA",
    "PANTOPRAZOL", "SMECTA", "REASEC", "LAGOSA",
    "DEGAN", "AMBROBENE", "PENTOXYPHILLINUM",
    "ACC", "CODEIN", "OXANTIL", "FRAXIPARIN",
    "CLEXANE", "FRAGMIN", "ASPIRIN", "ANOPYRIN",
]


for path in files:
    print(f"\nSpracúvam súbor: {path}")
    df = pd.read_excel(path)

    cols_to_drop = admin_cols + text_cols

    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
        else:
            print(f"   UPOZORNENIE: Stĺpec '{col}' sa nenašiel a nebol odstránený.")

    print("  Výsledný počet stĺpcov:", df.shape[1])

    df.to_excel(path, index=False)
