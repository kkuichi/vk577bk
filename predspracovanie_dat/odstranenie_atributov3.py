import pandas as pd

DATA_DIR = Path("../data")

files = [
    DATA_DIR / "1_vlna.xlsx",
    DATA_DIR / "2_vlna.xlsx",
    DATA_DIR / "3_vlna.xlsx",
    DATA_DIR / "4_vlna.xlsx",
]

# A04.7 + všetky lieky (podľa popisu dát)
cols_to_remove = [
    "A04.7",
    "MD652 | FABIFLU TABLETS",
    "MD656 IV-BECT 6MG (ivermectin)",
    "5042D | VEKLURY",
    "9547D | PAXLOVID",
    "LAGEVRIO",
    "00584 | PYRIDOXIN LÉČIVA INJ",
    "24836 | ACIDUM ASCORBICUM BBP",
    "24814 | CALCIFEROL BBP 7,5 MG/ML",
    "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
    "00449 | EREVIT 300 MG/ML",
    "89145 | VITAMIN C-INJEKTOPAS",
    "92973 ALPHA D3",
    "02963 | PREDNISON 20 LÉČIVA",
    "00269 | PREDNISON 5 LÉČIVA",
    "84090 | DEXAMED 6",
    "1275C | DEXAMETAZÓN KRKA",
    "MD661 BIODEXONE-DEXAMETHASONE",
    "2410B HYDROCORTISONE",
    "3242C | OLUMIANT 4 MG",
    "Anakinra",
    "RoActemra",
    "34045 | POLYOXIDONIUM 6 MG",
    "87299 | IMUNOR",
    "56930 IMMODIN",
    "Isoprinosine, ",
    "3879d INOMED",
    "35715 Azithromycin",
    "45954 Ceftriaxon",
    "0471B MOLOXIN",
    "9819A MOXIFLOXACIN",
    "58730 CIPROFLOXACIN KABI 200",
    "58746 CIPROFLOXACINKABI 400",
    "05044 OZZION",
    "4147C OMEMYL",
    "89662 NOLPAZA",
    "39397 PANTOPRAZOL",
    "62916 SMECTA",
    "30639 REASEC",
    "84370 LAGOSA",
    "93105 DEGAN ",
    "94918 AMBROBENE",
    "24859 PENTOXYPHILLINUM",
    "8893 ACC INJEKT",
    "24949 CODEIN ",
    "26846 OXANTIL",
    "FRAXIPARIN",
    "CLEXANE",
    "FRAGMIN",
    "ASPIRIN",
    "ANOPYRIN"
]

def drop_selected_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()

    not_found = []
    removed = []

    for col in cols_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            removed.append(col)
        else:
            not_found.append(col)

    print(f"\nSpracúvam súbor: {filename}")
    if removed:
        print("  Odstránené stĺpce:")
        for c in removed:
            print("   -", c)
    else:
        print("  Nebol odstránený žiadny z definovaných stĺpcov.")

    if not_found:
        print("  Nasledovné stĺpce sa v datasete nenašli (nebolo čo odstrániť):")
        print("   ", not_found)

    return df


for path in files:
    df = pd.read_excel(path)

    df = drop_selected_columns(df, path)

    print("  Počet zostávajúcich stĺpcov:", df.shape[1])
    print("  Zostávajúce stĺpce v datasete:")
    for col in df.columns:
        print("   •", col)

    df.to_excel(path, index=False)
    print("  Uložené späť do súboru:", path)



