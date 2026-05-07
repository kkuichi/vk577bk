import pandas as pd
import os

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path("../data")

files = [
    DATA_DIR / "1_vlna.xlsx",
    DATA_DIR / "2_vlna.xlsx",
    DATA_DIR / "3_vlna.xlsx",
    DATA_DIR / "4_vlna.xlsx",
]
# STĹPCE PRE MISSFOREST (8–25 % missing)

missforest_cols = [
    "S-Alb last", "S-Alb min", "S-Alb max", "S-Alb first",
    "NK last", "NK min", "NK max", "NK first",
    "CD19+ last", "CD19+ first", "CD19+ max", "CD19+ min",
    "S-PBNP first", "S-PBNP max", "S-PBNP min", "S-PBNP last",
    "CD3+ last", "CD3+ first", "CD3+ max", "CD3+ min",
    "CD4+ first", "CD4+ max", "CD4+ min", "CD4+ last",
    "CD8+ first", "CD8+ max", "CD8+ min", "CD8+ last",
    "CD4+/CD8+ first", "CD4+/CD8+ max", "CD4+/CD8+ min", "CD4+/CD8+ last",
    "P-Laktát first", "P-Laktát min", "P-Laktát last", "P-Laktát max",
]

# STĹPCE PRE MICE (< 8 % missing)

mice_cols = [
    "S-Bil-T last", "S-Bil-T first", "S-Bil-T max", "S-Bil-T min",
    "D-dimér HS max", "D-dimér HS first", "D-dimér HS min", "D-dimér HS last",
    "S-ALP max", "S-ALP last", "S-ALP min", "S-ALP first",
    "S-GMT max", "S-GMT last", "S-GMT first", "S-GMT min",
    "S-ALT first", "S-ALT last", "S-ALT max", "S-ALT min",
    "Fib max", "Fib first", "Fib min", "Fib last",
    "S-AST first", "S-AST last", "S-AST min", "S-AST max",
    "PT (INR) min", "PT (INR) last", "PT (INR) first", "PT (INR) max",
    "APTT-R last", "APTT-R min", "APTT-R first", "APTT-R max",
    "S-CL min", "S-CL max", "S-CL last", "S-CL first",
    "S-IL6 first", "S-IL6 min", "S-IL6 max", "S-IL6 last",
    "S-Kreat min", "S-Kreat first", "S-Kreat last", "S-Kreat max",
    "S-Gluk last", "S-Gluk first", "S-Gluk max", "S-Gluk min",
    "SatO2 %",
    "S-Urea min", "S-Urea last", "S-Urea max", "S-Urea first",
    "S-CRP max", "S-CRP min", "S-CRP last", "S-CRP first",
    "S-Na min", "S-Na max", "S-Na last", "S-Na first",
    "S-K first", "S-K last", "S-K min", "S-K max",
    "NE/LY(NLR) last", "NE/LY(NLR) first", "NE/LY(NLR) max", "NE/LY(NLR) min",
    "Eo abs first",
    "PLT max", "PLT min", "PLT last", "PLT first",
    "WBC max", "WBC min", "WBC last", "WBC first",
    "HGB max", "HGB min", "HGB last", "HGB first",
    "Neu abs max", "Neu abs min", "Neu abs last", "Neu abs first",
    "Eo abs last", "Ly abs first", "Eo abs max", "Eo abs min",
    "Ly abs max", "Ly abs last", "Ly abs min",
    "PDW first", "PDW last", "PDW min", "PDW max",
]

for path in files:
    print(f"\nSpracúvam súbor: {path}")
    df = pd.read_excel(path)

    # 1) MICE – atribúty s < 8 % missing
    mice_present = [c for c in mice_cols if c in df.columns]
    if mice_present:
        print(f"  MICE imputácia – počet stĺpcov: {len(mice_present)}")

        # istota, že sú to numerické stĺpce
        df[mice_present] = df[mice_present].apply(pd.to_numeric, errors="coerce")

        mice_imputer = IterativeImputer(
            random_state=0,
            max_iter=20,
            sample_posterior=False,
        )
        df[mice_present] = mice_imputer.fit_transform(df[mice_present])
    else:
        print("  MICE: nenašli sa žiadne z definovaných stĺpcov v tomto súbore.")

    # 2) MissForest – atribúty s 8–25 % missing
    mf_present = [c for c in missforest_cols if c in df.columns]
    if mf_present:
        print(f"  MissForest imputácia – počet stĺpcov: {len(mf_present)}")

        df[mf_present] = df[mf_present].apply(pd.to_numeric, errors="coerce")

        rf_estimator = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
            n_jobs=-1,
        )

        mf_imputer = IterativeImputer(
            estimator=rf_estimator,
            random_state=0,
            max_iter=15,
            sample_posterior=False,
        )

        df[mf_present] = mf_imputer.fit_transform(df[mf_present])
    else:
        print("  MissForest: nenašli sa žiadne z definovaných stĺpcov v tomto súbore.")

    # uloženie späť do toho istého súboru
    df.to_excel(path, index=False)
    print(f"  Uložené späť do: {path}")

print("\nHotovo – MICE (<8 %) a MissForest (8–25 %) boli aplikované na všetky 4 vlny.")
