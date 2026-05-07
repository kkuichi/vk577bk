# VYSVETLITEĽNÉ MODELY UMELEJ INTELIGENCIE PRE ANALÝZU DÁT O PACIENTOCH S COVID-19 


## Diplomová práca 
**Autor:** Bc. Veronika Kapcálová  
**Vedúci práce:** prof. Ing. Ján Paralič, PhD.  
**Konzultant:** Ing. Miroslava Matejová, PhD.  

Technická univerzita v Košiciach  
Fakulta elektrotechniky a informatiky  
Ústav umelej inteligencie

2026

Tento repozitár obsahuje implementáciu vytvorenú v rámci diplomovej práce zameranej na návrh a implementáciu prediktívnych modelov pre analýzu klinických dát pacientov s ochorením COVID-19, aplikáciu metód vysvetliteľnej umelej inteligencie (Explainable Artificial Intelligence – XAI) na interpretáciu rozhodnutí modelov a vyhodnotenie kvality generovaných vysvetlení.

## Obsah 

Repozitár obsahuje:

- analýzu klinických dát,
- preprocessing datasetu,
- imputáciu chýbajúcich hodnôt,
- trénovanie modelov strojového učenia,
- implementáciu metód vysvetliteľnej umelej inteligencie,
- generovanie lokálnych a globálnych vysvetlení,
- implementáciu technických metrík kvality vysvetlení,


---

## Použité technológie a knižnice

Projekt bol implementovaný v programovacom jazyku Python.

Použité knižnice:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- shap
- lime
- dalex
- alibi
- pytorch-tabnet
- torch

---

# Dataset

Použitý dataset obsahuje anonymizované klinické údaje pacientov hospitalizovaných s ochorením COVID-19.

Cieľová premenná:
- `Závažnosť priebehu ochorenia`

Triedy:
1. prepustenie domov alebo do sociálneho zariadenia,
2. preklad na iné oddelenie,
3. exitus.

Dataset nie je verejne dostupný z dôvodu ochrany osobných a zdravotných údajov pacientov.

---

## Preprocessing dát

Pred samotným trénovaním modelov boli realizované viaceré preprocessing kroky.
- odstránenie atribútov
- analýza chýbajúcich hodnôt
- imputácia dát


## Implementované modely

V práci boli implementované viaceré modely pre klasifikáciu závažnosti ochorenia.

- XGBoost
- LightGBM
- CatBoost
- ExtraTrees 
- Tabnet

## Implementované XAI metódy

- SHAP
- LIME
- BreakDown
- Anchor
- PDP

Repozitár obsahuje aj implementáciu viacerých technických metrík kvality vysvetlení.


# Štruktúra repozitára



```text
vk577bk-main/
│
├── README.md
│
├── analyza_dat/
│   ├── chyb_hod.py
│   ├── data.py
│   ├── korelacia.py
│   └── vsetky_chyb_hod.py
│
├── predspracovanie_dat/
│   ├── convert_binary.py
│   ├── imputation_indikator.py
│   ├── imputation_missforest_MICE.py
│   ├── odstranenie_atributov.py
│   ├── odstranenie_atributov3.py
│   ├── odstranenie_atributov_2.py
│   └── odstranenie_riadkov.py
│
└── models/
    ├── CatBoost.py
    ├── ExtraTrees.py
    ├── LightGBM.py
    ├── TabNet.py
    ├── XGBoost.py
    ├── XGBoost_xai_multi_patient_eval.py
    ├── catboost_multi_patient_eval.py
    ├── extratrees_multi_patient_eval.py
    └── lightgbm_multi_patient_eval.py
