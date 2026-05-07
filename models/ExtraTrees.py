# -*- coding: utf-8 -*-
"""
ExtraTrees + XAI (SHAP, LIME, DALEX BreakDown, Anchor,pdp) + metriky vysvetlení.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.ion()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors

from sklearn.ensemble import ExtraTreesClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import dalex as dx
import dice_ml
from dice_ml import Dice
from alibi.explainers import AnchorTabular
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dalex")
# 0) KONFIG – rýchlosť vs kvalita
FAST_MODE = True

RUN_PDP = True

# počet featur používaných v perturbation-based metrikách (completeness/necessity/sufficiency/monotonicity/...)
TOP_K_FEATURES = 20 if FAST_MODE else 60

# počet bootstrap replikácií pre globálnu stabilitu rankingu (Spearman)
BOOTSTRAP_ROUNDS = 10 if FAST_MODE else 50

# počet vzoriek pre trendability (Spearman medzi hodnotou feature a jej importance)
TRENDABILITY_SAMPLES = 80 if FAST_MODE else 200

RUN_ANCHOR = True
ANCHOR_THRESHOLD = 0.75
ANCHOR_BEAM_SIZE = 10 if FAST_MODE else 20
ANCHOR_BATCH_SIZE = 100
ANCHOR_COVERAGE_SAMPLES = 200 if FAST_MODE else 500
ANCHOR_DELTA = 0.15

RUN_LIME_SANITY_CHECK = True
SANITY_EXTRATREES_TREES = 60 if FAST_MODE else 150

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 40)

RUN_MULTI_PATIENT_EVAL = True
N_EVAL_PATIENTS = 20
MULTI_RANDOM_STATE = 42

SHOW_PLOTS = False
RUN_SINGLE_PATIENT_DEMO = False
RUN_LIME_SANITY_CHECK = False

# 1. NAČÍTANIE DÁT + TRÉNING EXTRATREES
PATH = "../data.xlsx"
TARGET = "Závažnosť priebehu ochorenia"

df = pd.read_excel(PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(np.unique(y_enc))
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_enc,
    test_size=0.15,
    random_state=42,
    stratify=y_enc
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.17647,
    random_state=42,
    stratify=y_temp
)

print("Train shape:", X_train.shape)
print("Val shape:  ", X_val.shape)
print("Test shape: ", X_test.shape)

feature_names = X.columns.tolist()

model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
print(" Trénujem ExtraTrees...")
model.fit(X_train, y_train)
print(" Model natrenovaný.")
# 2. KLASICKÉ METRIKY MODELU
def _ensure_df(X_any, feature_names):
    if isinstance(X_any, pd.DataFrame):
        return X_any

    arr = np.asarray(X_any)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return pd.DataFrame(arr, columns=feature_names)

def predict_proba_df(model, X_any, feature_names):
    return model.predict_proba(_ensure_df(X_any, feature_names))

def predict_df(model, X_any, feature_names, **kwargs):
    return model.predict(_ensure_df(X_any, feature_names), **kwargs)

def predict_raw_margin_df(model, X_any, feature_names):
    return predict_proba_df(model, X_any, feature_names)

def evaluate_model(model, X_test, y_test):
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

metrics = evaluate_model(model, X_test, y_test)
print("\n Klasické metriky ExtraTrees:")
for k, v in metrics.items():
    print(k, ":\n", v, "\n")

if RUN_MULTI_PATIENT_EVAL:
    rng = np.random.default_rng(MULTI_RANDOM_STATE)
    eval_patient_indices = rng.choice(len(X_test), size=min(N_EVAL_PATIENTS, len(X_test)), replace=False)
    eval_patient_indices = sorted(eval_patient_indices.tolist())
else:
    eval_patient_indices = [0]

patient_idx = 0
x_patient_df = X_test.iloc[patient_idx:patient_idx+1]
x_patient = x_patient_df.values[0]

probs = predict_proba_df(model, x_patient_df, feature_names)[0]
pred_class = int(np.argmax(probs))
pred_label = le.inverse_transform([pred_class])[0]

true_class = int(y_test[patient_idx])
true_label = le.inverse_transform([true_class])[0]

print(f"\n Pacient #{patient_idx}")
print(f"Predikovaná trieda {pred_label} | pravdepodobnosti: {probs}")
print(f"Skutočná trieda: {true_label}")
print("\nHodnoty pacienta:")

for col in X_test.columns:
    value = x_patient_df.iloc[0][col]
    print(f"{col}: {value}")

def _predict_proba(model, X_any):
    return predict_proba_df(model, X_any, feature_names)

def _topk_features_from_series(series, k):
    return series.sort_values(ascending=False).head(int(k)).index.tolist()

def _as_float_df(df1: pd.DataFrame) -> pd.DataFrame:
    return df1.copy().astype(float)

def _spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return np.nan
    # rank transform
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def _vectorize_importance(imp_dict, all_features):
    return np.array([imp_dict.get(f, 0.0) for f in all_features], dtype=float)

def _timeit(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, float(time.perf_counter() - t0)

def model_robustness(model, x_instance, noise_scale=0.01, trials=20):
    original_pred = int(np.argmax(_predict_proba(model, x_instance.reshape(1, -1))[0]))
    changes = 0
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        pred = int(np.argmax(_predict_proba(model, x_noisy.reshape(1, -1))[0]))
        if pred != original_pred:
            changes += 1
    return changes / trials

def explanation_robustness(
    explain_fn,
    x_instance,
    noise_scale=0.01,
    trials=10
):
    base = explain_fn(x_instance)

    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        noisy = explain_fn(x_noisy)

        keys = sorted(set(base.keys()) | set(noisy.keys()))
        b = np.array([base.get(k, 0.0) for k in keys])
        n = np.array([noisy.get(k, 0.0) for k in keys])

        diffs.append(np.linalg.norm(b - n))

    return float(np.mean(diffs))

def bootstrap_spearman_ranking(values_matrix, rounds=20, seed=42):
    rng = np.random.default_rng(seed)
    full = np.mean(values_matrix, axis=0)
    full_rank = full.argsort().argsort()
    rhos = []
    n = values_matrix.shape[0]
    for _ in range(int(rounds)):
        idx = rng.integers(0, n, size=n)
        boot = np.mean(values_matrix[idx], axis=0)
        boot_rank = boot.argsort().argsort()
        if np.std(full_rank) == 0 or np.std(boot_rank) == 0:
            rhos.append(np.nan)
        else:
            rhos.append(np.corrcoef(full_rank, boot_rank)[0, 1])
    rhos = np.array(rhos, dtype=float)
    return float(np.nanmean(rhos)), float(np.nanstd(rhos))

def explanation_parsimony_topk(importance_dict, k=10, threshold=0.95):
    if not importance_dict:
        return np.nan
    vals = np.array(sorted([abs(v) for v in importance_dict.values()], reverse=True), dtype=float)
    if vals.sum() == 0:
        return np.nan
    cum = np.cumsum(vals) / vals.sum()
    return int(np.searchsorted(cum, threshold) + 1)

def simplicity_entropy_and_gini(importance_dict):
    if not importance_dict:
        return np.nan, np.nan
    w = np.array([abs(v) for v in importance_dict.values()], dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan, np.nan
    p = w / s
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    gini_like = float(1.0 - np.sum(p ** 2))
    return gini_like, entropy

# Perturbation-based metriky
def _baseline_row(X_ref: pd.DataFrame) -> pd.Series:
    base = {}
    for c in X_ref.columns:
        col = X_ref[c]
        vals = pd.Series(col.dropna().unique())
        # binárna (0/1) – toleruj floaty typu 0.0/1.0
        if len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0}):
            # mód (najčastejšia hodnota)
            base[c] = float(col.mode(dropna=True).iloc[0]) if not col.mode(dropna=True).empty else 0.0
        else:
            base[c] = float(col.median(skipna=True))
    return pd.Series(base)

def completeness_drop_auc(model,x_df: pd.DataFrame,importance_order,baseline: pd.Series, pred_class: int,nonneg: bool = True,normalize: bool = True,):
    x_new = _as_float_df(x_df)
    p0 = float(_predict_proba(model, x_new)[0, pred_class])

    drops = [0.0]
    for f in importance_order:
        if f in x_new.columns:
            x_new.loc[:, f] = float(baseline[f])
        p = float(_predict_proba(model, x_new)[0, pred_class])
        d = (p0 - p)
        if nonneg:
            d = max(0.0, d)
        drops.append(float(d))

    auc = float(np.trapezoid(np.array(drops, dtype=float), dx=1.0))

    if not normalize:
        return auc

    p_all = float(_predict_proba(model, x_new)[0, pred_class])
    max_drop = max(0.0, p0 - p_all) if nonneg else (p0 - p_all)
    k = max(1, len(list(importance_order)))

    denom = float(max_drop * k)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(auc / denom, 0.0, 1.0))

def necessity_score(model, x_df: pd.DataFrame, top_features, baseline: pd.Series, pred_class: int, nonneg: bool = True):
    p0 = float(_predict_proba(model, x_df)[0, pred_class])
    deltas = []
    for f in top_features:
        if f not in x_df.columns:
            continue
        x_m = _as_float_df(x_df)
        x_m.loc[:, f] = float(baseline[f])
        pm = float(_predict_proba(model, x_m)[0, pred_class])
        d = (p0 - pm)
        if nonneg:
            d = max(0.0, d)
        deltas.append(float(d))
    return float(np.mean(deltas)) if deltas else np.nan

def sufficiency_retained_prob(model, x_df, top_features, baseline, pred_class):
    p0 = float(_predict_proba(model, x_df)[0, pred_class])
    x_new = _as_float_df(x_df)

    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])

    pk = float(_predict_proba(model, x_new)[0, pred_class])

    if p0 <= 0:
        return np.nan

    ratio = pk / p0
    return float(np.clip(ratio, 0.0, 1.0))

def sufficiency_class_stable(model, x_df: pd.DataFrame, top_features, baseline: pd.Series, pred_class: int):
    x_new = _as_float_df(x_df)
    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])
    pred_new = int(np.argmax(_predict_proba(model, x_new)[0]))
    return float(pred_new == pred_class)

def monotonicity_local(model, x_df: pd.DataFrame,feature_list,pred_class: int,attr_map: dict,X_ref: pd.DataFrame,delta_frac: float = 0.01,):
    hits = []
    for f in feature_list:
        if f not in x_df.columns:
            continue

        # atribúcia musí mať znamienko
        attr_sign = np.sign(float(attr_map.get(f, 0.0)))
        if attr_sign == 0:
            continue

        col = X_ref[f]
        vals = pd.Series(col.dropna().unique())
        is_binary = len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0})

        x_plus = _as_float_df(x_df)
        x_minus = _as_float_df(x_df)

        if is_binary:
            v = float(x_df.iloc[0][f])
            # ak v≈0 -> plus=1, minus=0; ak v≈1 -> plus=1, minus=0 (stále porovnáme 1 vs 0)
            x_plus.loc[:, f] = 1.0
            x_minus.loc[:, f] = 0.0
        else:
            if not pd.api.types.is_numeric_dtype(col):
                continue
            v = float(x_df.iloc[0][f])
            sd = float(col.std())
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            delta = float(delta_frac) * sd
            x_plus.loc[:, f] = v + delta
            x_minus.loc[:, f] = v - delta

        p_plus = float(_predict_proba(model, x_plus)[0, pred_class])
        p_minus = float(_predict_proba(model, x_minus)[0, pred_class])

        grad_sign = np.sign(p_plus - p_minus)
        if grad_sign == 0:
            continue

        hits.append(float(attr_sign == grad_sign))

    return float(np.mean(hits)) if hits else np.nan

def trendability_spearman(X_ref: pd.DataFrame, importance_matrix: np.ndarray, feature_list, use_abs=True):
    rhos = []
    for f in feature_list:
        j = X_ref.columns.get_loc(f)
        rho = _spearman(X_ref.iloc[:, j].values, importance_matrix[:, j])
        if not np.isnan(rho):
            rhos.append(abs(rho) if use_abs else rho)

    if len(rhos) == 0:
        return np.nan, np.nan

    rhos = np.array(rhos, dtype=float)
    return float(np.mean(rhos)), float(np.std(rhos))

# 4. SHAP – vysvetlenie + metriky
explainer_shap = shap.TreeExplainer(model)
shap_vals_patient = explainer_shap(x_patient_df)

print("\n SHAP waterfall (zobrazí sa okno s grafom)...")
values_for_class = shap_vals_patient.values[0, :, pred_class]      # (n_features,)
base_for_class   = shap_vals_patient.base_values[0, pred_class]    # scalar
data_for_class   = shap_vals_patient.data[0]                       # pôvodné hodnoty pacienta

shap_exp_single = shap.Explanation(values=values_for_class,base_values=base_for_class,data=data_for_class,feature_names=shap_vals_patient.feature_names)
def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    if s <= 0:
        return np.full_like(z, 1.0 / len(z))
    return ez / s

shap_prob_reconstructed_for_class = (float(shap_vals_patient.base_values[0, pred_class]) + float(np.sum(shap_vals_patient.values[0, :, pred_class])))

shap_prob_error = abs(shap_prob_reconstructed_for_class - float(probs[pred_class]))
shap_fidelity = float(max(0.0, 1.0 - shap_prob_error))

print(f" SHAP fidelity (probability-scale): {shap_fidelity:.6f}")
print(f"   reconstructed prob = {shap_prob_reconstructed_for_class:.8f}")
print(f"   model prob         = {float(probs[pred_class]):.8f}")
print(f"   abs prob error     = {shap_prob_error:.6e}")

shap.plots.waterfall(shap_exp_single, max_display=15, show=False)

plt.gcf().set_size_inches(14, 10)   # zväčší graf
plt.subplots_adjust(left=0.4)       # pridá priestor vľavo

plt.show()
def shap_stability(explainer, x_instance, feature_names, noise_scale=0.01, trials=10):
    base = explainer(pd.DataFrame([x_instance], columns=feature_names)).values.flatten()
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        shap_noisy = explainer(pd.DataFrame([x_noisy], columns=feature_names)).values.flatten()
        diffs.append(np.linalg.norm(base - shap_noisy))
    return float(np.mean(diffs))

def shap_consistency(explainer, X_test, x_instance, feature_names, k=5):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]

    base = explainer(pd.DataFrame([x_instance], columns=feature_names)).values.flatten()

    rhos = []
    for n in neighbors:
        shap_n = explainer(X_test.iloc[n:n+1]).values.flatten()
        rho = _spearman(base, shap_n)
        if not np.isnan(rho):
            rhos.append(abs(rho))  # 0–1

    return float(np.mean(rhos)) if rhos else np.nan

shap_stab = shap_stability(explainer_shap, x_patient, feature_names)
shap_cons = shap_consistency(explainer_shap, X_test, x_patient, feature_names)
#shap_rob = model_robustness(model, x_patient)
def shap_explain_fn(x):
    x_df = pd.DataFrame([x], columns=feature_names)
    sv = explainer_shap(x_df)
    vals = sv.values[0, :, pred_class]
    return dict(zip(feature_names, vals))

shap_rob = explanation_robustness(shap_explain_fn, x_patient)

# 4B. SHAP – GLOBAL (multi-class)
X_global = X_test.copy()   # alebo X_val
shap_vals_global = explainer_shap(X_global)  # Explanation
print("SHAP global values shape:", shap_vals_global.values.shape)

# 1) mean(|SHAP|) pre každú triedu zvlášť -> (n_features, n_classes)
mean_abs_by_class = np.mean(np.abs(shap_vals_global.values), axis=0)

# 2) agregácia cez triedy -> (n_features,)
mean_abs_global = mean_abs_by_class.mean(axis=1)  # mean cez triedy

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_global
}).sort_values("mean_abs_shap", ascending=False)

print("\n SHAP globálna dôležitosť (top 20):")
print(global_importance.head(20))

topN = 20
plt.figure(figsize=(8, 6))
plt.barh(global_importance["feature"].head(topN)[::-1],
         global_importance["mean_abs_shap"].head(topN)[::-1])
plt.xlabel("mean(|SHAP|) – agregované cez triedy")
plt.title(f"SHAP globálna dôležitosť atribútov (Top {topN})")
plt.tight_layout()
plt.show()

class_names = le.inverse_transform(np.arange(num_classes))
for c in range(num_classes):
    imp_c = pd.DataFrame({"feature": feature_names,"mean_abs_shap": mean_abs_by_class[:, c]}).sort_values("mean_abs_shap", ascending=False)

    print(f"\n Trieda {class_names[c]} – top 15 atribútov:")
    print(imp_c.head(15))

    plt.figure(figsize=(8, 6))
    plt.barh(imp_c["feature"].head(15)[::-1], imp_c["mean_abs_shap"].head(15)[::-1])
    plt.xlabel("mean(|SHAP|) v triede")
    plt.title(f"SHAP dôležitosť – trieda {class_names[c]}")
    plt.tight_layout()
    plt.show()

# globálne metriky SHAP: stabilita rankingu cez bootstrap
abs_shap_global_per_sample = np.mean(np.abs(shap_vals_global.values), axis=2)  # (n_samples, n_features)
shap_global_rank_mean_rho, shap_global_rank_std_rho = bootstrap_spearman_ranking(abs_shap_global_per_sample, rounds=BOOTSTRAP_ROUNDS)

# top-k featury podľa globálnej dôležitosti (na perturbation metriky)
top_features = _topk_features_from_series(global_importance.set_index("feature")["mean_abs_shap"], TOP_K_FEATURES)

def shap_global_importance_vector(explainer, X_ref):
    vals = explainer(X_ref).values              # (n, f, c)
    imp = np.mean(np.abs(vals), axis=0).mean(axis=1)   # (f,)
    return imp

def shap_global_robustness(explainer, X_ref, noise_scale=0.01, trials=10, seed=42):
    rng = np.random.default_rng(seed)

    base_vec = shap_global_importance_vector(explainer, X_ref)
    denom = float(np.linalg.norm(base_vec, ord=2)) + 1e-12

    num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    dists = []

    for _ in range(trials):
        X_noisy = X_ref.copy().astype({c: float for c in num_cols})
        noise = rng.normal(0, noise_scale, size=(len(X_noisy), len(num_cols)))
        X_noisy.loc[:, num_cols] = X_noisy[num_cols].to_numpy(dtype=float) + noise
        noisy_vec = shap_global_importance_vector(explainer, X_noisy)
        dists.append(float(np.linalg.norm(base_vec - noisy_vec, ord=2) / denom))

    return float(np.mean(dists))

shap_global_rob = shap_global_robustness(explainer_shap,X_global,noise_scale=0.01, trials=10 if FAST_MODE else 20,seed=42)

def shap_global_monotonicity(X_ref: pd.DataFrame, shap_values_matrix: np.ndarray, feature_list):
    rhos = []
    for f in feature_list:
        j = X_ref.columns.get_loc(f)
        rho = _spearman(X_ref.iloc[:, j].values, shap_values_matrix[:, j])
        if not np.isnan(rho):
            rhos.append(abs(rho))  # sila monotónneho vzťahu
    if len(rhos) == 0:
        return np.nan, np.nan
    rhos = np.asarray(rhos, dtype=float)
    return float(np.mean(rhos)), float(np.std(rhos))

X_shap_global_mono = X_global.sample(n=min(TRENDABILITY_SAMPLES, len(X_global)), random_state=42)

shap_signed_vals = explainer_shap(X_shap_global_mono).values[:, :, pred_class]   # (n, f)

shap_global_mono_mean, shap_global_mono_std = shap_global_monotonicity(X_shap_global_mono,shap_signed_vals,top_features)

# 5. LIME – vysvetlenie + metriky
X_train_np = X_train.values
class_names = le.inverse_transform(np.arange(num_classes))

def tune_lime_kernel_width(
    X_train_np,
    feature_names,
    class_names,
    x_patient,
    predict_fn,
    widths,
    pred_class,
    num_features=10,
    num_samples=8000,
    seed=42
):
    best = (-np.inf, None, None)  # (score, width, exp)

    for w in widths:
        expl = LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=feature_names,
            class_names=[str(c) for c in class_names],
            mode="classification",
            discretize_continuous=True,
            kernel_width=float(w),
            random_state=seed
        )

        exp = expl.explain_instance(data_row=x_patient,predict_fn=predict_fn,labels=[int(pred_class)],num_features=num_features,num_samples=int(num_samples))

        score = float(exp.score)

        if score > best[0]:
            best = (score, w, exp)

    return best

LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 8000  # o
TUNE_LIME_KERNEL_WIDTH = True

base = float(np.sqrt(len(feature_names)))
kernel_width_candidates = [0.25*base, 0.5*base, 1.0*base, 2.0*base, 3.0*base]

if TUNE_LIME_KERNEL_WIDTH:
    best_score, best_w, lime_exp = tune_lime_kernel_width(
        X_train_np, feature_names, class_names, x_patient, lambda z: _predict_proba(model, z),
        widths=kernel_width_candidates,
        pred_class=pred_class,
        num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES, seed=42
    )

    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        mode="classification",
        discretize_continuous=True,
        kernel_width=float(best_w),
        random_state=42
    )

    print(f" LIME kernel_width tuned: best_w={best_w:.3f}, fidelity(R2)={best_score:.4f}")
else:
    explainer_lime = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        mode='classification',
        discretize_continuous=True,
        kernel_width=base,
        random_state=42
    )
    lime_exp = explainer_lime.explain_instance(
        data_row=x_patient,
        predict_fn=lambda z: _predict_proba(model, z),
        labels=[int(pred_class)],
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES
    )
print("\n LIME vysvetlenie (atribút, prínos):")
for feat, val in lime_exp.as_list(label=int(pred_class)):
    print("  ", feat, ":", val)

fig = lime_exp.as_pyplot_figure(label=int(pred_class))
plt.title(f"LIME – lokálne vysvetlenie pre predikovanú triedu {pred_label}")
plt.xlabel("Vplyv atribútu na predikciu")
plt.tight_layout()
plt.show()

lime_fidelity = float(lime_exp.score)

def lime_importance_dict(lime_exp,pred_class):
    return {k: float(v) for k, v in lime_exp.as_list(label=int(pred_class))}

def lime_top_feature_names(lime_exp, feature_names, pred_class: int, k: int = 10):
    pairs = lime_exp.local_exp.get(int(pred_class), [])
    pairs = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:k]
    idxs = [int(i) for i, _ in pairs]
    return [feature_names[i] for i in idxs if 0 <= i < len(feature_names)]

def lime_attr_map(lime_exp, feature_names, pred_class: int):
    pairs = lime_exp.local_exp.get(int(pred_class), [])
    out = {}
    for i, w in pairs:
        i = int(i)
        if 0 <= i < len(feature_names):
            out[feature_names[i]] = float(w)
    return out

def lime_importance_vector(lime_exp, pred_class):
    d = dict(lime_exp.as_list(label=int(pred_class)))
    keys = sorted(d.keys())
    vec = np.array([d[k] for k in keys], dtype=float)
    return keys, vec

def lime_importance_matrix(explainer_lime, model, X_ref, pred_class, num_features=10, num_samples=8000):
    mats = []

    for i in range(len(X_ref)):
        x_i = X_ref.iloc[i].values
        exp = explainer_lime.explain_instance(
            data_row=x_i,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples)
        )

        vec = np.zeros(len(feature_names), dtype=float)
        pairs = exp.local_exp.get(int(pred_class), [])
        for feat_idx, w in pairs:
            feat_idx = int(feat_idx)
            if 0 <= feat_idx < len(feature_names):
                vec[feat_idx] = abs(float(w))

        mats.append(vec)

    return np.asarray(mats, dtype=float)

def lime_stability(explainer_lime, model, x_instance, pred_class,noise_scale=0.01, trials=10, num_features=10, num_samples=8000):
    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: _predict_proba(model, z),
        labels=[int(pred_class)],
        num_features=num_features,
        num_samples=int(num_samples)
    )
    base_dict = dict(base_exp.as_list(label=int(pred_class)))
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        exp_noisy = explainer_lime.explain_instance(
            data_row=x_noisy,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples)
        )
        dict_n = dict(exp_noisy.as_list(label=int(pred_class)))
        keys = sorted(set(base_dict.keys()) | set(dict_n.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([dict_n.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def lime_consistency(explainer_lime, model, X_test, x_instance, pred_class, k=5, num_features=10, num_samples=8000):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_test.values)
    distances, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]

    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: _predict_proba(model, z),
        labels = [int(pred_class)],
        num_features=num_features,
        num_samples=int(num_samples)
    )
    base_dict = dict(base_exp.as_list(label=int(pred_class)))

    diffs = []
    for n_idx in neighbors:
        x_n = X_test.iloc[n_idx].values
        exp_n = explainer_lime.explain_instance(
            data_row=x_n,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples)
        )
        dict_n = dict(exp_n.as_list(label=int(pred_class)))
        keys = sorted(set(base_dict.keys()) | set(dict_n.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([dict_n.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def lime_monotonicity(explainer_lime,model, x_instance,pred_class,num_features=10,num_samples=5000,delta=0.01):
    exp = explainer_lime.explain_instance(x_instance,lambda z: _predict_proba(model, z),labels=[int(pred_class)],num_features=num_features,num_samples=num_samples)

    weights = dict(exp.as_list(label=int(pred_class)))

    total = 0
    correct = 0

    base_pred = _predict_proba(model, x_instance.reshape(1, -1))[0][pred_class]

    for feat_str, w in weights.items():
        for i, name in enumerate(feature_names):
            if name in feat_str:
                idx = i
                break
        else:
            continue

        x_new = x_instance.copy()
        x_new[idx] += delta
        new_pred = _predict_proba(model, x_new.reshape(1, -1))[0][pred_class]
        delta_pred = new_pred - base_pred

        if w == 0:
            continue

        total += 1
        if (w > 0 and delta_pred > 0) or (w < 0 and delta_pred < 0):
            correct += 1

    if total == 0:
        return np.nan

    return correct / total

lime_stab = lime_stability(explainer_lime, model, x_patient, pred_class, trials=10 if FAST_MODE else 20, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
lime_cons = lime_consistency(explainer_lime, model, X_test, x_patient, pred_class, k=5, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
mono_lime = lime_monotonicity(explainer_lime,model,x_patient,pred_class, num_features=LIME_NUM_FEATURES,num_samples=LIME_NUM_SAMPLES)

def lime_explain_fn(x):
    exp = explainer_lime.explain_instance( x,lambda z: _predict_proba(model, z),labels=[int(pred_class)],num_features=LIME_NUM_FEATURES,num_samples=LIME_NUM_SAMPLES)
    return dict(exp.as_list(label=int(pred_class)))

lime_rob = explanation_robustness(lime_explain_fn, x_patient)

def lime_sanity_check_random_labels(X_train, y_train, X_instance, feature_names, class_names, kernel_width, pred_class, num_features=10, num_samples=8000, n_estimators=60, seed=42):
    rng = np.random.default_rng(seed)
    y_rand = rng.permutation(y_train)
    m_rand = ExtraTreesClassifier(n_estimators=int(n_estimators),max_depth=None,max_features="sqrt", bootstrap=False,random_state=seed,n_jobs=-1)
    m_rand.fit(X_train, y_rand)

    expl = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        mode="classification",
        discretize_continuous=True,
        kernel_width=float(kernel_width),
        random_state=seed
    )

    exp_orig = expl.explain_instance(X_instance,lambda z: _predict_proba(model, z),labels=[int(pred_class)],num_features=num_features,num_samples=int(num_samples),)
    exp_rand = expl.explain_instance(X_instance,lambda z: m_rand.predict_proba(_ensure_df(z, feature_names)),labels=[int(pred_class)],num_features=num_features,num_samples=int(num_samples), )

    d_orig = dict(exp_orig.as_list(label=int(pred_class)))
    d_rand = dict(exp_rand.as_list(label=int(pred_class)))
    keys = sorted(set(d_orig.keys()) | set(d_rand.keys()))
    v1 = np.array([d_orig.get(k, 0.0) for k in keys], dtype=float)
    v2 = np.array([d_rand.get(k, 0.0) for k in keys], dtype=float)
    rho = _spearman(v1, v2)
    return float(rho), float(exp_orig.score), float(exp_rand.score)

lime_sanity_rho = np.nan
lime_sanity_fid_orig = np.nan
lime_sanity_fid_rand = np.nan
if RUN_LIME_SANITY_CHECK:
    print("\n LIME sanity check (random labels model) ...")
    (lime_sanity_rho, lime_sanity_fid_orig, lime_sanity_fid_rand), lime_sanity_time = _timeit(
        lime_sanity_check_random_labels,
        X_train, y_train, x_patient, feature_names, class_names, best_w if TUNE_LIME_KERNEL_WIDTH else base,
        pred_class, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES, n_estimators=SANITY_EXTRATREES_TREES
    )
    print(f"  Spearman(importance_orig, importance_random) = {lime_sanity_rho:.4f} (nižšie je lepšie)")
    print(f"  Fidelity orig = {lime_sanity_fid_orig:.3f}, Fidelity random = {lime_sanity_fid_rand:.3f}")
    print(f"  Čas sanity check: {lime_sanity_time:.2f}s")
# 6. BREAK DOWN (DALEX) – vysvetlenie + metriky
def dalex_predict_proba_pred_class(model, data):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)
    return model.predict_proba(_ensure_df(data, feature_names))[:, pred_class]

exp_dalex = dx.Explainer(
    model=model,
    data=X_train,
    y=_predict_proba(model, X_train)[:, pred_class],
    predict_function=dalex_predict_proba_pred_class,
    label=f"ExtraTrees_prob_class_{pred_class}"
)

bd = exp_dalex.predict_parts(x_patient_df,type="break_down")

print("\n Break Down tabuľka:")
print(bd.result)

df_bd = bd.result.copy()

var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
if var_col is None:
    raise RuntimeError(f"Neočakávané stĺpce v bd.result: {df_bd.columns.tolist()}")

tmp = df_bd[var_col].astype(str).str.lower()

intercept_rows = df_bd[(tmp == "intercept") | (tmp == "baseline")]
if len(intercept_rows) > 0:
    bd_intercept = float(intercept_rows["contribution"].iloc[0])
else:
    bd_intercept = 0.0

df_bd = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")].copy()

bd_contrib_sum = float(df_bd["contribution"].sum())
bd_reconstructed_pred = float(bd_intercept + bd_contrib_sum)

bd_target_pred = float(probs[pred_class])
bd_fidelity_error = abs(bd_reconstructed_pred - bd_target_pred)
bd_fidelity = float(max(0.0, 1.0 - bd_fidelity_error))

print(f" BreakDown fidelity: {bd_fidelity:.6f}")
print(f"   reconstructed = {bd_reconstructed_pred:.6f}")
print(f"   model prob     = {bd_target_pred:.6f}")
print(f"   abs error      = {bd_fidelity_error:.6e}")

df_bd["abs_contribution"] = df_bd["contribution"].abs()
df_plot = df_bd.sort_values("abs_contribution", ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(df_plot[var_col], df_plot["contribution"])
plt.gca().invert_yaxis()
plt.xlabel(f"Príspevok k pravdepodobnosti triedy {pred_label}")
plt.title("Break Down – TOP 15 atribútov pre daného pacienta")
plt.tight_layout()
plt.show()

def bd_vector(bd):
    df_bd = bd.result.copy()
    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        return [], np.array([], dtype=float)
    tmp = df_bd[var_col].astype(str).str.lower()
    df_attr = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")]
    return df_attr[var_col].tolist(), df_attr["contribution"].values
def bd_stability(exp_dalex, x_instance_df, noise_scale=0.01, trials=10):
    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base_dict = dict(zip(base_keys, base_vals))

    diffs = []
    num_cols = x_instance_df.select_dtypes(include=["float", "float32", "float64"]).columns

    for _ in range(trials):
        x_noisy = x_instance_df.copy().astype(float)
        if len(num_cols) > 0:
            arr = x_noisy[num_cols].values.astype(float)
            arr_noisy = arr + np.random.normal(0, noise_scale, size=arr.shape)
            x_noisy[num_cols] = arr_noisy

        bd_noisy = exp_dalex.predict_parts(x_noisy, type="break_down")
        keys_n, vals_n = bd_vector(bd_noisy)
        n_dict = dict(zip(keys_n, vals_n))

        all_keys = sorted(set(base_dict.keys()) | set(n_dict.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in all_keys], dtype=float)
        n = np.array([n_dict.get(k, 0.0) for k in all_keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))

    return float(np.mean(diffs))

def bd_consistency(exp_dalex, X_test, x_instance_df, k=5):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_test.values)
    distances, idx = nn.kneighbors([x_instance_df.values[0]])
    neighbors = idx[0][1:]

    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base_dict = dict(zip(base_keys, base_vals))

    diffs = []
    for n_idx in neighbors:
        x_n_df = X_test.iloc[n_idx:n_idx+1].copy().astype(float)
        bd_n = exp_dalex.predict_parts(x_n_df, type="break_down")
        keys_n, vals_n = bd_vector(bd_n)
        n_dict = dict(zip(keys_n, vals_n))
        all_keys = sorted(set(base_dict.keys()) | set(n_dict.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in all_keys], dtype=float)
        n = np.array([n_dict.get(k, 0.0) for k in all_keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

bd_stab = bd_stability(exp_dalex, x_patient_df, trials=10 if FAST_MODE else 20)
bd_cons = bd_consistency(exp_dalex, X_test, x_patient_df, k=5)

def bd_explain_fn(x):
    obs = pd.DataFrame([x], columns=feature_names).astype(float)
    bd = exp_dalex.predict_parts(obs, type="break_down")
    df_bd = bd.result.copy()

    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        return {}

    tmp = df_bd[var_col].astype(str).str.lower()
    df_bd = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")]

    return dict(zip(df_bd[var_col], df_bd["contribution"]))

bd_rob = explanation_robustness(bd_explain_fn, x_patient)

def bd_importance_matrix(exp_dalex, X_ref):
    mats = []

    for i in range(len(X_ref)):
        x_i_df = X_ref.iloc[i:i+1].copy().astype(float)
        bd_i = exp_dalex.predict_parts(x_i_df, type="break_down")

        keys, vals = bd_vector(bd_i)
        d = {k: abs(float(v)) for k, v in zip(keys, vals) if k in feature_names}

        vec = np.array([d.get(f, 0.0) for f in feature_names], dtype=float)
        mats.append(vec)

    return np.asarray(mats, dtype=float)

# 7. ANCHOR (Alibi) – pravidlá + metriky
import numpy as np
import pandas as pd
from alibi.explainers import AnchorTabular

anchor_precision = np.nan
anchor_coverage = np.nan
anchor_rule_len = np.nan
anchor_rule = None
anchor_time = np.nan

anchor_all_rules = []
MAX_ANCHOR_RULES = 5
MIN_FILTER_ROWS = 80

def _to_scalar(x):
    arr = np.asarray(x).reshape(-1)
    return float(arr[0]) if arr.size > 0 else np.nan

def _extract_anchor_parts(anchor_exp):
    if hasattr(anchor_exp, "anchor"):
        rule = anchor_exp.anchor
    elif hasattr(anchor_exp, "data") and "anchor" in anchor_exp.data:
        rule = anchor_exp.data["anchor"]
    else:
        rule = []

    if hasattr(anchor_exp, "precision"):
        precision = _to_scalar(anchor_exp.precision)
    elif hasattr(anchor_exp, "data") and "precision" in anchor_exp.data:
        precision = _to_scalar(anchor_exp.data["precision"])
    else:
        precision = np.nan

    if hasattr(anchor_exp, "coverage"):
        coverage = _to_scalar(anchor_exp.coverage)
    elif hasattr(anchor_exp, "data") and "coverage" in anchor_exp.data:
        coverage = _to_scalar(anchor_exp.data["coverage"])
    else:
        coverage = np.nan

    return rule, precision, coverage

def _parse_single_condition(cond: str):
    cond = str(cond).strip()
    ops = ["<=", ">=", "<", ">", "="]
    for op in ops:
        if op in cond:
            left, right = cond.split(op, 1)
            feature = left.strip()
            value_str = right.strip()
            try:
                value = float(value_str)
            except ValueError:
                value = value_str.strip().strip("'").strip('"')
            return feature, op, value
    raise ValueError(f"Nepodarilo sa parsovať anchor podmienku: {cond}")

def _condition_mask(df: pd.DataFrame, cond: str) -> pd.Series:
    feature, op, value = _parse_single_condition(cond)
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' z anchor pravidla nie je v DataFrame.")
    s = df[feature]
    if op == "<=":
        return s.astype(float) <= float(value)
    elif op == ">=":
        return s.astype(float) >= float(value)
    elif op == "<":
        return s.astype(float) < float(value)
    elif op == ">":
        return s.astype(float) > float(value)
    elif op == "=":
        try:
            return s.astype(float) == float(value)
        except Exception:
            return s.astype(str) == str(value)
    raise ValueError(f"Neznámy operátor: {op}")

def _rule_mask(df: pd.DataFrame, rule):
    if rule is None or len(rule) == 0:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    for cond in rule:
        mask &= _condition_mask(df, cond)
    return mask

def anchor_rule_holds(rule, x_instance, feature_names):
    row = pd.DataFrame([x_instance], columns=feature_names)
    return bool(_rule_mask(row, rule).iloc[0])

def anchor_robustness(rule, x_instance, feature_names, X_ref, noise_frac=0.05, trials=50):
    row = pd.DataFrame([x_instance], columns=feature_names)
    hits = 0
    rule_features = []
    for cond in rule:
        f, _, _ = _parse_single_condition(cond)
        if f in feature_names:
            rule_features.append(f)

    rule_features = list(dict.fromkeys(rule_features))

    for _ in range(trials):
        x_noisy = row.copy()

        for f in rule_features:
            col = X_ref[f]
            vals = pd.Series(col.dropna().unique())
            is_binary = len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0})
            if is_binary:
                continue
            else:
                sd = float(col.std())
                if not np.isfinite(sd) or sd <= 0:
                    continue
                delta = np.random.normal(0, noise_frac * sd)
                x_noisy.loc[:, f] = float(x_noisy.iloc[0][f]) + delta
        if bool(_rule_mask(x_noisy, rule).iloc[0]):
            hits += 1
    return float(hits / trials)

if RUN_ANCHOR:
    print("\n Anchor – generujem viac alternatívnych pravidiel...")
    predict_fn = lambda x: np.asarray(predict_df(model, x, feature_names)).reshape(-1)
    X_anchor_pool = X_train.copy()
    for r_idx in range(MAX_ANCHOR_RULES):
        if len(X_anchor_pool) < MIN_FILTER_ROWS:
            print(f"  Stop: v pool-e ostalo už len {len(X_anchor_pool)} riadkov.")
            break
        print(f"\n  --- Anchor kolo {r_idx+1} | pool size = {len(X_anchor_pool)} ---")
        anchor_explainer = AnchorTabular(predict_fn, feature_names)
        anchor_explainer.fit(X_anchor_pool.values, disc_perc=(25, 50, 75))

        def _run_anchor():
            try:
                return anchor_explainer.explain(
                    x_patient,
                    threshold=ANCHOR_THRESHOLD,
                    delta=ANCHOR_DELTA,
                    batch_size=ANCHOR_BATCH_SIZE,
                    beam_size=ANCHOR_BEAM_SIZE,
                    coverage_samples=ANCHOR_COVERAGE_SAMPLES,
                    verbose=False
                )
            except TypeError:
                return anchor_explainer.explain(
                    x_patient,
                    threshold=ANCHOR_THRESHOLD
                )
        try:
            anchor_exp, this_time = _timeit(_run_anchor)
            rule, precision, coverage_anchor_est = _extract_anchor_parts(anchor_exp)
            if rule is None or len(rule) == 0:
                print("   Anchor vrátil prázdne pravidlo. Končím.")
                break
            rule_tuple = tuple(rule)
            if any(tuple(x["rule"]) == rule_tuple for x in anchor_all_rules):
                print("  Našlo sa duplicitné pravidlo. Končím.")
                break
            pool_size_before = len(X_anchor_pool)
            mask_hit = _rule_mask(X_anchor_pool, rule)
            n_hit = int(mask_hit.sum())
            coverage_pool_empirical = n_hit / pool_size_before if pool_size_before > 0 else np.nan
            print("  IF", " AND ".join(rule))
            print("  Precision:", precision)
            print("  Coverage (Anchor odhad):", coverage_anchor_est)
            print("  Coverage (empirická v pool-e):", coverage_pool_empirical)
            print(f"  Pokryté riadky: {n_hit} / {pool_size_before}")
            print(f"  Čas Anchor: {this_time:.2f}s")
            if n_hit == 0:
                print("  Pravidlo nepokrylo žiadne riadky v pool-e. Končím.")
                break
            anchor_all_rules.append({
                "rule": list(rule),
                "precision": float(precision),
                "coverage_anchor_est": float(coverage_anchor_est) if np.isfinite(coverage_anchor_est) else np.nan,
                "coverage_pool_empirical": float(coverage_pool_empirical),
                "n_hit": int(n_hit),
                "rule_len": int(len(rule)),
                "time_sec": float(this_time),
                "pool_size_before": int(pool_size_before)
            })
            print(f"  Odstraňujem {n_hit} riadkov, kde pravidlo platí.")
            X_anchor_pool = X_anchor_pool.loc[~mask_hit].copy()

        except Exception as e:
            print("  Anchor zlyhal:", repr(e))
            break

    anchor_hold_prob = np.nan
    anchor_fragility = np.nan
    if len(anchor_all_rules) > 0:
        first_rule = anchor_all_rules[0]
        anchor_rule = first_rule["rule"]
        anchor_precision = first_rule["precision"]
        anchor_coverage = first_rule["coverage_pool_empirical"]
        anchor_rule_len = first_rule["rule_len"]
        anchor_time = first_rule["time_sec"]
        anchor_hold_prob = anchor_robustness(anchor_rule, x_patient, feature_names, X_train, noise_frac=0.05, trials=50)
        anchor_fragility = 1.0 - anchor_hold_prob
        print("\n Zhrnutie Anchor pravidiel:")
        for i, rr in enumerate(anchor_all_rules, 1):
            print(f"  {i}. IF {' AND '.join(rr['rule'])}")
            print(
                f"     precision={rr['precision']:.4f}, "
                f"coverage_pool={rr['coverage_pool_empirical']:.4f}, "
                f"coverage_anchor={rr['coverage_anchor_est']:.4f}, "
                f"n_hit={rr['n_hit']}/{rr['pool_size_before']}, "
                f"len={rr['rule_len']}, time={rr['time_sec']:.2f}s"
            )
    else:
        print("\n Nepodarilo sa nájsť žiadne neprázdne Anchor pravidlo.")
        anchor_rule = []
        anchor_rule_len = 0

# 11. PDP – Partial Dependence (globálne vysvetlenie)
print("\n PDP – Partial Dependence Analysis")

PDP_GRID_RESOLUTION = 20 if FAST_MODE else 50
from sklearn.inspection import partial_dependence
def compute_pdp_for_feature(model, X_ref, feature, class_idx, grid_resolution=20):
    pd_result = partial_dependence(
        model,
        X_ref,
        features=[feature],
        kind="average",
        grid_resolution=int(grid_resolution),
    )

    if hasattr(pd_result, "grid_values"):
        grid = pd_result.grid_values[0]
    elif "grid_values" in pd_result:
        grid = pd_result["grid_values"][0]
    elif hasattr(pd_result, "values"):
        grid = pd_result.values[0]
    elif "values" in pd_result:
        grid = pd_result["values"][0]
    else:
        raise KeyError(f"PDP output neobsahuje grid kľúče. Dostupné: {list(pd_result.keys())}")

    if hasattr(pd_result, "average"):
        avg = pd_result.average
    elif "average" in pd_result:
        avg = pd_result["average"]
    elif hasattr(pd_result, "averages"):
        avg = pd_result.averages
    elif "averages" in pd_result:
        avg = pd_result["averages"]
    else:
        raise KeyError(f"PDP output neobsahuje priemery. Dostupné: {list(pd_result.keys())}")

    avg = np.asarray(avg)

    if avg.ndim == 2:
        pdp_values = avg[int(class_idx)]
    elif avg.ndim == 3:
        pdp_values = avg[0, int(class_idx)]
    elif avg.ndim == 1:
        pdp_values = avg
    else:
        raise ValueError(f"Neočakávaný tvar PDP average: {avg.shape}")

    return np.asarray(grid), np.asarray(pdp_values)

def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return np.nan
    rx = x.argsort().argsort()
    ry = y.argsort().argsort()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def pdp_monotonicity_spearman(grid, values):
    return _safe_spearman(grid, values)

def pdp_smoothness_var(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.nan
    diffs = np.diff(values)
    return float(np.var(diffs))

def pdp_amplitude(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.max(values) - np.min(values))

def pdp_trendability(model, X_ref, feature, class_idx,rounds=20, grid_resolution=20, seed=42):
    rng = np.random.default_rng(seed)

    _, full_vals = compute_pdp_for_feature(model, X_ref, feature, class_idx, grid_resolution=grid_resolution)

    rhos = []
    n = len(X_ref)

    for _ in range(int(rounds)):
        idx = rng.integers(0, n, size=n)
        X_boot = X_ref.iloc[idx]
        _, boot_vals = compute_pdp_for_feature( model, X_boot, feature, class_idx, grid_resolution=grid_resolution)
        rho = _safe_spearman(full_vals, boot_vals)  # ∈ [-1, 1]
        rhos.append(abs(rho))                       # ∈ [0, 1]

    rhos = np.asarray(rhos, dtype=float)
    return float(np.nanmean(rhos)), float(np.nanstd(rhos))

def pdp_linearity_r2(grid, values):
    x = np.asarray(grid, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 3 or np.std(x) == 0:
        return np.nan
    a, b = np.polyfit(x, y, deg=1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def pdp_stability_shape_and_value(model, X_ref, feature, class_idx, rounds=10, grid_resolution=20, seed=42):
    rng = np.random.default_rng(seed)
    grid_full, vals_full = compute_pdp_for_feature(
        model, X_ref, feature, class_idx, grid_resolution=grid_resolution
    )

    rhos = []
    maes = []
    n = len(X_ref)

    for _ in range(int(rounds)):
        idx = rng.integers(0, n, size=n)
        X_boot = X_ref.iloc[idx]
        _, vals_boot = compute_pdp_for_feature(model, X_boot, feature, class_idx, grid_resolution=grid_resolution )
        rhos.append(_safe_spearman(vals_full, vals_boot))
        maes.append(float(np.mean(np.abs(vals_full - vals_boot))))

    rhos = np.asarray(rhos, dtype=float)
    maes = np.asarray(maes, dtype=float)

    return {
        "stability_shape_rho_mean": float(np.nanmean(rhos)),
        "stability_shape_rho_std": float(np.nanstd(rhos)),
        "stability_value_mae_mean": float(np.nanmean(maes)),
        "stability_value_mae_std": float(np.nanstd(maes)),
    }
def pdp_stability(model, X_ref, feature, class_idx, rounds=10, grid_resolution=20, seed=42, eps=1e-12):
    rng = np.random.default_rng(seed)

    _, vals_full = compute_pdp_for_feature(model, X_ref, feature, class_idx, grid_resolution=grid_resolution)
    denom = float(np.linalg.norm(vals_full, ord=2)) + eps

    n = len(X_ref)
    dists = []

    for _ in range(int(rounds)):
        idx = rng.integers(0, n, size=n)
        X_boot = X_ref.iloc[idx]
        _, vals_boot = compute_pdp_for_feature(model, X_boot, feature, class_idx, grid_resolution=grid_resolution)
        dists.append(float(np.linalg.norm(vals_full - vals_boot, ord=2) / denom))

    return float(np.mean(dists))

def ice_curves_for_feature(model, X_ref, feature, class_idx, grid, sample_n=60, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_ref)
    sample_n = int(min(sample_n, n))
    idxs = rng.choice(n, size=sample_n, replace=False)

    ice = np.zeros((sample_n, len(grid)), dtype=float)

    for ii, ridx in enumerate(idxs):
        row = X_ref.iloc[ridx:ridx+1].copy()
        for gi, g in enumerate(grid):
            row.loc[:, feature] = float(g)
            ice[ii, gi] = float(_predict_proba(model, row)[0, class_idx])

    return ice

def ice_variance_score(model, X_ref, feature, class_idx, grid, sample_n=60, seed=42):

    ice = ice_curves_for_feature(model, X_ref, feature, class_idx, grid, sample_n=sample_n, seed=seed)
    return float(np.mean(np.var(ice, axis=0)))

def pdp_robustness(model, X_ref, feature, class_idx, noise_scale=0.01, trials=10, grid_resolution=20, seed=42):
    rng = np.random.default_rng(seed)

    _, vals_base = compute_pdp_for_feature(model, X_ref, feature, class_idx, grid_resolution=grid_resolution)
    denom = float(np.linalg.norm(vals_base, ord=2)) + 1e-12

    dists = []
    num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()

    for _ in range(trials):
        X_noisy = X_ref.copy().astype({c: float for c in num_cols})
        noise = rng.normal(0, noise_scale, size=(len(X_noisy), len(num_cols)))
        X_noisy.loc[:, num_cols] = X_noisy[num_cols].to_numpy(dtype=float) + noise
        _, vals_noisy = compute_pdp_for_feature( model, X_noisy, feature, class_idx, grid_resolution=grid_resolution)
        dists.append(float(np.linalg.norm(vals_base - vals_noisy, ord=2) / denom))

    return float(np.mean(dists))

if RUN_PDP:
    print("\n PDP – Partial Dependence Analysis")
    print("\n PDP – metriky (rozšírené)")

    PDP_GRID_RESOLUTION = 20 if FAST_MODE else 50
    PDP_TOP_M = 5
    PDP_BOOT_ROUNDS = 5 if FAST_MODE else 20
    ICE_SAMPLE_N = 40 if FAST_MODE else 120

    pdp_metrics = []

    for f in top_features[:PDP_TOP_M]:
        # --- efficiency: čas na výpočet PDP krivky ---
        (grid, vals), t_pdp = _timeit(compute_pdp_for_feature, model, X_train, f, pred_class, PDP_GRID_RESOLUTION)

        mono_rho = pdp_monotonicity_spearman(grid, vals)       # ∈ [-1, 1]
        (tr_mean, tr_std), t_tr = _timeit(pdp_trendability,model, X_train, f, pred_class,PDP_BOOT_ROUNDS, PDP_GRID_RESOLUTION, 42)
        smooth = pdp_smoothness_var(vals)                      # >=0
        amp = pdp_amplitude(vals)                              # >=0
        lin_r2 = pdp_linearity_r2(grid, vals)                  # ∈ [0, 1] (ideálne)

        stab, t_stab = _timeit(pdp_stability,model, X_train, f, pred_class,PDP_BOOT_ROUNDS, PDP_GRID_RESOLUTION, 42)

        ice_var, t_ice = _timeit(ice_variance_score,model, X_train, f, pred_class, grid,ICE_SAMPLE_N, 42)
        rob, t_rob = _timeit(pdp_robustness,model, X_train, f, pred_class,0.01, 10 if FAST_MODE else 20, PDP_GRID_RESOLUTION, 42)

        pdp_metrics.append({
            "top 5 atributov": f,
            "pdp_efficiency_sec": float(t_pdp), #čas výpočtu samotnej PDP krivky
            "stability": float(stab),
            "robustness": float(rob),
            "monotonicity": float(mono_rho), #rho
            "trendability": tr_mean,      #mean_rho
            "smoothness_var": float(smooth),
            "amplitude": float(amp),
            "linearity_r2": float(lin_r2),
            "ice_variance": float(ice_var),
        })

        # plot
        plt.figure(figsize=(6, 4))
        plt.plot(grid, vals)
        plt.xlabel(f)
        plt.ylabel(f"Pravdepodobnosť, že model predikuje triedu {pred_label}")
        plt.title(f"PDP – {f}")
        plt.tight_layout()
        plt.show()

    pdp_df = pd.DataFrame(pdp_metrics)

    print("\nPDP METRIKY – TABUĽKA")
    print(pdp_df)
    pdp_smoothness_global = float(pdp_df["smoothness_var"].mean())
    pdp_amplitude_global = float(pdp_df["amplitude"].mean())
    pdp_linearity_global = float(pdp_df["linearity_r2"].mean())
    pdp_ice_variance_global = float(pdp_df["ice_variance"].mean())
    pdp_robustness_global = float(pdp_df["robustness"].mean())

    print("\nPDP ŠPECIFICKÉ GLOBÁLNE METRIKY")
    print("  smoothness_var (nižšie lepšie):", pdp_smoothness_global)
    print("  amplitude:", pdp_amplitude_global)
    print("  linearity_r2 (vyššie = lineárnejší vzťah):", pdp_linearity_global)
    print("  ice_variance (nižšie = menej heterogénne):", pdp_ice_variance_global)
    print("  robustness (nizsie = lepsie):", pdp_robustness_global)

else:
    pdp_df = None
    pdp_smoothness_global = np.nan
    pdp_amplitude_global = np.nan
    pdp_linearity_global = np.nan
    pdp_ice_variance_global = np.nan
    pdp_robustness_global = np.nan

baseline = _baseline_row(X_train)
local_abs = pd.Series(np.abs(values_for_class), index=feature_names)
local_top_features = local_abs.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()

completeness_shap = completeness_drop_auc(model, x_patient_df, local_top_features, baseline, pred_class)
necessity_shap = necessity_score(model, x_patient_df, local_top_features, baseline, pred_class)
suff_prob_shap = sufficiency_retained_prob(model, x_patient_df, local_top_features, baseline, pred_class)
suff_class_shap = sufficiency_class_stable(model, x_patient_df, local_top_features, baseline, pred_class)

shap_attr_map = {f: float(pd.Series(values_for_class, index=feature_names).get(f, 0.0)) for f in feature_names}
mono_shap = monotonicity_local(model, x_patient_df, local_top_features, pred_class, shap_attr_map, X_train)

X_trend = X_global.sample(n=min(TRENDABILITY_SAMPLES, len(X_global)), random_state=42)

shap_trend_vals = explainer_shap(X_trend).values[:, :, pred_class]   # (n, f), signed SHAP pre triedu
trend_mean_rho, trend_std_rho = trendability_spearman( X_trend, shap_trend_vals,top_features,use_abs=True)

shap_imp_dict = {f: float(local_abs.loc[f]) for f in local_top_features}
parsimony_shap = explanation_parsimony_topk(shap_imp_dict, threshold=0.95)
simp_gini_shap, simp_entropy_shap = simplicity_entropy_and_gini(shap_imp_dict)

descriptive_accuracy_shap = np.nan

application_utility_shap = np.nan

lime_imp = lime_importance_dict(lime_exp, pred_class)
parsimony_lime = explanation_parsimony_topk(lime_imp, threshold=0.95)
simp_gini_lime, simp_entropy_lime = simplicity_entropy_and_gini(lime_imp)

lime_top_features = lime_top_feature_names(lime_exp, feature_names, pred_class, k=TOP_K_FEATURES)
completeness_lime = completeness_drop_auc(model, x_patient_df, lime_top_features, baseline, pred_class)
necessity_lime = necessity_score(model, x_patient_df, lime_top_features, baseline, pred_class)
suff_prob_lime = sufficiency_retained_prob(model, x_patient_df, lime_top_features, baseline, pred_class)
suff_class_lime = sufficiency_class_stable(model, x_patient_df, lime_top_features, baseline, pred_class)
lime_attr = lime_attr_map(lime_exp, feature_names, pred_class)

X_trend_lime = X_global.sample(n=min(TRENDABILITY_SAMPLES, len(X_global)), random_state=42)

lime_trend_imp = lime_importance_matrix(explainer_lime,model,X_trend_lime,pred_class,num_features=LIME_NUM_FEATURES,num_samples=LIME_NUM_SAMPLES)

trend_mean_rho_lime, trend_std_rho_lime = trendability_spearman(X_trend_lime,lime_trend_imp,lime_top_features)
descriptive_accuracy_lime = np.nan
application_utility_lime = np.nan

bd_keys, bd_vals = bd_vector(bd)
bd_imp_full = {k: float(v) for k, v in zip(bd_keys, bd_vals)}
bd_feat_candidates = [f for f in bd_keys if f in feature_names]
if len(bd_feat_candidates) == 0:
    bd_top_features = local_top_features
else:
    bd_top_features = sorted(
        bd_feat_candidates,
        key=lambda f: abs(bd_imp_full.get(f, 0.0)),
        reverse=True
    )[:TOP_K_FEATURES]

bd_imp = {f: float(bd_imp_full.get(f, 0.0)) for f in bd_top_features}
parsimony_bd = explanation_parsimony_topk(bd_imp, threshold=0.95)
simp_gini_bd, simp_entropy_bd = simplicity_entropy_and_gini(bd_imp)

completeness_bd = completeness_drop_auc(model, x_patient_df, bd_top_features, baseline, pred_class)
necessity_bd = necessity_score(model, x_patient_df, bd_top_features, baseline, pred_class)
suff_prob_bd = sufficiency_retained_prob(model, x_patient_df, bd_top_features, baseline, pred_class)
suff_class_bd = sufficiency_class_stable(model, x_patient_df, bd_top_features, baseline, pred_class)
bd_attr = {f: float(bd_imp.get(f, 0.0)) for f in bd_top_features}
mono_bd = monotonicity_local(model, x_patient_df, bd_top_features, pred_class, bd_attr, X_train)

X_trend_bd = X_global.sample(n=min(TRENDABILITY_SAMPLES, len(X_global)), random_state=42)

bd_trend_imp = bd_importance_matrix(exp_dalex, X_trend_bd)

trend_mean_rho_bd, trend_std_rho_bd = trendability_spearman( X_trend_bd,bd_trend_imp,bd_top_features)
descriptive_accuracy_bd = np.nan
application_utility_bd = np.nan

print("\n================= XAI METRIKY – SÚHRN =================")

print("\nSHAP:")
print("  Stabilita (nižšie lepšie):", shap_stab)
print("  Konzistentnosť (nižšie lepšie):", shap_cons)
print("  Robustnosť vysvetlenia:", shap_rob)
print("  SHAP global ranking stability (bootstrap Spearman):", shap_global_rank_mean_rho, "+/-", shap_global_rank_std_rho)

print("\nLIME:")
print("  Fidelity (lime_exp.score, bližšie k 1 lepšie):", lime_fidelity)
print("  Stabilita (nižšie lepšie):", lime_stab)
print("  Konzistentnosť (nižšie lepšie):", lime_cons)
print("  Robustnosť vysvetlenia:", lime_rob)
if RUN_LIME_SANITY_CHECK:
    print("  Sanity Spearman (orig vs random-label model) (nižšie lepšie):", lime_sanity_rho)

print("\nBreak Down (DALEX):")
print("  Stabilita (nižšie lepšie):", bd_stab)
print("  Konzistentnosť (nižšie lepšie):", bd_cons)
print("  Robustnosť vysvetlenia:", bd_rob)

if RUN_ANCHOR:
    print("\nAnchor:")
    print("  Precision:", anchor_precision)
    print("  Coverage:", anchor_coverage)
    print("  Dĺžka pravidla:", anchor_rule_len)
    print("  Čas:", anchor_time)
    print("  Robustnosť pravidla (pravdepodobnosť zachovania)-vyssie lepsie:", anchor_hold_prob)
    print("  Fragility pravidla (nižšie lepšie) ...1-robustnost:", anchor_fragility)

rows = []

rows.append({
    "method": "SHAP_local",
    "fidelity": shap_fidelity,
    "consistency": shap_cons,
    "completeness": completeness_shap,
    "stability": shap_stab,
    "robustness": shap_rob,
    "efficiency_sec": np.nan,
    "monotonicity": mono_shap,
    "trendability": trend_mean_rho,   #_mean_rho
    "necessity": necessity_shap,
    "sufficiency": suff_prob_shap,    #_retained_prob
    "explanation_parsimony": parsimony_shap,
    "simplicity": simp_entropy_shap,       #_entropy
    "precision": np.nan,
    "coverage": np.nan,

})
rows.append({
    "method": "SHAP_global",
    "fidelity": np.nan,
    "consistency": np.nan,
    "completeness": np.nan,
    "stability": 1.0 - shap_global_rank_mean_rho, #nizsie = lepsie
    "robustness": shap_global_rob,
    "efficiency_sec": np.nan,
    "monotonicity": shap_global_mono_mean,
    "trendability": trend_mean_rho,
    "necessity": np.nan,
    "sufficiency": np.nan,
    "explanation_parsimony": np.nan,
    "simplicity": np.nan,
    "precision": np.nan,
    "coverage": np.nan,
})
rows.append({
    "method": "LIME",
    "fidelity": lime_fidelity,
    "consistency": lime_cons,
    "completeness": completeness_lime,
    "stability": lime_stab,
    "robustness": lime_rob,
    "efficiency_sec": np.nan,
    "monotonicity": mono_lime,
    "trendability": trend_mean_rho_lime, #_mean_rho
    "necessity": necessity_lime,
    "sufficiency": suff_prob_lime,  #_retained_prob
    "explanation_parsimony": parsimony_lime,
    "simplicity": simp_entropy_lime,     #_entropy
    "precision": np.nan,
    "coverage": np.nan,
})

rows.append({
    "method": "BreakDown",
    "fidelity": bd_fidelity,
    "consistency": bd_cons,
    "completeness": completeness_bd,
    "stability": bd_stab,
    "robustness": bd_rob,
    "efficiency_sec": np.nan,
    "monotonicity": mono_bd,
    "trendability": trend_mean_rho_bd,         #_mean_rho
    "necessity": necessity_bd,
    "sufficiency": suff_prob_bd,       #_retained_prob
    "explanation_parsimony": parsimony_bd,
    "simplicity": simp_entropy_bd,    #_entropy
    "precision": np.nan,
    "coverage": np.nan,
})

if RUN_ANCHOR:
    rows.append({
        "method": "Anchor",
        "fidelity": np.nan,
        "consistency": np.nan,
        "completeness": np.nan,
        "stability": np.nan,
        "robustness": anchor_fragility,
        "efficiency_sec": anchor_time,
        "monotonicity": np.nan,
        "trendability": np.nan, #_mean_rho
        "necessity": np.nan,
        "sufficiency": np.nan,   #_retained_prob
        "explanation_parsimony": anchor_rule_len,
        "simplicity": np.nan,     #_entropy
        "precision": anchor_precision,
        "coverage": anchor_coverage,
    })

if RUN_PDP and pdp_df is not None and not pdp_df.empty:
    pdp_eff = float(pdp_df["pdp_efficiency_sec"].mean())
    pdp_mono = float(pdp_df["monotonicity"].abs().mean())  # mean signed rho
    pdp_trend_mean = float(pdp_df["trendability"].mean())
    pdp_trend_std = float(pdp_df["trendability"].std())

    pdp_stability_global = float(pdp_df["stability"].mean())

    rows.append({
        "method": "PDP",
        "fidelity": np.nan,
        "consistency": np.nan,
        "completeness": np.nan,
        "stability": pdp_stability_global,
        "robustness": pdp_robustness_global,
        "efficiency_sec": pdp_eff,
        "monotonicity": pdp_mono,  # spojité [-1,1]
        "trendability": float(pdp_df["trendability"].mean()), #_mean_rho
        "necessity": np.nan,
        "sufficiency": np.nan,  #_retained_prob
        "explanation_parsimony": np.nan,
        "simplicity": np.nan,  #_entropy
        "precision": np.nan,
        "coverage": np.nan,
    })

xai_table = pd.DataFrame(rows)

print("\n================= XAI METRIKY – TABUĽKA =================")
print(xai_table)

try:
    out_xlsx = "extratrees_xai_metriky_vysledky.xlsx"
    xai_table.to_excel(out_xlsx, index=False)
    print(f"\n Exportované: {out_xlsx}")
except Exception as e:
    print(" Export do Excel zlyhal:", repr(e))
