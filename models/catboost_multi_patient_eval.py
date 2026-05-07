# -*- coding: utf-8 -*-
"""
Samostatný skript pre vyhodnotenie XAI metrík na tabulkových klinických dátach pomocou:
- CatBoost
- SHAP local
- LIME
- DALEX Break Down
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.neighbors import NearestNeighbors

from catboost import CatBoostClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import dalex as dx

warnings.filterwarnings("ignore", category=FutureWarning, module="dalex")
# 0) KONFIG
PATH = "../data.xlsx"
TARGET = "Závažnosť priebehu ochorenia"

FAST_MODE = True

RUN_SINGLE_PATIENT_DEMO = True
SINGLE_PATIENT_IDX = 0
SHOW_PLOTS_SINGLE = True

# multi-patient evaluácia
RUN_MULTI_PATIENT_EVAL = True
N_EVAL_PATIENTS = 20
MULTI_RANDOM_STATE = 42
SHOW_PLOTS_MULTI = False

# metriky / výkon
TOP_K_FEATURES = 20 if FAST_MODE else 60
TRENDABILITY_SAMPLES = 80 if FAST_MODE else 200
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 3000 if FAST_MODE else 8000
TUNE_LIME_KERNEL_WIDTH = True
RUN_LIME_SANITY_CHECK_SINGLE = True   # len v single demo
SANITY_CATBOOST_TREES = 60 if FAST_MODE else 150

# export
EXPORT_DIR = "/mnt/data"
EXPORT_PREFIX = "xai_multi_patient"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 60)

# globálne pomocné premenné (naplnia sa v main)
feature_names: List[str] = []

# 1) HELPER FUNKCIE
def _predict_proba(model: CatBoostClassifier, X_any) -> np.ndarray:
    if isinstance(X_any, pd.DataFrame):
        return model.predict_proba(X_any)
    arr = np.asarray(X_any)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return model.predict_proba(pd.DataFrame(arr, columns=feature_names))

def _topk_features_from_series(series: pd.Series, k: int) -> List[str]:
    return series.sort_values(ascending=False).head(int(k)).index.tolist()

def _as_float_df(df1: pd.DataFrame) -> pd.DataFrame:
    return df1.copy().astype(float)

def _spearman(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return np.nan
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def _timeit(fn: Callable, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, float(time.perf_counter() - t0)

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    s = ez.sum()
    if s <= 0:
        return np.full_like(z, 1.0 / len(z))
    return ez / s

def _baseline_row(X_ref: pd.DataFrame) -> pd.Series:
    base = {}
    for c in X_ref.columns:
        col = X_ref[c]
        vals = pd.Series(col.dropna().unique())
        if len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0}):
            base[c] = float(col.mode(dropna=True).iloc[0]) if not col.mode(dropna=True).empty else 0.0
        else:
            base[c] = float(col.median(skipna=True))
    return pd.Series(base)

def completeness_drop_auc(
    model: CatBoostClassifier,
    x_df: pd.DataFrame,
    importance_order: List[str],
    baseline: pd.Series,
    pred_class: int,
    nonneg: bool = True,
    normalize: bool = True,
) -> float:
    x_new = _as_float_df(x_df)
    p0 = float(_predict_proba(model, x_new)[0, pred_class])

    drops = [0.0]
    for f in importance_order:
        if f in x_new.columns:
            x_new.loc[:, f] = float(baseline[f])
        p = float(_predict_proba(model, x_new)[0, pred_class])
        d = p0 - p
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

def necessity_score(
    model: CatBoostClassifier,
    x_df: pd.DataFrame,
    top_features: List[str],
    baseline: pd.Series,
    pred_class: int,
    nonneg: bool = True,
) -> float:
    p0 = float(_predict_proba(model, x_df)[0, pred_class])
    deltas: List[float] = []
    for f in top_features:
        if f not in x_df.columns:
            continue
        x_m = _as_float_df(x_df)
        x_m.loc[:, f] = float(baseline[f])
        pm = float(_predict_proba(model, x_m)[0, pred_class])
        d = p0 - pm
        if nonneg:
            d = max(0.0, d)
        deltas.append(float(d))
    return float(np.mean(deltas)) if deltas else np.nan

def _logit_safe(p, eps=1e-8):
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))
def sufficiency_retained_prob(model, x_df, top_features, baseline, pred_class, eps=1e-8):
    p0 = float(_predict_proba(model, x_df)[0, pred_class])

    x_new = _as_float_df(x_df)
    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])

    pk = float(_predict_proba(model, x_new)[0, pred_class])

    denom = max(p0, eps)
    score = pk / denom
    return float(np.clip(score, 0.0, 1.0))

def sufficiency_logit_retention(model, x_df, top_features, baseline, pred_class, eps=1e-8):
    p0 = float(_predict_proba(model, x_df)[0, pred_class])

    x_new = _as_float_df(x_df)
    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])

    pk = float(_predict_proba(model, x_new)[0, pred_class])

    l0 = _logit_safe(p0, eps=eps)
    lk = _logit_safe(pk, eps=eps)

    if abs(l0) < eps:
        return np.nan

    score = lk / l0
    return float(np.clip(score, 0.0, 1.0))

def monotonicity_local(
    model: CatBoostClassifier,
    x_df: pd.DataFrame,
    feature_list: List[str],
    pred_class: int,
    attr_map: Dict[str, float],
    X_ref: pd.DataFrame,
    delta_frac: float = 0.01,
) -> float:
    hits: List[float] = []
    for f in feature_list:
        if f not in x_df.columns:
            continue

        attr_sign = np.sign(float(attr_map.get(f, 0.0)))
        if attr_sign == 0:
            continue

        col = X_ref[f]
        vals = pd.Series(col.dropna().unique())
        is_binary = len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0})

        x_plus = _as_float_df(x_df)
        x_minus = _as_float_df(x_df)

        if is_binary:
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

def explanation_robustness(
    explain_fn: Callable[[np.ndarray], Dict[str, float]],
    x_instance: np.ndarray,
    noise_scale: float = 0.01,
    trials: int = 10,
) -> float:
    base = explain_fn(x_instance)
    diffs: List[float] = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        noisy = explain_fn(x_noisy)
        keys = sorted(set(base.keys()) | set(noisy.keys()))
        b = np.array([base.get(k, 0.0) for k in keys])
        n = np.array([noisy.get(k, 0.0) for k in keys])
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def explanation_parsimony_topk(importance_dict: Dict[str, float], threshold: float = 0.95) -> float:
    if not importance_dict:
        return np.nan
    vals = np.array(sorted([abs(v) for v in importance_dict.values()], reverse=True), dtype=float)
    if vals.sum() == 0:
        return np.nan
    cum = np.cumsum(vals) / vals.sum()
    return int(np.searchsorted(cum, threshold) + 1)

def simplicity_entropy_and_gini(importance_dict: Dict[str, float]) -> Tuple[float, float]:
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

def trendability_spearman(
    X_ref: pd.DataFrame,
    importance_matrix: np.ndarray,
    feature_list: List[str],
    use_abs: bool = True,
) -> Tuple[float, float]:
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

def evaluate_model(model: CatBoostClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, object]:
    y_pred = np.asarray(model.predict(X_test)).reshape(-1).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

# 2) SHAP / LIME / BREAKDOWN helper metriky
def shap_stability(explainer, x_instance: np.ndarray, feature_names: List[str], noise_scale: float = 0.01, trials: int = 10) -> float:
    base = explainer(pd.DataFrame([x_instance], columns=feature_names)).values.flatten()
    diffs: List[float] = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        shap_noisy = explainer(pd.DataFrame([x_noisy], columns=feature_names)).values.flatten()
        diffs.append(np.linalg.norm(base - shap_noisy))
    return float(np.mean(diffs))

def shap_consistency(explainer, X_test: pd.DataFrame, x_instance: np.ndarray, feature_names: List[str], k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]

    base = explainer(pd.DataFrame([x_instance], columns=feature_names)).values.flatten()
    rhos: List[float] = []
    for n in neighbors:
        shap_n = explainer(X_test.iloc[n:n + 1]).values.flatten()
        rho = _spearman(base, shap_n)
        if not np.isnan(rho):
            rhos.append(abs(rho))
    return float(np.mean(rhos)) if rhos else np.nan

def lime_importance_dict(lime_exp, pred_class: int) -> Dict[str, float]:
    return {k: float(v) for k, v in lime_exp.as_list(label=int(pred_class))}

def lime_top_feature_names(lime_exp, feature_names: List[str], pred_class: int, k: int = 10) -> List[str]:
    pairs = lime_exp.local_exp.get(int(pred_class), [])
    pairs = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)[:k]
    idxs = [int(i) for i, _ in pairs]
    return [feature_names[i] for i in idxs if 0 <= i < len(feature_names)]

def lime_attr_map(lime_exp, feature_names: List[str], pred_class: int) -> Dict[str, float]:
    pairs = lime_exp.local_exp.get(int(pred_class), [])
    out: Dict[str, float] = {}
    for i, w in pairs:
        i = int(i)
        if 0 <= i < len(feature_names):
            out[feature_names[i]] = float(w)
    return out

def lime_importance_matrix(
    explainer_lime,
    model: CatBoostClassifier,
    X_ref: pd.DataFrame,
    pred_class: int,
    feature_names: List[str],
    num_features: int = 10,
    num_samples: int = 3000,
) -> np.ndarray:
    mats = []
    for i in range(len(X_ref)):
        x_i = X_ref.iloc[i].values
        exp = explainer_lime.explain_instance(
            data_row=x_i,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples),
        )
        vec = np.zeros(len(feature_names), dtype=float)
        pairs = exp.local_exp.get(int(pred_class), [])
        for feat_idx, w in pairs:
            feat_idx = int(feat_idx)
            if 0 <= feat_idx < len(feature_names):
                vec[feat_idx] = abs(float(w))
        mats.append(vec)
    return np.asarray(mats, dtype=float)

def lime_stability(
    explainer_lime,
    model: CatBoostClassifier,
    x_instance: np.ndarray,
    pred_class: int,
    noise_scale: float = 0.01,
    trials: int = 10,
    num_features: int = 10,
    num_samples: int = 3000,
) -> float:
    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: _predict_proba(model, z),
        labels=[int(pred_class)],
        num_features=num_features,
        num_samples=int(num_samples),
    )
    base_dict = dict(base_exp.as_list(label=int(pred_class)))
    diffs: List[float] = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        exp_noisy = explainer_lime.explain_instance(
            data_row=x_noisy,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples),
        )
        dict_n = dict(exp_noisy.as_list(label=int(pred_class)))
        keys = sorted(set(base_dict.keys()) | set(dict_n.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([dict_n.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def lime_consistency(
    explainer_lime,
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    x_instance: np.ndarray,
    pred_class: int,
    k: int = 5,
    num_features: int = 10,
    num_samples: int = 3000,
) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]

    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: _predict_proba(model, z),
        labels=[int(pred_class)],
        num_features=num_features,
        num_samples=int(num_samples),
    )
    base_dict = dict(base_exp.as_list(label=int(pred_class)))

    diffs: List[float] = []
    for n_idx in neighbors:
        x_n = X_test.iloc[n_idx].values
        exp_n = explainer_lime.explain_instance(
            data_row=x_n,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples),
        )
        dict_n = dict(exp_n.as_list(label=int(pred_class)))
        keys = sorted(set(base_dict.keys()) | set(dict_n.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([dict_n.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def lime_monotonicity(
    explainer_lime,
    model: CatBoostClassifier,
    x_instance: np.ndarray,
    pred_class: int,
    feature_names: List[str],
    num_features: int = 10,
    num_samples: int = 3000,
    delta: float = 0.01,
) -> float:
    exp = explainer_lime.explain_instance(
        x_instance,
        lambda z: _predict_proba(model, z),
        labels=[int(pred_class)],
        num_features=num_features,
        num_samples=num_samples,
    )
    weights = dict(exp.as_list(label=int(pred_class)))
    total = 0
    correct = 0
    base_pred = _predict_proba(model, x_instance.reshape(1, -1))[0][pred_class]

    for feat_str, w in weights.items():
        idx = None
        for i, name in enumerate(feature_names):
            if name in feat_str:
                idx = i
                break
        if idx is None or w == 0:
            continue

        x_new = x_instance.copy()
        x_new[idx] += delta
        new_pred = _predict_proba(model, x_new.reshape(1, -1))[0][pred_class]
        delta_pred = new_pred - base_pred

        total += 1
        if (w > 0 and delta_pred > 0) or (w < 0 and delta_pred < 0):
            correct += 1

    if total == 0:
        return np.nan
    return correct / total

def bd_vector(bd) -> Tuple[List[str], np.ndarray]:
    df_bd = bd.result.copy()
    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        return [], np.array([], dtype=float)
    tmp = df_bd[var_col].astype(str).str.lower()
    df_attr = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")]
    return df_attr[var_col].tolist(), df_attr["contribution"].values

def bd_stability(exp_dalex, x_instance_df: pd.DataFrame, noise_scale: float = 0.01, trials: int = 10) -> float:
    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base_dict = dict(zip(base_keys, base_vals))

    diffs: List[float] = []
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

def bd_consistency(exp_dalex, X_test: pd.DataFrame, x_instance_df: pd.DataFrame, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance_df.values[0]])
    neighbors = idx[0][1:]

    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base_dict = dict(zip(base_keys, base_vals))

    diffs: List[float] = []
    for n_idx in neighbors:
        x_n_df = X_test.iloc[n_idx:n_idx + 1].copy().astype(float)
        bd_n = exp_dalex.predict_parts(x_n_df, type="break_down")
        keys_n, vals_n = bd_vector(bd_n)
        n_dict = dict(zip(keys_n, vals_n))
        all_keys = sorted(set(base_dict.keys()) | set(n_dict.keys()))
        b = np.array([base_dict.get(k, 0.0) for k in all_keys], dtype=float)
        n = np.array([n_dict.get(k, 0.0) for k in all_keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))

def build_lime_explainer(
    X_train: pd.DataFrame,
    feature_names: List[str],
    class_names: np.ndarray,
    x_patient: np.ndarray,
    model: CatBoostClassifier,
    pred_class: int,
    num_features: int,
    num_samples: int,
    tune_kernel_width: bool = True,
):
    X_train_np = X_train.values
    base = float(np.sqrt(len(feature_names)))
    kernel_width_candidates = [0.25 * base, 0.5 * base, 1.0 * base, 2.0 * base, 3.0 * base]

    def tune_lime_kernel_width(
        widths,
    ):
        best = (-np.inf, None, None)
        for w in widths:
            expl = LimeTabularExplainer(
                training_data=X_train_np,
                feature_names=feature_names,
                class_names=[str(c) for c in class_names],
                mode="classification",
                discretize_continuous=True,
                kernel_width=float(w),
                random_state=42,
            )
            exp = expl.explain_instance(
                data_row=x_patient,
                predict_fn=lambda z: _predict_proba(model, z),
                labels=[int(pred_class)],
                num_features=num_features,
                num_samples=int(num_samples),
            )
            score = float(exp.score)
            if score > best[0]:
                best = (score, w, exp)
        return best

    if tune_kernel_width:
        best_score, best_w, lime_exp = tune_lime_kernel_width(kernel_width_candidates)
        explainer_lime = LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=feature_names,
            class_names=[str(c) for c in class_names],
            mode="classification",
            discretize_continuous=True,
            kernel_width=float(best_w),
            random_state=42,
        )
    else:
        best_w = base
        explainer_lime = LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=feature_names,
            class_names=[str(c) for c in class_names],
            mode="classification",
            discretize_continuous=True,
            kernel_width=base,
            random_state=42,
        )
        lime_exp = explainer_lime.explain_instance(
            data_row=x_patient,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=num_features,
            num_samples=int(num_samples),
        )
        best_score = float(lime_exp.score)

    return explainer_lime, lime_exp, float(best_w), float(best_score)

def build_dalex_explainer(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    pred_class: int,
    feature_names: List[str],
):
    def dalex_predict_proba_pred_class(model_, data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=feature_names)
        return model_.predict_proba(data)[:, pred_class]

    exp_dalex = dx.Explainer(
        model=model,
        data=X_train,
        y=model.predict_proba(X_train)[:, pred_class],
        predict_function=dalex_predict_proba_pred_class,
        label=f"CatBoost_prob_class_{pred_class}",
        verbose=False,
    )
    return exp_dalex

def evaluate_single_patient(
    patient_idx: int,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model: CatBoostClassifier,
    le: LabelEncoder,
    explainer_shap,
    baseline: pd.Series,
    class_names: np.ndarray,
    show_plots: bool = True,
    run_lime_sanity: bool = True,
):
    x_patient_df = X_test.iloc[patient_idx:patient_idx + 1]
    x_patient = x_patient_df.values[0]

    probs = model.predict_proba(x_patient_df)[0]
    pred_class = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_class])[0]

    true_class = int(y_test[patient_idx])
    true_label = le.inverse_transform([true_class])[0]

    print(f"\nPacient #{patient_idx}")
    print(f"Predikovaná trieda {pred_label} | pravdepodobnosti: {probs}")
    print(f"Skutočná trieda: {true_label}")
    print("\n Hodnoty pacienta:")
    for col in X_test.columns:
        value = x_patient_df.iloc[0][col]
        print(f"{col}: {value}")

    # SHAP local
    shap_vals_patient = explainer_shap(x_patient_df)
    values_for_class = shap_vals_patient.values[0, :, pred_class]
    base_for_class = shap_vals_patient.base_values[0, pred_class]
    data_for_class = shap_vals_patient.data[0]

    shap_exp_single = shap.Explanation(
        values=values_for_class,
        base_values=base_for_class,
        data=data_for_class,
        feature_names=shap_vals_patient.feature_names,
    )

    shap_margins_reconstructed = (
        np.asarray(shap_vals_patient.base_values[0], dtype=float)
        + np.sum(shap_vals_patient.values[0], axis=0)
    )
    shap_probs_reconstructed = _softmax(shap_margins_reconstructed)
    model_margins = np.asarray(model.predict(x_patient_df, prediction_type="RawFormulaVal")).reshape(-1)
    shap_margin_error = float(np.max(np.abs(shap_margins_reconstructed - model_margins)))
    shap_prob_error = float(np.max(np.abs(shap_probs_reconstructed - probs)))
    shap_fidelity = float(max(0.0, 1.0 - shap_prob_error))

    print("\n SHAP fidelity (probability-scale):", shap_fidelity)
    print("   reconstructed probs =", shap_probs_reconstructed)
    print("   model probs         =", probs)
    print("   max abs prob error  =", shap_prob_error)
    print("   max abs margin err  =", shap_margin_error)

    if show_plots:
        shap.plots.waterfall(shap_exp_single, max_display=15, show=False)
        plt.gcf().set_size_inches(14, 10)
        plt.subplots_adjust(left=0.4)
        plt.show()

    shap_stab = shap_stability(explainer_shap, x_patient, feature_names)
    shap_cons = shap_consistency(explainer_shap, X_test, x_patient, feature_names)

    def shap_explain_fn(x):
        x_df = pd.DataFrame([x], columns=feature_names)
        sv = explainer_shap(x_df)
        vals = sv.values[0, :, pred_class]
        return dict(zip(feature_names, vals))

    shap_rob = explanation_robustness(shap_explain_fn, x_patient)

    local_abs = pd.Series(np.abs(values_for_class), index=feature_names)
    local_top_features = local_abs.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()
    completeness_shap = completeness_drop_auc(model, x_patient_df, local_top_features, baseline, pred_class)
    necessity_shap = necessity_score(model, x_patient_df, local_top_features, baseline, pred_class)
    suff_prob_shap = sufficiency_retained_prob(model, x_patient_df, local_top_features, baseline, pred_class)
    suff_logit_shap = sufficiency_logit_retention(model, x_patient_df, local_top_features, baseline, pred_class)
    shap_attr_map = {f: float(pd.Series(values_for_class, index=feature_names).get(f, 0.0)) for f in feature_names}
    mono_shap = monotonicity_local(model, x_patient_df, local_top_features, pred_class, shap_attr_map, X_train)
    shap_imp_dict = {f: float(local_abs.loc[f]) for f in local_top_features}
    parsimony_shap = explanation_parsimony_topk(shap_imp_dict, threshold=0.95)
    _, simp_entropy_shap = simplicity_entropy_and_gini(shap_imp_dict)

    # SHAP detail tabuľka
    shap_local_table = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values_for_class,
        "abs_shap_value": np.abs(values_for_class),
        "feature_value": x_patient_df.iloc[0].values,
    }).sort_values("abs_shap_value", ascending=False)

    print("\n SHAP local tabuľka (top 20):")
    print(shap_local_table.head(20))

    # LIME
    explainer_lime, lime_exp, best_w, best_score = build_lime_explainer(
        X_train,
        feature_names,
        class_names,
        x_patient,
        model,
        pred_class,
        LIME_NUM_FEATURES,
        LIME_NUM_SAMPLES,
        tune_kernel_width=TUNE_LIME_KERNEL_WIDTH,
    )
    print(f"\n LIME kernel_width tuned: best_w={best_w:.3f}, fidelity(R2)={best_score:.4f}")
    print("\n LIME vysvetlenie (atribút, prínos):")
    for feat, val in lime_exp.as_list(label=int(pred_class)):
        print("  ", feat, ":", val)

    if show_plots:
        fig = lime_exp.as_pyplot_figure(label=int(pred_class))
        plt.title(f"LIME – lokálne vysvetlenie pre predikovanú triedu {pred_label}")
        plt.xlabel("Vplyv atribútu na predikciu")
        plt.tight_layout()
        plt.show()

    lime_fidelity = float(lime_exp.score)
    lime_stab = lime_stability(explainer_lime, model, x_patient, pred_class, trials=10 if FAST_MODE else 20, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
    lime_cons = lime_consistency(explainer_lime, model, X_test, x_patient, pred_class, k=5, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)
    mono_lime = lime_monotonicity(explainer_lime, model, x_patient, pred_class, feature_names, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES)

    def lime_explain_fn(x):
        exp = explainer_lime.explain_instance(
            x,
            lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        return dict(exp.as_list(label=int(pred_class)))

    lime_rob = explanation_robustness(lime_explain_fn, x_patient)
    lime_imp = lime_importance_dict(lime_exp, pred_class)
    parsimony_lime = explanation_parsimony_topk(lime_imp, threshold=0.95)
    _, simp_entropy_lime = simplicity_entropy_and_gini(lime_imp)
    lime_top_features = lime_top_feature_names(lime_exp, feature_names, pred_class, k=TOP_K_FEATURES)
    completeness_lime = completeness_drop_auc(model, x_patient_df, lime_top_features, baseline, pred_class)
    necessity_lime = necessity_score(model, x_patient_df, lime_top_features, baseline, pred_class)
    suff_prob_lime = sufficiency_retained_prob(model, x_patient_df, lime_top_features, baseline, pred_class)
    suff_logit_lime = sufficiency_logit_retention(model, x_patient_df, lime_top_features, baseline, pred_class)

    lime_table = pd.DataFrame(lime_exp.as_list(label=int(pred_class)), columns=["feature_interval", "contribution"])
    lime_table["abs_contribution"] = lime_table["contribution"].abs()
    lime_table = lime_table.sort_values("abs_contribution", ascending=False)

    print("\n LIME tabuľka:")
    print(lime_table)

    if run_lime_sanity:
        def lime_sanity_check_random_labels(X_train_, y_train_, X_instance_, feature_names_, class_names_, kernel_width_, pred_class_, num_features=10, num_samples=3000, n_estimators=60, seed=42):
            rng = np.random.default_rng(seed)
            y_rand = rng.permutation(y_train_)
            m_rand = CatBoostClassifier(
                iterations=int(n_estimators),
                depth=5,
                learning_rate=0.1,
                loss_function="MultiClass",
                random_seed=seed,
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
            )
            m_rand.fit(X_train_, y_rand)

            expl = LimeTabularExplainer(
                training_data=X_train_.values,
                feature_names=feature_names_,
                class_names=[str(c) for c in class_names_],
                mode="classification",
                discretize_continuous=True,
                kernel_width=float(kernel_width_),
                random_state=seed,
            )

            exp_orig = expl.explain_instance(X_instance_, lambda z: _predict_proba(model, z), labels=[int(pred_class_)], num_features=num_features, num_samples=int(num_samples))
            exp_rand = expl.explain_instance(
                X_instance_,
                lambda z: m_rand.predict_proba(pd.DataFrame(np.asarray(z), columns=feature_names_)),
                labels=[int(pred_class_)],
                num_features=num_features,
                num_samples=int(num_samples),
            )
            d_orig = dict(exp_orig.as_list(label=int(pred_class_)))
            d_rand = dict(exp_rand.as_list(label=int(pred_class_)))
            keys = sorted(set(d_orig.keys()) | set(d_rand.keys()))
            v1 = np.array([d_orig.get(k, 0.0) for k in keys], dtype=float)
            v2 = np.array([d_rand.get(k, 0.0) for k in keys], dtype=float)
            rho = _spearman(v1, v2)
            return float(rho), float(exp_orig.score), float(exp_rand.score)

        (lime_sanity_rho, lime_sanity_fid_orig, lime_sanity_fid_rand), lime_sanity_time = _timeit(
            lime_sanity_check_random_labels,
            X_train, y_train, x_patient, feature_names, class_names, best_w,
            pred_class, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES, n_estimators=SANITY_CATBOOST_TREES,
        )
        print("\n LIME sanity check (random labels model) ...")
        print(f"  Spearman(importance_orig, importance_random) = {lime_sanity_rho:.4f} (nižšie je lepšie)")
        print(f"  Fidelity orig = {lime_sanity_fid_orig:.3f}, Fidelity random = {lime_sanity_fid_rand:.3f}")
        print(f"  Čas sanity check: {lime_sanity_time:.2f}s")

    # BreakDown
    exp_dalex = build_dalex_explainer(model, X_train, pred_class, feature_names)
    bd = exp_dalex.predict_parts(x_patient_df, type="break_down")
    print("\n Break Down tabuľka:")
    print(bd.result)

    df_bd = bd.result.copy()
    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        raise RuntimeError(f"Neočakávané stĺpce v bd.result: {df_bd.columns.tolist()}")
    tmp = df_bd[var_col].astype(str).str.lower()
    intercept_rows = df_bd[(tmp == "intercept") | (tmp == "baseline")]
    bd_intercept = float(intercept_rows["contribution"].iloc[0]) if len(intercept_rows) > 0 else 0.0
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

    if show_plots:
        df_bd_plot = df_bd.copy()
        df_bd_plot["abs_contribution"] = df_bd_plot["contribution"].abs()
        df_plot = df_bd_plot.sort_values("abs_contribution", ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        plt.barh(df_plot[var_col], df_plot["contribution"])
        plt.gca().invert_yaxis()
        plt.xlabel(f"Príspevok k pravdepodobnosti triedy {pred_label}")
        plt.title("Break Down – TOP 15 atribútov pre daného pacienta")
        plt.tight_layout()
        plt.show()

    bd_stab = bd_stability(exp_dalex, x_patient_df, trials=10 if FAST_MODE else 20)
    bd_cons = bd_consistency(exp_dalex, X_test, x_patient_df, k=5)

    def bd_explain_fn(x):
        obs = pd.DataFrame([x], columns=feature_names).astype(float)
        bd_loc = exp_dalex.predict_parts(obs, type="break_down")
        df_loc = bd_loc.result.copy()
        vc = "variable_name" if "variable_name" in df_loc.columns else ("variable" if "variable" in df_loc.columns else None)
        if vc is None:
            return {}
        tmp_loc = df_loc[vc].astype(str).str.lower()
        df_loc = df_loc[(tmp_loc != "intercept") & (tmp_loc != "baseline") & (tmp_loc != "")]
        return dict(zip(df_loc[vc], df_loc["contribution"]))

    bd_rob = explanation_robustness(bd_explain_fn, x_patient)
    bd_keys, bd_vals = bd_vector(bd)
    bd_imp_full = {k: float(v) for k, v in zip(bd_keys, bd_vals)}
    bd_feat_candidates = [f for f in bd_keys if f in feature_names]
    if len(bd_feat_candidates) == 0:
        bd_top_features = local_top_features
    else:
        bd_top_features = sorted(
            bd_feat_candidates,
            key=lambda f: abs(bd_imp_full.get(f, 0.0)),
            reverse=True,
        )[:TOP_K_FEATURES]

    bd_imp = {f: float(bd_imp_full.get(f, 0.0)) for f in bd_top_features}
    parsimony_bd = explanation_parsimony_topk(bd_imp, threshold=0.95)
    _, simp_entropy_bd = simplicity_entropy_and_gini(bd_imp)
    completeness_bd = completeness_drop_auc(model, x_patient_df, bd_top_features, baseline, pred_class)
    necessity_bd = necessity_score(model, x_patient_df, bd_top_features, baseline, pred_class)
    suff_prob_bd = sufficiency_retained_prob(model, x_patient_df, bd_top_features, baseline, pred_class)
    suff_logit_bd = sufficiency_logit_retention(model, x_patient_df, bd_top_features, baseline, pred_class)
    bd_attr = {f: float(bd_imp.get(f, 0.0)) for f in bd_top_features}
    mono_bd = monotonicity_local(model, x_patient_df, bd_top_features, pred_class, bd_attr, X_train)

    bd_table = df_bd.copy()
    bd_table["abs_contribution"] = bd_table["contribution"].abs()
    bd_table = bd_table.sort_values("abs_contribution", ascending=False)

    print("\n XAI METRIKY – SÚHRN (1 pacient) ")
    print("\nSHAP:")
    print("  Stabilita (nižšie lepšie):", shap_stab)
    print("  Konzistentnosť (nižšie lepšie):", shap_cons)
    print("  Robustnosť vysvetlenia:", shap_rob)
    print("\nLIME:")
    print("  Fidelity (R2 lokálneho surrogate modelu):", lime_fidelity)
    print("  Stabilita (nižšie lepšie):", lime_stab)
    print("  Konzistentnosť (nižšie lepšie):", lime_cons)
    print("  Robustnosť vysvetlenia:", lime_rob)
    print("\nBreak Down:")
    print("  Stabilita (nižšie lepšie):", bd_stab)
    print("  Konzistentnosť (nižšie lepšie):", bd_cons)
    print("  Robustnosť vysvetlenia:", bd_rob)

    single_rows = [
        {
            "patient_idx": patient_idx,
            "true_class": true_label,
            "pred_class": pred_label,
            "method": "SHAP_local",
            "fidelity": shap_fidelity,
            "consistency": shap_cons,
            "completeness": completeness_shap,
            "stability": shap_stab,
            "robustness": shap_rob,
            "monotonicity": mono_shap,
            "trendability": np.nan,
            "necessity": necessity_shap,
            "sufficiency": suff_prob_shap,    #_retained_prob
            "sufficiency_logit": suff_logit_shap,
            "explanation_parsimony": parsimony_shap,
            "simplicity": 1 / (1 + simp_entropy_shap),
        },
        {
            "patient_idx": patient_idx,
            "true_class": true_label,
            "pred_class": pred_label,
            "method": "LIME",
            "fidelity": lime_fidelity,
            "consistency": lime_cons,
            "completeness": completeness_lime,
            "stability": lime_stab,
            "robustness": lime_rob,
            "monotonicity": mono_lime,
            "trendability": np.nan,
            "necessity": necessity_lime,
            "sufficiency": suff_prob_lime,
            "sufficiency_logit": suff_logit_lime,
            "explanation_parsimony": parsimony_lime,
            "simplicity": 1 / (1 + simp_entropy_lime),
        },
        {
            "patient_idx": patient_idx,
            "true_class": true_label,
            "pred_class": pred_label,
            "method": "BreakDown",
            "fidelity": bd_fidelity,
            "consistency": bd_cons,
            "completeness": completeness_bd,
            "stability": bd_stab,
            "robustness": bd_rob,
            "monotonicity": mono_bd,
            "trendability": np.nan,
            "necessity": necessity_bd,
            "sufficiency": suff_prob_bd,
            "sufficiency_logit": suff_logit_bd,
            "explanation_parsimony": parsimony_bd,
            "simplicity": 1 / (1 + simp_entropy_bd),
        },
    ]

    single_metric_table = pd.DataFrame(single_rows)
    print("\nTABUĽKA METRÍK – 1 pacient")
    print(single_metric_table)

    try:
        shap_local_table.to_excel(os.path.join(EXPORT_DIR, f"{EXPORT_PREFIX}_patient_{patient_idx}_shap_table.xlsx"), index=False)
        lime_table.to_excel(os.path.join(EXPORT_DIR, f"{EXPORT_PREFIX}_patient_{patient_idx}_lime_table.xlsx"), index=False)
        bd_table.to_excel(os.path.join(EXPORT_DIR, f"{EXPORT_PREFIX}_patient_{patient_idx}_breakdown_table.xlsx"), index=False)
        single_metric_table.to_excel(os.path.join(EXPORT_DIR, f"{EXPORT_PREFIX}_patient_{patient_idx}_metrics.xlsx"), index=False)
        print("\n Exportované detailné tabuľky pre jedného pacienta.")
    except Exception as e:
        print(" Export detailných tabuliek zlyhal:", repr(e))

    return {
        "metric_table": single_metric_table,
        "shap_table": shap_local_table,
        "lime_table": lime_table,
        "breakdown_table": bd_table,
    }
# 4) MULTI-PATIENT EVAL
def evaluate_multi_patient(
    eval_patient_indices: List[int],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_train: pd.DataFrame,
    model: CatBoostClassifier,
    le: LabelEncoder,
    explainer_shap,
    baseline: pd.Series,
    explainer_lime,
    class_names: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multi_rows: List[Dict[str, object]] = []

    for patient_idx in eval_patient_indices:
        print(f"\nMULTI-EVAL PACIENT {patient_idx} ")

        x_patient_df = X_test.iloc[patient_idx:patient_idx + 1]
        x_patient = x_patient_df.values[0]

        probs = model.predict_proba(x_patient_df)[0]
        pred_class = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_class])[0]
        true_class = int(y_test[patient_idx])
        true_label = le.inverse_transform([true_class])[0]

        # SHAP local
        shap_vals_patient = explainer_shap(x_patient_df)
        values_for_class = shap_vals_patient.values[0, :, pred_class]
        shap_margins_reconstructed = (
            np.asarray(shap_vals_patient.base_values[0], dtype=float)
            + np.sum(shap_vals_patient.values[0], axis=0)
        )
        shap_probs_reconstructed = _softmax(shap_margins_reconstructed)
        shap_prob_error = float(np.max(np.abs(shap_probs_reconstructed - probs)))
        shap_fidelity = float(max(0.0, 1.0 - shap_prob_error))
        shap_stab = shap_stability(explainer_shap, x_patient, feature_names)
        shap_cons = shap_consistency(explainer_shap, X_test, x_patient, feature_names)

        def shap_explain_fn(x):
            x_df = pd.DataFrame([x], columns=feature_names)
            sv = explainer_shap(x_df)
            vals = sv.values[0, :, pred_class]
            return dict(zip(feature_names, vals))

        shap_rob = explanation_robustness(shap_explain_fn, x_patient)
        local_abs = pd.Series(np.abs(values_for_class), index=feature_names)
        local_top_features = local_abs.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()
        completeness_shap = completeness_drop_auc(model, x_patient_df, local_top_features, baseline, pred_class)
        necessity_shap = necessity_score(model, x_patient_df, local_top_features, baseline, pred_class)
        suff_prob_shap = sufficiency_retained_prob(model, x_patient_df, local_top_features, baseline, pred_class)
        suff_logit_shap = sufficiency_logit_retention(model, x_patient_df, local_top_features, baseline, pred_class)
        shap_attr_map = {f: float(pd.Series(values_for_class, index=feature_names).get(f, 0.0)) for f in feature_names}
        mono_shap = monotonicity_local(model, x_patient_df, local_top_features, pred_class, shap_attr_map, X_train)
        shap_imp_dict = {f: float(local_abs.loc[f]) for f in local_top_features}
        parsimony_shap = explanation_parsimony_topk(shap_imp_dict, threshold=0.95)
        _, simp_entropy_shap = simplicity_entropy_and_gini(shap_imp_dict)

        # LIME
        lime_exp = explainer_lime.explain_instance(
            data_row=x_patient,
            predict_fn=lambda z: _predict_proba(model, z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        lime_fidelity = float(lime_exp.score)
        lime_stab = lime_stability(
            explainer_lime, model, x_patient, pred_class,
            trials=10 if FAST_MODE else 20,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        lime_cons = lime_consistency(
            explainer_lime, model, X_test, x_patient, pred_class,
            k=5, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES,
        )
        mono_lime = lime_monotonicity(
            explainer_lime, model, x_patient, pred_class, feature_names,
            num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES,
        )

        def lime_explain_fn(x):
            exp = explainer_lime.explain_instance(
                x,
                lambda z: _predict_proba(model, z),
                labels=[int(pred_class)],
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )
            return dict(exp.as_list(label=int(pred_class)))

        lime_rob = explanation_robustness(lime_explain_fn, x_patient)
        lime_imp = lime_importance_dict(lime_exp, pred_class)
        parsimony_lime = explanation_parsimony_topk(lime_imp, threshold=0.95)
        _, simp_entropy_lime = simplicity_entropy_and_gini(lime_imp)
        lime_top_features = lime_top_feature_names(lime_exp, feature_names, pred_class, k=TOP_K_FEATURES)
        completeness_lime = completeness_drop_auc(model, x_patient_df, lime_top_features, baseline, pred_class)
        necessity_lime = necessity_score(model, x_patient_df, lime_top_features, baseline, pred_class)
        suff_prob_lime = sufficiency_retained_prob(model, x_patient_df, lime_top_features, baseline, pred_class)
        suff_logit_lime = sufficiency_logit_retention(model, x_patient_df, lime_top_features, baseline, pred_class)

        # BreakDown
        exp_dalex = build_dalex_explainer(model, X_train, pred_class, feature_names)
        bd = exp_dalex.predict_parts(x_patient_df, type="break_down")
        df_bd = bd.result.copy()
        var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
        tmp = df_bd[var_col].astype(str).str.lower()
        intercept_rows = df_bd[(tmp == "intercept") | (tmp == "baseline")]
        bd_intercept = float(intercept_rows["contribution"].iloc[0]) if len(intercept_rows) > 0 else 0.0
        df_bd = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")].copy()
        bd_contrib_sum = float(df_bd["contribution"].sum())
        bd_reconstructed_pred = float(bd_intercept + bd_contrib_sum)
        bd_target_pred = float(probs[pred_class])
        bd_fidelity_error = abs(bd_reconstructed_pred - bd_target_pred)
        bd_fidelity = float(max(0.0, 1.0 - bd_fidelity_error))
        bd_stab = bd_stability(exp_dalex, x_patient_df, trials=10 if FAST_MODE else 20)
        bd_cons = bd_consistency(exp_dalex, X_test, x_patient_df, k=5)

        def bd_explain_fn(x):
            obs = pd.DataFrame([x], columns=feature_names).astype(float)
            bd_loc = exp_dalex.predict_parts(obs, type="break_down")
            df_loc = bd_loc.result.copy()
            vc = "variable_name" if "variable_name" in df_loc.columns else ("variable" if "variable" in df_loc.columns else None)
            if vc is None:
                return {}
            tmp_loc = df_loc[vc].astype(str).str.lower()
            df_loc = df_loc[(tmp_loc != "intercept") & (tmp_loc != "baseline") & (tmp_loc != "")]
            return dict(zip(df_loc[vc], df_loc["contribution"]))

        bd_rob = explanation_robustness(bd_explain_fn, x_patient)
        bd_keys, bd_vals = bd_vector(bd)
        bd_imp_full = {k: float(v) for k, v in zip(bd_keys, bd_vals)}
        bd_feat_candidates = [f for f in bd_keys if f in feature_names]
        if len(bd_feat_candidates) == 0:
            bd_top_features = local_top_features
        else:
            bd_top_features = sorted(
                bd_feat_candidates,
                key=lambda f: abs(bd_imp_full.get(f, 0.0)),
                reverse=True,
            )[:TOP_K_FEATURES]
        bd_imp = {f: float(bd_imp_full.get(f, 0.0)) for f in bd_top_features}
        parsimony_bd = explanation_parsimony_topk(bd_imp, threshold=0.95)
        _, simp_entropy_bd = simplicity_entropy_and_gini(bd_imp)
        completeness_bd = completeness_drop_auc(model, x_patient_df, bd_top_features, baseline, pred_class)
        necessity_bd = necessity_score(model, x_patient_df, bd_top_features, baseline, pred_class)
        suff_prob_bd = sufficiency_retained_prob(model, x_patient_df, bd_top_features, baseline, pred_class)
        suff_logit_bd = sufficiency_logit_retention(model, x_patient_df, bd_top_features, baseline, pred_class)
        bd_attr = {f: float(bd_imp.get(f, 0.0)) for f in bd_top_features}
        mono_bd = monotonicity_local(model, x_patient_df, bd_top_features, pred_class, bd_attr, X_train)

        multi_rows.extend([
            {
                "method": "SHAP_local",
                "fidelity": shap_fidelity,
                "consistency": shap_cons,
                "completeness": completeness_shap,
                "stability": shap_stab,
                "robustness": shap_rob,
                "monotonicity": mono_shap,
                "trendability": np.nan,
                "necessity": necessity_shap,
                "sufficiency": suff_prob_shap,
                "sufficiency_logit": suff_logit_shap,
                "explanation_parsimony": parsimony_shap,
                "simplicity": 1 / (1 + simp_entropy_shap),
            },
            {
                "method": "LIME",
                "fidelity": lime_fidelity,
                "consistency": lime_cons,
                "completeness": completeness_lime,
                "stability": lime_stab,
                "robustness": lime_rob,
                "monotonicity": mono_lime,
                "trendability": np.nan,
                "necessity": necessity_lime,
                "sufficiency": suff_prob_lime,
                "sufficiency_logit": suff_logit_lime,
                "explanation_parsimony": parsimony_lime,
                "simplicity": 1 / (1 + simp_entropy_lime),
            },
            {
                "method": "BreakDown",
                "fidelity": bd_fidelity,
                "consistency": bd_cons,
                "completeness": completeness_bd,
                "stability": bd_stab,
                "robustness": bd_rob,
                "monotonicity": mono_bd,
                "trendability": np.nan,
                "necessity": necessity_bd,
                "sufficiency": suff_prob_bd,
                "sufficiency_logit": suff_logit_bd,
                "explanation_parsimony": parsimony_bd,
                "simplicity": 1 / (1 + simp_entropy_bd),
            },
        ])

    multi_patient_table = pd.DataFrame(multi_rows)
    multi_patient_mean = multi_patient_table.groupby("method").mean(numeric_only=True)
    multi_patient_median = multi_patient_table.groupby("method").median(numeric_only=True)
    multi_patient_std = multi_patient_table.groupby("method").std(numeric_only=True)

    return multi_patient_table, multi_patient_mean, multi_patient_median, multi_patient_std

# 5) MAIN
def main():
    global feature_names

    if not os.path.exists(PATH):
        raise FileNotFoundError(
            f"Dátový súbor neexistuje: {PATH}\n"
            f"Uprav premennú PATH v hlavičke skriptu."
        )

    df = pd.read_excel(PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(np.unique(y_enc))

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.15,
        random_state=42,
        stratify=y_enc,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.17647,
        random_state=42,
        stratify=y_temp,
    )

    print("Train shape:", X_train.shape)
    print("Val shape:  ", X_val.shape)
    print("Test shape: ", X_test.shape)

    feature_names = X.columns.tolist()
    class_names = le.inverse_transform(np.arange(num_classes))

    model = CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )

    print(" Trénujem CatBoost...")
    model.fit(X_train, y_train)
    print(" Model natrenovaný.")

    metrics = evaluate_model(model, X_test, y_test)
    print("\n Klasické metriky CatBoostu:")
    for k, v in metrics.items():
        print(k, ":\n", v, "\n")

    baseline = _baseline_row(X_train)
    explainer_shap = shap.TreeExplainer(model)

    demo_patient_df = X_test.iloc[SINGLE_PATIENT_IDX:SINGLE_PATIENT_IDX + 1]
    demo_patient = demo_patient_df.values[0]
    demo_probs = model.predict_proba(demo_patient_df)[0]
    demo_pred_class = int(np.argmax(demo_probs))

    explainer_lime, _, best_w, best_score = build_lime_explainer(
        X_train,
        feature_names,
        class_names,
        demo_patient,
        model,
        demo_pred_class,
        LIME_NUM_FEATURES,
        LIME_NUM_SAMPLES,
        tune_kernel_width=TUNE_LIME_KERNEL_WIDTH,
    )
    print(f"\n LIME global init: best_w={best_w:.3f}, demo fidelity(R2)={best_score:.4f}")

    if RUN_SINGLE_PATIENT_DEMO:
        evaluate_single_patient(
            SINGLE_PATIENT_IDX,
            X_test,
            y_test,
            X_train,
            y_train,
            model,
            le,
            explainer_shap,
            baseline,
            class_names,
            show_plots=SHOW_PLOTS_SINGLE,
            run_lime_sanity=RUN_LIME_SANITY_CHECK_SINGLE,
        )

    if RUN_MULTI_PATIENT_EVAL:
        rng = np.random.default_rng(MULTI_RANDOM_STATE)
        eval_patient_indices = rng.choice(len(X_test), size=min(N_EVAL_PATIENTS, len(X_test)), replace=False)
        eval_patient_indices = sorted(eval_patient_indices.tolist())
        print("\n Multi-patient evaluácia pre indexy:", eval_patient_indices)

        multi_patient_table, multi_patient_mean, multi_patient_median, multi_patient_std = evaluate_multi_patient(
            eval_patient_indices,
            X_test,
            y_test,
            X_train,
            model,
            le,
            explainer_shap,
            baseline,
            explainer_lime,
            class_names,
        )

        print("\nMULTI-PATIENT DETAIL")
        print(multi_patient_table)

        print("\n MULTI-PATIENT MEAN ")
        print(multi_patient_mean)

        print("\n MULTI-PATIENT MEDIAN ")
        print(multi_patient_median)

        print("\n MULTI-PATIENT STD ")
        print(multi_patient_std)

        try:
            out_xlsx = "CatBoost_xai_multi_patient_eval_detail.xlsx"
            out_xlsx_mean = "CatBoost_xai_multi_patient_eval_mean.xlsx"
            out_xlsx_median = "CatBoost_xai_multi_patient_eval_median.xlsx"
            out_xlsx_std = "CatBoost_xai_multi_patient_eval_std.xlsx"

            multi_patient_table.to_excel(out_xlsx, index=False)
            multi_patient_mean.to_excel(out_xlsx_mean, index=False)
            multi_patient_median.to_excel(out_xlsx_median, index=False)
            multi_patient_std.to_excel(out_xlsx_std, index=False)
            print("\nExport multi-patient tabuliek hotový.")
        except Exception as e:
            print(" Export multi-patient tabuliek zlyhal:", repr(e))


if __name__ == "__main__":
    main()
