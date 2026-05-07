# -*- coding: utf-8 -*-
"""
TabNet valid pipeline pre klinicke tabulkove data.

Obsahuje:
- nacitanie dat
- train/val/test split
- TabNetClassifier
- klasicke metriky + confusion matrix graf
- TabNet native explanations
- LIME lokalne vysvetlenie
- DALEX BreakDown
- multi-patient vyhodnotenie pre N pacientov

Dolezite:
- SHAP TreeExplainer tu nepouzivam, pretoze TabNet nie je stromovy model.
- Vsetky vstupy pre TabNet su konvertovane na numpy float32.
- Predikcie idu cez wrapper, aby mali vsetky XAI metody konzistentne rozhranie.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError as e:
    raise ImportError(
        "Chyba kniznica pytorch-tabnet alebo torch. Nainstaluj: pip install pytorch-tabnet torch\n"
        f"Povodna chyba: {e}"
    )

from lime.lime_tabular import LimeTabularExplainer
import dalex as dx

warnings.filterwarnings("ignore", category=FutureWarning, module="dalex")

# =========================================================
# 0) KONFIG
# =========================================================

PATH = "../data.xlsx"
TARGET = "Závažnosť priebehu ochorenia"

FAST_MODE = True
RUN_SINGLE_PATIENT_DEMO = True
RUN_MULTI_PATIENT_EVAL = True
SINGLE_PATIENT_IDX = 0
N_EVAL_PATIENTS = 20
MULTI_RANDOM_STATE = 42
SHOW_PLOTS = False

TOP_K_FEATURES = 20 if FAST_MODE else 60
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 2500 if FAST_MODE else 7000
TABNET_MAX_EPOCHS = 100 if FAST_MODE else 250
TABNET_PATIENCE = 20
TABNET_BATCH_SIZE = 128
TABNET_VIRTUAL_BATCH_SIZE = 32
SEED = 42

EXPORT_PREFIX = "tabnet_xai"

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 70)

feature_names: List[str] = []

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model: TabNetClassifier, feature_names_: List[str]):
        self.model = model
        self.feature_names = list(feature_names_)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def _to_numpy(self, X_any) -> np.ndarray:
        if isinstance(X_any, pd.DataFrame):
            arr = X_any.loc[:, self.feature_names].to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X_any, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(f"Ocakavanych {len(self.feature_names)} atributov, dostane {arr.shape[1]}.")
        return arr.astype(np.float32, copy=False)

    def predict_proba(self, X_any) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._to_numpy(X_any)), dtype=float)

    def predict(self, X_any) -> np.ndarray:
        return np.asarray(self.model.predict(self._to_numpy(X_any))).reshape(-1).astype(int)

    def explain_matrix(self, X_any) -> np.ndarray:
        M_explain, _ = self.model.explain(self._to_numpy(X_any))
        return np.asarray(M_explain, dtype=float)


def _as_float_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().astype(float)


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


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


def evaluate_model(model: TabNetWrapper, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, object]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def completeness_drop_auc(model, x_df, importance_order, baseline, pred_class, nonneg=True, normalize=True) -> float:
    x_new = _as_float_df(x_df)
    p0 = float(model.predict_proba(x_new)[0, pred_class])
    drops = [0.0]
    for f in importance_order:
        if f in x_new.columns:
            x_new.loc[:, f] = float(baseline[f])
        p = float(model.predict_proba(x_new)[0, pred_class])
        d = p0 - p
        if nonneg:
            d = max(0.0, d)
        drops.append(float(d))
    auc = float(np.trapezoid(np.array(drops, dtype=float), dx=1.0))
    if not normalize:
        return auc
    p_all = float(model.predict_proba(x_new)[0, pred_class])
    max_drop = max(0.0, p0 - p_all) if nonneg else (p0 - p_all)
    denom = float(max_drop * max(1, len(importance_order)))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(auc / denom, 0.0, 1.0))


def necessity_score(model, x_df, top_features, baseline, pred_class, nonneg=True) -> float:
    p0 = float(model.predict_proba(x_df)[0, pred_class])
    deltas = []
    for f in top_features:
        if f not in x_df.columns:
            continue
        x_m = _as_float_df(x_df)
        x_m.loc[:, f] = float(baseline[f])
        pm = float(model.predict_proba(x_m)[0, pred_class])
        d = p0 - pm
        if nonneg:
            d = max(0.0, d)
        deltas.append(float(d))
    return float(np.mean(deltas)) if deltas else np.nan


def sufficiency_retained_prob(model, x_df, top_features, baseline, pred_class, eps=1e-8) -> float:
    p0 = float(model.predict_proba(x_df)[0, pred_class])
    x_new = _as_float_df(x_df)
    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])
    pk = float(model.predict_proba(x_new)[0, pred_class])
    return float(np.clip(pk / max(p0, eps), 0.0, 1.0))


def _logit_safe(p, eps=1e-8):
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def sufficiency_logit_retention(model, x_df, top_features, baseline, pred_class, eps=1e-8) -> float:
    p0 = float(model.predict_proba(x_df)[0, pred_class])
    x_new = _as_float_df(x_df)
    keep = set(top_features)
    for c in x_new.columns:
        if c not in keep:
            x_new.loc[:, c] = float(baseline[c])
    pk = float(model.predict_proba(x_new)[0, pred_class])
    l0 = _logit_safe(p0, eps)
    lk = _logit_safe(pk, eps)
    if abs(l0) < eps:
        return np.nan
    return float(np.clip(lk / l0, 0.0, 1.0))


def explanation_parsimony_topk(importance_dict, threshold=0.95) -> float:
    if not importance_dict:
        return np.nan
    vals = np.array(sorted([abs(v) for v in importance_dict.values()], reverse=True), dtype=float)
    if vals.sum() == 0:
        return np.nan
    cum = np.cumsum(vals) / vals.sum()
    return int(np.searchsorted(cum, threshold) + 1)


def simplicity_entropy_and_gini(importance_dict) -> Tuple[float, float]:
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


def monotonicity_local(model, x_df, feature_list, pred_class, attr_map, X_ref, delta_frac=0.01) -> float:
    hits = []
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
            sd = float(col.std())
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            delta = delta_frac * sd
            v = float(x_df.iloc[0][f])
            x_plus.loc[:, f] = v + delta
            x_minus.loc[:, f] = v - delta
        p_plus = float(model.predict_proba(x_plus)[0, pred_class])
        p_minus = float(model.predict_proba(x_minus)[0, pred_class])
        grad_sign = np.sign(p_plus - p_minus)
        if grad_sign == 0:
            continue
        hits.append(float(attr_sign == grad_sign))
    return float(np.mean(hits)) if hits else np.nan


def explanation_robustness(explain_fn, x_instance, noise_scale=0.01, trials=10) -> float:
    base = explain_fn(x_instance)
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        noisy = explain_fn(x_noisy)
        keys = sorted(set(base.keys()) | set(noisy.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([noisy.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


def tabnet_importance_dict(model: TabNetWrapper, x_df: pd.DataFrame) -> Dict[str, float]:
    M = model.explain_matrix(x_df)
    vals = M[0]
    return {f: float(vals[i]) for i, f in enumerate(feature_names)}


def tabnet_stability(model, x_instance, trials=10, noise_scale=0.01) -> float:
    base = tabnet_importance_dict(model, pd.DataFrame([x_instance], columns=feature_names))
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        noisy = tabnet_importance_dict(model, pd.DataFrame([x_noisy], columns=feature_names))
        keys = sorted(set(base.keys()) | set(noisy.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([noisy.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


def tabnet_consistency(model, X_test, x_instance, k=5) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]
    base = np.array(list(tabnet_importance_dict(model, pd.DataFrame([x_instance], columns=feature_names)).values()))
    rhos = []
    for n_idx in neighbors:
        vec = np.array(list(tabnet_importance_dict(model, X_test.iloc[n_idx:n_idx+1]).values()))
        rho = _spearman(base, vec)
        if not np.isnan(rho):
            rhos.append(abs(rho))
    return float(np.mean(rhos)) if rhos else np.nan


# 4) LIME + BREAKDOWN
def build_lime_explainer(X_train, class_names, x_patient, model, pred_class):
    X_train_np = X_train.values.astype(np.float32)
    base = float(np.sqrt(len(feature_names)))
    widths = [0.25 * base, 0.5 * base, 1.0 * base, 2.0 * base, 3.0 * base]
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
            predict_fn=lambda z: model.predict_proba(z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        if float(exp.score) > best[0]:
            best = (float(exp.score), float(w), exp)
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        mode="classification",
        discretize_continuous=True,
        kernel_width=best[1],
        random_state=42,
    )
    return explainer, best[2], best[1], best[0]


def lime_top_feature_names(lime_exp, pred_class, k=10):
    pairs = sorted(lime_exp.local_exp.get(int(pred_class), []), key=lambda t: abs(t[1]), reverse=True)[:k]
    return [feature_names[int(i)] for i, _ in pairs if 0 <= int(i) < len(feature_names)]


def lime_importance_dict(lime_exp, pred_class):
    return {k: float(v) for k, v in lime_exp.as_list(label=int(pred_class))}


def lime_stability(explainer_lime, model, x_instance, pred_class, trials=10, noise_scale=0.01):
    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: model.predict_proba(z),
        labels=[int(pred_class)],
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    base = dict(base_exp.as_list(label=int(pred_class)))
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance + np.random.normal(0, noise_scale, size=x_instance.shape)
        exp = explainer_lime.explain_instance(
            data_row=x_noisy,
            predict_fn=lambda z: model.predict_proba(z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        d = dict(exp.as_list(label=int(pred_class)))
        keys = sorted(set(base.keys()) | set(d.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


def lime_consistency(explainer_lime, model, X_test, x_instance, pred_class, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance])
    neighbors = idx[0][1:]
    base_exp = explainer_lime.explain_instance(
        data_row=x_instance,
        predict_fn=lambda z: model.predict_proba(z),
        labels=[int(pred_class)],
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    base = dict(base_exp.as_list(label=int(pred_class)))
    diffs = []
    for n_idx in neighbors:
        x_n = X_test.iloc[n_idx].values.astype(np.float32)
        exp = explainer_lime.explain_instance(
            data_row=x_n,
            predict_fn=lambda z: model.predict_proba(z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )
        d = dict(exp.as_list(label=int(pred_class)))
        keys = sorted(set(base.keys()) | set(d.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


def lime_monotonicity(explainer_lime, model, x_instance, pred_class, X_ref, delta_frac=0.01):
    exp = explainer_lime.explain_instance(
        x_instance,
        lambda z: model.predict_proba(z),
        labels=[int(pred_class)],
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    pairs = exp.local_exp.get(int(pred_class), [])
    base_pred = float(model.predict_proba(x_instance.reshape(1, -1))[0][pred_class])
    scores = []
    for feat_idx, w in pairs:
        feat_idx = int(feat_idx)
        if abs(float(w)) < 1e-12:
            continue
        x_new = x_instance.copy()
        col = X_ref.iloc[:, feat_idx]
        vals = pd.Series(col.dropna().unique())
        is_binary = len(vals) > 0 and set(np.round(vals.astype(float), 6).tolist()).issubset({0.0, 1.0})
        if is_binary:
            x_new[feat_idx] = 1.0 if float(x_instance[feat_idx]) < 0.5 else 0.0
        else:
            sd = float(col.std())
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            x_new[feat_idx] = float(x_instance[feat_idx]) + delta_frac * sd
        new_pred = float(model.predict_proba(x_new.reshape(1, -1))[0][pred_class])
        delta_pred = new_pred - base_pred
        if abs(delta_pred) < 1e-12:
            continue
        scores.append(float(np.sign(w) == np.sign(delta_pred)))
    return float(np.mean(scores)) if scores else np.nan


def build_dalex_explainer(model, X_train, pred_class):
    def pred_fun(model_, data):
        return model_.predict_proba(data)[:, pred_class]
    return dx.Explainer(
        model=model,
        data=X_train,
        y=model.predict_proba(X_train)[:, pred_class],
        predict_function=pred_fun,
        label=f"TabNet_prob_class_{pred_class}",
        verbose=False,
    )


def bd_vector(bd):
    df_bd = bd.result.copy()
    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        return [], np.array([], dtype=float)
    tmp = df_bd[var_col].astype(str).str.lower()
    df_attr = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")]
    return df_attr[var_col].tolist(), df_attr["contribution"].values


def bd_stability(exp_dalex, x_instance_df, trials=10, noise_scale=0.01):
    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base = dict(zip(base_keys, base_vals))
    diffs = []
    for _ in range(trials):
        x_noisy = x_instance_df.copy().astype(float)
        x_noisy.loc[:, :] = x_noisy.values + np.random.normal(0, noise_scale, size=x_noisy.shape)
        bd_n = exp_dalex.predict_parts(x_noisy, type="break_down")
        keys_n, vals_n = bd_vector(bd_n)
        d = dict(zip(keys_n, vals_n))
        keys = sorted(set(base.keys()) | set(d.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


def bd_consistency(exp_dalex, X_test, x_instance_df, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_test.values)
    _, idx = nn.kneighbors([x_instance_df.values[0]])
    neighbors = idx[0][1:]
    bd_base = exp_dalex.predict_parts(x_instance_df, type="break_down")
    base_keys, base_vals = bd_vector(bd_base)
    base = dict(zip(base_keys, base_vals))
    diffs = []
    for n_idx in neighbors:
        bd_n = exp_dalex.predict_parts(X_test.iloc[n_idx:n_idx+1].copy().astype(float), type="break_down")
        keys_n, vals_n = bd_vector(bd_n)
        d = dict(zip(keys_n, vals_n))
        keys = sorted(set(base.keys()) | set(d.keys()))
        b = np.array([base.get(k, 0.0) for k in keys], dtype=float)
        n = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        diffs.append(np.linalg.norm(b - n))
    return float(np.mean(diffs))


# SINGLE PATIENT EVAL

def evaluate_patient(patient_idx, X_test, y_test, X_train, model, le, baseline, explainer_lime=None):
    x_df = X_test.iloc[patient_idx:patient_idx+1]
    x = x_df.values.astype(np.float32)[0]
    probs = model.predict_proba(x_df)[0]
    pred_class = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_class])[0]
    true_label = le.inverse_transform([int(y_test[patient_idx])])[0]

    print(f"\n🧍 Pacient #{patient_idx}")
    print(f"Predikovaná trieda {pred_label} | pravdepodobnosti: {probs}")
    print(f"Skutočná trieda: {true_label}")

    # TabNet native
    tab_imp = tabnet_importance_dict(model, x_df)
    tab_abs = pd.Series({k: abs(v) for k, v in tab_imp.items()})
    tab_top = tab_abs.sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist()
    tab_imp_dict = {f: tab_abs.loc[f] for f in tab_top}
    _, tab_entropy = simplicity_entropy_and_gini(tab_imp_dict)

    def tab_explain_fn(z):
        return tabnet_importance_dict(model, pd.DataFrame([z], columns=feature_names))

    row_tab = {
        "patient_idx": patient_idx, "true_class": true_label, "pred_class": pred_label, "method": "TabNet_native",
        "fidelity": np.nan,
        "consistency": tabnet_consistency(model, X_test, x),
        "completeness": completeness_drop_auc(model, x_df, tab_top, baseline, pred_class),
        "stability": tabnet_stability(model, x),
        "robustness": explanation_robustness(tab_explain_fn, x),
        "monotonicity": monotonicity_local(model, x_df, tab_top, pred_class, tab_imp, X_train),
        "trendability": np.nan,
        "necessity": necessity_score(model, x_df, tab_top, baseline, pred_class),
        "sufficiency": sufficiency_retained_prob(model, x_df, tab_top, baseline, pred_class),
        "sufficiency_logit": sufficiency_logit_retention(model, x_df, tab_top, baseline, pred_class),
        "explanation_parsimony": explanation_parsimony_topk(tab_imp_dict),
        "simplicity": 1 / (1 + tab_entropy),
    }

    tab_table = pd.DataFrame({
        "feature": feature_names,
        "importance": [tab_imp[f] for f in feature_names],
        "abs_importance": [abs(tab_imp[f]) for f in feature_names],
        "feature_value": x_df.iloc[0].values,
    }).sort_values("abs_importance", ascending=False)

    # LIME
    if explainer_lime is None:
        class_names = le.inverse_transform(np.arange(len(le.classes_)))
        explainer_lime, lime_exp, best_w, best_score = build_lime_explainer(X_train, class_names, x, model, pred_class)
    else:
        lime_exp = explainer_lime.explain_instance(
            data_row=x,
            predict_fn=lambda z: model.predict_proba(z),
            labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )

    lime_imp = lime_importance_dict(lime_exp, pred_class)
    lime_top = lime_top_feature_names(lime_exp, pred_class, TOP_K_FEATURES)
    _, lime_entropy = simplicity_entropy_and_gini(lime_imp)

    def lime_explain_fn(z):
        exp = explainer_lime.explain_instance(
            z, lambda a: model.predict_proba(a), labels=[int(pred_class)],
            num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES
        )
        return dict(exp.as_list(label=int(pred_class)))

    row_lime = {
        "patient_idx": patient_idx, "true_class": true_label, "pred_class": pred_label, "method": "LIME",
        "fidelity": float(lime_exp.score),
        "consistency": lime_consistency(explainer_lime, model, X_test, x, pred_class),
        "completeness": completeness_drop_auc(model, x_df, lime_top, baseline, pred_class),
        "stability": lime_stability(explainer_lime, model, x, pred_class),
        "robustness": explanation_robustness(lime_explain_fn, x),
        "monotonicity": lime_monotonicity(explainer_lime, model, x, pred_class, X_train),
        "trendability": np.nan,
        "necessity": necessity_score(model, x_df, lime_top, baseline, pred_class),
        "sufficiency": sufficiency_retained_prob(model, x_df, lime_top, baseline, pred_class),
        "sufficiency_logit": sufficiency_logit_retention(model, x_df, lime_top, baseline, pred_class),
        "explanation_parsimony": explanation_parsimony_topk(lime_imp),
        "simplicity": 1 / (1 + lime_entropy),
    }

    lime_table = pd.DataFrame(lime_exp.as_list(label=int(pred_class)), columns=["feature_interval", "contribution"])
    lime_table["abs_contribution"] = lime_table["contribution"].abs()
    lime_table = lime_table.sort_values("abs_contribution", ascending=False)

    # BreakDown
    exp_dalex = build_dalex_explainer(model, X_train, pred_class)
    bd = exp_dalex.predict_parts(x_df, type="break_down")
    df_bd = bd.result.copy()
    var_col = "variable_name" if "variable_name" in df_bd.columns else ("variable" if "variable" in df_bd.columns else None)
    if var_col is None:
        raise RuntimeError(f"Neočakávané stĺpce v bd.result: {df_bd.columns.tolist()}")
    tmp = df_bd[var_col].astype(str).str.lower()
    intercept_rows = df_bd[(tmp == "intercept") | (tmp == "baseline")]
    bd_intercept = float(intercept_rows["contribution"].iloc[0]) if len(intercept_rows) > 0 else 0.0
    df_bd_attr = df_bd[(tmp != "intercept") & (tmp != "baseline") & (tmp != "")].copy()
    bd_reconstructed = float(bd_intercept + df_bd_attr["contribution"].sum())
    bd_fidelity = float(max(0.0, 1.0 - abs(bd_reconstructed - probs[pred_class])))
    bd_keys, bd_vals = bd_vector(bd)
    bd_imp_full = {k: float(v) for k, v in zip(bd_keys, bd_vals)}
    bd_candidates = [f for f in bd_keys if f in feature_names]
    bd_top = sorted(bd_candidates, key=lambda f: abs(bd_imp_full.get(f, 0.0)), reverse=True)[:TOP_K_FEATURES] if bd_candidates else tab_top
    bd_imp = {f: bd_imp_full.get(f, 0.0) for f in bd_top}
    _, bd_entropy = simplicity_entropy_and_gini(bd_imp)

    def bd_explain_fn(z):
        obs = pd.DataFrame([z], columns=feature_names).astype(float)
        bd_loc = exp_dalex.predict_parts(obs, type="break_down")
        keys, vals = bd_vector(bd_loc)
        return dict(zip(keys, vals))

    row_bd = {
        "patient_idx": patient_idx, "true_class": true_label, "pred_class": pred_label, "method": "BreakDown",
        "fidelity": bd_fidelity,
        "consistency": bd_consistency(exp_dalex, X_test, x_df),
        "completeness": completeness_drop_auc(model, x_df, bd_top, baseline, pred_class),
        "stability": bd_stability(exp_dalex, x_df),
        "robustness": explanation_robustness(bd_explain_fn, x),
        "monotonicity": monotonicity_local(model, x_df, bd_top, pred_class, bd_imp, X_train),
        "trendability": np.nan,
        "necessity": necessity_score(model, x_df, bd_top, baseline, pred_class),
        "sufficiency": sufficiency_retained_prob(model, x_df, bd_top, baseline, pred_class),
        "sufficiency_logit": sufficiency_logit_retention(model, x_df, bd_top, baseline, pred_class),
        "explanation_parsimony": explanation_parsimony_topk(bd_imp),
        "simplicity": 1 / (1 + bd_entropy),
    }

    bd_table = df_bd_attr.copy()
    bd_table["abs_contribution"] = bd_table["contribution"].abs()
    bd_table = bd_table.sort_values("abs_contribution", ascending=False)

    metric_table = pd.DataFrame([row_tab, row_lime, row_bd])
    print("\n================= TABUĽKA METRÍK – PACIENT =================")
    print(metric_table)

    return metric_table, tab_table, lime_table, bd_table


def main():
    global feature_names

    if not os.path.exists(PATH):
        raise FileNotFoundError(f"Dátový súbor neexistuje: {PATH}\nUprav premennú PATH v hlavičke skriptu.")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_excel(PATH)
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError("TabNet očakáva numerické vstupy. Nenumerické stĺpce: " + ", ".join(non_numeric))

    X = X.astype(np.float32)
    feature_names = X.columns.tolist()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(np.unique(y_enc))
    class_names = le.inverse_transform(np.arange(num_classes))

    X_temp, X_test, y_temp, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=42, stratify=y_temp)

    print("Train shape:", X_train.shape)
    print("Val shape:  ", X_val.shape)
    print("Test shape: ", X_test.shape)

    tabnet = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=SEED,
        verbose=0,
        device_name="auto",
    )

    print("🔹 Trénujem TabNet...")
    tabnet.fit(
        X_train=X_train.to_numpy(dtype=np.float32),
        y_train=y_train.astype(int),
        eval_set=[(X_val.to_numpy(dtype=np.float32), y_val.astype(int))],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=TABNET_MAX_EPOCHS,
        patience=TABNET_PATIENCE,
        batch_size=TABNET_BATCH_SIZE,
        virtual_batch_size=TABNET_VIRTUAL_BATCH_SIZE,
        num_workers=0,
        drop_last=False,
    )
    print(" Model natrenovaný.")

    model = TabNetWrapper(tabnet, feature_names)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print("\n Klasické metriky TabNet:")
    for k, v in metrics.items():
        print(k, ":\n", v, "\n")

    # confusion matrix graf
    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix – TabNet")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("Skutočná trieda")
    plt.xlabel("Predikovaná trieda")
    plt.tight_layout()
    plt.savefig(f"{EXPORT_PREFIX}_confusion_matrix.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    baseline = _baseline_row(X_train)

    demo_df = X_test.iloc[SINGLE_PATIENT_IDX:SINGLE_PATIENT_IDX+1]
    demo_x = demo_df.values.astype(np.float32)[0]
    demo_pred = int(np.argmax(model.predict_proba(demo_df)[0]))
    explainer_lime, _, best_w, best_score = build_lime_explainer(X_train, class_names, demo_x, model, demo_pred)
    print(f"\n🔧 LIME init: best_w={best_w:.3f}, demo fidelity(R2)={best_score:.4f}")

    if RUN_SINGLE_PATIENT_DEMO:
        single_metrics, tab_table, lime_table, bd_table = evaluate_patient(
            SINGLE_PATIENT_IDX, X_test, y_test, X_train, model, le, baseline, explainer_lime
        )
        single_metrics.to_excel(f"{EXPORT_PREFIX}_single_patient_metrics.xlsx", index=False)
        tab_table.to_excel(f"{EXPORT_PREFIX}_single_patient_tabnet_native.xlsx", index=False)
        lime_table.to_excel(f"{EXPORT_PREFIX}_single_patient_lime.xlsx", index=False)
        bd_table.to_excel(f"{EXPORT_PREFIX}_single_patient_breakdown.xlsx", index=False)

    if RUN_MULTI_PATIENT_EVAL:
        rng = np.random.default_rng(MULTI_RANDOM_STATE)
        eval_indices = sorted(rng.choice(len(X_test), size=min(N_EVAL_PATIENTS, len(X_test)), replace=False).tolist())
        print("\n🔹 Multi-patient evaluácia pre indexy:", eval_indices)
        rows = []
        for idx in eval_indices:
            metric_table, _, _, _ = evaluate_patient(idx, X_test, y_test, X_train, model, le, baseline, explainer_lime)
            rows.append(metric_table)
        detail = pd.concat(rows, ignore_index=True)
        summary_base = detail.drop(columns=[c for c in ["patient_idx", "true_class", "pred_class"] if c in detail.columns])
        mean_df = summary_base.groupby("method").mean(numeric_only=True)
        median_df = summary_base.groupby("method").median(numeric_only=True)
        std_df = summary_base.groupby("method").std(numeric_only=True)

        print("\n================= MULTI-PATIENT DETAIL =================")
        print(detail)
        print("\n================= MULTI-PATIENT MEAN =================")
        print(mean_df)
        print("\n================= MULTI-PATIENT MEDIAN =================")
        print(median_df)
        print("\n================= MULTI-PATIENT STD =================")
        print(std_df)

        patient_correctness = detail[["patient_idx", "true_class", "pred_class"]].drop_duplicates("patient_idx").copy()
        patient_correctness["is_correct"] = (patient_correctness["true_class"] == patient_correctness["pred_class"]).astype(int)
        n_eval = len(patient_correctness)
        n_correct = int(patient_correctness["is_correct"].sum())
        y_pred_test = model.predict(X_test)
        test_correct = int((y_pred_test == y_test).sum())
        print("\n================= SPRÁVNOSŤ PREDIKCIE =================")
        print(f"Eval vzorka: {n_correct}/{n_eval} ({100*n_correct/n_eval:.2f} %)")
        print(f"Celý test set: {test_correct}/{len(y_test)} ({100*test_correct/len(y_test):.2f} %)")

        detail.to_excel(f"{EXPORT_PREFIX}_multi_patient_detail.xlsx", index=False)
        mean_df.to_excel(f"{EXPORT_PREFIX}_multi_patient_mean.xlsx")
        median_df.to_excel(f"{EXPORT_PREFIX}_multi_patient_median.xlsx")
        std_df.to_excel(f"{EXPORT_PREFIX}_multi_patient_std.xlsx")
        patient_correctness.to_excel(f"{EXPORT_PREFIX}_patient_correctness.xlsx", index=False)
        print("\n Export tabuliek hotový.")


if __name__ == "__main__":
    main()
