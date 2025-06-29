import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from itertools import islice
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import shap
import joblib
import json

# load baseline-feature dataset
baseline_df = pd.read_csv("../data/processed/clean_survival_baseline.csv")

# survival targets
y_time  = baseline_df['time'].values
y_event = baseline_df['event'].values

# feature matrix — drop id & survival columns
X = baseline_df.drop(columns=['patient_id', 'time', 'event']).copy()

# column categorisation
cat_cols = ['sex', 'frykman_pattern',
            'dorsal_comminution', 'volar_cortex_malalignment',
            'ulnar_styloid_fx', 'intra_articular_extension']

bin_cols = ['has_diabetes_t2', 'has_oa', 'has_ra', 'has_htn',
            'has_depression', 'has_anxiety', 'has_anemia']

num_cols = [c for c in X.columns if c not in cat_cols + bin_cols]

# preprocessing pipeline
preproc = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ],
    remainder='drop'
)

# stratified train–test split (80/20)
bucket = pd.cut(baseline_df['time'],
                bins=[0, 3.5, 4.5, np.inf],
                labels=[0, 1, 2]).astype(int)

X_train, X_test, y_time_tr, y_time_te, y_event_tr, y_event_te = train_test_split(
    X, y_time, y_event,
    test_size=0.2, random_state=42,
    stratify=bucket
)

# fit/transform
X_train_enc = preproc.fit_transform(X_train)
X_test_enc  = preproc.transform(X_test)

# XGBoost-Cox model – baseline features (randomised search)
# parameters to sample
param_space = {
    'n_estimators'     : lambda: random.randint(100, 1000),
    'learning_rate'    : lambda: 10 ** random.uniform(-2.3, -0.4),   
    'max_depth'        : lambda: random.randint(1, 100),
    'subsample'        : lambda: random.uniform(0.1, 1.0),
    'colsample_bytree' : lambda: random.uniform(0.1, 1.0),
    'min_child_weight' : lambda: random.uniform(0.1, 10),
    'gamma'            : lambda: random.uniform(0, 10),
}
def sample_params():
    return {k: f() for k, f in param_space.items()}

# CV function (3-fold, concordance index)
def cv_cindex(params, X, t, e, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        est = XGBRegressor(objective="survival:cox",
                           random_state=0, **params)
        est.fit(X[tr_idx], t[tr_idx],
                sample_weight=e[tr_idx])      # event flags as weights
        risk_val = est.predict(X[val_idx])
        ci       = concordance_index(t[val_idx], -risk_val, e[val_idx])
        scores.append(ci)
    return np.mean(scores)

# random search loop
n_iter = 100
best_score, best_params = -np.inf, None

for i, params in enumerate(islice(iter(sample_params, None), n_iter), start=1):
    score = cv_cindex(params, X_train_enc, y_time_tr, y_event_tr)
    print(f"[{i:02}/{n_iter}]  CV C-index: {score:.3f}")
    if score > best_score:
        best_score, best_params = score, params

# fit final model on full training split 
final_baseline_model = XGBRegressor(
    objective="survival:cox",
    random_state=0,
    **best_params
)
final_baseline_model.fit(X_train_enc, y_time_tr,
                         sample_weight=y_event_tr)

# test-set concordance
risk_test_baseline = final_baseline_model.predict(X_test_enc)
ci_test   = concordance_index(y_time_te, -risk_test_baseline, y_event_te)

# save model and parameters
joblib.dump(final_baseline_model, "../models/baseline_model.joblib")
with open("../models/baseline_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
with open("../models/baseline_test_cindex.txt", "w") as f:
    f.write(f"Test-set concordance index: {ci_test:.4f}\n")

# Δ-feature XGBoost-Cox model (randomised search)
# load delta-feature dataset
delta_df = pd.read_csv("../data/processed/clean_survival_delta.csv")

y_time  = delta_df['time'].values
y_event = delta_df['event'].values
X_delta = delta_df.drop(columns=['patient_id', 'time', 'event']).copy()

# column categorisation (same as baseline)
cat_cols = ['sex', 'frykman_pattern',
            'dorsal_comminution', 'volar_cortex_malalignment',
            'ulnar_styloid_fx', 'intra_articular_extension']

bin_cols = ['has_diabetes_t2', 'has_oa', 'has_ra', 'has_htn',
            'has_depression', 'has_anxiety', 'has_anemia']

num_cols = [c for c in X_delta.columns if c not in cat_cols + bin_cols]

preproc_delta = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ],
    remainder='drop'
)

# train–test split (same stratification)
bucket = pd.cut(delta_df['time'],
                bins=[0, 3.5, 4.5, np.inf],
                labels=[0, 1, 2]).astype(int)

X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
    X_delta, y_time, y_event,
    test_size=0.20, random_state=42,
    stratify=bucket
)

X_tr_enc = preproc_delta.fit_transform(X_tr)
X_te_enc = preproc_delta.transform(X_te)

# randomised hyper-param search
param_space = {
    'n_estimators'     : lambda: random.randint(100, 1000),
    'learning_rate'    : lambda: 10 ** random.uniform(-2.3, -0.4),  
    'max_depth'        : lambda: random.randint(1, 100),
    'subsample'        : lambda: random.uniform(0.1, 1.0),
    'colsample_bytree' : lambda: random.uniform(0.1, 1.0),
    'min_child_weight' : lambda: random.uniform(0.1, 10),
    'gamma'            : lambda: random.uniform(0, 10),
}
def sample_params():
    return {k: f() for k, f in param_space.items()}

def cv_cindex(params, X, t, e, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        est = XGBRegressor(objective="survival:cox",
                           random_state=0, **params)
        est.fit(X[tr_idx], t[tr_idx], sample_weight=e[tr_idx])
        risk_val = est.predict(X[val_idx])
        ci       = concordance_index(t[val_idx], -risk_val, e[val_idx])
        scores.append(ci)
    return np.mean(scores)

best_score, best_params = -np.inf, None
n_iter = 100
for i, params in enumerate(islice(iter(sample_params, None), n_iter), start=1):
    score = cv_cindex(params, X_tr_enc, t_tr, e_tr)
    print(f"[{i:02}/{n_iter}]  CV C-index: {score:.3f}")
    if score > best_score:
        best_score, best_params = score, params

# fit final Δ-model & evaluate
delta_model = XGBRegressor(objective="survival:cox",
                           random_state=0, **best_params)
delta_model.fit(X_tr_enc, t_tr, sample_weight=e_tr)

risk_test_delta = delta_model.predict(X_te_enc)
ci_te   = concordance_index(t_te, -risk_test_delta, e_te)

# save model and parameters
joblib.dump(delta_model, "../models/delta_model.joblib")
with open("../models/delta_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
with open("../models/delta_test_cindex.txt", "w") as f:
    f.write(f"Test-set concordance index: {ci_te:.4f}\n")

# combined feature XGBoost-Cox model (randomised search)
# load combined feature dataset
combined_df = pd.read_csv("../data/processed/clean_survival_combined.csv")

y_time  = combined_df['time'].values
y_event = combined_df['event'].values
X_delta = combined_df.drop(columns=['patient_id', 'time', 'event']).copy()

# column categorisation (same as baseline)
cat_cols = ['sex', 'frykman_pattern',
            'dorsal_comminution', 'volar_cortex_malalignment',
            'ulnar_styloid_fx', 'intra_articular_extension']

bin_cols = ['has_diabetes_t2', 'has_oa', 'has_ra', 'has_htn',
            'has_depression', 'has_anxiety', 'has_anemia']

num_cols = [c for c in X_delta.columns if c not in cat_cols + bin_cols]

preproc_delta = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ],
    remainder='drop'
)

# train–test split (same stratification)
bucket = pd.cut(combined_df['time'],
                bins=[0, 3.5, 4.5, np.inf],
                labels=[0, 1, 2]).astype(int)

X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
    X_delta, y_time, y_event,
    test_size=0.20, random_state=42,
    stratify=bucket
)

X_tr_enc = preproc_delta.fit_transform(X_tr)
X_te_enc = preproc_delta.transform(X_te)

# randomised hyper-param search
param_space = {
    'n_estimators'     : lambda: random.randint(100, 1000),
    'learning_rate'    : lambda: 10 ** random.uniform(-2.3, -0.4),   
    'max_depth'        : lambda: random.randint(1, 100),
    'subsample'        : lambda: random.uniform(0.1, 1.0),
    'colsample_bytree' : lambda: random.uniform(0.1, 1.0),
    'min_child_weight' : lambda: random.uniform(0.1, 10),
    'gamma'            : lambda: random.uniform(0, 10),
}
def sample_params():
    return {k: f() for k, f in param_space.items()}

def cv_cindex(params, X, t, e, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        est = XGBRegressor(objective="survival:cox",
                           random_state=0, **params)
        est.fit(X[tr_idx], t[tr_idx], sample_weight=e[tr_idx])
        risk_val = est.predict(X[val_idx])
        ci       = concordance_index(t[val_idx], -risk_val, e[val_idx])
        scores.append(ci)
    return np.mean(scores)

best_score, best_params = -np.inf, None
n_iter = 100
for i, params in enumerate(islice(iter(sample_params, None), n_iter), start=1):
    score = cv_cindex(params, X_tr_enc, t_tr, e_tr)
    print(f"[{i:02}/{n_iter}]  CV C-index: {score:.3f}")
    if score > best_score:
        best_score, best_params = score, params

# 5. fit final combined model & evaluate
combined_model = XGBRegressor(objective="survival:cox",
                           random_state=0, **best_params)
combined_model.fit(X_tr_enc, t_tr, sample_weight=e_tr)

risk_train_combo = combined_model.predict(X_tr_enc)
risk_test_combo = combined_model.predict(X_te_enc)
ci_te   = concordance_index(t_te, -risk_test_combo, e_te)

# save model and parameters
joblib.dump(combined_model, "../models/combined_model.joblib")
with open("../models/combined_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
with open("../models/combined_test_cindex.txt", "w") as f:
    f.write(f"Test-set concordance index: {ci_te:.4f}\n")

risk_dict = {
    "Baseline": risk_test_baseline,
    "Delta":    risk_test_delta,
    "Combined": risk_test_combo,
}

# plot ROC curves for each horizon
times = [4, 8, 16, 30]    # weeks
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, horizon in zip(axes, times):
    # binary labels for this horizon
    y_bin = ((y_event_te == 1) & (y_time_te <= horizon)).astype(int)
    
    # to avoid degenerate ROC when y_bin is all 0 or all 1
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        ax.text(0.5, 0.5, "No positives\nor negatives", ha='center', va='center')
        ax.set_title(f"Week ≤ {horizon}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        continue
    
    for label, scores in risk_dict.items():
        fpr, tpr, _ = roc_curve(y_bin, scores)   # -scores: higher risk → shorter time
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(f"Week = {horizon}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("../plots/roc_comparison_by_horizon.png",
            dpi=1200)
plt.savefig("../plots/roc_comparison_by_horizon.svg",
            dpi=1200)
plt.close()

# SHAP beeswarm plots – combined model
# get feature names from the ColumnTransformer
feature_names = preproc_delta.get_feature_names_out()

# compute SHAP values on TEST split
explainer = shap.TreeExplainer(combined_model)
shap_vals = explainer(X_te_enc, check_additivity=False)   # SHAP values array

# split feature indices into two groups
radiographic_keywords = ['_base', '_delta1']   # rough filter
is_radio = np.array([any(k in f for k in radiographic_keywords) for f in feature_names])

radio_idx   = np.where(is_radio)[0]
patient_idx = np.where(~is_radio)[0]

# Filter out rows with any NaN in the radiographic features
radio_X = X_te_enc[:, radio_idx]
radio_shap = shap_vals[:, radio_idx]

# Find rows with no NaN in any radiographic feature
not_nan_mask = ~np.isnan(radio_X).any(axis=1)

# Subset to only rows without NaN
radio_X_clean = radio_X[not_nan_mask]
radio_shap_clean = radio_shap[not_nan_mask]

plt.figure(figsize=(6, 4))  # Compress the plot
shap.summary_plot(
    radio_shap_clean,
    radio_X_clean,
    feature_names=feature_names[radio_idx],
    show=False,
    cmap=plt.get_cmap("inferno"),
    plot_size=(6, 4)
)
plt.tight_layout()
plt.savefig("../plots/shap_beeswarm_radiographic_metrics.png",
            dpi=1200)
plt.savefig("../plots/shap_beeswarm_radiographic_metrics.svg",
            dpi=1200)
plt.close()

sns.set(style="whitegrid")
# feature names out of the transformer
feature_names = preproc_delta.get_feature_names_out()

# SHAP values for the test split (fast on tree models)
explainer = shap.TreeExplainer(combined_model)
shap_vals = explainer(X_te_enc, check_additivity=False)         # (n_samples, n_features)

# 3select “patient” feature indices
radio_kw   = ['_base', '_delta1', '_delta2']                    # radiographic keywords
is_radio   = np.array([any(k in f for k in radio_kw) for f in feature_names])
patient_idx = np.where(~is_radio)[0]                           # everything else

# mean |SHAP| per patient feature
patient_shap = np.abs(shap_vals.values[:, patient_idx]).mean(axis=0)

patient_df = (
    pd.DataFrame({"feature": feature_names[patient_idx],
                  "mean_abs_shap": patient_shap})
      .sort_values("mean_abs_shap", ascending=True)
      .tail(20)                              # top 20
)

# map features to categories
def get_category(feat):
    if feat.startswith("num__age"):
        return "Age"
    if feat.startswith("cat__sex"):
        return "Sex"
    if feat.startswith("cat__frykman_pattern"):
        return "Frykman pattern"
    if feat.startswith("cat__dorsal_comminution"):
        return "Dorsal comminution"
    if feat.startswith("cat__volar_cortex_malalignment"):
        return "Volar cortex malalignment"
    if feat.startswith("cat__ulnar_styloid_fx"):
        return "Ulnar styloid fx"
    if feat.startswith("cat__intra_articular_extension"):
        return "Intra-articular extension"
    if feat.startswith("bin__"):
        return "Comorbidities"
    return "Other"

patient_df["category"] = patient_df["feature"].map(get_category)

# aggregate mean |SHAP| by category
cat_importance = (
    patient_df.groupby("category")["mean_abs_shap"]
    .sum()  # or .mean(), but sum is more interpretable for importance
    .sort_values(ascending=True)
)

# plot
plt.figure(figsize=(6, 4))
palette = sns.color_palette("Set2", n_colors=len(cat_importance))
bars = plt.barh(cat_importance.index, cat_importance.values, color=palette, linewidth=1, edgecolor='black')

plt.xlabel("Total mean |SHAP value|", fontsize=13, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.savefig("../plots/feature_group_shap_importance.png",
            dpi=1200)
plt.savefig("../plots/feature_group_shap_importance.svg",
            dpi=1200)
plt.close()

import numpy as np, matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

models = {
    "XGBoost Model": (risk_train_combo, risk_test_combo),
}

horizons = [4, 8, 16, 30]
fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.flat

def net_benefit(y_true, prob, thresh):
    pred = prob >= thresh
    TP = (pred & y_true).sum()
    FP = (pred & ~y_true).sum()
    n  = len(y_true)
    return TP/n - FP/n * (thresh / (1 - thresh))

for ax, H in zip(axes, horizons):
    y_tr_H = ((y_event_tr == 1) & (y_time_tr <= H)).astype(int)
    y_te_H = ((e_te      == 1) & (t_te      <= H)).astype(int)

    base_prev = y_te_H.mean()
    thresholds = np.linspace(0.01, min(0.9, base_prev + 0.1), 60)

    # reference lines
    treat_all  = [base_prev - (1 - base_prev)*(t/(1-t)) for t in thresholds]
    ax.plot(thresholds, treat_all, '--', color='gray', label='Treat all')
    ax.plot(thresholds, np.zeros_like(thresholds), ':', color='black', label='Treat none')

    for name, (r_tr, r_te) in models.items():
        # orient scores
        if np.corrcoef(r_tr, y_time_tr)[0,1] > 0:
            r_tr, r_te = -r_tr, -r_te
        # isotonic calibration
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(r_tr, y_tr_H)
        p_te = iso.transform(r_te)
        nb = [net_benefit(y_te_H.astype(bool), p_te, t) for t in thresholds]
        ax.plot(thresholds, nb, lw=2, label=name, color='green' if name == "Combined" else 'orange')

    ax.set_title(f"Week = {H}")
    ax.set_xlabel("Risk tolerance")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig("../plots/decision_curve_net_benefit_by_horizon.png",
            dpi=1200)
plt.savefig("../plots/decision_curve_net_benefit_by_horizon.svg",
            dpi=1200)
plt.close()

# Kaplan–Meier curves – 3 risk strata
if np.corrcoef(risk_test_combo, t_te)[0, 1] > 0:  
    risk_test_combo = -risk_test_combo

# split into tertiles
q_low, q_high = np.quantile(risk_test_combo, [1/3, 2/3])
strata = np.where(risk_test_combo <  q_low,  "High risk",
         np.where(risk_test_combo < q_high, "Medium risk", "Low risk"))

# fit & plot KM 
fig, ax = plt.subplots(figsize=(7, 5))

kmf = KaplanMeierFitter()
for group, color in zip(["High risk", "Medium risk", "Low risk"],
                        sns.color_palette("viridis", 3)):
    mask = strata == group
    kmf.fit(durations=t_te[mask],
            event_observed=e_te[mask],
            label=group)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

ax.set_ylabel("Cumulative fraction unstable")
ax.set_xlabel("Weeks since baseline")

plt.tight_layout()
plt.savefig("../plots/km_curve_by_risk_strata.png",
            dpi=1200)
plt.savefig("../plots/km_curve_by_risk_strata.svg",
            dpi=1200)
plt.close()



