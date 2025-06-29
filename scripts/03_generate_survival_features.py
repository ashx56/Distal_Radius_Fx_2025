# package import
import pandas as pd, numpy as np, re

# helper functions 
def snake(name):
    name = re.sub(r"[ /()-]", "_", name.strip())
    name = re.sub(r"__+", "_", name).strip("_").lower()
    return name

def to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def coerce_numeric_series(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(',', '.', regex=False)
              .str.replace(r'[^\d\.\-]', '', regex=True),
        errors='coerce'
    )

# load cleaned data
raw_path = "../data/processed/distal_radius_fx_clean.csv"
df = pd.read_csv(raw_path)

metric_prefixes = ['RHT','UV','RI','VT']
for col in df.columns:
    if any(col.startswith(prefix.lower()) for prefix in metric_prefixes):
        df[col] = coerce_numeric_series(df[col])

metrics = [m.lower() for m in metric_prefixes]
pres_cols = {m: [c for c in df.columns if c.startswith(f"{m}_") and 'pres' in c][0] for m in metrics}
w1_cols = {m: [c for c in df.columns if c.startswith(f"{m}_") and 'week1' in c][0] for m in metrics}

pres_complete = df[[pres_cols[m] for m in metrics]].notna().all(axis=1)
w1_complete = df[[w1_cols[m] for m in metrics]].notna().all(axis=1)

df['baseline_source'] = np.where(pres_complete,'presentation', np.where(w1_complete,'week1', np.nan))
df = df.dropna(subset=['baseline_source'])

for m in metrics:
    df[f'{m}_base'] = np.where(df['baseline_source']=='presentation', df[pres_cols[m]], df[w1_cols[m]])
    if m in ['ri','vt']:
        df[f'{m}_base_abs'] = df[f'{m}_base'].abs()

# compute deltas
week_order = [int(re.search(r'week(\d+)', c).group(1)) for c in df.columns if re.search(r'week(\d+)', c)]
# per metric build ordered list
ordered = {}
for m in metrics:
    ordered_cols = [c for c in df.columns if c.startswith(f'{m}_week')]
    ordered_cols = sorted(ordered_cols, key=lambda x: int(re.search(r'week(\d+)', x).group(1)))
    ordered[m] = ordered_cols

def flex_deltas(row, m):
    base = row[f'{m}_base']
    # gather all weeks >=1 numerically sorted
    values = [(int(re.search(r'week(\d+)', c).group(1)), row[c]) for c in ordered[m] if pd.notna(row[c])]
    if len(values)==0:
        return pd.Series([np.nan,np.nan])
    values.sort(key=lambda x: x[0])
    d1 = values[0][1] - base
    return pd.Series([d1])

for m in metrics:
    df[[f'{m}_delta1']] = df.apply(lambda r, m=m: flex_deltas(r,m), axis=1)
    if m in ['ri','vt']:
        df[f'{m}_delta1_abs'] = df[f'{m}_delta1'].abs()

# determine first stability week bucket 
thr = {'ri_abs':3,'vt_abs':5,'uv':1,'rht':-2}
bucket_edges = [3,4,8,16,30]  # 31 for censored
def week_instable(d, m):
    if np.isnan(d): return True
    if m in ['ri','vt']:
        return abs(d) > thr[f'{m}_abs']
    if m=='uv':
        return d > thr['uv']
    if m=='rht':
        return d <= thr['rht']
    return True

def first_stable(row):
    # compute instability each week vs prev available week
    prev_vals = {m: row[f'{m}_base'] for m in metrics}
    for wk in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
        stable_here = True
        for m in metrics:
            col = f'{m}_week{wk}'
            if pd.isna(row.get(col)):
                stable_here = False
                break
            d = row[col] - prev_vals[m]
            if week_instable(d, m):
                stable_here = False
                break
            prev_vals[m] = row[col]
        if stable_here:
            return wk
    return np.nan

df['first_stable_week'] = df.apply(first_stable, axis=1)
df['event'] = df['first_stable_week'].notna().astype(int)
df['time'] = df['first_stable_week'].fillna(31)

# build feature sets 
baseline_metrics = [f'{m}_base' for m in metrics] + ['ri_base_abs','vt_base_abs']
delta_metrics = [f'{m}_delta{d}_abs' for m in ['ri','vt'] for d in [1]]

demo_cols = ['age','sex','frykman_pattern','dorsal_comminution','volar_cortex_malalignment',
             'ulnar_styloid_fx','intra_articular_extension']

baseline_cols = ['patient_id','time','event'] + demo_cols + baseline_metrics
delta_cols = ['patient_id','time','event'] + demo_cols + delta_metrics

baseline_df = df[baseline_cols].copy()
delta_df = df[delta_cols].copy()

baseline_path = "../data/processed/clean_survival_baseline.csv"
delta_path = "../data/processed/clean_survival_delta.csv"
baseline_df.to_csv(baseline_path, index=False)
delta_df.to_csv(delta_path, index=False)