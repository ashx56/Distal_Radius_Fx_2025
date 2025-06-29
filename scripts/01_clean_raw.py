# package import
import pandas as pd
import re

# load raw dataset
file_path = "../data/Non_Op_Distal_Radius_Fracture_Updated.xlsx"
df_raw = pd.read_excel(file_path)

# rename columns
rename_map = {
    "A": "patient_id",
    "Comorbidities": "comorbidities",
    "# Pattern (Fryk) Type:": "frykman_pattern",
    "Sex": "sex",
    "Sex.1": "sex",                 # in case only the dup column exists
    "Age_at_Procedure": "age",
    "Age_at_Procedure.1": "age",    # dup column
    "Dorsal Communition": "dorsal_comminution",
    "Volar cortex malalignment": "volar_cortex_malalignment",
    "Ulnar styloid FX": "ulnar_styloid_fx",
    "Intra-articular extension": "intra_articular_extension",
}

rename_map = {k: v for k, v in rename_map.items() if k in df_raw.columns}
df = df_raw.rename(columns=rename_map)

# drop duplicate sex/age columns
if "sex.1" in df.columns and "sex" in df.columns:
    df.drop(columns=["sex.1"], inplace=True)
if "age.1" in df.columns and "age" in df.columns:
    df.drop(columns=["age.1"], inplace=True)

# collect radiographic metric columns
metric_regex = re.compile(r"^(RI|VT|UV|RHT)_(Pres|Presentation|Week\d{1,2}|Week5-8|Week9-16|Week17-30)$", re.I)
metric_cols = [c for c in df.columns if metric_regex.match(c)]

# filter out irrelevant columns
keep_cols = ["patient_id", "comorbidities", "frykman_pattern",
             "sex", "age", "dorsal_comminution", "volar_cortex_malalignment",
             "ulnar_styloid_fx", "intra_articular_extension"] + metric_cols

keep_cols = [c for c in keep_cols if c in df.columns]
df_clean = df[keep_cols].copy()
df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

# lowercase + snake_case remaining columns
def to_snake(name):
    name = name.strip().lower()
    name = re.sub(r"[ /()-]", "_", name)
    name = re.sub(r"__+", "_", name).strip("_")
    return name

df_clean.columns = [to_snake(c) for c in df_clean.columns]
df_cat = df_clean.copy()  

# clean categorical columns
df_cat['frykman_pattern'] = (
    df_cat['frykman_pattern']
      .replace({'A2+B2': 'B2'})
      .str.strip()
      .replace('', pd.NA)
)
df_cat = df_cat[df_cat['frykman_pattern'].notna()]

sex_map = {
    'female': 'Female', 'f': 'Female', 'male': 'Male', 'm': 'Male',
    'Female': 'Female', 'Male': 'Male'
}
df_cat['sex'] = (
    df_cat['sex']
      .str.strip()
      .str.lower()
      .map(sex_map)
)
df_cat = df_cat[df_cat['sex'].notna()]

yes_no_map = {'yes': 'Yes', 'no': 'No'}

for col in ['dorsal_comminution',
            'volar_cortex_malalignment',
            'ulnar_styloid_fx']:
    
    # 1) normalise text → lower case → map to proper capitalisation
    df_cat.loc[:, col] = (
        df_cat[col]                   # preserves original slice semantics
            .astype(str)              # in case of non-string dtypes
            .str.strip()
            .str.lower()
            .map(yes_no_map)          # returns NaN for anything not yes/no
    )

df_cat = df_cat.dropna(subset=[
    'dorsal_comminution',
    'volar_cortex_malalignment',
    'ulnar_styloid_fx'
])

# fix intra-articular extension
def intra_ext_to_yes_no(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip().str.lower()
    yes_terms = {'sagittal', 'saggittal', 'coronal', 'both'}
    series = series.map(lambda x: 'Yes' if x in yes_terms
                                  else ('No' if x == 'no' else pd.NA))
    return series

df_cat['intra_articular_extension'] = intra_ext_to_yes_no(df_cat['intra_articular_extension'])
df_cat = df_cat.dropna(subset=['intra_articular_extension'])

# one-hot encode comorbities of interest
patterns = {
    'Diabetes': re.compile(
        r'(?:t2d|dm2)'                                
        r'|(?=.*\b(diabet|dm)\b)(?=.*\b(2|ii)\b)', re.I),
    'Osteoarthritis'         : re.compile(r'\b(osteoarth|oa|degenerative joint disease|djd)\b', re.I),
    'Rheumatoid arthritis'         : re.compile(r'\b(rheumatoid|ra)\b', re.I),
    'Hypertension'        : re.compile(r'\b(htn|hypertension|high blood pressure|hbp)\b', re.I),
    'Depression' : re.compile(r'\b(depression|mdd|major depressive)\b', re.I),
    'Anxiety'    : re.compile(r'\b(anxiety|gad|generalized anxiety)\b', re.I),
    'Anemia'     : re.compile(r'\banemia|anaemia|iron deficiency\b', re.I),
}

def map_comorbidities(cell_text: str) -> dict:
    txt = str(cell_text).lower()
    return {col: bool(pat.search(txt)) for col, pat in patterns.items()}

comorb_df = df_cat['comorbidities'].apply(map_comorbidities).apply(pd.Series)
df_cat = pd.concat([df_cat.drop(columns=['comorbidities']), comorb_df], axis=1)

# save cleaned dataset
df_cat.to_csv("../data/processed/distal_radius_fx_clean.csv", index=False)

