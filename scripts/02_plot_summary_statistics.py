# package import 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# load cleaned dataset
df_cat = pd.read_csv("../data/processed/distal_radius_fx_clean.csv")

# age distribution
plt.figure(figsize=(5, 5), dpi=150)
sns.histplot(df_cat['age'].dropna(), bins=20, kde=True, color='purple')
plt.title("Age distribution")
plt.xlabel("Age (years)")
plt.ylabel("Patient count")
plt.savefig("../plots/age_distribution.png",
            dpi=1200)
plt.savefig("../plots/age_distribution.svg",
            dpi=1200)
plt.close()

# pie charts for binary categoricals
plt.figure(figsize=(5, 5), dpi=150)
cat_cols = ['sex',
            'dorsal_comminution',
            'volar_cortex_malalignment',
            'ulnar_styloid_fx',
            'intra_articular_extension']

n_cols = 3
n_rows = int(len(cat_cols) / n_cols + 0.999)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 4, n_rows * 4))
axes = axes.flatten()

for ax, col in zip(axes, cat_cols):
    counts = df_cat[col].value_counts()
    palette = sns.color_palette("Set2", n_colors=len(counts))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=palette,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        explode=[0.05] * len(counts)
    )
    # Set label font size and weight
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    ax.axis('equal')
    sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    sns.set_style("whitegrid")
    ax.set_title(col.replace('_', ' ').title(), fontsize=15, fontweight='bold')

# hide any unused subplots
for ax in axes[len(cat_cols):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig("../plots/categoricals_distribution.png",
            dpi=1200)
plt.savefig("../plots/categoricals_distribution.svg",
            dpi=1200)
plt.close()

# frykman pattern distribution
plt.figure(figsize=(7, 5), dpi=150)
palette = sns.color_palette("Set2", n_colors=df_cat['frykman_pattern'].nunique())
sns.countplot(
    y='frykman_pattern',
    data=df_cat,
    order=df_cat['frykman_pattern'].value_counts().index,
    palette=palette,
    edgecolor='black'
)
plt.title("Frykman Pattern Distribution", fontsize=16, fontweight='bold')
plt.xlabel("Patient Count", fontsize=13, fontweight='bold')
plt.ylabel("Pattern", fontsize=13, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(top=True, right=True)
plt.tight_layout()
plt.savefig("../plots/frykman_distribution.png", dpi=1200)
plt.savefig("../plots/frykman_distribution.svg", dpi=1200)
plt.close()

# comorbidity distribution
comorb_cols = ['Diabetes', 'Osteoarthritis', 'Rheumatoid arthritis', 'Hypertension',
               'Depression', 'Anxiety', 'Anemia']

tot = len(df_cat)
perc = (df_cat[comorb_cols].sum() / tot * 100).sort_values(ascending=False)
perc_df = perc.reset_index()
perc_df.columns = ['comorbidity', 'percent']

plt.figure(figsize=(8, 5), dpi=150)
palette = sns.color_palette("Set2", n_colors=len(perc_df))
sns.barplot(
    data=perc_df,
    x='percent',
    y='comorbidity',
    palette=palette,
    edgecolor='black'
)
plt.xlabel('Patients (%)', fontsize=14, fontweight='bold')
plt.ylabel('', fontsize=14)
plt.title('Comorbidity Distribution', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(top=True, right=True)

# Add bold percentage labels
for i, pct in enumerate(perc_df['percent']):
    plt.text(pct + 0.5, i, f"{pct:.1f}%", va='center', fontsize=13, fontweight='bold')

plt.xlim(0, perc_df['percent'].max() + 5)
plt.tight_layout()
plt.savefig("../plots/comorbidity_distribution.png", dpi=1200)
plt.savefig("../plots/comorbidity_distribution.svg", dpi=1200)
plt.close()





