# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from scipy import stats
import statsmodels.formula.api as smf

COLORS = {
    'hip': '#F4A7B9',
    'spine': '#90C9A0',
}

PHASE_ORDER = [
    'PRE_TEST',
    'IN_TEST',
    'POST_TEST',
    'FOLLOWUP_6MO',
    'FOLLOWUP_1Y'
]

PHASE_LABELS = [
    'Pre',
    'In-Bedrest',
    'Post',
    '6-Month',
    '1-Year'
]

PHASE_POS = {phase: i for i, phase in enumerate(PHASE_ORDER)}

# %%
# %%


def load_hip_files(folder='Hip'):
    all_dfs = []

    for fname in os.listdir(folder):

        if not fname.endswith('.csv'):
            continue

        if 'LH_Bone' not in fname and 'RH_Bone' not in fname:
            continue

        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)

        df.columns = df.columns.str.strip()

        # Reconstruct missing follow-up labels from filenames
        if '_6MO_' in fname:
            df['TEST_PHASE'] = df['TEST_PHASE'].fillna('FOLLOWUP_6MO')
            df['BR_DAY'] = df['BR_DAY'].fillna(180)

        if '_1Y_' in fname:
            df['TEST_PHASE'] = df['TEST_PHASE'].fillna('FOLLOWUP_1Y')
            df['BR_DAY'] = df['BR_DAY'].fillna(365)

        side = 'L' if 'LH_Bone' in fname else 'R'
        df['SIDE'] = side

        keep = [
            'SUBJECT',
            'CAMPAIGN',
            'TEST_PHASE',
            'BR_DAY',
            'HTOT_BMD',
            'SIDE'
        ]

        df = df[[c for c in keep if c in df.columns]]
        all_dfs.append(df)

    hip = pd.concat(all_dfs, ignore_index=True)

    hip = hip.dropna(subset=['TEST_PHASE'])

    # Average repeated measurements within each side
    hip = (
        hip.groupby(
            ['SUBJECT', 'TEST_PHASE', 'BR_DAY', 'SIDE'],
            as_index=False
        )
        .agg({'HTOT_BMD': 'mean'})
    )

    # Average left and right hip measurements
    hip = (
        hip.groupby(
            ['SUBJECT', 'TEST_PHASE', 'BR_DAY'],
            as_index=False
        )
        .agg({'HTOT_BMD': 'mean'})
    )

    return hip


hip = load_hip_files()

print(
    f"Hip data loaded: "
    f"{hip['SUBJECT'].nunique()} subjects, "
    f"{len(hip)} records"
)

print(hip['TEST_PHASE'].unique())

# %%
# %%


def load_spine_files(folder='Spine'):
    all_dfs = []

    for fname in os.listdir(folder):

        if not fname.endswith('.csv'):
            continue

        if 'Spine' not in fname or 'S_Bone' in fname:
            continue

        fpath = os.path.join(folder, fname)
        df = pd.read_csv(fpath)

        df.columns = df.columns.str.strip()

        # Reconstruct missing follow-up labels from filenames
        if '_6MO_' in fname:
            df['TEST_PHASE'] = df['TEST_PHASE'].fillna('FOLLOWUP_6MO')
            df['BR_DAY'] = df['BR_DAY'].fillna(180)

        if '_1Y_' in fname:
            df['TEST_PHASE'] = df['TEST_PHASE'].fillna('FOLLOWUP_1Y')
            df['BR_DAY'] = df['BR_DAY'].fillna(365)

        keep = [
            'SUBJECT',
            'CAMPAIGN',
            'TEST_PHASE',
            'BR_DAY',
            'TOT_BMD'
        ]

        df = df[[c for c in keep if c in df.columns]]
        all_dfs.append(df)

    spine = pd.concat(all_dfs, ignore_index=True)

    spine = spine.dropna(subset=['TEST_PHASE'])

    spine = (
        spine.groupby(
            ['SUBJECT', 'TEST_PHASE', 'BR_DAY'],
            as_index=False
        )
        .agg({'TOT_BMD': 'mean'})
    )

    return spine


spine = load_spine_files()

print(
    f"Spine data loaded: "
    f"{spine['SUBJECT'].nunique()} subjects, "
    f"{len(spine)} records"
)

print(spine['TEST_PHASE'].unique())

# %%
# %%


def add_pct_change(df, bmd_col):

    df = df.drop(
        columns=['BASELINE', 'PCT_CHANGE'],
        errors='ignore'
    )

    baseline = (
        df[df['TEST_PHASE'] == 'PRE_TEST']
        .groupby('SUBJECT')[bmd_col]
        .mean()
        .reset_index()
        .rename(columns={bmd_col: 'BASELINE'})
    )

    df = df.merge(
        baseline,
        on='SUBJECT',
        how='left'
    )

    df['PCT_CHANGE'] = (
        (df[bmd_col] - df['BASELINE'])
        / df['BASELINE']
    ) * 100

    return df


hip = add_pct_change(hip, 'HTOT_BMD')
spine = add_pct_change(spine, 'TOT_BMD')

print("Hip:")
print(
    hip[
        ['SUBJECT', 'TEST_PHASE', 'BR_DAY', 'HTOT_BMD', 'BASELINE', 'PCT_CHANGE']
    ].head()
)

print("\nSpine:")
print(
    spine[
        ['SUBJECT', 'TEST_PHASE', 'BR_DAY', 'TOT_BMD', 'BASELINE', 'PCT_CHANGE']
    ].head()
)

# %%
# %%
hip_long = hip[
    ['SUBJECT', 'TEST_PHASE', 'BR_DAY', 'HTOT_BMD']
].copy()

hip_long = hip_long.rename(
    columns={'HTOT_BMD': 'BMD'}
)

hip_long['SITE'] = 'Hip'


spine_long = spine[
    ['SUBJECT', 'TEST_PHASE', 'BR_DAY', 'TOT_BMD']
].copy()

spine_long = spine_long.rename(
    columns={'TOT_BMD': 'BMD'}
)

spine_long['SITE'] = 'Spine'


combined = pd.concat(
    [hip_long, spine_long],
    ignore_index=True
)

combined = combined.dropna(subset=['BMD'])

combined['TEST_PHASE'] = pd.Categorical(
    combined['TEST_PHASE'],
    categories=PHASE_ORDER,
    ordered=True
)

combined['SITE'] = pd.Categorical(
    combined['SITE'],
    categories=['Hip', 'Spine']
)

print(f"Total observations: {len(combined)}")

print(
    combined
    .groupby(['SITE', 'TEST_PHASE'], observed=True)
    .size()
)

# %%
# %%


def paired_cohens_dz(df, bmd_col):
    """
    Paired Cohen's dz for PRE_TEST vs POST_TEST.

    Positive values indicate a decrease from pre-bedrest
    to post-bedrest.
    """

    pre = (
        df[df['TEST_PHASE'] == 'PRE_TEST']
        [['SUBJECT', bmd_col]]
        .rename(columns={bmd_col: 'PRE'})
    )

    post = (
        df[df['TEST_PHASE'] == 'POST_TEST']
        [['SUBJECT', bmd_col]]
        .rename(columns={bmd_col: 'POST'})
    )

    paired = pre.merge(
        post,
        on='SUBJECT',
        how='inner'
    ).dropna()

    differences = paired['PRE'] - paired['POST']

    dz = (
        differences.mean()
        / differences.std(ddof=1)
    )

    return dz, len(paired)


effect_rows = []

for site_name, df, bmd_col in [
    ('Hip', hip, 'HTOT_BMD'),
    ('Spine', spine, 'TOT_BMD')
]:

    dz, n = paired_cohens_dz(
        df,
        bmd_col
    )

    effect_rows.append({
        'Site': site_name,
        'Comparison': 'Pre vs Post',
        'Cohens_dz': dz,
        'Paired_n': n
    })

effect_sizes = pd.DataFrame(effect_rows)

print(effect_sizes.round(3))

effect_sizes.to_csv(
    'effect_sizes.csv',
    index=False
)

# %%
# %%
model = smf.mixedlm(
    "BMD ~ C(TEST_PHASE, Treatment(reference='PRE_TEST')) * SITE",
    data=combined,
    groups=combined["SUBJECT"]
)

result = model.fit(
    method='lbfgs',
    reml=True
)

print(result.summary())

# %%
# %%
conf_int = result.conf_int()

summary_table = pd.DataFrame({
    'Coefficient': result.params,
    'Std_Error': result.bse,
    'p_value': result.pvalues,
})

summary_table['CI_lower_95'] = conf_int[0]
summary_table['CI_upper_95'] = conf_int[1]

summary_table = summary_table.round(4)

print(summary_table)

summary_table.to_csv(
    'mixed_model_results.csv'
)

print("\nSaved to mixed_model_results.csv")

# %%
# Build a subject x phase presence matrix (1 = data present, 0 = missing)
phase_order = ['PRE_TEST', 'IN_TEST',
               'POST_TEST', 'FOLLOWUP_6MO', 'FOLLOWUP_1Y']

hip_presence = (
    hip.pivot_table(index='SUBJECT', columns='TEST_PHASE',
                    values='HTOT_BMD', aggfunc='count')
    .reindex(columns=phase_order)
    .notna()
    .astype(int)
)

print("Missingness pattern (1 = present, 0 = missing), Hip:")
print(hip_presence)

print("\nTotal subjects with data at each phase:")
print(hip_presence.sum())

print("\nSubjects missing at least one follow-up (6MO or 1Y):")
missing_followup = hip_presence[(hip_presence['FOLLOWUP_6MO'] == 0) | (
    hip_presence['FOLLOWUP_1Y'] == 0)]
print(f"{len(missing_followup)} out of {len(hip_presence)} subjects")

# %%
# %%
campaign_map = {}

for fname in os.listdir('Hip'):

    if not fname.endswith('.csv'):
        continue

    if fname == '.DS_Store':
        continue

    parts = fname.split('_')

    if len(parts) <= 5:
        continue

    campaign_code = parts[4]
    subject_id = parts[5]

    if subject_id == 'All':
        continue

    campaign_map[subject_id] = campaign_code


campaign_df = pd.DataFrame(
    list(campaign_map.items()),
    columns=['SUBJECT', 'CAMPAIGN_CODE']
)

campaign_df['SUBJECT'] = pd.to_numeric(
    campaign_df['SUBJECT'],
    errors='coerce'
)

campaign_df = campaign_df.dropna(
    subset=['SUBJECT']
)

campaign_df['SUBJECT'] = campaign_df['SUBJECT'].astype(int)

merged = (
    hip_presence
    .reset_index()
    .merge(campaign_df, on='SUBJECT')
)

print(
    merged[
        ['SUBJECT', 'CAMPAIGN_CODE', 'IN_TEST']
    ].sort_values('CAMPAIGN_CODE')
)

print("\nIN_TEST presence by campaign:")

print(
    merged
    .groupby('CAMPAIGN_CODE')['IN_TEST']
    .agg(['sum', 'count'])
)

# %%
# %%
hip_presence = (
    hip.pivot_table(
        index='SUBJECT',
        columns='TEST_PHASE',
        values='HTOT_BMD',
        aggfunc='count'
    )
    .reindex(columns=PHASE_ORDER)
    .notna()
    .astype(int)
)


spine_presence = (
    spine.pivot_table(
        index='SUBJECT',
        columns='TEST_PHASE',
        values='TOT_BMD',
        aggfunc='count'
    )
    .reindex(columns=PHASE_ORDER)
    .notna()
    .astype(int)
)

# Make sure both matrices contain the same subjects
all_subjects = sorted(
    set(hip_presence.index) | set(spine_presence.index)
)

hip_presence = hip_presence.reindex(
    index=all_subjects,
    fill_value=0
)

spine_presence = spine_presence.reindex(
    index=all_subjects,
    fill_value=0
)

# Verify that hip and spine have identical availability
missingness_matches = hip_presence.equals(spine_presence)

print(
    f"Hip and spine missingness patterns identical: "
    f"{missingness_matches}"
)

if not missingness_matches:
    print("WARNING: Hip and spine availability differ.")
else:
    print(
        "The hip missingness matrix is therefore used as the "
        "representative Hip & Spine availability figure."
    )

print("\nTotal subjects with data at each phase:")
print(hip_presence.sum())

missing_followup = hip_presence[
    (hip_presence['FOLLOWUP_6MO'] == 0) |
    (hip_presence['FOLLOWUP_1Y'] == 0)
]

print(
    f"\nSubjects missing at least one follow-up: "
    f"{len(missing_followup)} / {len(hip_presence)}"
)

# %%
# %%
fig, ax = plt.subplots(figsize=(8, 10))

sns.heatmap(
    hip_presence,
    cmap=["#D4C7C7", "#309836"],
    cbar=False,
    linewidths=0.5,
    linecolor='white',
    ax=ax
)

ax.set_title(
    'Data Availability by Subject and Test Phase\n(Hip & Spine)',
    fontsize=12
)

ax.set_xlabel('Test Phase')
ax.set_ylabel('Subject ID')

present_patch = mpatches.Patch(
    color='#2E7D32',
    label='Present'
)

missing_patch = mpatches.Patch(
    color='#D1BFBF',
    label='Missing'
)

ax.legend(
    handles=[present_patch, missing_patch],
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

plt.tight_layout()

plt.savefig(
    'figure_missingness.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# %%


def add_pct_change(df, bmd_col):
    df = df.drop(columns=['BASELINE', 'PCT_CHANGE'], errors='ignore')
    baseline = (
        df[df['TEST_PHASE'] == 'PRE_TEST']
        .groupby('SUBJECT')[bmd_col]
        .mean()
        .reset_index()
        .rename(columns={bmd_col: 'BASELINE'})
    )
    df = df.merge(baseline, on='SUBJECT', how='left')
    df['PCT_CHANGE'] = ((df[bmd_col] - df['BASELINE']) / df['BASELINE']) * 100
    return df


hip = add_pct_change(hip, 'HTOT_BMD')
spine = add_pct_change(spine, 'TOT_BMD')
print(hip.columns.tolist())
print(spine.columns.tolist())

# %%
print(hip.columns.tolist())

# %%
# %%
fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

for ax, df, label, color in [
    (axes[0], hip, 'Hip', COLORS['hip']),
    (axes[1], spine, 'Spine', COLORS['spine'])
]:

    df_plot = df[
        df['TEST_PHASE'].isin(PHASE_ORDER)
    ].copy()

    df_plot['X'] = (
        df_plot['TEST_PHASE']
        .map(PHASE_POS)
    )

    # Individual participant trajectories
    for subj, sub_df in df_plot.groupby('SUBJECT'):

        sub_df = sub_df.sort_values('X')

        ax.plot(
            sub_df['X'],
            sub_df['PCT_CHANGE'],
            color=color,
            alpha=0.15,
            linewidth=1
        )

    # Group mean
    group_mean = (
        df_plot
        .groupby('X')['PCT_CHANGE']
        .mean()
    )

    ax.plot(
        group_mean.index,
        group_mean.values,
        color=color,
        linewidth=3,
        marker='o',
        markersize=6,
        label='Group mean'
    )

    ax.axhline(
        0,
        color='#999999',
        linewidth=1,
        linestyle='--'
    )

    ax.set_xticks(
        range(len(PHASE_ORDER))
    )

    ax.set_xticklabels(
        PHASE_LABELS,
        rotation=20
    )

    ax.set_ylabel(
        '% Change from Baseline'
    )

    ax.set_title(
        f'{label}: Individual Trajectories'
    )

    ax.legend()


plt.tight_layout()

plt.savefig(
    'figure_trajectories.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# %%
# %%
# Build model-based estimates for each site and phase

fixed_effects = result.fe_params
cov_fixed = result.cov_params().loc[
    fixed_effects.index,
    fixed_effects.index
]


def get_design_vector(site, phase):
    """
    Create the fixed-effect design vector corresponding
    to one site/phase combination.
    """

    row = pd.DataFrame({
        'TEST_PHASE': pd.Categorical(
            [phase],
            categories=PHASE_ORDER,
            ordered=True
        ),
        'SITE': pd.Categorical(
            [site],
            categories=['Hip', 'Spine']
        )
    })

    # Use the same formula structure as the fitted model
    from patsy import build_design_matrices

    design_info = result.model.data.design_info

    X = build_design_matrices(
        [design_info],
        row,
        return_type='dataframe'
    )[0]

    return X.iloc[0].reindex(
        fixed_effects.index,
        fill_value=0
    ).values


rows = []

for site in ['Hip', 'Spine']:

    # Baseline model estimate
    x_baseline = get_design_vector(
        site,
        'PRE_TEST'
    )

    baseline_mean = np.dot(
        x_baseline,
        fixed_effects.values
    )

    for phase in PHASE_ORDER:

        x_phase = get_design_vector(
            site,
            phase
        )

        phase_mean = np.dot(
            x_phase,
            fixed_effects.values
        )

        # Difference from baseline
        x_diff = x_phase - x_baseline

        change = np.dot(
            x_diff,
            fixed_effects.values
        )

        # Standard error of the contrast
        variance = np.dot(
            x_diff,
            np.dot(cov_fixed, x_diff)
        )

        se = np.sqrt(
            max(variance, 0)
        )

        ci_lower_change = change - 1.96 * se
        ci_upper_change = change + 1.96 * se

        # Convert absolute change to percentage of baseline
        pct_change = (
            change / baseline_mean
        ) * 100

        pct_ci_lower = (
            ci_lower_change / baseline_mean
        ) * 100

        pct_ci_upper = (
            ci_upper_change / baseline_mean
        ) * 100

        rows.append({
            'SITE': site,
            'TEST_PHASE': phase,
            'BASELINE_BMD': baseline_mean,
            'MODEL_BMD': phase_mean,
            'CHANGE_BMD': change,
            'CI_lower_BMD': ci_lower_change,
            'CI_upper_BMD': ci_upper_change,
            'PCT_CHANGE': pct_change,
            'CI_lower_pct': pct_ci_lower,
            'CI_upper_pct': pct_ci_upper
        })


model_change_df = pd.DataFrame(rows)

print(
    model_change_df.round(4)
)

# %%
# %%
fig, ax = plt.subplots(
    figsize=(9, 6)
)

offset = 0.08

for site, color, shift in [
    ('Hip', COLORS['hip'], -offset),
    ('Spine', COLORS['spine'], offset)
]:

    site_data = model_change_df[
        model_change_df['SITE'] == site
    ].copy()

    site_data['X'] = (
        site_data['TEST_PHASE']
        .map(PHASE_POS)
        + shift
    )

    site_data = site_data.sort_values('X')

    yerr = [
        site_data['PCT_CHANGE']
        - site_data['CI_lower_pct'],

        site_data['CI_upper_pct']
        - site_data['PCT_CHANGE']
    ]

    ax.errorbar(
        site_data['X'],
        site_data['PCT_CHANGE'],
        yerr=yerr,
        fmt='none',
        ecolor=color,
        elinewidth=1.2,
        alpha=0.5,
        capsize=4,
        capthick=1.2
    )

    ax.plot(
        site_data['X'],
        site_data['PCT_CHANGE'],
        color=color,
        linewidth=2.5,
        marker='o',
        markersize=7,
        label=site
    )


ax.axhline(
    0,
    color='#999999',
    linewidth=1,
    linestyle='--'
)

ax.set_ylabel(
    '% Change from Baseline\n(model-estimated, 95% CI)'
)

ax.set_title(
    'Hip vs Spine: Model-Estimated Change from Baseline'
)

ax.set_xticks(
    range(len(PHASE_ORDER))
)

ax.set_xticklabels(
    PHASE_LABELS,
    rotation=20
)

ax.legend()

plt.tight_layout()

plt.savefig(
    'figure_hip_vs_spine_model.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# %%
# %%
model_change_df.to_csv(
    'model_estimated_changes.csv',
    index=False
)

print(
    "Saved model_estimated_changes.csv"
)

# %%
# %%
final_results = model_change_df[
    [
        'SITE',
        'TEST_PHASE',
        'PCT_CHANGE',
        'CI_lower_pct',
        'CI_upper_pct'
    ]
].copy()

final_results = final_results.rename(
    columns={
        'TEST_PHASE': 'Phase',
        'PCT_CHANGE': 'Estimated % Change',
        'CI_lower_pct': '95% CI Lower',
        'CI_upper_pct': '95% CI Upper'
    }
)

final_results = final_results.round(2)

print(final_results)

final_results.to_csv(
    'final_model_results.csv',
    index=False
)

# %%
print(results_table.to_markdown())
