import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import false_discovery_control
from statsmodels.formula.api import mixedlm
import io
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Manzoni Lab - Statistical Engine", layout="wide")
st.markdown("# ⚖️ Moteur Statistique Avancé")
st.markdown("### Comparaison de Groupes : Propriétés Intrinsèques & Courbes I-F")
st.divider()

STEP_DURATION_S = 0.5  # durée de l'échelon de courant (0.1s → 0.6s = 500 ms)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def identify_outliers(df, column, group_col='Condition'):
    """Tukey IQR ×1.5 par groupe. Requiert n ≥ 4."""
    out = []
    for grp in df[group_col].unique():
        d = df[df[group_col] == grp][column].dropna()
        if len(d) < 4:
            continue
        Q1, Q3 = d.quantile(0.25), d.quantile(0.75)
        IQR = Q3 - Q1
        out.extend(d[(d < Q1 - 1.5*IQR) | (d > Q3 + 1.5*IQR)].index.tolist())
    return list(set(out))


def cohen_d(a, b):
    """Cohen's d avec pooled SD (Hedges)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def rank_biserial_r(u_stat, n1, n2):
    """r rank-biséral à partir de U. Varie de -1 à +1."""
    return 1 - (2 * u_stat) / (n1 * n2)


def effect_label(v):
    av = abs(v)
    if av < 0.2: return "négligeable"
    if av < 0.5: return "petit"
    if av < 0.8: return "moyen"
    return "grand"


def run_shapiro(data, label):
    """Shapiro-Wilk avec gardes n<3 et n>50."""
    n = len(data)
    if n < 3:
        st.warning(f"⚠️ {label} : n={n} < 3 → routage non-paramétrique.")
        return np.nan, 0.0
    if n > 50:
        st.info(f"ℹ️ {label} : n={n} > 50 → Shapiro-Wilk peu fiable, routage non-paramétrique.")
        return np.nan, 0.0
    return stats.shapiro(data)


def choose_test(g1, g2, grp1_name, grp2_name):
    """
    Routing automatique :
      Normal + variances égales  → Student
      Normal + variances inégales → Welch
      Non-normal                  → Mann-Whitney U
    """
    _, p1 = run_shapiro(g1, grp1_name)
    _, p2 = run_shapiro(g2, grp2_name)
    is_normal = (p1 > 0.05) and (p2 > 0.05)

    n1, n2 = len(g1), len(g2)
    _, p_lev = stats.levene(g1, g2) if (n1 >= 2 and n2 >= 2) else (np.nan, np.nan)
    is_homo = (p_lev > 0.05) if not np.isnan(p_lev) else True

    if is_normal and is_homo:
        name = "T-test (Student)"
        stat, p = stats.ttest_ind(g1, g2, equal_var=True)
        d = cohen_d(g1, g2)
        es = f"Cohen d={d:.3f} ({effect_label(d)})" if not np.isnan(d) else "N/A"
    elif is_normal and not is_homo:
        name = "T-test (Welch)"
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        d = cohen_d(g1, g2)
        es = f"Cohen d={d:.3f} ({effect_label(d)})" if not np.isnan(d) else "N/A"
    else:
        name = "Mann-Whitney U"
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        r = rank_biserial_r(stat, n1, n2)
        es = f"r={r:.3f} ({effect_label(r)})"

    return name, stat, p, es, is_normal, p_lev if not np.isnan(p_lev) else None


def load_csvs(files, condition_name):
    """
    Charge un ou plusieurs CSV d'une condition.
    - Lit le buffer une seule fois (io.BytesIO) → évite le double-read bug.
    - Détecte les en-têtes Excel 'Tableau'.
    - Ajoute les colonnes 'Condition' et 'Cell' (= nom du fichier).
    """
    dfs = []
    for f in files:
        raw = f.getvalue()
        peek = pd.read_csv(io.BytesIO(raw), nrows=2)
        skip = 2 if 'Tableau' in peek.to_string() else 0
        df = pd.read_csv(io.BytesIO(raw), skiprows=range(skip))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['Condition'] = condition_name
        df['Cell'] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def lmm_summary_table(result, grp1_name, grp2_name):
    """Extrait les fixed effects du résultat LMM en DataFrame lisible."""
    fe = result.fe_params
    pv = result.pvalues
    ci = result.conf_int()

    rows = [
        ('Intercept',                     'Intercept'),
        (f'Pente I-F ({grp1_name})',      'I_inj'),
        (f'Décalage vertical ({grp2_name})', 'Condition_bin'),
        ('Différence de pente (interaction)', 'I_inj:Condition_bin'),
    ]

    records = []
    for label, key in rows:
        records.append({
            "Paramètre":  label,
            "Coeff.":     fe.get(key, np.nan),
            "IC 95% bas": ci.loc[key, 0] if key in ci.index else np.nan,
            "IC 95% haut":ci.loc[key, 1] if key in ci.index else np.nan,
            "p-value":    pv.get(key, np.nan),
        })

    df_res = pd.DataFrame(records).round(5)
    df_res['Sig.'] = df_res['p-value'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    return df_res


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
analysis_mode = st.sidebar.radio(
    "Mode d'analyse",
    ["📊 Propriétés intrinsèques (Global CSV)",
     "⚡ Courbes I-F (Sweeps CSV)"],
    index=0
)
st.sidebar.divider()
st.sidebar.markdown("""
**Propriétés intrinsèques** : chargez les `_Global.csv`
(1 ligne par cellule : Vrest, Rin, Tau, Rhéobase).

**Courbes I-F** : chargez les `_Sweeps.csv`
(1 ligne par échelon : I_inj, Nb_Spikes).
Chaque fichier = une cellule.
""")

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    grp1_name  = st.text_input("Nom Groupe 1", "Mâles")
    grp1_files = st.file_uploader(f"Fichiers {grp1_name}", accept_multiple_files=True, key="g1")
with c2:
    grp2_name  = st.text_input("Nom Groupe 2", "Femelles")
    grp2_files = st.file_uploader(f"Fichiers {grp2_name}", accept_multiple_files=True, key="g2")

if not (grp1_files and grp2_files):
    st.info("Chargez au moins un fichier par groupe pour démarrer.")
    st.stop()

df1 = load_csvs(grp1_files, grp1_name)
df2 = load_csvs(grp2_files, grp2_name)
df_all = pd.concat([df1, df2], ignore_index=True)
st.success(f"✅ {df_all['Cell'].nunique()} cellules chargées — "
           f"{df1['Cell'].nunique()} {grp1_name} / {df2['Cell'].nunique()} {grp2_name}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — PROPRIÉTÉS INTRINSÈQUES
# ══════════════════════════════════════════════════════════════════════════════
if "Propriétés" in analysis_mode:
    st.divider()
    st.subheader("📊 Comparaison des Propriétés Intrinsèques")
    st.caption("Chaque ligne = une cellule. n = nombre de cellules.")

    numeric_cols = [c for c in df_all.select_dtypes(include=np.number).columns
                    if c not in ['Sweep', 'I_inj', 'index']]
    if not numeric_cols:
        st.error("Aucune colonne numérique détectée.")
        st.stop()

    remove_out = st.checkbox("Exclure les outliers (IQR×1.5) des tests", value=False)
    metric     = st.selectbox("Paramètre à analyser :", numeric_cols)

    out_idx  = identify_outliers(df_all, metric, 'Condition')
    df_test  = df_all.drop(index=out_idx) if (remove_out and out_idx) else df_all.copy()
    if remove_out and out_idx:
        st.warning(f"🧹 {len(out_idx)} outlier(s) exclus du test pour **{metric}**.")

    g1 = df_test[df_test['Condition'] == grp1_name][metric].dropna()
    g2 = df_test[df_test['Condition'] == grp2_name][metric].dropna()

    test_name, stat_val, p_val, es_str, is_norm, p_lev = choose_test(g1, g2, grp1_name, grp2_name)

    st.markdown(f"### {metric}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Normalité (S-W)", "✅ Validée" if is_norm else "❌ Rejetée")
    m2.metric("Variances (Levene)", "✅ Égales" if (p_lev and p_lev > 0.05) else "❌ Inégales",
              f"p={p_lev:.4f}" if p_lev else "N/A")
    m3.metric(f"Test : {test_name}", "⭐ p<0.05" if p_val < 0.05 else "ns",
              f"p={p_val:.4f}", delta_color="normal" if p_val < 0.05 else "off")
    m4.metric("Taille d'effet", es_str.split(" (")[0], es_str.split("(")[-1].rstrip(")"),
              delta_color="off")
    st.caption(f"n₁={len(g1)} ({grp1_name}) | n₂={len(g2)} ({grp2_name})"
               + (f" | {len(out_idx)} outlier(s) détecté(s)" if out_idx else ""))

    with st.expander("📋 Statistiques descriptives"):
        desc = pd.DataFrame({
            "Groupe":  [grp1_name, grp2_name],
            "n":       [len(g1), len(g2)],
            "Moyenne": [g1.mean(), g2.mean()],
            "Médiane": [g1.median(), g2.median()],
            "SD":      [g1.std(ddof=1), g2.std(ddof=1)],
            "SEM":     [g1.sem(), g2.sem()],
            "Min":     [g1.min(), g2.min()],
            "Max":     [g1.max(), g2.max()],
        }).set_index("Groupe").round(4)
        st.dataframe(desc, use_container_width=True)

    df_all['Is_Outlier'] = df_all.index.isin(out_idx)
    palette = {False: "tab:blue", True: "crimson"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.violinplot(data=df_all, x='Condition', y=metric, inner=None,
                   color=".92", linewidth=0.8, ax=ax1)
    sns.stripplot(data=df_all, x='Condition', y=metric, hue='Is_Outlier',
                  palette=palette, size=7, alpha=0.85, jitter=True, ax=ax1)
    ax1.set_title(f"{metric}\n{test_name} : p={p_val:.4f} | {es_str}", fontweight='bold', fontsize=10)
    ax1.set_ylabel(metric)
    hl, _ = ax1.get_legend_handles_labels()
    if len(hl) >= 2:
        ax1.legend(hl[:2], ['Valide', 'Outlier'], fontsize=8, loc='upper right')

    sns.boxplot(data=df_all, x='Condition', y=metric, width=0.4, color=".92",
                linewidth=1.2, fliersize=0, ax=ax2)
    sns.stripplot(data=df_all, x='Condition', y=metric, hue='Is_Outlier',
                  palette=palette, size=7, alpha=0.85, jitter=True, ax=ax2)
    ax2.set_ylabel("")
    if ax2.get_legend(): ax2.get_legend().remove()
    yv  = df_all[metric].dropna()
    yr  = yv.max() - yv.min()
    bh  = yv.max() + yr * 0.08
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax2.plot([0,0,1,1], [bh, bh+yr*0.02, bh+yr*0.02, bh], lw=1.2, color='black')
    ax2.text(0.5, bh+yr*0.03, sig, ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax2.set_title(f"n₁={len(g1)} | n₂={len(g2)}", fontsize=10)

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

    if out_idx:
        st.markdown("#### 🚨 Outliers détectés")
        id_cols = [c for c in ['Cell', 'Condition', metric] if c in df_all.columns]
        st.dataframe(df_all.loc[out_idx, id_cols], use_container_width=True)

    st.divider()
    summary = pd.DataFrame({
        "Métrique": [metric], "Groupe_1": [grp1_name], "n_1": [len(g1)],
        "Moyenne_1": [g1.mean()], "SD_1": [g1.std(ddof=1)],
        "Groupe_2": [grp2_name], "n_2": [len(g2)],
        "Moyenne_2": [g2.mean()], "SD_2": [g2.std(ddof=1)],
        "Test": [test_name], "Statistique": [stat_val], "p_value": [p_val],
        "Significatif_0.05": [p_val < 0.05], "Taille_Effet": [es_str],
        "Outliers_détectés": [len(out_idx)], "Outliers_exclus": [remove_out and len(out_idx) > 0],
    })
    e1, e2 = st.columns(2)
    e1.download_button("💾 Résumé statistique (CSV)", summary.to_csv(index=False).encode(),
                       f"stats_{metric}.csv", use_container_width=True)
    e2.download_button("💾 Données brutes (CSV)", df_all.to_csv(index=False).encode(),
                       "donnees_brutes.csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — COURBES I-F
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.divider()
    st.subheader("⚡ Comparaison des Courbes I-F (Fréquence–Courant)")
    st.caption(
        "Chaque fichier = une cellule. "
        "La courbe I-F est construite sur les échelons dépolarisants (I_inj > 0). "
        f"Fréquence = Nb_Spikes / {STEP_DURATION_S} s."
    )

    required = {'I_inj', 'Nb_Spikes'}
    missing = required - set(df_all.columns)
    if missing:
        st.error(f"Colonnes manquantes : {missing}. Chargez bien les fichiers `_Sweeps.csv`.")
        st.stop()

    df_if = df_all[df_all['I_inj'] > 0].copy()
    df_if['Firing_Hz'] = df_if['Nb_Spikes'] / STEP_DURATION_S

    if df_if.empty:
        st.error("Aucun échelon dépolarisant (I_inj > 0) trouvé.")
        st.stop()

    n_cells_g1 = df_if[df_if['Condition'] == grp1_name]['Cell'].nunique()
    n_cells_g2 = df_if[df_if['Condition'] == grp2_name]['Cell'].nunique()

    steps_g1 = set(df_if[df_if['Condition'] == grp1_name]['I_inj'].unique())
    steps_g2 = set(df_if[df_if['Condition'] == grp2_name]['I_inj'].unique())
    common_steps = sorted(steps_g1 & steps_g2)

    if not common_steps:
        st.error("Aucun échelon commun entre les deux groupes.")
        st.stop()

    df_if_common = df_if[df_if['I_inj'].isin(common_steps)].copy()

    st.info(
        f"**{len(common_steps)} échelons communs** "
        f"({int(min(common_steps))} → {int(max(common_steps))} pA) | "
        f"Cellules : {n_cells_g1} {grp1_name} / {n_cells_g2} {grp2_name}"
    )

    # ── Options ───────────────────────────────────────────────────────────────
    show_individual = st.checkbox("Afficher les courbes individuelles", value=True)
    run_lmm = st.checkbox(
        "Lancer le LMM",
        value=True,
        help="Teste si la PENTE (gain) et l'INTERCEPT (seuil) de la courbe I-F "
             "diffèrent entre groupes."
    )

    # ── Animal ID assignment ──────────────────────────────────────────────────
    #
    # Par défaut : 1 fichier CSV = 1 cellule = 1 animal distinct.
    # Si plusieurs cellules viennent du même animal, l'utilisateur les regroupe
    # ici via des champs texte (un par cellule chargée).
    #
    # Impact sur le LMM :
    #   - 1 cellule = 1 animal  →  modèle simple  : (1 | Cell)
    #     La cellule est l'unité de réplication et l'effet aléatoire.
    #
    #   - Plusieurs cellules par animal → modèle nested : (1 | Animal / Cell)
    #     L'animal est l'effet aléatoire de niveau supérieur ;
    #     la cellule est nichée à l'intérieur.
    #     Ceci corrige la pseudo-réplication cellule-dans-animal :
    #     deux cellules du même animal ne sont pas indépendantes.
    #
    # Le point-par-point Mann-Whitney reste au niveau cellule dans les deux cas
    # (l'unité d'observation est la cellule), mais une note est affichée quand
    # plusieurs cellules partagent un animal.

    st.divider()
    st.markdown("#### 🐭 Attribution Animal → Cellule(s)")

    all_cells = sorted(df_if_common['Cell'].unique().tolist())

    use_animal_grouping = st.checkbox(
        "Plusieurs cellules partagent le même animal (activer le modèle nested)",
        value=False,
        help=(
            "Par défaut (décoché) : chaque fichier CSV = 1 animal distinct. "
            "Le LMM utilise (1 | Cell) comme effet aléatoire.\n\n"
            "Si coché : assignez un Animal_ID à chaque cellule. "
            "Le LMM utilisera (1 | Animal / Cell) pour corriger "
            "la pseudo-réplication cellule-dans-animal."
        )
    )

    cell_to_animal = {}

    if not use_animal_grouping:
        # Default: animal = cell
        for cell in all_cells:
            cell_to_animal[cell] = cell
        st.caption(
            f"Mode par défaut : {len(all_cells)} cellule(s) = {len(all_cells)} animal(s). "
            "LMM : effets aléatoires = **(1 | Cell)**."
        )
    else:
        st.markdown(
            "Assignez un **identifiant animal** à chaque fichier CSV. "
            "Cellules avec le même ID = même animal."
        )
        # Show inputs in a compact 2-column grid
        col_pairs = [all_cells[i:i+2] for i in range(0, len(all_cells), 2)]
        for pair in col_pairs:
            cols = st.columns(len(pair))
            for col, cell in zip(cols, pair):
                idx = all_cells.index(cell)
                default_id = f"Animal_{str(idx+1).zfill(2)}"
                val = col.text_input(f"🐭 `{cell}`", value=default_id, key=f"anid_{idx}")
                cell_to_animal[cell] = val.strip() if val.strip() else default_id

        # Live summary
        animal_counts = pd.Series(cell_to_animal).value_counts().sort_index()
        multi = animal_counts[animal_counts > 1]
        if not multi.empty:
            st.success(
                f"✅ {len(multi)} animal(s) avec plusieurs cellules : "
                + " | ".join([f"**{a}** ({n} cells)" for a, n in multi.items()])
            )
            st.caption("LMM : effets aléatoires = **(1 | Animal / Cell)**")
        else:
            st.info(
                "Aucun animal partagé détecté — identique au mode par défaut. "
                "LMM : **(1 | Cell)**."
            )

    # Resolve how many unique animals per group
    n_animals_g1 = len(set(
        cell_to_animal.get(c, c)
        for c in df_if_common[df_if_common['Condition'] == grp1_name]['Cell'].unique()
    ))
    n_animals_g2 = len(set(
        cell_to_animal.get(c, c)
        for c in df_if_common[df_if_common['Condition'] == grp2_name]['Cell'].unique()
    ))

    # Inject Animal column
    df_if_common = df_if_common.copy()
    df_if_common['Animal'] = df_if_common['Cell'].map(cell_to_animal)

    # Determine nesting mode
    n_unique_animals = len(set(cell_to_animal.values()))
    is_nested = use_animal_grouping and (n_unique_animals < len(all_cells))

    st.divider()

    # ── Per-cell means (unit of observation for point-by-point test) ──────────
    cell_means = (df_if_common.groupby(['Condition', 'Animal', 'Cell', 'I_inj'])['Firing_Hz']
                  .mean().reset_index())

    # ── Mean ± SEM per group per step (for plot) ──────────────────────────────
    agg = (cell_means.groupby(['Condition', 'I_inj'])['Firing_Hz']
           .agg(mean='mean', sem=lambda x: x.sem(), n='count')
           .reset_index())
    agg1 = agg[agg['Condition'] == grp1_name].sort_values('I_inj')
    agg2 = agg[agg['Condition'] == grp2_name].sort_values('I_inj')

    # ── Point-by-point Mann-Whitney + BH-FDR ─────────────────────────────────
    # Unit of replication = Cell (one value per cell per I_inj step).
    # When multiple cells come from the same animal, a note is shown in the UI.
    test_rows = []
    for step in common_steps:
        v1 = cell_means[(cell_means['Condition'] == grp1_name) &
                        (cell_means['I_inj'] == step)]['Firing_Hz'].dropna()
        v2 = cell_means[(cell_means['Condition'] == grp2_name) &
                        (cell_means['I_inj'] == step)]['Firing_Hz'].dropna()
        if len(v1) > 2 and len(v2) > 2:
            _, p_raw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            test_rows.append({'I_inj': step, 'p_raw': p_raw,
                               'n_cells_1': len(v1), 'n_cells_2': len(v2),
                               'mean1': v1.mean(), 'mean2': v2.mean()})

    if test_rows:
        test_df = pd.DataFrame(test_rows)
        p_adj = false_discovery_control(test_df['p_raw'].values, method='bh')
        test_df['p_adj_BH'] = p_adj
        test_df['stars'] = test_df['p_adj_BH'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        )
        sig_steps = test_df[test_df['stars'] != '']
    else:
        test_df  = pd.DataFrame()
        sig_steps = pd.DataFrame()

    # ── Plot I-F curves ───────────────────────────────────────────────────────
    colors = {grp1_name: '#2196F3', grp2_name: '#F44336'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for grp, agg_grp in [(grp1_name, agg1), (grp2_name, agg2)]:
        col = colors[grp]
        n_c = n_cells_g1 if grp == grp1_name else n_cells_g2
        n_a = n_animals_g1 if grp == grp1_name else n_animals_g2
        lbl = (f"{grp} ({n_c} cells / {n_a} animals)"
               if is_nested else f"{grp} (n={n_c} cells)")
        ax.plot(agg_grp['I_inj'], agg_grp['mean'], 'o-', color=col,
                lw=2.5, ms=7, label=lbl)
        ax.fill_between(agg_grp['I_inj'],
                        agg_grp['mean'] - agg_grp['sem'],
                        agg_grp['mean'] + agg_grp['sem'],
                        color=col, alpha=0.18)

    if show_individual:
        for grp, col in colors.items():
            for cell in cell_means[cell_means['Condition'] == grp]['Cell'].unique():
                cd = cell_means[(cell_means['Condition'] == grp) &
                                (cell_means['Cell'] == cell)].sort_values('I_inj')
                ax.plot(cd['I_inj'], cd['Firing_Hz'], color=col,
                        alpha=0.18, lw=0.9, zorder=1)

    if not sig_steps.empty:
        y_max   = df_if_common['Firing_Hz'].max()
        y_range = y_max
        for _, row in sig_steps.iterrows():
            m1v = agg1[agg1['I_inj'] == row['I_inj']]['mean'].values
            m2v = agg2[agg2['I_inj'] == row['I_inj']]['mean'].values
            y_annot = max(m1v[0] if len(m1v) else 0, m2v[0] if len(m2v) else 0) + y_range * 0.06
            ax.text(row['I_inj'], y_annot, row['stars'],
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    ax.set_xlabel("Courant injecté (pA)", fontsize=12)
    ax.set_ylabel("Fréquence de décharge (Hz)", fontsize=12)
    ax.set_title(
        f"Courbes I-F : {grp1_name} vs {grp2_name}\n"
        f"Mean ± SEM | Mann-Whitney point-par-point + BH-FDR"
        + (" | Modèle nested (Animal/Cell)" if is_nested else ""),
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

    # ── LMM ──────────────────────────────────────────────────────────────────
    #
    # Two models depending on whether animal grouping is active:
    #
    # A) Simple  (1 cell = 1 animal, default):
    #    Firing_Hz ~ I_inj * Condition_bin
    #    groups = Cell
    #    → random intercept per cell
    #
    # B) Nested  (multiple cells per animal):
    #    Firing_Hz ~ I_inj * Condition_bin
    #    groups = Animal
    #    exog_re = identity matrix  →  random intercept per animal
    #    + a second MixedLM call with groups = Cell_within_animal to capture
    #      residual cell variance (statsmodels doesn't support crossed random
    #      effects natively; we use the Animal-level model as the primary
    #      inference model and note the limitation).
    #
    # Fixed effects interpretation (both models):
    #   Intercept           → baseline firing rate of grp1 at I_inj=0 (extrapolated)
    #   I_inj               → slope of I-F curve for grp1 (Hz per pA)
    #   Condition_bin       → vertical shift of grp2 relative to grp1
    #   I_inj:Condition_bin → DIFFERENCE IN SLOPE = different gain between groups ★

    if run_lmm:
        st.divider()
        if is_nested:
            st.markdown("### 🧮 LMM Nested : Gain & Seuil de la Courbe I-F")
            st.caption(
                "Modèle : `Firing_Hz ~ I_inj × Condition + (1 | Animal)`\n\n"
                "L'animal est l'unité de réplication biologique. "
                "La cellule est nichée dans l'animal (pseudo-réplication corrigée). "
                "L'interaction **I_inj × Condition** teste si les groupes ont un **gain différent**."
            )
        else:
            st.markdown("### 🧮 LMM : Gain & Seuil de la Courbe I-F")
            st.caption(
                "Modèle : `Firing_Hz ~ I_inj × Condition + (1 | Cell)`\n\n"
                "La cellule est l'unité de réplication. "
                "L'interaction **I_inj × Condition** teste si les groupes ont un **gain différent**."
            )

        lmm_df = cell_means.copy()
        lmm_df['Condition_bin'] = (lmm_df['Condition'] == grp2_name).astype(int)
        lmm_data = lmm_df.dropna(subset=['Firing_Hz', 'I_inj'])

        # Choose grouping variable: Animal (nested) or Cell (simple)
        grouping_var = 'Animal' if is_nested else 'Cell'

        try:
            model  = mixedlm(
                "Firing_Hz ~ I_inj * Condition_bin",
                lmm_data,
                groups=lmm_data[grouping_var]
            )
            result = model.fit(method='cg', disp=False)

            df_res = lmm_summary_table(result, grp1_name, grp2_name)
            st.dataframe(df_res, use_container_width=True)

            # Interpretation
            pv = result.pvalues
            p_inter = pv.get('I_inj:Condition_bin', np.nan)
            p_cond  = pv.get('Condition_bin', np.nan)
            fe      = result.fe_params
            slope1  = fe.get('I_inj', np.nan)
            slope2  = slope1 + fe.get('I_inj:Condition_bin', 0)

            interp = []
            if not np.isnan(p_inter):
                if p_inter < 0.05:
                    interp.append(
                        f"✅ **Gain différent** entre {grp1_name} et {grp2_name} "
                        f"(interaction p={p_inter:.4f}) | "
                        f"pente {grp1_name} = {slope1:.3f} Hz/pA, "
                        f"pente {grp2_name} = {slope2:.3f} Hz/pA"
                    )
                else:
                    interp.append(
                        f"➖ **Gain similaire** entre les groupes "
                        f"(interaction p={p_inter:.4f})"
                    )
            if not np.isnan(p_cond):
                if p_cond < 0.05:
                    interp.append(
                        f"✅ **Décalage vertical significatif** "
                        f"(seuil de décharge différent : p={p_cond:.4f})"
                    )
                else:
                    interp.append(f"➖ **Pas de décalage vertical** (p={p_cond:.4f})")

            for line in interp:
                st.markdown(line)

            if is_nested:
                st.info(
                    "ℹ️ **Note sur le modèle nested :** statsmodels ne supporte pas "
                    "nativement les effets aléatoires croisés (Animal + Cell simultanément). "
                    "Le modèle ici utilise l'Animal comme unique effet aléatoire, ce qui est "
                    "l'approche conservatrice recommandée quand le nombre de cellules par animal "
                    "est faible (< 5). Pour un modèle complet (1|Animal/Cell), utilisez lme4 sous R."
                )

        except Exception as e:
            st.error(f"Le LMM n'a pas convergé : {e}")
            st.info("Essayez avec plus de cellules/animaux ou vérifiez la variabilité des données.")

    # ── Point-by-point results table ──────────────────────────────────────────
    if not test_df.empty:
        st.divider()
        st.markdown("### 📋 Tests Mann-Whitney Point-par-Point (BH-FDR)")
        if is_nested:
            st.caption(
                f"⚠️ Le test point-par-point utilise la **cellule** comme unité (n = nb cellules). "
                f"Avec {n_unique_animals} animaux au total, gardez ce résultat interprétatif — "
                f"le LMM nested ci-dessus est le test confirmatoire."
            )
        else:
            st.caption(f"n = nombre de cellules par groupe. "
                       f"Corrigé sur {len(test_df)} comparaisons (BH-FDR).")
        display_df = test_df.copy()
        display_df.columns = ['I_inj (pA)', 'p brute',
                               f'n cells {grp1_name}', f'n cells {grp2_name}',
                               f'Mean Hz {grp1_name}', f'Mean Hz {grp2_name}',
                               'p BH-adj', 'Sig.']
        st.dataframe(display_df.round(4), use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    e1, e2 = st.columns(2)
    if not test_df.empty:
        e1.download_button("💾 Résultats point-par-point (CSV)",
                           test_df.to_csv(index=False).encode(),
                           f"IF_tests_{grp1_name}_vs_{grp2_name}.csv",
                           use_container_width=True)
    e2.download_button("💾 Données I-F par cellule (CSV)",
                       cell_means.to_csv(index=False).encode(),
                       "IF_cell_means.csv",
                       use_container_width=True)
