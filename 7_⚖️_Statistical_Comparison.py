import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Manzoni Lab - Statistical Engine", layout="wide")
st.markdown("# ⚖️ Moteur Statistique Avancé : Comparaison de Groupes")
st.markdown("### Standards de Publication : Détection d'Outliers, Tests de Normalité & Sélection Automatique du Test")
st.divider()

# ── HELPERS ──────────────────────────────────────────────────────────────────

def identify_outliers(df, column, group_col='Condition'):
    """
    Détecte les outliers par la méthode IQR de Tukey (seuil 1.5×IQR) au sein
    de chaque groupe. Retourne une liste de labels d'index (issus du DataFrame
    passé en argument).
    Nécessite au moins 4 observations par groupe pour être robuste.
    """
    outlier_indices = []
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][column].dropna()
        if len(group_data) < 4:
            continue
        Q1, Q3 = group_data.quantile(0.25), group_data.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_indices.extend(
            group_data[(group_data < lo) | (group_data > hi)].index.tolist()
        )
    return list(set(outlier_indices))


def cohen_d(a, b):
    """
    Cohen's d avec correction de pooled SD (Hedges' pooling).
    Signe positif = groupe A > groupe B.
    """
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def rank_biserial_r(u_stat, n1, n2):
    """
    Corrélation rank-bisérale r à partir de la statistique U de Mann-Whitney.
    r = 1 - 2U / (n1*n2). Varie de -1 à +1.
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def effect_size_label(val):
    """Interprétation de Cohen (1988) : petit < 0.2 ≤ moyen < 0.5 ≤ grand."""
    av = abs(val)
    if av < 0.2:   return "négligeable"
    if av < 0.5:   return "petit"
    if av < 0.8:   return "moyen"
    return "grand"


# ── FIX 1: process_files ─────────────────────────────────────────────────────
# ORIGINAL BUG: pd.read_csv was called TWICE on the same UploadedFile buffer.
# The inner call (to detect 'Tableau' headers) consumed the stream pointer,
# leaving the outer call reading from EOF → empty DataFrame.
# Additionally, the skiprows lambda re-called read_csv thousands of times
# (once per row), which was O(n²) in file size.
#
# FIX: read the file content into bytes once with f.read() / f.getvalue(),
# then use io.BytesIO() to get independent readable buffers for each call.
# Header detection is now a single cheap check on the first 2 rows.

def process_files(files, condition_name):
    """
    Charge et fusionne les fichiers CSV/exports d'une condition.
    Gère les en-têtes complexes Excel (lignes 'Tableau …') en les sautant.
    Fusionne plusieurs fichiers sur la colonne 'File' si elle est présente.
    """
    df_list = []
    for f in files:
        # Read bytes once — gives us reusable content regardless of Streamlit's
        # UploadedFile stream state after previous reads.
        raw = f.getvalue()

        # Peek at first 2 rows to detect Excel-style Tableau header
        peek = pd.read_csv(io.BytesIO(raw), nrows=2)
        has_tableau_header = 'Tableau' in peek.to_string()
        skip = 2 if has_tableau_header else 0

        df = pd.read_csv(io.BytesIO(raw), skiprows=range(skip))
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['Condition'] = condition_name
        df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    master = df_list[0]
    for other_df in df_list[1:]:
        if 'File' in master.columns and 'File' in other_df.columns:
            master = pd.merge(master, other_df, on=['File', 'Condition'], how='outer')
        else:
            # No common key: stack rows instead of joining columns
            master = pd.concat([master, other_df], ignore_index=True)

    return master


# ── UI: FILE UPLOAD ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Groupe 1")
    grp1_name  = st.text_input("Nom du Groupe 1", "Mâles")
    grp1_files = st.file_uploader(
        f"Fichiers CSV pour {grp1_name}", accept_multiple_files=True, key="g1"
    )

with col2:
    st.subheader("Groupe 2")
    grp2_name  = st.text_input("Nom du Groupe 2", "Femelles")
    grp2_files = st.file_uploader(
        f"Fichiers CSV pour {grp2_name}", accept_multiple_files=True, key="g2"
    )

if grp1_files and grp2_files:

    df_g1 = process_files(grp1_files, grp1_name)
    df_g2 = process_files(grp2_files, grp2_name)
    df_master = pd.concat([df_g1, df_g2], ignore_index=True)

    numeric_cols = [
        c for c in df_master.select_dtypes(include=[np.number]).columns
        if c not in ['Sweep', 'I_inj', 'index']
    ]

    if not numeric_cols:
        st.error("Aucune métrique numérique détectée. Vérifiez le format de vos fichiers.")
        st.stop()

    st.divider()
    st.subheader("🕵️ Analyse Statistique Rigoureuse")

    remove_outliers = st.checkbox(
        "Exclure les Outliers (IQR×1.5) de l'analyse statistique",
        value=False,
        help="Les outliers restent visibles sur le graphique (points rouges) mais sont retirés du test."
    )
    selected_metric = st.selectbox("Paramètre biophysique à analyser :", numeric_cols)

    # Outlier detection on the full dataset
    data_clean    = df_master.copy()
    outliers_idx  = identify_outliers(data_clean, selected_metric, 'Condition')

    if remove_outliers and outliers_idx:
        data_to_test = data_clean.drop(index=outliers_idx)
        st.warning(
            f"🧹 **{len(outliers_idx)} outlier(s)** retirés du test statistique "
            f"pour **{selected_metric}**. Ils restent affichés en rouge sur le graphique."
        )
    else:
        data_to_test = data_clean

    g1 = data_to_test[data_to_test['Condition'] == grp1_name][selected_metric].dropna()
    g2 = data_to_test[data_to_test['Condition'] == grp2_name][selected_metric].dropna()

    n1, n2 = len(g1), len(g2)

    # ── FIX 2: Shapiro-Wilk guard ────────────────────────────────────────────
    # ORIGINAL BUG: stats.shapiro() crashes (or gives nonsensical results) when
    # n < 3. No guard existed. Added explicit n >= 3 check; if too few points,
    # normality is treated as unknown → non-parametric routing for safety.
    #
    # Also: Shapiro-Wilk loses power for n > 50 (nearly always rejects).
    # For n > 50 we automatically fall back to Mann-Whitney for robustness
    # and flag this to the user.

    SHAPIRO_MIN = 3
    SHAPIRO_MAX = 50   # above this, S-W is too sensitive to trivial deviations

    def run_shapiro(data, label):
        n = len(data)
        if n < SHAPIRO_MIN:
            st.warning(f"⚠️ {label} : n={n} < {SHAPIRO_MIN} → normalité supposée inconnue.")
            return np.nan, 0.0   # force non-parametric
        if n > SHAPIRO_MAX:
            st.info(
                f"ℹ️ {label} : n={n} > {SHAPIRO_MAX}. Shapiro-Wilk perd en puissance "
                f"à grand n → routage non-paramétrique automatique."
            )
            return np.nan, 0.0   # force non-parametric
        return stats.shapiro(data)

    _, p_shap1 = run_shapiro(g1, grp1_name)
    _, p_shap2 = run_shapiro(g2, grp2_name)
    is_normal = (p_shap1 > 0.05) and (p_shap2 > 0.05)

    stat_levene, p_levene = stats.levene(g1, g2) if (n1 >= 2 and n2 >= 2) else (np.nan, np.nan)
    is_homoscedastic = (p_levene > 0.05) if not np.isnan(p_levene) else True

    # ── Statistical routing ───────────────────────────────────────────────────
    if is_normal and is_homoscedastic:
        test_name = "T-test (Student)"
        stat_val, p_val = stats.ttest_ind(g1, g2, equal_var=True)
        d = cohen_d(g1, g2)
        es_label = f"Cohen d = {d:.3f} ({effect_size_label(d)})"
    elif is_normal and not is_homoscedastic:
        test_name = "T-test (Welch)"
        stat_val, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        d = cohen_d(g1, g2)
        es_label = f"Cohen d = {d:.3f} ({effect_size_label(d)})"
    else:
        test_name = "Mann-Whitney U"
        stat_val, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        r = rank_biserial_r(stat_val, n1, n2)
        es_label = f"r = {r:.3f} ({effect_size_label(r)})"

    significance = "Significatif ⭐" if p_val < 0.05 else "Non Significatif (ns)"

    # ── Dashboard metrics ─────────────────────────────────────────────────────
    st.markdown(f"### Paramètre : **{selected_metric}**")
    m1, m2, m3, m4 = st.columns(4)

    shap_display = "Validée ✅" if is_normal else "Rejetée ❌"
    m1.metric(
        "Normalité (Shapiro-Wilk)",
        shap_display,
        f"p₁={p_shap1:.4f} | p₂={p_shap2:.4f}" if not np.isnan(p_shap1) else "n trop petit/grand",
        delta_color="off"
    )

    lev_display = "Validée ✅" if is_homoscedastic else "Rejetée ❌"
    m2.metric(
        "Variances égales (Levene)",
        lev_display,
        f"p={p_levene:.4f}" if not np.isnan(p_levene) else "N/A",
        delta_color="off"
    )

    m3.metric(
        f"Test : {test_name}",
        significance,
        f"p = {p_val:.4f}",
        delta_color="normal" if p_val < 0.05 else "off"
    )

    # FIX 3: Effect size — was completely absent in original.
    # Cohen's d for parametric tests, rank-biserial r for Mann-Whitney.
    # Both are mandatory for publication (APA, Nature, etc.).
    m4.metric("Taille d'Effet", es_label.split(" (")[0], es_label.split(" (")[1].rstrip(")"), delta_color="off")

    # Sample sizes
    st.caption(
        f"n₁ ({grp1_name}) = **{n1}** | n₂ ({grp2_name}) = **{n2}**"
        + (f" | {len(outliers_idx)} outlier(s) détecté(s)" if outliers_idx else "")
    )

    # ── Descriptive table ─────────────────────────────────────────────────────
    with st.expander("📋 Statistiques Descriptives", expanded=False):
        desc = pd.DataFrame({
            "Groupe":   [grp1_name, grp2_name],
            "n":        [n1, n2],
            "Moyenne":  [g1.mean(), g2.mean()],
            "Médiane":  [g1.median(), g2.median()],
            "SD":       [g1.std(ddof=1), g2.std(ddof=1)],
            "SEM":      [g1.sem(), g2.sem()],
            "Min":      [g1.min(), g2.min()],
            "Max":      [g1.max(), g2.max()],
        }).set_index("Groupe").round(4)
        st.dataframe(desc, use_container_width=True)

    # ── Visualisation ─────────────────────────────────────────────────────────
    # The violin always uses df_master (all data) so the full distribution shape
    # is visible regardless of the outlier-exclusion checkbox.
    # Outliers are coloured red so the reader can judge their influence.
    df_master['Is_Outlier'] = df_master.index.isin(outliers_idx)
    palette = {False: "tab:blue", True: "crimson"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin + swarm
    ax = axes[0]
    sns.violinplot(
        data=df_master, x='Condition', y=selected_metric,
        inner=None, color=".92", linewidth=0.8, ax=ax
    )
    sns.stripplot(
        data=df_master, x='Condition', y=selected_metric,
        hue='Is_Outlier', palette=palette, size=7, alpha=0.85,
        jitter=True, ax=ax
    )
    ax.set_title(
        f"{selected_metric}\n{test_name}: p={p_val:.4f} | {es_label}",
        fontweight='bold', fontsize=11
    )
    ax.set_ylabel(selected_metric)
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(handles[:2], ['Donnée valide', 'Outlier (IQR)'], loc='upper right', fontsize=9)
    elif len(handles) == 1:
        ax.legend(handles[:1], ['Donnée valide'], loc='upper right', fontsize=9)
    else:
        ax.get_legend().remove() if ax.get_legend() else None

    # Right: box + swarm (complementary view)
    ax2 = axes[1]
    sns.boxplot(
        data=df_master, x='Condition', y=selected_metric,
        width=0.4, color=".92", linewidth=1.2, fliersize=0, ax=ax2
    )
    sns.stripplot(
        data=df_master, x='Condition', y=selected_metric,
        hue='Is_Outlier', palette=palette, size=7, alpha=0.85,
        jitter=True, ax=ax2
    )
    ax2.set_title(
        f"Vue Boîte à Moustaches\nn₁={n1} | n₂={n2}",
        fontweight='bold', fontsize=11
    )
    ax2.set_ylabel("")
    if ax2.get_legend():
        ax2.get_legend().remove()

    # Significance bracket
    y_max = df_master[selected_metric].dropna().max()
    y_range = df_master[selected_metric].dropna().max() - df_master[selected_metric].dropna().min()
    bracket_h = y_max + y_range * 0.08
    sig_text = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    ax2.plot([0, 0, 1, 1], [bracket_h, bracket_h + y_range*0.02, bracket_h + y_range*0.02, bracket_h],
             lw=1.2, color='black')
    ax2.text(0.5, bracket_h + y_range*0.03, sig_text, ha='center', va='bottom', fontsize=13, fontweight='bold')

    sns.despine()
    plt.tight_layout()
    st.pyplot(fig)

    # ── Outlier detail table ──────────────────────────────────────────────────
    # FIX 4: 'File' column is not guaranteed to exist.
    # ORIGINAL BUG: df_master.loc[outliers_idx, ['File', 'Condition', metric]]
    # raises KeyError if the uploaded files have no 'File' column.
    # FIX: build the column list dynamically.
    if outliers_idx:
        st.markdown("#### 🚨 Détail des Outliers Détectés")
        id_cols = [c for c in ['File', 'Condition', selected_metric] if c in df_master.columns]
        if selected_metric not in id_cols:
            id_cols.append(selected_metric)
        st.dataframe(df_master.loc[outliers_idx, id_cols], use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Exportation")

    summary = pd.DataFrame({
        "Métrique":           [selected_metric],
        "Groupe_1":           [grp1_name],
        "n_1":                [n1],
        "Moyenne_1":          [g1.mean()],
        "SD_1":               [g1.std(ddof=1)],
        "Groupe_2":           [grp2_name],
        "n_2":                [n2],
        "Moyenne_2":          [g2.mean()],
        "SD_2":               [g2.std(ddof=1)],
        "Test":               [test_name],
        "Statistique":        [stat_val],
        "p_value":            [p_val],
        "Significatif_0.05":  [p_val < 0.05],
        "Taille_Effet":       [es_label],
        "Normalité_validée":  [is_normal],
        "Variances_égales":   [is_homoscedastic],
        "N_outliers_détectés":[len(outliers_idx)],
        "Outliers_exclus":    [remove_outliers and len(outliers_idx) > 0],
    })

    col_exp1, col_exp2 = st.columns(2)
    col_exp1.download_button(
        "💾 Exporter le Résumé Statistique (CSV)",
        summary.to_csv(index=False).encode('utf-8'),
        f"stats_{selected_metric}.csv",
        use_container_width=True
    )
    col_exp2.download_button(
        "💾 Exporter les Données Brutes (CSV)",
        df_master.to_csv(index=False).encode('utf-8'),
        "donnees_brutes.csv",
        use_container_width=True
    )
