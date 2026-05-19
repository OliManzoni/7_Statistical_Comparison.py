import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# --- CONFIGURATION ---
st.set_page_config(page_title="Manzoni Lab - Statistical Engine", layout="wide")
st.markdown("# ⚖️ Moteur Statistique Avancé : Comparaison de Groupes")
st.markdown("### Standards de Publication : Détection d'Outliers, Tests de Normalité & Modèles Mixtes")
st.divider()

# --- FONCTION DE DÉTECTION DES OUTLIERS (TUKEY'S IQR) ---
def identify_outliers(df, column, group_col='Condition'):
    """Identifie les outliers par la méthode IQR (1.5) au sein de chaque groupe."""
    outlier_indices = []
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][column].dropna()
        if len(group_data) < 4:
            continue # Pas assez de données pour faire un IQR robuste
        
        Q1 = group_data.quantile(0.25)
        Q3 = group_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = group_data[(group_data < lower_bound) | (group_data > upper_bound)]
        outlier_indices.extend(outliers.index.tolist())
    
    return list(set(outlier_indices))

# --- CHARGEMENT DES DONNÉES ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Groupe 1 (ex: Mâles)")
    grp1_name = st.text_input("Nom du Groupe 1", "Mâles")
    grp1_files = st.file_uploader(f"Chargez les fichiers 'Propriétés intrinsèques' et 'Propriétés PA' des {grp1_name}", accept_multiple_files=True, key="g1")

with col2:
    st.subheader("Groupe 2 (ex: Femelles)")
    grp2_name = st.text_input("Nom du Groupe 2", "Femelles")
    grp2_files = st.file_uploader(f"Chargez les fichiers 'Propriétés intrinsèques' et 'Propriétés PA' des {grp2_name}", accept_multiple_files=True, key="g2")

if grp1_files and grp2_files:
    # --- PARSING DES FICHIERS ---
    def process_files(files, condition_name):
        df_list = []
        for f in files:
            # Sécurité pour sauter les en-têtes complexes générés par l'export Excel (Tableau 1, etc.)
            df = pd.read_csv(f, skiprows=lambda x: x < 2 if 'Tableau' in pd.read_csv(f, nrows=2).to_string() else 0)
            # Nettoyage des colonnes Unnamed
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df['Condition'] = condition_name
            df_list.append(df)
        
        if df_list:
            # Tente de fusionner les propriétés intrinsèques et PA sur la colonne 'File'
            master = df_list[0]
            for other_df in df_list[1:]:
                if 'File' in master.columns and 'File' in other_df.columns:
                    master = pd.merge(master, other_df, on=['File', 'Condition'], how='outer')
            return master
        return pd.DataFrame()

    df_g1 = process_files(grp1_files, grp1_name)
    df_g2 = process_files(grp2_files, grp2_name)
    
    df_master = pd.concat([df_g1, df_g2], ignore_index=True)
    
    # Identification des colonnes numériques (métriques)
    numeric_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
    # Retrait des colonnes non pertinentes
    numeric_cols = [c for c in numeric_cols if c not in ['Sweep', 'I_inj', 'index']]

    if not numeric_cols:
        st.error("Aucune métrique numérique détectée. Vérifiez le format de vos fichiers.")
        st.stop()

    st.divider()
    st.subheader("🕵️‍♂️ Analyse des Valeurs Aberrantes & Routage Statistique")
    
    # Options de l'utilisateur
    remove_outliers = st.checkbox("Exclure dynamiquement les Outliers (IQR) de l'analyse statistique", value=False)
    selected_metric = st.selectbox("Sélectionnez le paramètre biophysique à analyser :", numeric_cols)

    # Extraction des données pour le test
    data_clean = df_master.copy()
    outliers_idx = identify_outliers(data_clean, selected_metric, 'Condition')
    
    if remove_outliers and outliers_idx:
        data_to_test = data_clean.drop(index=outliers_idx)
        st.warning(f"🧹 {len(outliers_idx)} Outlier(s) retiré(s) du jeu de données pour le paramètre {selected_metric}.")
    else:
        data_to_test = data_clean

    group1_data = data_to_test[data_to_test['Condition'] == grp1_name][selected_metric].dropna()
    group2_data = data_to_test[data_to_test['Condition'] == grp2_name][selected_metric].dropna()

    # --- ROUTAGE STATISTIQUE RIGOUREUX ---
    st.markdown(f"### Paramètre : **{selected_metric}**")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    # 1. Test de Normalité (Shapiro-Wilk)
    stat_shapiro1, p_shap1 = stats.shapiro(group1_data)
    stat_shapiro2, p_shap2 = stats.shapiro(group2_data)
    is_normal = (p_shap1 > 0.05) and (p_shap2 > 0.05)
    
    stat_col1.metric(
        "Normalité (Shapiro-Wilk)", 
        "Validée" if is_normal else "Rejetée", 
        f"p={min(p_shap1, p_shap2):.4f}", 
        delta_color="off" if is_normal else "inverse"
    )

    # 2. Test d'Homoscédasticité (Levene)
    stat_levene, p_levene = stats.levene(group1_data, group2_data)
    is_homoscedastic = p_levene > 0.05
    
    stat_col2.metric(
        "Égalité des Variances (Levene)", 
        "Validée" if is_homoscedastic else "Rejetée", 
        f"p={p_levene:.4f}",
        delta_color="off" if is_homoscedastic else "inverse"
    )

    # 3. Choix du Test & p-value
    if is_normal and is_homoscedastic:
        test_name = "T-test (Student)"
        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=True)
    elif is_normal and not is_homoscedastic:
        test_name = "T-test (Welch)"
        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    else:
        test_name = "Mann-Whitney U"
        t_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')

    significance = "Significatif (p<0.05) ⭐" if p_val < 0.05 else "Non Significatif (ns)"
    stat_col3.metric(f"Test Appliqué : {test_name}", significance, f"p = {p_val:.4f}", delta_color="normal" if p_val < 0.05 else "off")

    # --- VISUALISATION (VIOLIN + SWARM) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Trace un Violin plot pour la distribution globale
    sns.violinplot(data=df_master, x='Condition', y=selected_metric, inner=None, color=".9", ax=ax, linewidth=0)
    
    # Définition des couleurs pour identifier les outliers sur le graphique
    df_master['Is_Outlier'] = df_master.index.isin(outliers_idx)
    palette = {False: "tab:blue", True: "crimson"}
    
    # Trace les points individuels
    sns.stripplot(data=df_master, x='Condition', y=selected_metric, hue='Is_Outlier', palette=palette, size=7, alpha=0.8, jitter=True, ax=ax)
    
    ax.set_title(f"Distribution & Outliers : {selected_metric}\n({test_name} : p={p_val:.4f})", fontweight='bold')
    ax.set_ylabel(selected_metric)
    
    # Légende personnalisée
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[handles[0], handles[1]], labels=['Data validée', 'Outlier (IQR)'], loc='upper right')
    
    sns.despine()
    st.pyplot(fig)
    
    # --- TABLEAU DÉTAILLÉ DES OUTLIERS ---
    if outliers_idx:
        st.markdown("#### 🚨 Détail des Outliers Détectés")
        outlier_df = df_master.loc[outliers_idx, ['File', 'Condition', selected_metric]]
        st.dataframe(outlier_df, use_container_width=True)