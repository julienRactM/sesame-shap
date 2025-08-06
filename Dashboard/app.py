"""
Dashboard interactif pour l'analyse d'interprétabilité COMPAS

Ce dashboard Streamlit permet d'explorer les résultats SHAP, LIME et SAGE
pour l'analyse des biais dans les modèles COMPAS.

Auteur: Projet SESAME-SHAP
Date: 2025

Usage: streamlit run Dashboard/app.py
"""

import os
import sys
import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ajouter le répertoire parent au path pour importer les modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

try:
    from src.data_acquisition import CompasDataAcquisition
    from src.exploratory_analysis import CompasEDA
    from src.feature_engineering import COMPASFeatureEngineer
    from src.model_training import CompasModelTrainer
    from src.shap_analysis import CompasShapAnalyzer
    from src.bias_analysis import CompasBiasAnalyzer
    from src.bias_mitigation import BiasMitigationFramework
    from src.fairness_evaluation import FairnessEvaluator
    from src.interpretability_comparison import InterpretabilityComparator
except ImportError as e:
    st.error(f"Erreur d'importation des modules: {e}")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="COMPAS SHAP Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bias-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Charge des données d'exemple pour la démonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    # Simuler des données COMPAS
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], 
                                n_samples, p=[0.45, 0.40, 0.10, 0.05]),
        'priors_count': np.random.poisson(2, n_samples),
        'c_charge_degree': np.random.choice(['F', 'M'], n_samples, p=[0.6, 0.4]),
        'compas_score': np.random.randint(1, 11, n_samples),
        'two_year_recid': np.random.binomial(1, 0.4, n_samples)
    }
    
    # Introduire un biais racial
    african_american_mask = data['race'] == 'African-American'
    data['compas_score'][african_american_mask] += np.random.randint(0, 3, np.sum(african_american_mask))
    data['compas_score'] = np.clip(data['compas_score'], 1, 10)
    
    df = pd.DataFrame(data)
    return df

def create_bias_metrics_demo():
    """Crée des métriques de biais d'exemple."""
    return {
        'demographic_parity_difference': 0.18,
        'equal_opportunity_difference': 0.15,
        'disparate_impact_ratio': 1.35,
        'passes_80_rule': False,
        'chi2_pvalue': 0.001,
        'mannwhitney_pvalue': 0.0005
    }

def create_shap_values_demo(n_samples=100, n_features=8):
    """Crée des valeurs SHAP d'exemple."""
    np.random.seed(42)
    
    feature_names = ['age', 'priors_count', 'compas_score', 'sex_Male', 
                    'race_African-American', 'race_Caucasian', 'c_charge_degree_F', 'age_group']
    
    # Simuler des valeurs SHAP avec biais
    shap_values = np.random.normal(0, 0.3, (n_samples, n_features))
    
    # Introduire un biais pour race_African-American
    shap_values[:, 4] += np.random.normal(0.2, 0.1, n_samples)  # Biais positif
    
    return {
        'values': shap_values,
        'feature_names': feature_names,
        'expected_value': 0.4
    }

def main():
    """Fonction principale du dashboard."""
    
    # En-tête principal
    st.markdown('<div class="main-header">⚖️ COMPAS SHAP Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Analyse d\'Interprétabilité et de Biais pour les Modèles COMPAS</div>', unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une section",
        [
            "🏠 Accueil",
            "📊 Analyse Exploratoire", 
            "🤖 Modèles et Performance",
            "🔍 Analyse SHAP",
            "⚖️ Détection des Biais",
            "🛡️ Mitigation des Biais",
            "📈 Évaluation d'Équité",
            "🔄 Comparaison d'Interprétabilité"
        ]
    )
    
    # Charger les données d'exemple
    if 'demo_data' not in st.session_state:
        with st.spinner("Chargement des données d'exemple..."):
            st.session_state.demo_data = load_sample_data()
            st.session_state.bias_metrics = create_bias_metrics_demo()
            st.session_state.shap_values = create_shap_values_demo()
    
    # Routage des pages
    if page == "🏠 Accueil":
        show_home_page()
    elif page == "📊 Analyse Exploratoire":
        show_eda_page()
    elif page == "🤖 Modèles et Performance":
        show_models_page()
    elif page == "🔍 Analyse SHAP":
        show_shap_page()
    elif page == "⚖️ Détection des Biais":
        show_bias_detection_page()
    elif page == "🛡️ Mitigation des Biais":
        show_bias_mitigation_page()
    elif page == "📈 Évaluation d'Équité":
        show_fairness_evaluation_page()
    elif page == "🔄 Comparaison d'Interprétabilité":
        show_interpretability_comparison_page()

def show_home_page():
    """Page d'accueil du dashboard."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Bienvenue sur le Dashboard COMPAS SHAP
        
        Ce dashboard interactif vous permet d'explorer l'interprétabilité et l'équité 
        des modèles de prédiction de récidive COMPAS.
        
        ### 🎯 Objectifs du Projet
        - **Détecter** les biais raciaux dans les prédictions COMPAS
        - **Analyser** l'interprétabilité avec SHAP, LIME et SAGE
        - **Mitiger** les biais identifiés
        - **Évaluer** l'efficacité des stratégies de mitigation
        
        ### 📋 Sections Disponibles
        """)
    
    # Cartes de navigation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Analyse Exploratoire</h4>
        <p>Exploration des données COMPAS avec focus sur les biais démographiques</p>
        </div>
        
        <div class="metric-card">
        <h4>🤖 Modèles ML</h4>
        <p>Entraînement et évaluation de modèles de classification</p>
        </div>
        
        <div class="metric-card">
        <h4>🔍 Analyse SHAP</h4>
        <p>Interprétabilité des modèles avec les valeurs de Shapley</p>
        </div>
        
        <div class="metric-card">
        <h4>⚖️ Détection des Biais</h4>
        <p>Métriques d'équité et identification des disparités</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🛡️ Mitigation des Biais</h4>
        <p>Stratégies pour réduire les biais détectés</p>
        </div>
        
        <div class="metric-card">
        <h4>📈 Évaluation d'Équité</h4>
        <p>Comparaison avant/après mitigation</p>
        </div>
        
        <div class="metric-card">
        <h4>🔄 Comparaison Interprétabilité</h4>
        <p>SHAP vs LIME vs SAGE (BONUS)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques globales
    st.markdown('<div class="section-header">📈 Vue d\'Ensemble du Dataset</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    df = st.session_state.demo_data
    
    with col1:
        st.metric("Échantillons Total", f"{len(df):,}")
        
    with col2:
        recidivism_rate = df['two_year_recid'].mean()
        st.metric("Taux de Récidive", f"{recidivism_rate:.1%}")
        
    with col3:
        african_american_pct = (df['race'] == 'African-American').mean()
        st.metric("% African-American", f"{african_american_pct:.1%}")
        
    with col4:
        avg_compas_score = df['compas_score'].mean()
        st.metric("Score COMPAS Moyen", f"{avg_compas_score:.1f}")

def show_eda_page():
    """Page d'analyse exploratoire."""
    
    st.markdown('<div class="section-header">📊 Analyse Exploratoire des Données</div>', unsafe_allow_html=True)
    
    df = st.session_state.demo_data
    
    # Statistiques descriptives
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des Variables")
        
        # Distribution par race
        race_dist = df['race'].value_counts()
        fig_race = px.pie(values=race_dist.values, names=race_dist.index, 
                         title="Distribution Raciale")
        st.plotly_chart(fig_race, use_container_width=True)
        
        # Distribution des scores COMPAS
        fig_compas = px.histogram(df, x='compas_score', title="Distribution des Scores COMPAS",
                                 nbins=10, color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_compas, use_container_width=True)
    
    with col2:
        st.subheader("Analyse des Biais")
        
        # Scores COMPAS par race
        fig_bias = px.box(df, x='race', y='compas_score', 
                         title="Scores COMPAS par Groupe Racial")
        fig_bias.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bias, use_container_width=True)
        
        # Taux de récidive par race
        recid_by_race = df.groupby('race')['two_year_recid'].agg(['mean', 'count']).reset_index()
        fig_recid = px.bar(recid_by_race, x='race', y='mean', 
                          title="Taux de Récidive par Groupe Racial",
                          text='count')
        fig_recid.update_traces(texttemplate='n=%{text}', textposition='outside')
        fig_recid.update_xaxes(tickangle=45)
        st.plotly_chart(fig_recid, use_container_width=True)
    
    # Matrice de corrélation
    st.subheader("Corrélations entre Variables")
    
    # Encoder les variables catégorielles pour la corrélation
    df_corr = df.copy()
    df_corr['sex_encoded'] = (df_corr['sex'] == 'Male').astype(int)
    df_corr['race_aa'] = (df_corr['race'] == 'African-American').astype(int)
    df_corr['charge_felony'] = (df_corr['c_charge_degree'] == 'F').astype(int)
    
    corr_cols = ['age', 'priors_count', 'compas_score', 'two_year_recid', 
                'sex_encoded', 'race_aa', 'charge_felony']
    corr_matrix = df_corr[corr_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Matrice de Corrélation",
                        color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Alerte sur les biais détectés
    st.markdown("""
    <div class="bias-alert">
    <strong>⚠️ Biais Potentiels Détectés:</strong><br>
    • Scores COMPAS plus élevés pour les défendeurs African-American<br>
    • Corrélation significative entre race et scores de risque<br>
    • Disparités dans les taux de récidive prédits vs réels
    </div>
    """, unsafe_allow_html=True)

def show_models_page():
    """Page des modèles et performance."""
    
    st.markdown('<div class="section-header">🤖 Modèles et Performance</div>', unsafe_allow_html=True)
    
    # Simuler des résultats de modèles
    model_results = {
        'RandomForest': {'accuracy': 0.72, 'precision': 0.68, 'recall': 0.75, 'f1': 0.71, 'auc': 0.78},
        'LogisticRegression': {'accuracy': 0.69, 'precision': 0.65, 'recall': 0.72, 'f1': 0.68, 'auc': 0.74},
        'XGBoost': {'accuracy': 0.74, 'precision': 0.70, 'recall': 0.77, 'f1': 0.73, 'auc': 0.80},
        'SVM': {'accuracy': 0.67, 'precision': 0.63, 'recall': 0.69, 'f1': 0.66, 'auc': 0.71}
    }
    
    # Tableau de comparaison des modèles
    st.subheader("Comparaison des Performances")
    
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.round(3)
    
    st.dataframe(results_df, use_container_width=True)
    
    # Graphiques de performance
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en barres des métriques
        fig_metrics = go.Figure()
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            fig_metrics.add_trace(go.Bar(
                name=metric.title(),
                x=list(model_results.keys()),
                y=[model_results[model][metric] for model in model_results.keys()]
            ))
        
        fig_metrics.update_layout(
            title="Métriques de Performance par Modèle",
            xaxis_title="Modèles",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Graphique radar de comparaison
        fig_radar = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for model_name, results in model_results.items():
            fig_radar.add_trace(go.Scatterpolar(
                r=[results[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Comparaison Radar des Modèles"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Recommandations
    st.subheader("Recommandations")
    
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['f1'])
    
    st.markdown(f"""
    <div class="success-alert">
    <strong>✅ Meilleur Modèle Identifié:</strong> {best_model}<br>
    • F1-Score: {model_results[best_model]['f1']:.3f}<br>
    • AUC: {model_results[best_model]['auc']:.3f}<br>
    • Recommandé pour l'analyse SHAP approfondie
    </div>
    """, unsafe_allow_html=True)

def show_shap_page():
    """Page d'analyse SHAP."""
    
    st.markdown('<div class="section-header">🔍 Analyse SHAP - Interprétabilité</div>', unsafe_allow_html=True)
    
    shap_data = st.session_state.shap_values
    
    # Sélecteur de modèle
    model_selected = st.selectbox("Sélectionner un modèle", 
                                 ['RandomForest', 'XGBoost', 'LogisticRegression'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Importance Globale des Features")
        
        # Importance moyenne des features
        feature_importance = np.abs(shap_data['values']).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': shap_data['feature_names'],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h', title="Importance SHAP des Features")
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("Distribution des Valeurs SHAP")
        
        # Distribution des valeurs SHAP pour la feature la plus importante
        top_feature_idx = np.argmax(feature_importance)
        top_feature_name = shap_data['feature_names'][top_feature_idx]
        top_feature_values = shap_data['values'][:, top_feature_idx]
        
        fig_dist = px.histogram(x=top_feature_values, nbins=30,
                               title=f"Distribution SHAP - {top_feature_name}")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Analyse des biais dans les valeurs SHAP
    st.subheader("Analyse des Biais via SHAP")
    
    # Simuler une analyse de biais par race
    np.random.seed(42)
    n_samples = len(shap_data['values'])
    races = np.random.choice(['African-American', 'Caucasian'], n_samples, p=[0.5, 0.5])
    
    # Créer un DataFrame pour l'analyse
    bias_df = pd.DataFrame({
        'race': races,
        'race_shap': shap_data['values'][:, 4],  # Feature race_African-American
        'compas_score_shap': shap_data['values'][:, 2]  # Feature compas_score
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot des valeurs SHAP par race
        fig_bias_box = px.box(bias_df, x='race', y='race_shap',
                             title="Valeurs SHAP 'Race' par Groupe")
        st.plotly_chart(fig_bias_box, use_container_width=True)
    
    with col2:
        # Scatter plot SHAP values
        fig_scatter = px.scatter(bias_df, x='compas_score_shap', y='race_shap',
                                color='race', title="Corrélation SHAP: Score vs Race")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Métriques de biais SHAP
    st.subheader("Métriques de Biais SHAP")
    
    aa_mask = bias_df['race'] == 'African-American'
    caucasian_mask = bias_df['race'] == 'Caucasian'
    
    aa_mean_shap = bias_df.loc[aa_mask, 'race_shap'].mean()
    caucasian_mean_shap = bias_df.loc[caucasian_mask, 'race_shap'].mean()
    shap_bias_diff = aa_mean_shap - caucasian_mean_shap
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SHAP Moyen - African-American", f"{aa_mean_shap:.4f}")
    
    with col2:
        st.metric("SHAP Moyen - Caucasian", f"{caucasian_mean_shap:.4f}")
    
    with col3:
        st.metric("Différence SHAP", f"{shap_bias_diff:.4f}")
    
    if abs(shap_bias_diff) > 0.1:
        st.markdown("""
        <div class="bias-alert">
        <strong>⚠️ Biais Détecté dans les Explications SHAP:</strong><br>
        Différence significative dans les contributions SHAP entre groupes raciaux
        </div>
        """, unsafe_allow_html=True)

def show_bias_detection_page():
    """Page de détection des biais."""
    
    st.markdown('<div class="section-header">⚖️ Détection des Biais</div>', unsafe_allow_html=True)
    
    bias_metrics = st.session_state.bias_metrics
    
    # Métriques d'équité principales
    st.subheader("Métriques d'Équité")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dp_diff = bias_metrics['demographic_parity_difference']
        st.metric("Parité Démographique", f"{dp_diff:.3f}")
        if abs(dp_diff) > 0.1:
            st.error("Biais détecté!")
        else:
            st.success("Acceptable")
    
    with col2:
        eo_diff = bias_metrics['equal_opportunity_difference']
        st.metric("Égalité des Chances", f"{eo_diff:.3f}")
        if abs(eo_diff) > 0.1:
            st.error("Biais détecté!")
        else:
            st.success("Acceptable")
    
    with col3:
        di_ratio = bias_metrics['disparate_impact_ratio']
        st.metric("Impact Disparate", f"{di_ratio:.3f}")
        if di_ratio < 0.8 or di_ratio > 1.25:
            st.error("Biais détecté!")
        else:
            st.success("Acceptable")
    
    with col4:
        passes_80 = bias_metrics['passes_80_rule']
        st.metric("Règle des 80%", "✅" if passes_80 else "❌")
        if not passes_80:
            st.error("Échec règle 80%")
        else:
            st.success("Conforme")
    
    # Graphique des métriques
    st.subheader("Visualisation des Biais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en barres des métriques de biais
        metrics_names = ['Parité Démographique', 'Égalité des Chances']
        metrics_values = [bias_metrics['demographic_parity_difference'], 
                         bias_metrics['equal_opportunity_difference']]
        
        fig_metrics = px.bar(x=metrics_names, y=metrics_values,
                            title="Métriques de Biais (Différences)",
                            color=metrics_values,
                            color_continuous_scale='RdYlBu_r')
        fig_metrics.add_hline(y=0.1, line_dash="dash", line_color="red", 
                             annotation_text="Seuil d'alerte")
        fig_metrics.add_hline(y=-0.1, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # Simuler une matrice de confusion par groupe
        conf_matrix_data = {
            'African-American': {'TP': 180, 'FP': 120, 'TN': 150, 'FN': 50},
            'Caucasian': {'TP': 140, 'FP': 80, 'TN': 180, 'FN': 40}
        }
        
        # Calculer les taux
        rates_data = []
        for group, matrix in conf_matrix_data.items():
            fpr = matrix['FP'] / (matrix['FP'] + matrix['TN'])
            fnr = matrix['FN'] / (matrix['FN'] + matrix['TP'])
            rates_data.append({'Groupe': group, 'Taux FP': fpr, 'Taux FN': fnr})
        
        rates_df = pd.DataFrame(rates_data)
        
        fig_rates = px.bar(rates_df, x='Groupe', y=['Taux FP', 'Taux FN'],
                          title="Taux d'Erreur par Groupe",
                          barmode='group')
        st.plotly_chart(fig_rates, use_container_width=True)
    
    # Tests statistiques
    st.subheader("Significativité Statistique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chi2_p = bias_metrics['chi2_pvalue']
        st.metric("Test Chi²", f"p = {chi2_p:.4f}")
        if chi2_p < 0.05:
            st.error("Dépendance significative")
        else:
            st.success("Indépendant")
    
    with col2:
        mw_p = bias_metrics['mannwhitney_pvalue']
        st.metric("Test Mann-Whitney", f"p = {mw_p:.4f}")
        if mw_p < 0.05:
            st.error("Différence significative")
        else:
            st.success("Pas de différence")
    
    # Recommandations
    st.subheader("Recommandations")
    
    if not passes_80 or abs(dp_diff) > 0.1 or abs(eo_diff) > 0.1:
        st.markdown("""
        <div class="bias-alert">
        <strong>🚨 Actions Recommandées:</strong><br>
        • Appliquer des techniques de mitigation des biais<br>
        • Réévaluer les features utilisées<br>
        • Considérer un rééchantillonnage des données<br>
        • Ajuster les seuils de décision par groupe
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-alert">
        <strong>✅ Modèle Acceptable:</strong><br>
        Les métriques d'équité sont dans les limites acceptables
        </div>
        """, unsafe_allow_html=True)

def show_bias_mitigation_page():
    """Page de mitigation des biais."""
    
    st.markdown('<div class="section-header">🛡️ Mitigation des Biais</div>', unsafe_allow_html=True)
    
    st.subheader("Stratégies de Mitigation Disponibles")
    
    # Sélecteur de stratégie
    strategy = st.selectbox(
        "Choisir une stratégie de mitigation",
        [
            "Suppression des Features Sensibles",
            "Rééchantillonnage SMOTE Équitable", 
            "Calibration par Groupe",
            "Optimisation des Seuils",
            "Entraînement avec Contraintes d'Équité"
        ]
    )
    
    # Simuler les résultats avant/après mitigation
    baseline_metrics = {
        'demographic_parity_difference': 0.18,
        'equal_opportunity_difference': 0.15,
        'disparate_impact_ratio': 1.35,
        'accuracy': 0.72
    }
    
    mitigated_metrics = {
        'demographic_parity_difference': 0.08,
        'equal_opportunity_difference': 0.06,
        'disparate_impact_ratio': 1.12,
        'accuracy': 0.69  # Légère baisse de performance
    }
    
    # Comparaison avant/après
    st.subheader("Impact de la Mitigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Avant Mitigation**")
        st.metric("Parité Démographique", f"{baseline_metrics['demographic_parity_difference']:.3f}")
        st.metric("Égalité des Chances", f"{baseline_metrics['equal_opportunity_difference']:.3f}")
        st.metric("Impact Disparate", f"{baseline_metrics['disparate_impact_ratio']:.3f}")
        st.metric("Précision", f"{baseline_metrics['accuracy']:.3f}")
    
    with col2:
        st.markdown("**Après Mitigation**")
        dp_improvement = baseline_metrics['demographic_parity_difference'] - mitigated_metrics['demographic_parity_difference']
        eo_improvement = baseline_metrics['equal_opportunity_difference'] - mitigated_metrics['equal_opportunity_difference']
        di_improvement = abs(baseline_metrics['disparate_impact_ratio'] - 1.0) - abs(mitigated_metrics['disparate_impact_ratio'] - 1.0)
        acc_change = mitigated_metrics['accuracy'] - baseline_metrics['accuracy']
        
        st.metric("Parité Démographique", f"{mitigated_metrics['demographic_parity_difference']:.3f}", 
                 delta=f"{-dp_improvement:.3f}")
        st.metric("Égalité des Chances", f"{mitigated_metrics['equal_opportunity_difference']:.3f}",
                 delta=f"{-eo_improvement:.3f}")
        st.metric("Impact Disparate", f"{mitigated_metrics['disparate_impact_ratio']:.3f}",
                 delta=f"{-di_improvement:.3f}")
        st.metric("Précision", f"{mitigated_metrics['accuracy']:.3f}",
                 delta=f"{acc_change:.3f}")
    
    # Graphique de comparaison
    st.subheader("Visualisation de l'Amélioration")
    
    comparison_data = {
        'Métrique': ['Parité Démographique', 'Égalité des Chances', 'Impact Disparate (écart à 1)'],
        'Avant': [abs(baseline_metrics['demographic_parity_difference']),
                 abs(baseline_metrics['equal_opportunity_difference']),
                 abs(baseline_metrics['disparate_impact_ratio'] - 1.0)],
        'Après': [abs(mitigated_metrics['demographic_parity_difference']),
                 abs(mitigated_metrics['equal_opportunity_difference']),
                 abs(mitigated_metrics['disparate_impact_ratio'] - 1.0)]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig_comparison = px.bar(comparison_df, x='Métrique', y=['Avant', 'Après'],
                           title="Comparaison des Métriques de Biais",
                           barmode='group')
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Trade-off performance vs équité
    st.subheader("Trade-off Performance vs Équité")
    
    fairness_improvement = (dp_improvement + eo_improvement + di_improvement) / 3 * 100
    performance_loss = abs(acc_change) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Amélioration Équité", f"{fairness_improvement:.1f}%")
    
    with col2:
        st.metric("Impact Performance", f"{performance_loss:.1f}%")
    
    with col3:
        tradeoff_ratio = fairness_improvement / (performance_loss + 0.01)
        st.metric("Ratio Trade-off", f"{tradeoff_ratio:.1f}")
    
    # Évaluation du trade-off
    if tradeoff_ratio > 3:
        st.markdown("""
        <div class="success-alert">
        <strong>✅ Excellent Trade-off:</strong><br>
        Amélioration significative de l'équité avec impact minimal sur la performance
        </div>
        """, unsafe_allow_html=True)
    elif tradeoff_ratio > 1:
        st.markdown("""
        <div class="success-alert">
        <strong>👍 Trade-off Acceptable:</strong><br>
        Amélioration de l'équité justifie la légère baisse de performance
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="bias-alert">
        <strong>⚠️ Trade-off Problématique:</strong><br>
        Impact sur la performance trop important - considérer d'autres stratégies
        </div>
        """, unsafe_allow_html=True)

def show_fairness_evaluation_page():
    """Page d'évaluation de l'équité."""
    
    st.markdown('<div class="section-header">📈 Évaluation de l\'Équité</div>', unsafe_allow_html=True)
    
    st.subheader("Efficacité des Stratégies de Mitigation")
    
    # Simuler des scores d'efficacité
    effectiveness_data = {
        'Stratégie': [
            'Suppression Features',
            'SMOTE Équitable',
            'Calibration par Groupe',
            'Optimisation Seuils',
            'Entraînement Contraint'
        ],
        'Amélioration Équité (%)': [25, 35, 20, 30, 40],
        'Impact Performance (%)': [-8, -5, -2, -3, -12],
        'Score Composite': [17, 30, 18, 27, 28]
    }
    
    effectiveness_df = pd.DataFrame(effectiveness_data)
    
    # Tableau des résultats
    st.dataframe(effectiveness_df, use_container_width=True)
    
    # Graphique scatter plot
    fig_scatter = px.scatter(effectiveness_df, 
                           x='Impact Performance (%)', 
                           y='Amélioration Équité (%)',
                           size='Score Composite',
                           text='Stratégie',
                           title="Trade-off Performance vs Équité par Stratégie")
    
    # Ajouter des lignes de référence
    fig_scatter.add_hline(y=20, line_dash="dash", line_color="green", 
                         annotation_text="Seuil amélioration équité")
    fig_scatter.add_vline(x=-10, line_dash="dash", line_color="red",
                         annotation_text="Seuil impact performance")
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Recommandations basées sur les scores
    st.subheader("Recommandations")
    
    best_strategy = effectiveness_df.loc[effectiveness_df['Score Composite'].idxmax()]
    
    st.markdown(f"""
    <div class="success-alert">
    <strong>🏆 Meilleure Stratégie:</strong> {best_strategy['Stratégie']}<br>
    • Amélioration équité: {best_strategy['Amélioration Équité (%)']}%<br>
    • Impact performance: {best_strategy['Impact Performance (%)']}%<br>
    • Score composite: {best_strategy['Score Composite']}
    </div>
    """, unsafe_allow_html=True)
    
    # Analyse longitudinale simulée
    st.subheader("Évolution des Métriques d'Équité")
    
    # Simuler l'évolution dans le temps
    timeline_data = {
        'Étape': ['Baseline', 'Après Feature Engineering', 'Après Mitigation', 'Après Optimisation'],
        'Parité Démographique': [0.25, 0.20, 0.10, 0.05],
        'Égalité des Chances': [0.22, 0.18, 0.08, 0.04],
        'Impact Disparate': [1.45, 1.35, 1.15, 1.08]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig_timeline = px.line(timeline_df, x='Étape', y=['Parité Démographique', 'Égalité des Chances'],
                          title="Évolution des Métriques de Biais",
                          markers=True)
    st.plotly_chart(fig_timeline, use_container_width=True)

def show_interpretability_comparison_page():
    """Page de comparaison des méthodes d'interprétabilité."""
    
    st.markdown('<div class="section-header">🔄 Comparaison SHAP vs LIME vs SAGE</div>', unsafe_allow_html=True)
    
    # Simuler des résultats de comparaison
    comparison_data = {
        'Méthode': ['SHAP', 'LIME', 'SAGE'],
        'Corrélation avec SHAP': [1.0, 0.72, 0.68],
        'Stabilité': [0.85, 0.62, 0.78],
        'Temps de Calcul': ['Rapide', 'Modéré', 'Lent'],
        'Consistance Top Features': [1.0, 0.6, 0.7]
    }
    
    st.subheader("Comparaison des Méthodes")
    
    # Afficher le tableau
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Graphique radar de comparaison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_radar = go.Figure()
        
        methods = ['SHAP', 'LIME', 'SAGE']
        metrics = ['Corrélation', 'Stabilité', 'Consistance', 'Facilité d\'usage']
        
        # Données simulées pour le radar
        shap_scores = [1.0, 0.85, 1.0, 0.9]
        lime_scores = [0.72, 0.62, 0.6, 0.8]
        sage_scores = [0.68, 0.78, 0.7, 0.3]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=shap_scores,
            theta=metrics,
            fill='toself',
            name='SHAP'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=lime_scores,
            theta=metrics,
            fill='toself',
            name='LIME'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=sage_scores,
            theta=metrics,
            fill='toself',
            name='SAGE'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparaison Radar des Méthodes"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Graphique de corrélation
        corr_data = {
            'SHAP-LIME': 0.72,
            'SHAP-SAGE': 0.68,
            'LIME-SAGE': 0.55
        }
        
        fig_corr = px.bar(x=list(corr_data.keys()), y=list(corr_data.values()),
                         title="Corrélations entre Méthodes")
        fig_corr.add_hline(y=0.7, line_dash="dash", line_color="green",
                          annotation_text="Seuil forte corrélation")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Analyse des forces et faiblesses
    st.subheader("Forces et Faiblesses")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **SHAP**
        
        ✅ **Forces:**
        - Base théorique solide
        - Explications cohérentes
        - Rapide (TreeExplainer)
        
        ❌ **Faiblesses:**
        - Lent (KernelExplainer)
        - Complexité conceptuelle
        """)
    
    with col2:
        st.markdown("""
        **LIME**
        
        ✅ **Forces:**
        - Intuitivement compréhensible
        - Flexible (tout modèle)
        - Explications locales
        
        ❌ **Faiblesses:**
        - Instabilité
        - Approximations locales
        """)
    
    with col3:
        st.markdown("""
        **SAGE**
        
        ✅ **Forces:**
        - Gestion interactions
        - Théoriquement robuste
        - Explications globales
        
        ❌ **Faiblesses:**
        - Très lent
        - Complexité calcul
        """)
    
    # Recommandations finales
    st.subheader("Recommandations d'Usage")
    
    st.markdown("""
    <div class="success-alert">
    <strong>🎯 Recommandations par Contexte:</strong><br><br>
    
    <strong>Production:</strong> Utiliser SHAP (TreeExplainer) pour sa rapidité et fiabilité<br>
    <strong>Exploration:</strong> Combiner SHAP et LIME pour validation croisée<br>
    <strong>Recherche:</strong> SAGE pour analyse approfondie des interactions<br>
    <strong>Audit:</strong> SHAP + LIME pour robustesse des conclusions
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()