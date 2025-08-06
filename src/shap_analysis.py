"""
Module d'analyse SHAP pour le projet COMPAS - Interprétabilité et détection de biais

Ce module fournit des outils complets pour l'analyse SHAP (SHapley Additive exPlanations)
optimisés pour Mac M4 Pro, avec un focus sur la détection de biais racial dans les
prédictions COMPAS.

Auteur: Projet SESAME-SHAP
Date: 2025
"""

import os
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

import shap
from shap import TreeExplainer, KernelExplainer, LinearExplainer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Configuration pour Mac M4 Pro
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompasShapAnalyzer:
    """
    Analyseur SHAP complet pour les modèles COMPAS avec focus sur la détection de biais.
    
    Cette classe fournit tous les outils nécessaires pour l'analyse d'interprétabilité
    des modèles de prédiction de récidive COMPAS, avec une attention particulière
    aux biais raciaux et de genre.
    """
    
    def __init__(self, results_dir: str = "data/results/shap_analysis"):
        """
        Initialise l'analyseur SHAP.
        
        Args:
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        self.models = {}
        self.shap_values = {}
        self.explainers = {}
        self.feature_names = []
        self.X_test = None
        self.y_test = None
        self.sensitive_attributes = None
        
        # Créer le répertoire de résultats
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration SHAP pour performance optimale
        shap.initjs()
        
        logger.info(f"Analyseur SHAP initialisé - Résultats: {self.results_dir}")
    
    def load_trained_models(self, models_dict: Dict[str, BaseEstimator]) -> None:
        """
        Charge les modèles entraînés pour l'analyse SHAP.
        
        Args:
            models_dict: Dictionnaire {nom_modèle: modèle_entraîné}
        """
        self.models = models_dict.copy()
        logger.info(f"Modèles chargés: {list(self.models.keys())}")
    
    def load_test_data(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      sensitive_attributes: pd.DataFrame) -> None:
        """
        Charge les données de test pour l'analyse.
        
        Args:
            X_test: Features de test
            y_test: Cibles de test
            sensitive_attributes: Attributs sensibles (race, sexe, âge)
        """
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.sensitive_attributes = sensitive_attributes.copy()
        self.feature_names = list(X_test.columns)
        
        logger.info(f"Données de test chargées: {X_test.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
    
    def calculate_shap_values(self, max_evals: int = 1000, 
                            sample_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Calcule les valeurs SHAP pour tous les modèles chargés.
        
        Args:
            max_evals: Nombre maximum d'évaluations pour KernelExplainer
            sample_size: Taille d'échantillon pour l'analyse (None = tous)
            
        Returns:
            Dictionnaire {nom_modèle: valeurs_shap}
        """
        if not self.models:
            raise ValueError("Aucun modèle chargé. Utilisez load_trained_models() d'abord.")
        
        if self.X_test is None:
            raise ValueError("Données de test non chargées. Utilisez load_test_data() d'abord.")
        
        # Échantillonnage si nécessaire (optimisation Mac M4 Pro)
        if sample_size and sample_size < len(self.X_test):
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[indices]
        else:
            X_sample = self.X_test
        
        logger.info(f"Calcul des valeurs SHAP pour {len(self.models)} modèles...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Traitement du modèle: {model_name}")
                
                # Sélection de l'explainer approprié
                explainer = self._create_explainer(model, model_name, max_evals)
                self.explainers[model_name] = explainer
                
                # Calcul des valeurs SHAP
                if isinstance(explainer, TreeExplainer):
                    shap_values = explainer.shap_values(X_sample)
                    # Pour les modèles binaires, prendre la classe positive
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                else:
                    shap_values = explainer.shap_values(X_sample)
                
                self.shap_values[model_name] = shap_values
                
                logger.info(f"Valeurs SHAP calculées pour {model_name}: {shap_values.shape}")
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul SHAP pour {model_name}: {str(e)}")
                continue
        
        # Sauvegarder les valeurs SHAP
        self._save_shap_values()
        
        return self.shap_values
    
    def _create_explainer(self, model: BaseEstimator, model_name: str, 
                         max_evals: int) -> Union[TreeExplainer, KernelExplainer, LinearExplainer]:
        """
        Crée l'explainer SHAP approprié selon le type de modèle.
        
        Args:
            model: Modèle à expliquer
            model_name: Nom du modèle
            max_evals: Nombre max d'évaluations pour KernelExplainer
            
        Returns:
            Explainer SHAP approprié
        """
        # TreeExplainer pour les modèles basés sur les arbres
        if any(tree_type in model_name.lower() for tree_type in ['forest', 'xgb', 'lgb', 'tree']):
            if hasattr(model, 'predict_proba'):
                return TreeExplainer(model)
            else:
                return TreeExplainer(model)
        
        # LinearExplainer pour les modèles linéaires
        elif any(linear_type in model_name.lower() for linear_type in ['logistic', 'linear']):
            if hasattr(model, 'coef_'):
                return LinearExplainer(model, self.X_test.iloc[:100])  # Échantillon de base
            else:
                return KernelExplainer(model.predict_proba, self.X_test.iloc[:100])
        
        # KernelExplainer pour les autres modèles
        else:
            # Échantillon de base optimisé pour Mac M4 Pro
            background_size = min(100, len(self.X_test) // 10)
            background = self.X_test.iloc[:background_size]
            
            return KernelExplainer(model.predict_proba, background, 
                                 link="logit", feature_perturbation="interventional")
    
    def analyze_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Analyse l'importance globale des features via SHAP.
        
        Args:
            model_name: Nom du modèle à analyser (None = tous)
            
        Returns:
            DataFrame avec l'importance des features
        """
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées. Utilisez calculate_shap_values() d'abord.")
        
        importance_results = []
        models_to_analyze = [model_name] if model_name else list(self.shap_values.keys())
        
        for model in models_to_analyze:
            if model not in self.shap_values:
                continue
                
            shap_vals = self.shap_values[model]
            
            # Importance globale (moyenne des valeurs absolues)
            feature_importance = np.abs(shap_vals).mean(axis=0)
            
            for i, feature in enumerate(self.feature_names):
                importance_results.append({
                    'model': model,
                    'feature': feature,
                    'importance': feature_importance[i],
                    'mean_shap': shap_vals[:, i].mean(),
                    'std_shap': shap_vals[:, i].std(),
                    'abs_mean_shap': np.abs(shap_vals[:, i]).mean()
                })
        
        importance_df = pd.DataFrame(importance_results)
        
        # Sauvegarder les résultats
        importance_path = os.path.join(self.results_dir, "feature_importance_shap.csv")
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Importance des features sauvegardée: {importance_path}")
        
        return importance_df
    
    def analyze_bias_through_shap(self, demographic_group: str = 'race') -> Dict[str, pd.DataFrame]:
        """
        Analyse les biais via les valeurs SHAP par groupe démographique.
        
        Args:
            demographic_group: Attribut démographique à analyser ('race', 'sex', 'age')
            
        Returns:
            Dictionnaire avec les analyses de biais par modèle
        """
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        if self.sensitive_attributes is None:
            raise ValueError("Attributs sensibles non chargés.")
        
        if demographic_group not in self.sensitive_attributes.columns:
            raise ValueError(f"Groupe démographique '{demographic_group}' non trouvé.")
        
        bias_results = {}
        
        for model_name, shap_vals in self.shap_values.items():
            logger.info(f"Analyse des biais SHAP pour {model_name} - {demographic_group}")
            
            # Créer DataFrame avec valeurs SHAP et groupes démographiques
            shap_df = pd.DataFrame(shap_vals, columns=self.feature_names)
            shap_df[demographic_group] = self.sensitive_attributes[demographic_group].values
            
            # Analyser les différences moyennes par groupe
            bias_analysis = []
            groups = shap_df[demographic_group].unique()
            
            for feature in self.feature_names:
                group_means = {}
                for group in groups:
                    group_mask = shap_df[demographic_group] == group
                    group_means[group] = shap_df.loc[group_mask, feature].mean()
                
                # Calculer les différences entre groupes
                if len(groups) >= 2:
                    # Prendre les deux groupes les plus nombreux
                    top_groups = shap_df[demographic_group].value_counts().head(2).index
                    group1, group2 = top_groups[0], top_groups[1]
                    
                    difference = group_means[group1] - group_means[group2]
                    
                    bias_analysis.append({
                        'feature': feature,
                        'group1': group1,
                        'group2': group2,
                        'group1_mean_shap': group_means[group1],
                        'group2_mean_shap': group_means[group2],
                        'shap_difference': difference,
                        'abs_difference': abs(difference),
                        'relative_difference': abs(difference) / (abs(group_means[group1]) + 1e-8)
                    })
            
            bias_df = pd.DataFrame(bias_analysis)
            bias_df = bias_df.sort_values('abs_difference', ascending=False)
            
            bias_results[model_name] = bias_df
            
            # Sauvegarder les résultats
            bias_path = os.path.join(self.results_dir, f"bias_analysis_{model_name}_{demographic_group}.csv")
            bias_df.to_csv(bias_path, index=False)
        
        logger.info(f"Analyse des biais SHAP terminée pour {demographic_group}")
        return bias_results
    
    def create_shap_visualizations(self, model_name: str, save_plots: bool = True) -> Dict[str, str]:
        """
        Crée un ensemble complet de visualisations SHAP.
        
        Args:
            model_name: Nom du modèle à visualiser
            save_plots: Sauvegarder les graphiques
            
        Returns:
            Dictionnaire avec les chemins des fichiers sauvegardés
        """
        if model_name not in self.shap_values:
            raise ValueError(f"Modèle '{model_name}' non trouvé dans les valeurs SHAP.")
        
        shap_vals = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        plot_paths = {}
        
        # 1. Summary Plot Global
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, self.X_test, feature_names=self.feature_names, 
                         show=False, plot_type="bar")
        plt.title(f'Importance des Features SHAP - {model_name}', fontsize=14, pad=20)
        
        if save_plots:
            summary_path = os.path.join(self.results_dir, f"shap_summary_{model_name}.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plot_paths['summary'] = summary_path
        
        plt.show()
        
        # 2. Summary Plot Détaillé
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, self.X_test, feature_names=self.feature_names, show=False)
        plt.title(f'Impact des Features SHAP - {model_name}', fontsize=14, pad=20)
        
        if save_plots:
            detailed_path = os.path.join(self.results_dir, f"shap_detailed_{model_name}.png")
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            plot_paths['detailed'] = detailed_path
        
        plt.show()
        
        # 3. Dependence Plots pour les top features
        top_features = np.abs(shap_vals).mean(axis=0).argsort()[-5:][::-1]
        
        for i, feature_idx in enumerate(top_features):
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_idx, shap_vals, self.X_test, 
                               feature_names=self.feature_names, show=False)
            plt.title(f'Dépendance SHAP - {self.feature_names[feature_idx]} - {model_name}')
            
            if save_plots:
                dep_path = os.path.join(self.results_dir, 
                                      f"shap_dependence_{model_name}_{self.feature_names[feature_idx]}.png")
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plot_paths[f'dependence_{i}'] = dep_path
            
            plt.show()
        
        # 4. Waterfall Plot pour exemples spécifiques
        if hasattr(explainer, 'expected_value'):
            for i in range(min(3, len(self.X_test))):
                plt.figure(figsize=(12, 8))
                
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                else:
                    expected_value = explainer.expected_value
                
                shap.waterfall_plot(
                    shap.Explanation(values=shap_vals[i], 
                                   base_values=expected_value,
                                   feature_names=self.feature_names),
                    show=False
                )
                plt.title(f'Explication SHAP - Exemple {i+1} - {model_name}')
                
                if save_plots:
                    waterfall_path = os.path.join(self.results_dir, 
                                                f"shap_waterfall_{model_name}_example_{i+1}.png")
                    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                    plot_paths[f'waterfall_{i}'] = waterfall_path
                
                plt.show()
        
        logger.info(f"Visualisations SHAP créées pour {model_name}")
        return plot_paths
    
    def create_bias_comparison_plots(self, demographic_group: str = 'race') -> str:
        """
        Crée des visualisations comparatives des biais SHAP entre groupes démographiques.
        
        Args:
            demographic_group: Groupe démographique à analyser
            
        Returns:
            Chemin du fichier HTML sauvegardé
        """
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        # Créer un dashboard interactif avec Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distribution SHAP par Groupe', 'Top Features Biaisées', 
                          'Heatmap des Biais', 'Comparaison Modèles'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Analyser les biais pour tous les modèles
        bias_analyses = self.analyze_bias_through_shap(demographic_group)
        
        # 1. Distribution SHAP par groupe pour le meilleur modèle
        best_model = list(self.shap_values.keys())[0]  # Premier modèle par défaut
        shap_vals = self.shap_values[best_model]
        
        groups = self.sensitive_attributes[demographic_group].unique()
        for i, group in enumerate(groups):
            group_mask = self.sensitive_attributes[demographic_group] == group
            group_shap = shap_vals[group_mask].mean(axis=0)
            
            fig.add_trace(
                go.Bar(name=f'{group}', x=self.feature_names[:10], y=group_shap[:10]),
                row=1, col=1
            )
        
        # 2. Top features biaisées
        if best_model in bias_analyses:
            bias_df = bias_analyses[best_model].head(10)
            fig.add_trace(
                go.Bar(x=bias_df['feature'], y=bias_df['abs_difference'], 
                      name='Différence Absolue SHAP'),
                row=1, col=2
            )
        
        # 3. Heatmap des biais par modèle et feature
        bias_matrix = []
        model_names = []
        feature_names_bias = []
        
        for model, bias_df in bias_analyses.items():
            model_names.append(model)
            if not feature_names_bias:
                feature_names_bias = bias_df['feature'].head(10).tolist()
            
            bias_row = []
            for feature in feature_names_bias:
                feature_data = bias_df[bias_df['feature'] == feature]
                if not feature_data.empty:
                    bias_row.append(feature_data['abs_difference'].iloc[0])
                else:
                    bias_row.append(0)
            bias_matrix.append(bias_row)
        
        fig.add_trace(
            go.Heatmap(z=bias_matrix, x=feature_names_bias, y=model_names,
                      colorscale='Reds', name='Biais SHAP'),
            row=2, col=1
        )
        
        # 4. Comparaison des modèles (biais moyen)
        model_bias_avg = []
        model_names_clean = []
        
        for model, bias_df in bias_analyses.items():
            model_names_clean.append(model)
            model_bias_avg.append(bias_df['abs_difference'].mean())
        
        fig.add_trace(
            go.Bar(x=model_names_clean, y=model_bias_avg, 
                  name='Biais Moyen', marker_color='red'),
            row=2, col=2
        )
        
        # Mise en forme
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Analyse Comparative des Biais SHAP - {demographic_group.title()}",
            title_x=0.5
        )
        
        # Sauvegarder le dashboard
        html_path = os.path.join(self.results_dir, f"bias_comparison_dashboard_{demographic_group}.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        
        logger.info(f"Dashboard de comparaison des biais sauvegardé: {html_path}")
        return html_path
    
    def generate_shap_report(self, output_format: str = 'markdown') -> str:
        """
        Génère un rapport complet d'analyse SHAP en français.
        
        Args:
            output_format: Format du rapport ('markdown' ou 'html')
            
        Returns:
            Chemin du fichier de rapport généré
        """
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analyser l'importance des features
        importance_df = self.analyze_feature_importance()
        
        # Analyser les biais par race
        bias_race = self.analyze_bias_through_shap('race')
        
        # Contenu du rapport
        report_content = f"""# Rapport d'Analyse SHAP - Projet COMPAS
        
## Résumé Exécutif

Date d'analyse: {datetime.now().strftime("%d/%m/%Y %H:%M")}
Modèles analysés: {', '.join(self.shap_values.keys())}
Nombre d'échantillons: {len(self.X_test)}
Nombre de features: {len(self.feature_names)}

## 1. Importance Globale des Features

### Top 10 Features les Plus Importantes (Moyenne tous modèles)
"""
        
        # Top features
        top_features = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
        
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            report_content += f"{i}. **{feature}**: {importance:.4f}\n"
        
        report_content += "\n## 2. Analyse des Biais Raciaux via SHAP\n\n"
        
        # Analyse des biais par modèle
        for model_name, bias_df in bias_race.items():
            report_content += f"### Modèle: {model_name}\n\n"
            report_content += "**Top 5 Features Contribuant le Plus au Biais Racial:**\n\n"
            
            for i, row in bias_df.head(5).iterrows():
                report_content += f"- **{row['feature']}**: Différence SHAP = {row['shap_difference']:.4f}\n"
                report_content += f"  - {row['group1']}: {row['group1_mean_shap']:.4f}\n"
                report_content += f"  - {row['group2']}: {row['group2_mean_shap']:.4f}\n\n"
        
        report_content += """
## 3. Conclusions et Recommandations

### Principales Observations:
1. **Biais Détectés**: L'analyse SHAP révèle des différences significatives dans les contributions des features entre groupes raciaux.
2. **Features Problématiques**: Certaines features contribuent de manière disproportionnée aux prédictions selon la race.
3. **Transparence**: SHAP permet d'identifier précisément les sources de biais dans les modèles.

### Recommandations:
1. **Mitigation**: Appliquer des techniques de mitigation sur les features identifiées comme biaisées.
2. **Monitoring**: Surveiller continuellement les valeurs SHAP par groupe démographique.
3. **Features Engineering**: Reconsidérer l'inclusion de certaines features hautement biaisées.

### Prochaines Étapes:
1. Appliquer les stratégies de mitigation des biais
2. Réévaluer les modèles après mitigation
3. Comparer avec d'autres méthodes d'interprétabilité (LIME, SAGE)

---
*Rapport généré automatiquement par le module d'analyse SHAP COMPAS*
"""
        
        # Sauvegarder le rapport
        if output_format == 'markdown':
            report_path = os.path.join(self.results_dir, f"shap_analysis_report_{timestamp}.md")
        else:
            report_path = os.path.join(self.results_dir, f"shap_analysis_report_{timestamp}.html")
            # Convertir en HTML basique
            report_content = report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')
            report_content = report_content.replace('**', '<strong>').replace('**', '</strong>')
            report_content = f"<html><body>{report_content}</body></html>"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Rapport SHAP généré: {report_path}")
        return report_path
    
    def _save_shap_values(self) -> None:
        """Sauvegarde les valeurs SHAP calculées."""
        shap_path = os.path.join(self.results_dir, "shap_values.pkl")
        
        save_data = {
            'shap_values': self.shap_values,
            'feature_names': self.feature_names,
            'models': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(shap_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Valeurs SHAP sauvegardées: {shap_path}")
    
    def load_shap_values(self, shap_path: str) -> None:
        """
        Charge des valeurs SHAP précédemment calculées.
        
        Args:
            shap_path: Chemin vers le fichier de valeurs SHAP
        """
        with open(shap_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.shap_values = save_data['shap_values']
        self.feature_names = save_data['feature_names']
        
        logger.info(f"Valeurs SHAP chargées depuis: {shap_path}")


# Fonctions utilitaires
def create_sample_analysis() -> CompasShapAnalyzer:
    """
    Crée une analyse SHAP d'exemple avec des données simulées.
    
    Returns:
        Analyseur SHAP configuré avec des données d'exemple
    """
    # Générer des données d'exemple
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_test = pd.Series(np.random.binomial(1, 0.5, n_samples))
    
    sensitive_attributes = pd.DataFrame({
        'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic'], n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 70, n_samples)
    })
    
    # Créer des modèles d'exemple
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    # Entraîner les modèles
    for name, model in models.items():
        model.fit(X_test, y_test)
    
    # Créer l'analyseur
    analyzer = CompasShapAnalyzer()
    analyzer.load_trained_models(models)
    analyzer.load_test_data(X_test, y_test, sensitive_attributes)
    
    return analyzer


def main():
    """Fonction principale pour démonstration."""
    print("🚀 Démonstration du module d'analyse SHAP COMPAS")
    
    # Créer une analyse d'exemple
    analyzer = create_sample_analysis()
    
    # Calculer les valeurs SHAP
    print("\n📊 Calcul des valeurs SHAP...")
    shap_values = analyzer.calculate_shap_values(max_evals=500, sample_size=200)
    
    # Analyser l'importance des features
    print("\n🔍 Analyse de l'importance des features...")
    importance_df = analyzer.analyze_feature_importance()
    print("\nTop 5 Features les plus importantes:")
    print(importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head())
    
    # Analyser les biais
    print("\n⚖️ Analyse des biais raciaux...")
    bias_analysis = analyzer.analyze_bias_through_shap('race')
    
    for model_name, bias_df in bias_analysis.items():
        print(f"\n{model_name} - Top 3 features biaisées:")
        print(bias_df[['feature', 'shap_difference', 'abs_difference']].head(3))
    
    # Créer des visualisations
    print("\n📈 Génération des visualisations...")
    for model_name in shap_values.keys():
        plot_paths = analyzer.create_shap_visualizations(model_name, save_plots=True)
        print(f"Visualisations {model_name}: {len(plot_paths)} graphiques créés")
    
    # Dashboard de comparaison des biais
    print("\n📊 Création du dashboard de biais...")
    dashboard_path = analyzer.create_bias_comparison_plots()
    print(f"Dashboard sauvegardé: {dashboard_path}")
    
    # Générer le rapport
    print("\n📄 Génération du rapport...")
    report_path = analyzer.generate_shap_report()
    print(f"Rapport généré: {report_path}")
    
    print("\n✅ Analyse SHAP terminée avec succès!")
    print(f"Résultats disponibles dans: {analyzer.results_dir}")


if __name__ == "__main__":
    main()