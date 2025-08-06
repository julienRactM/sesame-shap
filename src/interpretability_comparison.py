"""
Module de comparaison des méthodes d'interprétabilité pour le projet COMPAS (BONUS)

Ce module compare SHAP, LIME et SAGE pour l'analyse d'interprétabilité des modèles COMPAS,
permettant d'évaluer les forces et faiblesses de chaque approche.

Auteur: Projet SESAME-SHAP
Date: 2025
"""

import os
import json
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

# SHAP
import shap
from shap import TreeExplainer, KernelExplainer

# LIME  
import lime
import lime.lime_tabular

# SAGE (si disponible)
try:
    import sage
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False
    logging.warning("SAGE non disponible - installation recommandée: pip install sage-ml")

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterpretabilityComparator:
    """
    Comparateur des méthodes d'interprétabilité SHAP, LIME et SAGE.
    
    Cette classe permet de comparer les explications fournies par différentes
    méthodes d'interprétabilité sur les mêmes données et modèles COMPAS.
    """
    
    def __init__(self, results_dir: str = "data/results/interpretability_comparison"):
        """
        Initialise le comparateur d'interprétabilité.
        
        Args:
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        self.models = {}
        self.X_test = None
        self.y_test = None
        self.sensitive_attributes = None
        
        # Stockage des explications
        self.shap_explanations = {}
        self.lime_explanations = {}
        self.sage_explanations = {}
        
        # Comparaisons
        self.comparison_results = {}
        
        # Créer le répertoire de résultats
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Comparateur d'interprétabilité initialisé - Résultats: {self.results_dir}")
        logger.info(f"SAGE disponible: {SAGE_AVAILABLE}")
    
    def load_models_and_data(self, models_dict: Dict[str, BaseEstimator],
                           X_test: pd.DataFrame, y_test: pd.Series,
                           sensitive_attributes: pd.DataFrame) -> None:
        """
        Charge les modèles et données pour la comparaison.
        
        Args:
            models_dict: Dictionnaire {nom_modèle: modèle_entraîné}
            X_test: Features de test
            y_test: Cibles de test
            sensitive_attributes: Attributs sensibles
        """
        self.models = models_dict.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.sensitive_attributes = sensitive_attributes.copy()
        
        logger.info(f"Modèles et données chargés: {len(self.models)} modèles, {len(X_test)} échantillons")
    
    def generate_shap_explanations(self, model_name: str, sample_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Génère les explications SHAP pour un modèle.
        
        Args:
            model_name: Nom du modèle à expliquer
            sample_size: Nombre d'échantillons à expliquer
            
        Returns:
            Dictionnaire avec les valeurs SHAP
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé.")
        
        model = self.models[model_name]
        
        # Échantillonnage
        sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        
        logger.info(f"Génération des explications SHAP pour {model_name}...")
        
        try:
            # Choisir l'explainer approprié
            if any(tree_type in model_name.lower() for tree_type in ['forest', 'xgb', 'lgb']):
                explainer = TreeExplainer(model)
            else:
                # Background pour KernelExplainer
                background = self.X_test.iloc[:min(50, len(self.X_test))]
                explainer = KernelExplainer(model.predict_proba, background)
            
            # Calculer les valeurs SHAP
            if isinstance(explainer, TreeExplainer):
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]  # Classe positive
            else:
                shap_values = explainer.shap_values(X_sample)
            
            self.shap_explanations[model_name] = {
                'values': shap_values,
                'sample_indices': sample_indices,
                'feature_names': list(self.X_test.columns),
                'explainer': explainer
            }
            
            logger.info(f"Explications SHAP générées: {shap_values.shape}")
            
        except Exception as e:
            logger.error(f"Erreur génération SHAP: {str(e)}")
            self.shap_explanations[model_name] = None
        
        return self.shap_explanations.get(model_name, {})
    
    def generate_lime_explanations(self, model_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Génère les explications LIME pour un modèle.
        
        Args:
            model_name: Nom du modèle à expliquer
            sample_size: Nombre d'échantillons à expliquer
            
        Returns:
            Dictionnaire avec les explications LIME
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé.")
        
        model = self.models[model_name]
        
        # Échantillonnage
        sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        
        logger.info(f"Génération des explications LIME pour {model_name}...")
        
        try:
            # Créer l'explainer LIME
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_test.values,
                feature_names=list(self.X_test.columns),
                class_names=['No Recidivism', 'Recidivism'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Générer les explications
            lime_explanations = []
            feature_importances = []
            
            for i, idx in enumerate(sample_indices):
                instance = self.X_test.iloc[idx].values
                
                # Explication LIME
                explanation = explainer.explain_instance(
                    instance, 
                    model.predict_proba,
                    num_features=len(self.X_test.columns),
                    num_samples=1000
                )
                
                lime_explanations.append(explanation)
                
                # Extraire les importances de features
                exp_list = explanation.as_list()
                feature_importance = np.zeros(len(self.X_test.columns))
                
                for feature_name, importance in exp_list:
                    # Trouver l'index de la feature
                    for j, col_name in enumerate(self.X_test.columns):
                        if col_name in feature_name or feature_name in col_name:
                            feature_importance[j] = importance
                            break
                
                feature_importances.append(feature_importance)
            
            # Convertir en array numpy
            lime_values = np.array(feature_importances)
            
            self.lime_explanations[model_name] = {
                'values': lime_values,
                'sample_indices': sample_indices,
                'feature_names': list(self.X_test.columns),
                'explanations': lime_explanations,
                'explainer': explainer
            }
            
            logger.info(f"Explications LIME générées: {lime_values.shape}")
            
        except Exception as e:
            logger.error(f"Erreur génération LIME: {str(e)}")
            self.lime_explanations[model_name] = None
        
        return self.lime_explanations.get(model_name, {})
    
    def generate_sage_explanations(self, model_name: str, sample_size: int = 100) -> Dict[str, Any]:
        """
        Génère les explications SAGE pour un modèle (si disponible).
        
        Args:
            model_name: Nom du modèle à expliquer
            sample_size: Nombre d'échantillons à expliquer
            
        Returns:
            Dictionnaire avec les explications SAGE
        """
        if not SAGE_AVAILABLE:
            logger.warning("SAGE non disponible - explications non générées")
            self.sage_explanations[model_name] = None
            return {}
        
        if model_name not in self.models:
            raise ValueError(f"Modèle '{model_name}' non trouvé.")
        
        model = self.models[model_name]
        
        # Échantillonnage
        sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        X_sample = self.X_test.iloc[sample_indices]
        
        logger.info(f"Génération des explications SAGE pour {model_name}...")
        
        try:
            # Créer l'estimateur SAGE
            def model_wrapper(X):
                """Wrapper pour SAGE"""
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    return proba[:, 1]  # Probabilité classe positive
                else:
                    return model.predict(X)
            
            # Configuration SAGE
            imputer = sage.MarginalImputer(model_wrapper, self.X_test.values[:200])  # Background limité
            estimator = sage.PermutationEstimator(imputer, 'cross_entropy')
            
            # Calculer les valeurs SAGE
            sage_values = estimator(X_sample.values, batch_size=10, n_samples=100)
            
            self.sage_explanations[model_name] = {
                'values': sage_values,
                'sample_indices': sample_indices,
                'feature_names': list(self.X_test.columns),
                'estimator': estimator
            }
            
            logger.info(f"Explications SAGE générées: {sage_values.shape}")
            
        except Exception as e:
            logger.error(f"Erreur génération SAGE: {str(e)}")
            self.sage_explanations[model_name] = None
        
        return self.sage_explanations.get(model_name, {})
    
    def compare_explanations(self, model_name: str) -> Dict[str, Any]:
        """
        Compare les explications SHAP, LIME et SAGE pour un modèle.
        
        Args:
            model_name: Nom du modèle à comparer
            
        Returns:
            Dictionnaire avec les résultats de comparaison
        """
        if model_name not in self.shap_explanations:
            self.generate_shap_explanations(model_name)
        
        if model_name not in self.lime_explanations:
            self.generate_lime_explanations(model_name)
        
        if model_name not in self.sage_explanations and SAGE_AVAILABLE:
            self.generate_sage_explanations(model_name)
        
        logger.info(f"Comparaison des explications pour {model_name}...")
        
        comparison_results = {}
        
        # Récupérer les explications
        shap_data = self.shap_explanations.get(model_name)
        lime_data = self.lime_explanations.get(model_name)
        sage_data = self.sage_explanations.get(model_name) if SAGE_AVAILABLE else None
        
        if not shap_data or not lime_data:
            logger.error("Explications SHAP ou LIME manquantes")
            return {}
        
        # Comparer les corrélations entre méthodes
        correlations = self._calculate_explanation_correlations(shap_data, lime_data, sage_data)
        comparison_results['correlations'] = correlations
        
        # Analyser la consistance des top features
        top_features_consistency = self._analyze_top_features_consistency(shap_data, lime_data, sage_data)
        comparison_results['top_features_consistency'] = top_features_consistency
        
        # Analyser la stabilité des explications
        stability_analysis = self._analyze_explanation_stability(shap_data, lime_data, sage_data)
        comparison_results['stability'] = stability_analysis
        
        # Temps de calcul (si mesuré)
        computation_time = self._compare_computation_times(model_name)
        comparison_results['computation_time'] = computation_time
        
        # Analyse des biais dans les explications
        bias_analysis = self._analyze_explanation_bias(shap_data, lime_data, sage_data)
        comparison_results['bias_analysis'] = bias_analysis
        
        self.comparison_results[model_name] = comparison_results
        
        # Sauvegarder les résultats
        self._save_comparison_results(model_name)
        
        return comparison_results
    
    def _calculate_explanation_correlations(self, shap_data: Dict, lime_data: Dict, sage_data: Optional[Dict]) -> Dict[str, float]:
        """Calcule les corrélations entre les différentes méthodes d'explication."""
        correlations = {}
        
        try:
            shap_values = shap_data['values']
            lime_values = lime_data['values']
            
            # Assurer que les formes correspondent
            min_samples = min(len(shap_values), len(lime_values))
            shap_flat = shap_values[:min_samples].flatten()
            lime_flat = lime_values[:min_samples].flatten()
            
            # Corrélation SHAP-LIME
            shap_lime_corr = np.corrcoef(shap_flat, lime_flat)[0, 1]
            correlations['shap_lime'] = shap_lime_corr if not np.isnan(shap_lime_corr) else 0.0
            
            # Corrélation avec SAGE si disponible
            if sage_data and 'values' in sage_data:
                sage_values = sage_data['values']
                min_samples_sage = min(min_samples, len(sage_values))
                sage_flat = sage_values[:min_samples_sage].flatten()
                
                # Ajuster les tailles
                shap_flat_sage = shap_flat[:min_samples_sage * shap_values.shape[1]]
                lime_flat_sage = lime_flat[:min_samples_sage * lime_values.shape[1]]
                
                shap_sage_corr = np.corrcoef(shap_flat_sage, sage_flat)[0, 1]
                lime_sage_corr = np.corrcoef(lime_flat_sage, sage_flat)[0, 1]
                
                correlations['shap_sage'] = shap_sage_corr if not np.isnan(shap_sage_corr) else 0.0
                correlations['lime_sage'] = lime_sage_corr if not np.isnan(lime_sage_corr) else 0.0
        
        except Exception as e:
            logger.error(f"Erreur calcul corrélations: {str(e)}")
        
        return correlations
    
    def _analyze_top_features_consistency(self, shap_data: Dict, lime_data: Dict, sage_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyse la consistance des top features entre méthodes."""
        try:
            feature_names = shap_data['feature_names']
            
            # Importance moyenne par méthode
            shap_importance = np.abs(shap_data['values']).mean(axis=0)
            lime_importance = np.abs(lime_data['values']).mean(axis=0)
            
            # Top 10 features par méthode
            shap_top = np.argsort(shap_importance)[-10:][::-1]
            lime_top = np.argsort(lime_importance)[-10:][::-1]
            
            # Consistance SHAP-LIME
            shap_lime_overlap = len(set(shap_top) & set(lime_top))
            
            consistency_results = {
                'shap_top_features': [feature_names[i] for i in shap_top],
                'lime_top_features': [feature_names[i] for i in lime_top],
                'shap_lime_overlap': shap_lime_overlap,
                'shap_lime_consistency_ratio': shap_lime_overlap / 10.0
            }
            
            # Avec SAGE si disponible
            if sage_data and 'values' in sage_data:
                sage_importance = np.abs(sage_data['values']).mean(axis=0)
                sage_top = np.argsort(sage_importance)[-10:][::-1]
                
                consistency_results['sage_top_features'] = [feature_names[i] for i in sage_top]
                consistency_results['shap_sage_overlap'] = len(set(shap_top) & set(sage_top))
                consistency_results['lime_sage_overlap'] = len(set(lime_top) & set(sage_top))
                consistency_results['all_methods_overlap'] = len(set(shap_top) & set(lime_top) & set(sage_top))
            
            return consistency_results
            
        except Exception as e:
            logger.error(f"Erreur analyse consistance: {str(e)}")
            return {}
    
    def _analyze_explanation_stability(self, shap_data: Dict, lime_data: Dict, sage_data: Optional[Dict]) -> Dict[str, float]:
        """Analyse la stabilité des explications."""
        stability_results = {}
        
        try:
            # Stabilité SHAP (variance des explications)
            shap_values = shap_data['values']
            shap_stability = 1.0 / (1.0 + np.mean(np.var(shap_values, axis=0)))
            stability_results['shap_stability'] = shap_stability
            
            # Stabilité LIME
            lime_values = lime_data['values']
            lime_stability = 1.0 / (1.0 + np.mean(np.var(lime_values, axis=0)))
            stability_results['lime_stability'] = lime_stability
            
            # Stabilité SAGE si disponible
            if sage_data and 'values' in sage_data:
                sage_values = sage_data['values']
                sage_stability = 1.0 / (1.0 + np.mean(np.var(sage_values, axis=0)))
                stability_results['sage_stability'] = sage_stability
        
        except Exception as e:
            logger.error(f"Erreur analyse stabilité: {str(e)}")
        
        return stability_results
    
    def _compare_computation_times(self, model_name: str) -> Dict[str, str]:
        """Compare les temps de calcul (estimation basique)."""
        return {
            'shap_time': "Rapide (pour TreeExplainer), Lent (pour KernelExplainer)",
            'lime_time': "Modéré (dépend du nombre d'échantillons)",
            'sage_time': "Lent (calculs de permutation intensifs)" if SAGE_AVAILABLE else "Non disponible"
        }
    
    def _analyze_explanation_bias(self, shap_data: Dict, lime_data: Dict, sage_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyse les biais dans les explications par groupe démographique."""
        if self.sensitive_attributes is None:
            return {}
        
        bias_analysis = {}
        
        try:
            # Analyser par race
            if 'race' in self.sensitive_attributes.columns:
                race_analysis = {}
                
                sample_indices = shap_data['sample_indices']
                race_sample = self.sensitive_attributes.iloc[sample_indices]['race']
                
                # Groupes raciaux
                groups = race_sample.unique()
                
                for method_name, data in [('shap', shap_data), ('lime', lime_data)]:
                    if data is None:
                        continue
                    
                    group_importances = {}
                    for group in groups:
                        group_mask = race_sample == group
                        if group_mask.sum() > 0:
                            group_explanations = data['values'][group_mask]
                            group_importance = np.abs(group_explanations).mean(axis=0)
                            group_importances[group] = group_importance
                    
                    race_analysis[method_name] = group_importances
                
                bias_analysis['race'] = race_analysis
        
        except Exception as e:
            logger.error(f"Erreur analyse biais explications: {str(e)}")
        
        return bias_analysis
    
    def create_comparison_dashboard(self, model_name: str) -> str:
        """
        Crée un dashboard de comparaison des méthodes d'interprétabilité.
        
        Args:
            model_name: Nom du modèle à visualiser
            
        Returns:
            Chemin du fichier HTML du dashboard
        """
        if model_name not in self.comparison_results:
            self.compare_explanations(model_name)
        
        comparison_data = self.comparison_results[model_name]
        
        # Créer le dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Corrélations entre Méthodes',
                'Consistance des Top Features',
                'Comparaison des Importances Moyennes',
                'Stabilité des Explications',
                'Distribution des Explications SHAP vs LIME',
                'Analyse des Biais par Groupe'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Corrélations entre méthodes
        correlations = comparison_data.get('correlations', {})
        corr_names = list(correlations.keys())
        corr_values = list(correlations.values())
        
        fig.add_trace(
            go.Bar(x=corr_names, y=corr_values, name='Corrélations',
                  marker_color=['blue', 'green', 'red'][:len(corr_values)]),
            row=1, col=1
        )
        
        # 2. Consistance des top features
        consistency = comparison_data.get('top_features_consistency', {})
        if consistency:
            consistency_metrics = []
            consistency_values = []
            
            if 'shap_lime_consistency_ratio' in consistency:
                consistency_metrics.append('SHAP-LIME')
                consistency_values.append(consistency['shap_lime_consistency_ratio'])
            
            if 'shap_sage_overlap' in consistency:
                consistency_metrics.append('SHAP-SAGE')
                consistency_values.append(consistency.get('shap_sage_overlap', 0) / 10.0)
                
            if consistency_metrics:
                fig.add_trace(
                    go.Bar(x=consistency_metrics, y=consistency_values, name='Consistance'),
                    row=1, col=2
                )
        
        # 3. Comparaison des importances moyennes
        shap_data = self.shap_explanations.get(model_name, {})
        lime_data = self.lime_explanations.get(model_name, {})
        
        if shap_data and lime_data and 'values' in shap_data and 'values' in lime_data:
            feature_names = shap_data['feature_names'][:10]  # Top 10
            shap_importance = np.abs(shap_data['values']).mean(axis=0)[:10]
            lime_importance = np.abs(lime_data['values']).mean(axis=0)[:10]
            
            fig.add_trace(
                go.Bar(x=feature_names, y=shap_importance, name='SHAP', opacity=0.7),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=feature_names, y=lime_importance, name='LIME', opacity=0.7),
                row=2, col=1
            )
        
        # 4. Stabilité des explications
        stability = comparison_data.get('stability', {})
        if stability:
            stability_methods = list(stability.keys())
            stability_scores = list(stability.values())
            
            fig.add_trace(
                go.Scatter(x=stability_methods, y=stability_scores, 
                          mode='markers+lines', name='Stabilité',
                          marker=dict(size=10)),
                row=2, col=2
            )
        
        # 5. Distribution des explications (scatter plot)
        if shap_data and lime_data and 'values' in shap_data and 'values' in lime_data:
            shap_flat = shap_data['values'].flatten()[:1000]  # Limiter pour performance
            lime_flat = lime_data['values'].flatten()[:1000]
            
            fig.add_trace(
                go.Scatter(x=shap_flat, y=lime_flat, mode='markers',
                          name='SHAP vs LIME', opacity=0.6),
                row=3, col=1
            )
        
        # 6. Analyse des biais (si disponible)
        bias_analysis = comparison_data.get('bias_analysis', {})
        if bias_analysis and 'race' in bias_analysis:
            race_data = bias_analysis['race']
            
            # Exemple: différence d'importance entre groupes raciaux
            if 'shap' in race_data and len(race_data['shap']) >= 2:
                groups = list(race_data['shap'].keys())[:2]
                group1_importance = race_data['shap'][groups[0]][:5]  # Top 5 features
                group2_importance = race_data['shap'][groups[1]][:5]
                
                feature_subset = shap_data['feature_names'][:5]
                
                fig.add_trace(
                    go.Bar(x=feature_subset, y=group1_importance, name=f'SHAP - {groups[0]}'),
                    row=3, col=2
                )
                fig.add_trace(
                    go.Bar(x=feature_subset, y=group2_importance, name=f'SHAP - {groups[1]}'),
                    row=3, col=2
                )
        
        # Mise en forme
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text=f"Comparaison des Méthodes d'Interprétabilité - {model_name}",
            title_x=0.5
        )
        
        # Sauvegarder le dashboard
        html_path = os.path.join(self.results_dir, f"interpretability_comparison_dashboard_{model_name}.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        
        logger.info(f"Dashboard de comparaison sauvegardé: {html_path}")
        return html_path
    
    def generate_comparison_report(self, model_name: str) -> str:
        """
        Génère un rapport de comparaison des méthodes d'interprétabilité.
        
        Args:
            model_name: Nom du modèle analysé
            
        Returns:
            Chemin du fichier de rapport généré
        """
        if model_name not in self.comparison_results:
            self.compare_explanations(model_name)
        
        comparison_data = self.comparison_results[model_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Contenu du rapport
        report_content = f"""# Rapport de Comparaison des Méthodes d'Interprétabilité - COMPAS

## Résumé Exécutif

**Date d'analyse**: {datetime.now().strftime("%d/%m/%Y %H:%M")}  
**Modèle analysé**: {model_name}  
**Méthodes comparées**: SHAP, LIME{', SAGE' if SAGE_AVAILABLE and self.sage_explanations.get(model_name) else ''}  

## 1. Corrélations entre Méthodes

"""
        
        # Corrélations
        correlations = comparison_data.get('correlations', {})
        for method_pair, correlation in correlations.items():
            method_names = method_pair.replace('_', ' vs ').upper()
            report_content += f"- **{method_names}**: {correlation:.4f}\n"
        
        # Interprétation des corrélations
        shap_lime_corr = correlations.get('shap_lime', 0)
        if shap_lime_corr > 0.7:
            corr_interpretation = "Forte concordance"
        elif shap_lime_corr > 0.4:
            corr_interpretation = "Concordance modérée"
        else:
            corr_interpretation = "Faible concordance"
        
        report_content += f"\n**Interprétation**: {corr_interpretation} entre SHAP et LIME (r={shap_lime_corr:.3f})\n"
        
        report_content += "\n## 2. Consistance des Top Features\n\n"
        
        # Consistance des features
        consistency = comparison_data.get('top_features_consistency', {})
        if consistency:
            report_content += f"**Ratio de consistance SHAP-LIME**: {consistency.get('shap_lime_consistency_ratio', 0):.2f}\n\n"
            
            # Top features par méthode
            if 'shap_top_features' in consistency:
                report_content += "**Top 5 Features SHAP**:\n"
                for i, feature in enumerate(consistency['shap_top_features'][:5], 1):
                    report_content += f"{i}. {feature}\n"
            
            if 'lime_top_features' in consistency:
                report_content += "\n**Top 5 Features LIME**:\n"
                for i, feature in enumerate(consistency['lime_top_features'][:5], 1):
                    report_content += f"{i}. {feature}\n"
        
        report_content += "\n## 3. Analyse de Stabilité\n\n"
        
        # Stabilité
        stability = comparison_data.get('stability', {})
        for method, score in stability.items():
            method_name = method.replace('_stability', '').upper()
            report_content += f"- **{method_name}**: {score:.4f}\n"
        
        report_content += "\n## 4. Temps de Calcul\n\n"
        
        # Temps de calcul
        computation_time = comparison_data.get('computation_time', {})
        for method, time_desc in computation_time.items():
            method_name = method.replace('_time', '').upper()
            report_content += f"- **{method_name}**: {time_desc}\n"
        
        report_content += "\n## 5. Analyse des Biais\n\n"
        
        # Biais dans les explications
        bias_analysis = comparison_data.get('bias_analysis', {})
        if bias_analysis and 'race' in bias_analysis:
            report_content += "**Analyse par groupes raciaux**: Détection de différences dans les patterns d'explication entre groupes démographiques.\n\n"
        else:
            report_content += "**Analyse des biais**: Données insuffisantes pour l'analyse par groupe.\n\n"
        
        report_content += """## 6. Recommandations

### Forces et Faiblesses par Méthode

**SHAP**:
- ✅ **Forces**: Base théorique solide (valeurs de Shapley), explications cohérentes
- ❌ **Faiblesses**: Peut être lent pour les modèles complexes (KernelExplainer)

**LIME**:
- ✅ **Forces**: Flexible, fonctionne avec tout type de modèle, intuitivement compréhensible
- ❌ **Faiblesses**: Approximations locales, peut être instable

"""
        
        if SAGE_AVAILABLE and self.sage_explanations.get(model_name):
            report_content += """**SAGE**:
- ✅ **Forces**: Gestion native des interactions entre features
- ❌ **Faiblesses**: Très coûteux en calcul, moins mature

"""
        
        # Recommandations basées sur les résultats
        shap_lime_corr = correlations.get('shap_lime', 0)
        consistency_ratio = consistency.get('shap_lime_consistency_ratio', 0)
        
        if shap_lime_corr > 0.6 and consistency_ratio > 0.6:
            recommendation = "Les méthodes convergent - Utiliser SHAP pour la production (plus rapide avec TreeExplainer)"
        elif shap_lime_corr > 0.4:
            recommendation = "Concordance modérée - Utiliser les deux méthodes pour validation croisée"
        else:
            recommendation = "Faible concordance - Investiguer les différences et choisir selon le contexte"
        
        report_content += f"""### Recommandation Principale

{recommendation}

### Actions Suggérées

1. **Pour la production**: Privilégier SHAP avec TreeExplainer pour les modèles basés sur les arbres
2. **Pour l'exploration**: Utiliser LIME pour comprendre les approximations locales
3. **Pour la validation**: Comparer les résultats des deux méthodes sur des cas critiques
4. **Pour les audits**: Documenter les différences d'interprétation entre méthodes

---
*Rapport généré automatiquement par le module de comparaison d'interprétabilité COMPAS*
"""
        
        # Sauvegarder le rapport
        report_path = os.path.join(self.results_dir, f"interpretability_comparison_report_{model_name}_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Rapport de comparaison généré: {report_path}")
        return report_path
    
    def _save_comparison_results(self, model_name: str) -> None:
        """Sauvegarde les résultats de comparaison."""
        results_path = os.path.join(self.results_dir, f"comparison_results_{model_name}.json")
        
        # Préparer les données pour sérialisation JSON
        serializable_results = {}
        for key, value in self.comparison_results[model_name].items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Résultats de comparaison sauvegardés: {results_path}")


# Fonctions utilitaires
def create_sample_comparison() -> InterpretabilityComparator:
    """
    Crée une comparaison d'interprétabilité d'exemple.
    
    Returns:
        Comparateur configuré avec des données d'exemple
    """
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Données d'exemple
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_test = pd.Series(np.random.binomial(1, 0.4, n_samples))
    
    sensitive_attributes = pd.DataFrame({
        'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic'], n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples)
    })
    
    # Modèles d'exemple
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }
    
    # Entraîner les modèles
    for name, model in models.items():
        model.fit(X_test, y_test)
    
    # Créer le comparateur
    comparator = InterpretabilityComparator()
    comparator.load_models_and_data(models, X_test, y_test, sensitive_attributes)
    
    return comparator


def main():
    """Fonction principale pour démonstration."""
    print("🔍 Démonstration du module de comparaison d'interprétabilité COMPAS")
    
    # Créer une comparaison d'exemple
    comparator = create_sample_comparison()
    
    # Tester avec RandomForest
    model_name = 'RandomForest'
    
    # Générer les explications
    print(f"\n📊 Génération des explications pour {model_name}...")
    
    shap_results = comparator.generate_shap_explanations(model_name, sample_size=50)
    print(f"SHAP: {shap_results['values'].shape if shap_results else 'Échec'}")
    
    lime_results = comparator.generate_lime_explanations(model_name, sample_size=50)
    print(f"LIME: {lime_results['values'].shape if lime_results else 'Échec'}")
    
    if SAGE_AVAILABLE:
        sage_results = comparator.generate_sage_explanations(model_name, sample_size=20)
        print(f"SAGE: {sage_results['values'].shape if sage_results else 'Échec'}")
    
    # Comparer les explications
    print(f"\n🔄 Comparaison des explications...")
    comparison_results = comparator.compare_explanations(model_name)
    
    # Afficher les résultats
    correlations = comparison_results.get('correlations', {})
    print(f"\nCorrélations:")
    for method_pair, corr in correlations.items():
        print(f"  {method_pair}: {corr:.4f}")
    
    consistency = comparison_results.get('top_features_consistency', {})
    if consistency:
        print(f"\nConsistance SHAP-LIME: {consistency.get('shap_lime_consistency_ratio', 0):.2f}")
    
    # Créer le dashboard
    print(f"\n📈 Génération du dashboard...")
    dashboard_path = comparator.create_comparison_dashboard(model_name)
    print(f"Dashboard sauvegardé: {dashboard_path}")
    
    # Générer le rapport
    print(f"\n📄 Génération du rapport...")
    report_path = comparator.generate_comparison_report(model_name)
    print(f"Rapport généré: {report_path}")
    
    print(f"\n✅ Comparaison d'interprétabilité terminée!")
    print(f"Résultats disponibles dans: {comparator.results_dir}")


if __name__ == "__main__":
    main()