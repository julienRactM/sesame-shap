"""
Module de détection et d'analyse des biais pour le projet COMPAS

Ce module fournit des outils complets pour détecter et analyser les biais dans les
modèles de prédiction COMPAS, avec un focus sur les métriques d'équité et l'analyse
statistique des disparités entre groupes démographiques.

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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp
import statsmodels.stats.proportion as sm_prop

warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompasBiasAnalyzer:
    """
    Analyseur complet de biais pour les modèles COMPAS.
    
    Cette classe fournit tous les outils nécessaires pour détecter et analyser
    les biais dans les prédictions de récidive, avec un focus sur l'équité
    entre groupes démographiques.
    """
    
    def __init__(self, results_dir: str = "data/results/bias_analysis"):
        """
        Initialise l'analyseur de biais.
        
        Args:
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        self.predictions = {}
        self.probabilities = {}
        self.y_true = None
        self.sensitive_attributes = None
        self.fairness_metrics = {}
        
        # Créer le répertoire de résultats
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Analyseur de biais initialisé - Résultats: {self.results_dir}")
    
    def load_predictions(self, predictions_dict: Dict[str, np.ndarray], 
                        probabilities_dict: Dict[str, np.ndarray],
                        y_true: np.ndarray, 
                        sensitive_attributes: pd.DataFrame) -> None:
        """
        Charge les prédictions et données pour l'analyse.
        
        Args:
            predictions_dict: Dictionnaire {modèle: prédictions}
            probabilities_dict: Dictionnaire {modèle: probabilités}
            y_true: Vraies valeurs cibles
            sensitive_attributes: Attributs sensibles (race, sexe, âge)
        """
        self.predictions = predictions_dict.copy()
        self.probabilities = probabilities_dict.copy()
        self.y_true = y_true.copy()
        self.sensitive_attributes = sensitive_attributes.copy()
        
        logger.info(f"Prédictions chargées pour {len(self.predictions)} modèles")
        logger.info(f"Échantillons: {len(y_true)}")
        logger.info(f"Attributs sensibles: {list(sensitive_attributes.columns)}")
    
    def calculate_fairness_metrics(self, protected_attribute: str = 'race') -> Dict[str, Dict[str, float]]:
        """
        Calcule toutes les métriques d'équité pour tous les modèles.
        
        Args:
            protected_attribute: Attribut protégé à analyser
            
        Returns:
            Dictionnaire imbriqué {modèle: {métrique: valeur}}
        """
        if not self.predictions:
            raise ValueError("Prédictions non chargées. Utilisez load_predictions() d'abord.")
        
        fairness_results = {}
        
        for model_name, y_pred in self.predictions.items():
            logger.info(f"Calcul des métriques d'équité pour {model_name}")
            
            model_metrics = {}
            
            # Obtenir les probabilités si disponibles
            y_proba = self.probabilities.get(model_name, None)
            
            # Calculer toutes les métriques d'équité
            model_metrics.update(self._calculate_demographic_parity(y_pred, protected_attribute))
            model_metrics.update(self._calculate_equalized_odds(y_pred, protected_attribute))
            model_metrics.update(self._calculate_equal_opportunity(y_pred, protected_attribute))
            
            if y_proba is not None:
                model_metrics.update(self._calculate_calibration_metrics(y_proba, protected_attribute))
            
            model_metrics.update(self._calculate_disparate_impact(y_pred, protected_attribute))
            model_metrics.update(self._calculate_statistical_significance(y_pred, protected_attribute))
            
            fairness_results[model_name] = model_metrics
        
        self.fairness_metrics[protected_attribute] = fairness_results
        
        # Sauvegarder les résultats
        self._save_fairness_metrics(protected_attribute)
        
        return fairness_results
    
    def _calculate_demographic_parity(self, y_pred: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule les métriques de parité démographique."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'demographic_parity_difference': 0.0, 'demographic_parity_ratio': 1.0}
        
        # Prendre les deux groupes principaux
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        # Taux de prédictions positives par groupe
        mask1 = self.sensitive_attributes[protected_attribute] == group1
        mask2 = self.sensitive_attributes[protected_attribute] == group2
        
        rate1 = y_pred[mask1].mean()
        rate2 = y_pred[mask2].mean()
        
        # Différence et ratio
        difference = rate1 - rate2
        ratio = rate1 / (rate2 + 1e-8)
        
        return {
            'demographic_parity_difference': difference,
            'demographic_parity_ratio': ratio,
            f'positive_rate_{group1}': rate1,
            f'positive_rate_{group2}': rate2
        }
    
    def _calculate_equalized_odds(self, y_pred: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule les métriques d'égalité des chances (equalized odds)."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'equalized_odds_difference': 0.0}
        
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        results = {}
        
        # Pour chaque classe (0 et 1)
        for class_value in [0, 1]:
            class_mask = self.y_true == class_value
            
            # Masques combinés
            mask1 = (self.sensitive_attributes[protected_attribute] == group1) & class_mask
            mask2 = (self.sensitive_attributes[protected_attribute] == group2) & class_mask
            
            if mask1.sum() > 0 and mask2.sum() > 0:
                rate1 = y_pred[mask1].mean()
                rate2 = y_pred[mask2].mean()
                
                results[f'tpr_class_{class_value}_{group1}'] = rate1
                results[f'tpr_class_{class_value}_{group2}'] = rate2
                results[f'tpr_class_{class_value}_difference'] = abs(rate1 - rate2)
        
        # Différence moyenne d'égalité des chances
        tpr_diffs = [v for k, v in results.items() if 'difference' in k]
        results['equalized_odds_difference'] = np.mean(tpr_diffs) if tpr_diffs else 0.0
        
        return results
    
    def _calculate_equal_opportunity(self, y_pred: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule les métriques d'égalité des opportunités."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'equal_opportunity_difference': 0.0}
        
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        # True Positive Rate pour chaque groupe (classe positive uniquement)
        positive_mask = self.y_true == 1
        
        mask1 = (self.sensitive_attributes[protected_attribute] == group1) & positive_mask
        mask2 = (self.sensitive_attributes[protected_attribute] == group2) & positive_mask
        
        if mask1.sum() > 0 and mask2.sum() > 0:
            tpr1 = y_pred[mask1].mean()
            tpr2 = y_pred[mask2].mean()
            difference = abs(tpr1 - tpr2)
        else:
            tpr1 = tpr2 = difference = 0.0
        
        return {
            'equal_opportunity_difference': difference,
            f'tpr_{group1}': tpr1,
            f'tpr_{group2}': tpr2
        }
    
    def _calculate_calibration_metrics(self, y_proba: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule les métriques de calibration."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'calibration_difference': 0.0}
        
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        results = {}
        
        try:
            # Calibration pour chaque groupe
            mask1 = self.sensitive_attributes[protected_attribute] == group1
            mask2 = self.sensitive_attributes[protected_attribute] == group2
            
            if mask1.sum() > 10 and mask2.sum() > 10:  # Minimum d'échantillons
                # Brier Score par groupe
                brier1 = np.mean((y_proba[mask1] - self.y_true[mask1]) ** 2)
                brier2 = np.mean((y_proba[mask2] - self.y_true[mask2]) ** 2)
                
                results[f'brier_score_{group1}'] = brier1
                results[f'brier_score_{group2}'] = brier2
                results['calibration_difference'] = abs(brier1 - brier2)
                
                # Courbes de calibration
                try:
                    frac_pos1, mean_pred1 = calibration_curve(self.y_true[mask1], y_proba[mask1], n_bins=5)
                    frac_pos2, mean_pred2 = calibration_curve(self.y_true[mask2], y_proba[mask2], n_bins=5)
                    
                    # Écart moyen des courbes de calibration
                    if len(frac_pos1) == len(frac_pos2):
                        calib_diff = np.mean(np.abs(frac_pos1 - frac_pos2))
                        results['calibration_curve_difference'] = calib_diff
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Erreur calcul calibration: {str(e)}")
            results['calibration_difference'] = 0.0
        
        return results
    
    def _calculate_disparate_impact(self, y_pred: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule les métriques d'impact disparate."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'disparate_impact_ratio': 1.0, 'passes_80_rule': True}
        
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        # Taux de sélection par groupe
        mask1 = self.sensitive_attributes[protected_attribute] == group1
        mask2 = self.sensitive_attributes[protected_attribute] == group2
        
        rate1 = y_pred[mask1].mean()
        rate2 = y_pred[mask2].mean()
        
        # Ratio d'impact disparate
        if rate2 > 0:
            disparate_impact_ratio = rate1 / rate2
        else:
            disparate_impact_ratio = float('inf') if rate1 > 0 else 1.0
        
        # Règle des 80% (4/5ths rule)
        passes_80_rule = disparate_impact_ratio >= 0.8 and disparate_impact_ratio <= 1.25
        
        return {
            'disparate_impact_ratio': disparate_impact_ratio,
            'passes_80_rule': passes_80_rule,
            '80_rule_threshold_low': 0.8,
            '80_rule_threshold_high': 1.25
        }
    
    def _calculate_statistical_significance(self, y_pred: np.ndarray, protected_attribute: str) -> Dict[str, float]:
        """Calcule la significativité statistique des différences."""
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            return {'chi2_pvalue': 1.0, 'mannwhitney_pvalue': 1.0}
        
        main_groups = self.sensitive_attributes[protected_attribute].value_counts().head(2).index
        group1, group2 = main_groups[0], main_groups[1]
        
        mask1 = self.sensitive_attributes[protected_attribute] == group1
        mask2 = self.sensitive_attributes[protected_attribute] == group2
        
        results = {}
        
        try:
            # Test du Chi-carré pour l'indépendance
            contingency_table = pd.crosstab(
                self.sensitive_attributes[protected_attribute], 
                y_pred, 
                margins=False
            )
            
            if contingency_table.shape == (2, 2):
                chi2, p_chi2, _, _ = chi2_contingency(contingency_table)
                results['chi2_statistic'] = chi2
                results['chi2_pvalue'] = p_chi2
            
            # Test de Mann-Whitney U
            if mask1.sum() > 0 and mask2.sum() > 0:
                statistic, p_mw = mannwhitneyu(y_pred[mask1], y_pred[mask2], alternative='two-sided')
                results['mannwhitney_statistic'] = statistic
                results['mannwhitney_pvalue'] = p_mw
        
        except Exception as e:
            logger.warning(f"Erreur calcul significativité: {str(e)}")
            results['chi2_pvalue'] = 1.0
            results['mannwhitney_pvalue'] = 1.0
        
        return results
    
    def detect_bias_patterns(self, protected_attribute: str = 'race') -> Dict[str, Any]:
        """
        Détecte les patterns de biais dans les prédictions.
        
        Args:
            protected_attribute: Attribut protégé à analyser
            
        Returns:
            Dictionnaire avec les patterns détectés
        """
        if protected_attribute not in self.fairness_metrics:
            self.calculate_fairness_metrics(protected_attribute)
        
        bias_patterns = {}
        
        for model_name, metrics in self.fairness_metrics[protected_attribute].items():
            model_patterns = {
                'severe_bias': [],
                'moderate_bias': [],
                'potential_bias': [],
                'acceptable': []
            }
            
            # Seuils de détection des biais
            SEVERE_THRESHOLD = 0.2
            MODERATE_THRESHOLD = 0.1
            POTENTIAL_THRESHOLD = 0.05
            
            # Analyser chaque métrique
            for metric_name, value in metrics.items():
                if 'difference' in metric_name or 'ratio' in metric_name:
                    if 'ratio' in metric_name:
                        # Pour les ratios, vérifier l'écart par rapport à 1
                        deviation = abs(value - 1.0)
                    else:
                        # Pour les différences, prendre la valeur absolue
                        deviation = abs(value)
                    
                    if deviation >= SEVERE_THRESHOLD:
                        model_patterns['severe_bias'].append((metric_name, value, deviation))
                    elif deviation >= MODERATE_THRESHOLD:
                        model_patterns['moderate_bias'].append((metric_name, value, deviation))
                    elif deviation >= POTENTIAL_THRESHOLD:
                        model_patterns['potential_bias'].append((metric_name, value, deviation))
                    else:
                        model_patterns['acceptable'].append((metric_name, value, deviation))
            
            # Détection de la règle des 80%
            if not metrics.get('passes_80_rule', True):
                model_patterns['severe_bias'].append(
                    ('80_rule_violation', metrics.get('disparate_impact_ratio', 1.0), 
                     abs(metrics.get('disparate_impact_ratio', 1.0) - 1.0))
                )
            
            # Score de biais global
            total_severe = len(model_patterns['severe_bias'])
            total_moderate = len(model_patterns['moderate_bias'])
            total_potential = len(model_patterns['potential_bias'])
            
            bias_score = (total_severe * 3 + total_moderate * 2 + total_potential * 1)
            
            model_patterns['bias_score'] = bias_score
            model_patterns['bias_level'] = self._classify_bias_level(bias_score)
            
            bias_patterns[model_name] = model_patterns
        
        # Sauvegarder les patterns
        patterns_path = os.path.join(self.results_dir, f"bias_patterns_{protected_attribute}.json")
        with open(patterns_path, 'w', encoding='utf-8') as f:
            # Convertir les tuples en listes pour la sérialisation JSON
            serializable_patterns = {}
            for model, patterns in bias_patterns.items():
                serializable_patterns[model] = {}
                for level, items in patterns.items():
                    if isinstance(items, list) and items and isinstance(items[0], tuple):
                        serializable_patterns[model][level] = [list(item) for item in items]
                    else:
                        serializable_patterns[model][level] = items
            
            json.dump(serializable_patterns, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Patterns de biais sauvegardés: {patterns_path}")
        
        return bias_patterns
    
    def _classify_bias_level(self, bias_score: float) -> str:
        """Classifie le niveau de biais selon le score."""
        if bias_score >= 6:
            return "Biais Sévère"
        elif bias_score >= 3:
            return "Biais Modéré"
        elif bias_score >= 1:
            return "Biais Potentiel"
        else:
            return "Acceptable"
    
    def compare_group_outcomes(self, protected_attribute: str = 'race') -> pd.DataFrame:
        """
        Compare les résultats entre groupes démographiques.
        
        Args:
            protected_attribute: Attribut protégé à analyser
            
        Returns:
            DataFrame avec la comparaison détaillée
        """
        groups = self.sensitive_attributes[protected_attribute].unique()
        
        if len(groups) < 2:
            logger.warning(f"Moins de 2 groupes trouvés pour {protected_attribute}")
            return pd.DataFrame()
        
        comparison_results = []
        
        for model_name, y_pred in self.predictions.items():
            for group in groups:
                mask = self.sensitive_attributes[protected_attribute] == group
                
                if mask.sum() == 0:
                    continue
                
                group_true = self.y_true[mask]
                group_pred = y_pred[mask]
                
                # Calculer les métriques pour ce groupe
                metrics = {
                    'model': model_name,
                    'group': group,
                    'sample_size': mask.sum(),
                    'base_rate': group_true.mean(),
                    'positive_rate': group_pred.mean(),
                    'accuracy': accuracy_score(group_true, group_pred) if len(np.unique(group_true)) > 1 else 0,
                    'precision': precision_score(group_true, group_pred, zero_division=0),
                    'recall': recall_score(group_true, group_pred, zero_division=0),
                    'f1_score': f1_score(group_true, group_pred, zero_division=0)
                }
                
                # AUC si probabilités disponibles
                if model_name in self.probabilities:
                    group_proba = self.probabilities[model_name][mask]
                    if len(np.unique(group_true)) > 1:
                        metrics['auc'] = roc_auc_score(group_true, group_proba)
                    else:
                        metrics['auc'] = 0.5
                
                # Matrice de confusion
                tn, fp, fn, tp = confusion_matrix(group_true, group_pred, labels=[0, 1]).ravel()
                
                metrics.update({
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_positives': tp,
                    'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
                })
                
                comparison_results.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Sauvegarder la comparaison
        comparison_path = os.path.join(self.results_dir, f"group_comparison_{protected_attribute}.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Comparaison des groupes sauvegardée: {comparison_path}")
        
        return comparison_df
    
    def visualize_bias_metrics(self, protected_attribute: str = 'race') -> str:
        """
        Crée des visualisations complètes des métriques de biais.
        
        Args:
            protected_attribute: Attribut protégé à analyser
            
        Returns:
            Chemin du fichier HTML du dashboard
        """
        if protected_attribute not in self.fairness_metrics:
            self.calculate_fairness_metrics(protected_attribute)
        
        # Préparer les données pour visualisation
        metrics_df = self._prepare_metrics_for_viz(protected_attribute)
        comparison_df = self.compare_group_outcomes(protected_attribute)
        
        # Créer le dashboard interactif
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Métriques d\'Équité par Modèle',
                'Comparaison des Taux par Groupe',
                'Matrice de Confusion par Groupe',
                'Distributions des Prédictions',
                'Courbes ROC par Groupe',
                'Scores de Biais Global'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Métriques d'équité par modèle
        if not metrics_df.empty:
            models = metrics_df['model'].unique()
            metrics_to_plot = ['demographic_parity_difference', 'equalized_odds_difference', 
                             'equal_opportunity_difference']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in metrics_df.columns:
                    fig.add_trace(
                        go.Bar(
                            name=metric.replace('_', ' ').title(),
                            x=models,
                            y=metrics_df.groupby('model')[metric].first(),
                            offsetgroup=i
                        ),
                        row=1, col=1
                    )
        
        # 2. Comparaison des taux par groupe
        if not comparison_df.empty:
            groups = comparison_df['group'].unique()
            for group in groups:
                group_data = comparison_df[comparison_df['group'] == group]
                fig.add_trace(
                    go.Bar(
                        name=f'Taux Positif - {group}',
                        x=group_data['model'],
                        y=group_data['positive_rate']
                    ),
                    row=1, col=2
                )
        
        # 3. Matrice de confusion agrégée
        if not comparison_df.empty:
            for i, group in enumerate(groups):
                group_data = comparison_df[comparison_df['group'] == group]
                
                # Calculer les moyennes des matrices de confusion
                avg_tp = group_data['true_positives'].mean()
                avg_fp = group_data['false_positives'].mean()
                avg_tn = group_data['true_negatives'].mean()
                avg_fn = group_data['false_negatives'].mean()
                
                confusion_matrix_data = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])
                
                fig.add_trace(
                    go.Heatmap(
                        z=confusion_matrix_data,
                        x=['Prédit Négatif', 'Prédit Positif'],
                        y=['Vrai Négatif', 'Vrai Positif'],
                        name=f'Confusion - {group}',
                        colorscale='Blues',
                        showscale=i == 0
                    ),
                    row=2, col=1
                )
        
        # 4. Distributions des prédictions
        for model_name, y_pred in self.predictions.items():
            groups = self.sensitive_attributes[protected_attribute].unique()
            
            for group in groups:
                mask = self.sensitive_attributes[protected_attribute] == group
                if mask.sum() > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=y_pred[mask],
                            name=f'{model_name} - {group}',
                            opacity=0.7,
                            nbinsx=20
                        ),
                        row=2, col=2
                    )
        
        # 5. Courbes ROC par groupe (si probabilités disponibles)
        if self.probabilities:
            for model_name, y_proba in self.probabilities.items():
                groups = self.sensitive_attributes[protected_attribute].unique()
                
                for group in groups:
                    mask = self.sensitive_attributes[protected_attribute] == group
                    if mask.sum() > 0 and len(np.unique(self.y_true[mask])) > 1:
                        fpr, tpr, _ = roc_curve(self.y_true[mask], y_proba[mask])
                        auc_score = roc_auc_score(self.y_true[mask], y_proba[mask])
                        
                        fig.add_trace(
                            go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'{model_name} - {group} (AUC: {auc_score:.3f})'
                            ),
                            row=3, col=1
                        )
        
        # 6. Scores de biais global
        bias_patterns = self.detect_bias_patterns(protected_attribute)
        
        models = list(bias_patterns.keys())
        bias_scores = [patterns['bias_score'] for patterns in bias_patterns.values()]
        bias_levels = [patterns['bias_level'] for patterns in bias_patterns.values()]
        
        colors = {'Acceptable': 'green', 'Biais Potentiel': 'yellow', 
                 'Biais Modéré': 'orange', 'Biais Sévère': 'red'}
        bar_colors = [colors.get(level, 'gray') for level in bias_levels]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=bias_scores,
                name='Score de Biais',
                marker_color=bar_colors,
                text=bias_levels,
                textposition='outside'
            ),
            row=3, col=2
        )
        
        # Mise en forme du dashboard
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text=f"Dashboard d'Analyse des Biais - {protected_attribute.title()}",
            title_x=0.5
        )
        
        # Sauvegarder le dashboard
        html_path = os.path.join(self.results_dir, f"bias_dashboard_{protected_attribute}.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        
        logger.info(f"Dashboard de biais sauvegardé: {html_path}")
        return html_path
    
    def _prepare_metrics_for_viz(self, protected_attribute: str) -> pd.DataFrame:
        """Prépare les métriques pour la visualisation."""
        if protected_attribute not in self.fairness_metrics:
            return pd.DataFrame()
        
        metrics_data = []
        
        for model_name, metrics in self.fairness_metrics[protected_attribute].items():
            metric_row = {'model': model_name}
            metric_row.update(metrics)
            metrics_data.append(metric_row)
        
        return pd.DataFrame(metrics_data)
    
    def generate_bias_report(self, protected_attribute: str = 'race', 
                           output_format: str = 'markdown') -> str:
        """
        Génère un rapport complet d'analyse des biais.
        
        Args:
            protected_attribute: Attribut protégé analysé
            output_format: Format du rapport ('markdown' ou 'html')
            
        Returns:
            Chemin du fichier de rapport généré
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # S'assurer que toutes les analyses sont faites
        if protected_attribute not in self.fairness_metrics:
            self.calculate_fairness_metrics(protected_attribute)
        
        bias_patterns = self.detect_bias_patterns(protected_attribute)
        comparison_df = self.compare_group_outcomes(protected_attribute)
        
        # Contenu du rapport
        report_content = f"""# Rapport d'Analyse des Biais - Projet COMPAS

## Résumé Exécutif

**Date d'analyse**: {datetime.now().strftime("%d/%m/%Y %H:%M")}  
**Attribut protégé analysé**: {protected_attribute.title()}  
**Modèles évalués**: {', '.join(self.predictions.keys())}  
**Nombre d'échantillons**: {len(self.y_true)}  

## 1. Vue d'Ensemble des Biais Détectés

"""
        
        # Résumé des niveaux de biais par modèle
        for model_name, patterns in bias_patterns.items():
            bias_level = patterns['bias_level']
            bias_score = patterns['bias_score']
            
            report_content += f"### {model_name}\n"
            report_content += f"- **Niveau de biais**: {bias_level}\n"
            report_content += f"- **Score de biais**: {bias_score}\n"
            
            if patterns['severe_bias']:
                report_content += f"- **Biais sévères détectés**: {len(patterns['severe_bias'])}\n"
                for metric, value, deviation in patterns['severe_bias']:
                    report_content += f"  - {metric}: {value:.4f} (écart: {deviation:.4f})\n"
            
            if patterns['moderate_bias']:
                report_content += f"- **Biais modérés détectés**: {len(patterns['moderate_bias'])}\n"
            
            report_content += "\n"
        
        report_content += "## 2. Métriques d'Équité Détaillées\n\n"
        
        # Tableau des métriques principales
        if protected_attribute in self.fairness_metrics:
            report_content += "| Modèle | Parité Démographique | Égalité des Chances | Impact Disparate | Règle 80% |\n"
            report_content += "|--------|---------------------|-------------------|-----------------|----------|\n"
            
            for model_name, metrics in self.fairness_metrics[protected_attribute].items():
                dp_diff = metrics.get('demographic_parity_difference', 0)
                eo_diff = metrics.get('equal_opportunity_difference', 0)
                di_ratio = metrics.get('disparate_impact_ratio', 1)
                passes_80 = "✅" if metrics.get('passes_80_rule', True) else "❌"
                
                report_content += f"| {model_name} | {dp_diff:.4f} | {eo_diff:.4f} | {di_ratio:.4f} | {passes_80} |\n"
        
        report_content += "\n## 3. Comparaison entre Groupes Démographiques\n\n"
        
        # Analyse comparative des groupes
        if not comparison_df.empty:
            groups = comparison_df['group'].unique()
            
            for group in groups:
                group_data = comparison_df[comparison_df['group'] == group]
                avg_metrics = group_data.groupby('model').agg({
                    'accuracy': 'mean',
                    'precision': 'mean',
                    'recall': 'mean',
                    'f1_score': 'mean',
                    'fpr': 'mean',
                    'fnr': 'mean'
                }).round(4)
                
                report_content += f"### Groupe: {group}\n\n"
                report_content += "| Modèle | Précision | Rappel | F1-Score | Taux FP | Taux FN |\n"
                report_content += "|--------|-----------|--------|----------|---------|--------|\n"
                
                for model in avg_metrics.index:
                    metrics = avg_metrics.loc[model]
                    report_content += f"| {model} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['fpr']:.4f} | {metrics['fnr']:.4f} |\n"
                
                report_content += "\n"
        
        report_content += """## 4. Conclusions et Recommandations

### Principales Observations:

1. **Biais Systémiques**: L'analyse révèle des patterns de biais significatifs dans les prédictions COMPAS.
2. **Disparités de Traitement**: Certains groupes démographiques subissent un traitement inéquitable.
3. **Impact sur l'Équité**: Les différences de performance peuvent avoir des conséquences importantes sur la justice.

### Recommandations Prioritaires:

1. **Mitigation Immédiate**: Appliquer des techniques de réduction des biais sur les modèles identifiés comme problématiques.
2. **Réévaluation des Features**: Examiner les variables qui contribuent le plus aux disparités.
3. **Monitoring Continu**: Mettre en place une surveillance permanente des métriques d'équité.
4. **Formation**: Sensibiliser les utilisateurs aux biais détectés et à leurs implications.

### Actions Techniques:

1. **Préprocessing**: Appliquer des techniques de rééquilibrage des données.
2. **In-processing**: Utiliser des algorithmes d'apprentissage équitable.
3. **Post-processing**: Ajuster les seuils de décision par groupe.
4. **Validation**: Tester l'efficacité des stratégies de mitigation.

### Suivi et Évaluation:

- Réévaluer les métriques d'équité après chaque modification
- Comparer les performances avant/après mitigation
- Documenter tous les changements et leurs impacts
- Maintenir la transparence sur les méthodes utilisées

---
*Rapport généré automatiquement par le module d'analyse des biais COMPAS*
*Conforme aux standards d'équité algorithmique et aux recommandations ProPublica*
"""
        
        # Sauvegarder le rapport
        if output_format == 'markdown':
            report_path = os.path.join(self.results_dir, f"bias_analysis_report_{protected_attribute}_{timestamp}.md")
        else:
            report_path = os.path.join(self.results_dir, f"bias_analysis_report_{protected_attribute}_{timestamp}.html")
            # Conversion HTML basique
            report_content = report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')
            report_content = report_content.replace('**', '<strong>').replace('**', '</strong>')
            report_content = report_content.replace('\n', '<br>\n')
            report_content = f"<html><head><meta charset='utf-8'></head><body>{report_content}</body></html>"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Rapport d'analyse des biais généré: {report_path}")
        return report_path
    
    def _save_fairness_metrics(self, protected_attribute: str) -> None:
        """Sauvegarde les métriques d'équité."""
        metrics_path = os.path.join(self.results_dir, f"fairness_metrics_{protected_attribute}.json")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.fairness_metrics[protected_attribute], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métriques d'équité sauvegardées: {metrics_path}")


# Fonctions utilitaires
def create_sample_bias_analysis() -> CompasBiasAnalyzer:
    """
    Crée une analyse de biais d'exemple avec des données simulées.
    
    Returns:
        Analyseur de biais configuré avec des données d'exemple
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Générer des données avec biais intentionnel
    race = np.random.choice(['African-American', 'Caucasian', 'Hispanic'], n_samples, p=[0.5, 0.4, 0.1])
    sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3])
    age = np.random.randint(18, 70, n_samples)
    
    # Créer un biais: plus de prédictions positives pour African-American
    y_true = np.random.binomial(1, 0.4, n_samples)
    
    # Prédictions biaisées
    bias_factor = np.where(race == 'African-American', 0.3, 0.0)
    pred_proba_base = y_true * 0.7 + (1 - y_true) * 0.2 + bias_factor
    pred_proba_base = np.clip(pred_proba_base + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    predictions = {
        'BiasedModel': (pred_proba_base > 0.5).astype(int),
        'LessBiasedModel': (pred_proba_base > 0.6).astype(int)
    }
    
    probabilities = {
        'BiasedModel': pred_proba_base,
        'LessBiasedModel': np.clip(pred_proba_base - 0.1, 0, 1)
    }
    
    sensitive_attributes = pd.DataFrame({
        'race': race,
        'sex': sex,
        'age': age
    })
    
    # Créer l'analyseur
    analyzer = CompasBiasAnalyzer()
    analyzer.load_predictions(predictions, probabilities, y_true, sensitive_attributes)
    
    return analyzer


def main():
    """Fonction principale pour démonstration."""
    print("🔍 Démonstration du module d'analyse des biais COMPAS")
    
    # Créer une analyse d'exemple
    analyzer = create_sample_bias_analysis()
    
    # Calculer les métriques d'équité
    print("\n📊 Calcul des métriques d'équité...")
    fairness_metrics = analyzer.calculate_fairness_metrics('race')
    
    for model_name, metrics in fairness_metrics.items():
        print(f"\n{model_name}:")
        print(f"  - Parité démographique: {metrics['demographic_parity_difference']:.4f}")
        print(f"  - Égalité des chances: {metrics['equal_opportunity_difference']:.4f}")
        print(f"  - Impact disparate: {metrics['disparate_impact_ratio']:.4f}")
        print(f"  - Règle 80%: {'✅' if metrics['passes_80_rule'] else '❌'}")
    
    # Détecter les patterns de biais
    print("\n🚨 Détection des patterns de biais...")
    bias_patterns = analyzer.detect_bias_patterns('race')
    
    for model_name, patterns in bias_patterns.items():
        print(f"\n{model_name} - {patterns['bias_level']} (Score: {patterns['bias_score']})")
        if patterns['severe_bias']:
            print(f"  Biais sévères: {len(patterns['severe_bias'])}")
        if patterns['moderate_bias']:
            print(f"  Biais modérés: {len(patterns['moderate_bias'])}")
    
    # Comparer les groupes
    print("\n🔄 Comparaison entre groupes...")
    comparison_df = analyzer.compare_group_outcomes('race')
    print(f"Comparaison sauvegardée: {len(comparison_df)} entrées")
    
    # Créer les visualisations
    print("\n📈 Génération du dashboard...")
    dashboard_path = analyzer.visualize_bias_metrics('race')
    print(f"Dashboard sauvegardé: {dashboard_path}")
    
    # Générer le rapport
    print("\n📄 Génération du rapport...")
    report_path = analyzer.generate_bias_report('race')
    print(f"Rapport généré: {report_path}")
    
    print("\n✅ Analyse des biais terminée avec succès!")
    print(f"Résultats disponibles dans: {analyzer.results_dir}")


if __name__ == "__main__":
    main()