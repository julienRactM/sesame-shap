"""
Framework d'analyse de biais et métriques d'équité pour le projet COMPAS

Ce module fournit une suite complète d'outils pour détecter et analyser les biais
algorithmiques dans les prédictions du système COMPAS, en se basant sur les découvertes
de l'enquête ProPublica.

Classes principales:
- BiasAnalyzer: Analyseur principal pour la détection de biais
- FairnessMetrics: Calculateur de métriques d'équité
- BiasVisualizer: Générateur de visualisations de biais
- StatisticalTester: Tests statistiques pour la détection de biais

Auteur: Claude AI - Assistant Data Engineer
Date: 2025-08-05
"""

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kstest
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, calibration_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer

# Suppression des warnings pour une sortie plus propre
warnings.filterwarnings('ignore')

@dataclass
class FairnessResult:
    """Structure pour stocker les résultats d'équité"""
    metric_name: str
    value: float
    group_values: Dict[str, float]
    statistical_significance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    threshold_passed: bool = True


@dataclass
class BiasAnalysisConfig:
    """Configuration pour l'analyse de biais"""
    protected_attributes: List[str] = None
    fairness_threshold: float = 0.8  # Seuil de 80% pour la disparate impact rule
    statistical_significance_level: float = 0.05
    bootstrap_samples: int = 1000
    save_visualizations: bool = True
    results_dir: str = "data/results/bias_analysis"
    
    def __post_init__(self):
        if self.protected_attributes is None:
            self.protected_attributes = ['race', 'sex', 'age_group']


class StatisticalTester:
    """Classe pour effectuer des tests statistiques de détection de biais"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialise le testeur statistique
        
        Args:
            alpha: Niveau de signification statistique
        """
        self.alpha = alpha
    
    def chi_square_test(self, contingency_table: np.ndarray) -> Dict[str, float]:
        """
        Test du chi-carré pour l'indépendance
        
        Args:
            contingency_table: Table de contingence
            
        Returns:
            Dictionnaire avec statistique et p-value
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < self.alpha
        }
    
    def proportion_test(self, counts1: np.ndarray, nobs1: np.ndarray, 
                       counts2: np.ndarray, nobs2: np.ndarray) -> Dict[str, float]:
        """
        Test de différence de proportions entre deux groupes
        
        Args:
            counts1, nobs1: Comptages et observations groupe 1
            counts2, nobs2: Comptages et observations groupe 2
            
        Returns:
            Dictionnaire avec statistique et p-value
        """
        z_stat, p_value = proportions_ztest([counts1, counts2], [nobs1, nobs2])
        return {
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha
        }
    
    def mann_whitney_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """
        Test de Mann-Whitney U pour comparer les distributions
        
        Args:
            group1, group2: Échantillons des deux groupes
            
        Returns:
            Dictionnaire avec statistique et p-value
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        return {
            'u_statistic': statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha
        }
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    metric_func: callable,
                                    n_bootstrap: int = 1000,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calcule l'intervalle de confiance par bootstrap
        
        Args:
            data: Données d'entrée
            metric_func: Fonction de métrique à calculer
            n_bootstrap: Nombre d'échantillons bootstrap
            confidence_level: Niveau de confiance
            
        Returns:
            Intervalle de confiance (borne_inf, borne_sup)
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(metric_func(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower_bound, upper_bound)


class FairnessMetrics:
    """Classe pour calculer les métriques d'équité"""
    
    def __init__(self):
        """Initialise le calculateur de métriques d'équité"""
        self.metrics_results = {}
    
    def demographic_parity(self, y_pred: np.ndarray, 
                          protected_attr: np.ndarray) -> Dict[str, Any]:
        """
        Calcule la parité démographique (Statistical Parity)
        
        La parité démographique est respectée si P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
        pour tous les groupes a, b de l'attribut protégé A.
        
        Args:
            y_pred: Prédictions binaires
            protected_attr: Attribut protégé
            
        Returns:
            Dictionnaire avec métriques de parité démographique
        """
        unique_groups = np.unique(protected_attr)
        group_rates = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            positive_rate = np.mean(y_pred[mask])
            group_rates[str(group)] = positive_rate
        
        # Calcul du ratio min/max (Disparate Impact)
        rates_values = list(group_rates.values())
        disparate_impact = min(rates_values) / max(rates_values) if max(rates_values) > 0 else 0
        
        # Différence maximale entre groupes
        max_difference = max(rates_values) - min(rates_values)
        
        return {
            'group_rates': group_rates,
            'disparate_impact': disparate_impact,
            'max_difference': max_difference,
            'passes_80_rule': disparate_impact >= 0.8,
            'interpretation': self._interpret_demographic_parity(disparate_impact)
        }
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      protected_attr: np.ndarray) -> Dict[str, Any]:
        """
        Calcule l'égalité des chances (Equalized Odds)
        
        L'égalité des chances est respectée si:
        - P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) (Taux de vrais positifs égaux)
        - P(Ŷ=1|Y=0,A=a) = P(Ŷ=1|Y=0,A=b) (Taux de faux positifs égaux)
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            protected_attr: Attribut protégé
            
        Returns:
            Dictionnaire avec métriques d'égalité des chances
        """
        unique_groups = np.unique(protected_attr)
        group_metrics = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Taux de vrais positifs (TPR)
            tpr = np.mean(y_pred_group[y_true_group == 1]) if np.sum(y_true_group == 1) > 0 else 0
            
            # Taux de faux positifs (FPR)
            fpr = np.mean(y_pred_group[y_true_group == 0]) if np.sum(y_true_group == 0) > 0 else 0
            
            group_metrics[str(group)] = {
                'tpr': tpr,
                'fpr': fpr,
                'tnr': 1 - fpr,  # Spécificité
                'fnr': 1 - tpr   # Taux de faux négatifs
            }
        
        # Calcul des différences entre groupes
        tpr_values = [metrics['tpr'] for metrics in group_metrics.values()]
        fpr_values = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_difference = max(tpr_values) - min(tpr_values)
        fpr_difference = max(fpr_values) - min(fpr_values)
        
        return {
            'group_metrics': group_metrics,
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'equalized_odds_difference': max(tpr_difference, fpr_difference),
            'interpretation': self._interpret_equalized_odds(tpr_difference, fpr_difference)
        }
    
    def equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         protected_attr: np.ndarray) -> Dict[str, Any]:
        """
        Calcule l'égalité d'opportunité (Equal Opportunity)
        
        L'égalité d'opportunité est respectée si P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)
        (Taux de vrais positifs égaux entre groupes)
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            protected_attr: Attribut protégé
            
        Returns:
            Dictionnaire avec métriques d'égalité d'opportunité
        """
        unique_groups = np.unique(protected_attr)
        group_tpr = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Taux de vrais positifs
            tpr = np.mean(y_pred_group[y_true_group == 1]) if np.sum(y_true_group == 1) > 0 else 0
            group_tpr[str(group)] = tpr
        
        tpr_values = list(group_tpr.values())
        tpr_difference = max(tpr_values) - min(tpr_values)
        
        return {
            'group_tpr': group_tpr,
            'tpr_difference': tpr_difference,
            'interpretation': self._interpret_equal_opportunity(tpr_difference)
        }
    
    def calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           protected_attr: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Calcule les métriques de calibration par groupe
        
        La calibration mesure si les probabilités prédites correspondent aux fréquences observées.
        
        Args:
            y_true: Vraies étiquettes
            y_prob: Probabilités prédites
            protected_attr: Attribut protégé
            n_bins: Nombre de bins pour la calibration
            
        Returns:
            Dictionnaire avec métriques de calibration
        """
        unique_groups = np.unique(protected_attr)
        group_calibration = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_prob_group = y_prob[mask]
            
            if len(y_true_group) == 0:
                continue
                
            # Calcul de la calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_group, y_prob_group, n_bins=n_bins
            )
            
            # Brier Score (mesure de calibration)
            brier_score = np.mean((y_prob_group - y_true_group) ** 2)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob_group > bin_lower) & (y_prob_group <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true_group[in_bin].mean()
                    avg_confidence_in_bin = y_prob_group[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            group_calibration[str(group)] = {
                'brier_score': brier_score,
                'expected_calibration_error': ece,
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        
        # Différence de calibration entre groupes
        brier_scores = [cal['brier_score'] for cal in group_calibration.values()]
        ece_scores = [cal['expected_calibration_error'] for cal in group_calibration.values()]
        
        brier_difference = max(brier_scores) - min(brier_scores) if brier_scores else 0
        ece_difference = max(ece_scores) - min(ece_scores) if ece_scores else 0
        
        return {
            'group_calibration': group_calibration,
            'brier_score_difference': brier_difference,
            'ece_difference': ece_difference,
            'interpretation': self._interpret_calibration(brier_difference, ece_difference)
        }
    
    def individual_fairness_metrics(self, X: np.ndarray, y_pred: np.ndarray, 
                                   distance_func: callable = None) -> Dict[str, Any]:
        """
        Calcule des mesures d'équité individuelle
        
        L'équité individuelle stipule que des individus similaires doivent recevoir
        des prédictions similaires.
        
        Args:
            X: Caractéristiques des individus
            y_pred: Prédictions
            distance_func: Fonction de distance (défaut: distance euclidienne)
            
        Returns:
            Dictionnaire avec métriques d'équité individuelle
        """
        if distance_func is None:
            from sklearn.metrics.pairwise import euclidean_distances
            distance_func = euclidean_distances
        
        # Calcul des distances entre individus
        distances = distance_func(X)
        n = len(X)
        
        # Calcul des différences de prédictions
        pred_differences = np.abs(y_pred[:, np.newaxis] - y_pred[np.newaxis, :])
        
        # Correlation entre similarité et différence de prédiction
        # (plus les individus sont similaires, moins leurs prédictions devraient différer)
        similarity_scores = 1 / (1 + distances)  # Transformation en similarité
        
        # Calcul de la corrélation de Spearman
        correlation_values = []
        for i in range(n):
            for j in range(i + 1, n):
                correlation_values.append((similarity_scores[i, j], pred_differences[i, j]))
        
        if correlation_values:
            similarities, pred_diffs = zip(*correlation_values)
            correlation, p_value = stats.spearmanr(similarities, pred_diffs)
        else:
            correlation, p_value = 0, 1
        
        return {
            'similarity_prediction_correlation': correlation,
            'correlation_p_value': p_value,
            'interpretation': self._interpret_individual_fairness(correlation)
        }
    
    def _interpret_demographic_parity(self, disparate_impact: float) -> str:
        """Interprète les résultats de parité démographique"""
        if disparate_impact >= 0.8:
            return "Parité démographique respectée (règle des 80%)"
        elif disparate_impact >= 0.5:
            return "Disparité modérée détectée"
        else:
            return "Disparité importante détectée - biais significatif probable"
    
    def _interpret_equalized_odds(self, tpr_diff: float, fpr_diff: float) -> str:
        """Interprète les résultats d'égalité des chances"""
        max_diff = max(tpr_diff, fpr_diff)
        if max_diff <= 0.05:
            return "Égalité des chances respectée"
        elif max_diff <= 0.1:
            return "Légère différence dans l'égalité des chances"
        else:
            return "Violation significative de l'égalité des chances"
    
    def _interpret_equal_opportunity(self, tpr_diff: float) -> str:
        """Interprète les résultats d'égalité d'opportunité"""
        if tpr_diff <= 0.05:
            return "Égalité d'opportunité respectée"
        elif tpr_diff <= 0.1:
            return "Légère différence dans l'égalité d'opportunité"
        else:
            return "Violation significative de l'égalité d'opportunité"
    
    def _interpret_calibration(self, brier_diff: float, ece_diff: float) -> str:
        """Interprète les résultats de calibration"""
        if max(brier_diff, ece_diff) <= 0.05:
            return "Calibration équitable entre groupes"
        elif max(brier_diff, ece_diff) <= 0.1:
            return "Légères différences de calibration"
        else:
            return "Différences significatives de calibration entre groupes"
    
    def _interpret_individual_fairness(self, correlation: float) -> str:
        """Interprète les résultats d'équité individuelle"""
        if abs(correlation) <= 0.1:
            return "Équité individuelle respectée"
        elif abs(correlation) <= 0.3:
            return "Légère violation de l'équité individuelle"
        else:
            return "Violation significative de l'équité individuelle"


class BiasVisualizer:
    """Classe pour créer des visualisations de biais"""
    
    def __init__(self, config: BiasAnalysisConfig):
        """
        Initialise le visualiseur de biais
        
        Args:
            config: Configuration d'analyse de biais
        """
        self.config = config
        self.figures = {}
    
    def create_fairness_metrics_dashboard(self, fairness_results: Dict[str, Any], 
                                        protected_attr_name: str) -> go.Figure:
        """
        Crée un tableau de bord des métriques d'équité
        
        Args:
            fairness_results: Résultats des métriques d'équité
            protected_attr_name: Nom de l'attribut protégé
            
        Returns:
            Figure Plotly du tableau de bord
        """
        # Création des sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Parité Démographique', 'Égalité des Chances (TPR)', 
                'Égalité des Chances (FPR)', 'Calibration (Brier Score)'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Parité démographique
        if 'demographic_parity' in fairness_results:
            dp_data = fairness_results['demographic_parity']
            groups = list(dp_data['group_rates'].keys())
            rates = list(dp_data['group_rates'].values())
            
            fig.add_trace(
                go.Bar(x=groups, y=rates, name='Taux de prédictions positives',
                       marker_color='lightblue'),
                row=1, col=1
            )
            
            # Ligne de référence pour la parité parfaite
            mean_rate = np.mean(rates)
            fig.add_hline(y=mean_rate, line_dash="dash", line_color="red", 
                         row=1, col=1, annotation_text="Parité parfaite")
        
        # Égalité des chances - TPR
        if 'equalized_odds' in fairness_results:
            eo_data = fairness_results['equalized_odds']
            groups = list(eo_data['group_metrics'].keys())
            tpr_values = [eo_data['group_metrics'][g]['tpr'] for g in groups]
            
            fig.add_trace(
                go.Bar(x=groups, y=tpr_values, name='Taux de vrais positifs',
                       marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Égalité des chances - FPR
        if 'equalized_odds' in fairness_results:
            fpr_values = [eo_data['group_metrics'][g]['fpr'] for g in groups]
            
            fig.add_trace(
                go.Bar(x=groups, y=fpr_values, name='Taux de faux positifs',
                       marker_color='lightcoral'),
                row=2, col=1
            )
        
        # Calibration
        if 'calibration' in fairness_results:
            cal_data = fairness_results['calibration']
            if 'group_calibration' in cal_data:
                groups = list(cal_data['group_calibration'].keys())
                brier_scores = [cal_data['group_calibration'][g]['brier_score'] for g in groups]
                
                fig.add_trace(
                    go.Bar(x=groups, y=brier_scores, name='Brier Score',
                           marker_color='lightyellow'),
                    row=2, col=2
                )
        
        # Mise à jour du layout
        fig.update_layout(
            title=f'Tableau de Bord des Métriques d\'Équité - {protected_attr_name}',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_bias_heatmap(self, bias_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Crée une carte de chaleur des biais
        
        Args:
            bias_results: Résultats de biais par attribut protégé
            
        Returns:
            Figure Plotly de la carte de chaleur
        """
        # Préparation des données pour la heatmap
        metrics = ['disparate_impact', 'tpr_difference', 'fpr_difference', 'ece_difference']
        attributes = list(bias_results.keys())
        
        heatmap_data = []
        for attr in attributes:
            row = []
            attr_results = bias_results[attr]
            
            # Disparate Impact
            dp_value = attr_results.get('demographic_parity', {}).get('disparate_impact', 1.0)
            row.append(1 - dp_value)  # Convertir en "niveau de biais"
            
            # TPR Difference
            tpr_diff = attr_results.get('equalized_odds', {}).get('tpr_difference', 0.0)
            row.append(tpr_diff)
            
            # FPR Difference
            fpr_diff = attr_results.get('equalized_odds', {}).get('fpr_difference', 0.0)
            row.append(fpr_diff)
            
            # ECE Difference
            ece_diff = attr_results.get('calibration', {}).get('ece_difference', 0.0)
            row.append(ece_diff)
            
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Impact Disparate', 'Diff. TPR', 'Diff. FPR', 'Diff. Calibration'],
            y=attributes,
            colorscale='Reds',
            colorbar=dict(title="Niveau de Biais"),
            text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Carte de Chaleur des Biais par Attribut Protégé',
            xaxis_title='Métriques de Biais',
            yaxis_title='Attributs Protégés',
            height=500
        )
        
        return fig
    
    def create_roc_curves_by_group(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  protected_attr: np.ndarray, 
                                  protected_attr_name: str) -> go.Figure:
        """
        Crée les courbes ROC par groupe démographique
        
        Args:
            y_true: Vraies étiquettes
            y_prob: Probabilités prédites
            protected_attr: Attribut protégé
            protected_attr_name: Nom de l'attribut protégé
            
        Returns:
            Figure Plotly des courbes ROC
        """
        fig = go.Figure()
        
        unique_groups = np.unique(protected_attr)
        colors = px.colors.qualitative.Set1[:len(unique_groups)]
        
        for i, group in enumerate(unique_groups):
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_prob_group = y_prob[mask]
            
            if len(np.unique(y_true_group)) > 1:
                fpr, tpr, _ = roc_curve(y_true_group, y_prob_group)
                auc_score = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{group} (AUC = {auc_score:.3f})',
                    line=dict(color=colors[i], width=2)
                ))
        
        # Ligne de référence (classification aléatoire)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Classification Aléatoire',
            line=dict(color='gray', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=f'Courbes ROC par {protected_attr_name}',
            xaxis_title='Taux de Faux Positifs (1 - Spécificité)',
            yaxis_title='Taux de Vrais Positifs (Sensibilité)',
            width=600, height=500,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    def create_calibration_plots(self, y_true: np.ndarray, y_prob: np.ndarray, 
                               protected_attr: np.ndarray, 
                               protected_attr_name: str) -> go.Figure:
        """
        Crée les graphiques de calibration par groupe
        
        Args:
            y_true: Vraies étiquettes
            y_prob: Probabilités prédites
            protected_attr: Attribut protégé
            protected_attr_name: Nom de l'attribut protégé
            
        Returns:
            Figure Plotly des graphiques de calibration
        """
        fig = go.Figure()
        
        unique_groups = np.unique(protected_attr)
        colors = px.colors.qualitative.Set1[:len(unique_groups)]
        
        for i, group in enumerate(unique_groups):
            mask = protected_attr == group
            y_true_group = y_true[mask]
            y_prob_group = y_prob[mask]
            
            if len(y_true_group) > 0:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_group, y_prob_group, n_bins=10
                )
                
                fig.add_trace(go.Scatter(
                    x=mean_predicted_value, y=fraction_of_positives,
                    mode='lines+markers',
                    name=f'{group}',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8)
                ))
        
        # Ligne de calibration parfaite
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Calibration Parfaite',
            line=dict(color='gray', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=f'Graphiques de Calibration par {protected_attr_name}',
            xaxis_title='Probabilité Prédite Moyenne',
            yaxis_title='Fraction de Positifs',
            width=600, height=500,
            legend=dict(x=0.1, y=0.9)
        )
        
        return fig
    
    def create_disparate_impact_visualization(self, bias_results: Dict[str, Any]) -> go.Figure:
        """
        Crée une visualisation de l'impact disparate
        
        Args:
            bias_results: Résultats d'analyse de biais
            
        Returns:
            Figure Plotly de l'impact disparate
        """
        # Extraction des données d'impact disparate
        impact_data = []
        
        for attr_name, attr_results in bias_results.items():
            if 'demographic_parity' in attr_results:
                dp_data = attr_results['demographic_parity']
                disparate_impact = dp_data.get('disparate_impact', 1.0)
                passes_80_rule = dp_data.get('passes_80_rule', True)
                
                impact_data.append({
                    'attribute': attr_name,
                    'disparate_impact': disparate_impact,
                    'passes_80_rule': passes_80_rule,
                    'color': 'green' if passes_80_rule else 'red'
                })
        
        if not impact_data:
            return go.Figure()
        
        # Création du graphique
        fig = go.Figure()
        
        attributes = [d['attribute'] for d in impact_data]
        impacts = [d['disparate_impact'] for d in impact_data]
        colors = [d['color'] for d in impact_data]
        
        fig.add_trace(go.Bar(
            x=attributes,
            y=impacts,
            marker_color=colors,
            text=[f"{impact:.3f}" for impact in impacts],
            textposition='auto',
            name='Impact Disparate'
        ))
        
        # Ligne de référence pour la règle des 80%
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", 
                     annotation_text="Seuil de 80% (Règle d'Impact Disparate)")
        
        # Ligne de parité parfaite
        fig.add_hline(y=1.0, line_dash="solid", line_color="blue", 
                     annotation_text="Parité Parfaite")
        
        fig.update_layout(
            title='Analyse d\'Impact Disparate par Attribut Protégé',
            xaxis_title='Attributs Protégés',
            yaxis_title='Ratio d\'Impact Disparate',
            yaxis=dict(range=[0, 1.2]),
            height=500,
            showlegend=False
        )
        
        return fig
    
    def save_all_visualizations(self, figures_dict: Dict[str, go.Figure], 
                              timestamp: str) -> List[str]:
        """
        Sauvegarde toutes les visualisations
        
        Args:
            figures_dict: Dictionnaire des figures à sauvegarder
            timestamp: Horodatage pour les noms de fichiers
            
        Returns:
            Liste des chemins de fichiers sauvegardés
        """
        saved_files = []
        
        if not self.config.save_visualizations:
            return saved_files
        
        # Création du répertoire s'il n'existe pas
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        for fig_name, figure in figures_dict.items():
            # Sauvegarde en HTML (interactif)
            html_filename = f"{fig_name}_{timestamp}.html"
            html_path = os.path.join(self.config.results_dir, html_filename)
            figure.write_html(html_path)
            saved_files.append(html_path)
            
            # Sauvegarde en PNG (statique)
            png_filename = f"{fig_name}_{timestamp}.png"
            png_path = os.path.join(self.config.results_dir, png_filename)
            try:
                figure.write_image(png_path, width=1200, height=800)
                saved_files.append(png_path)
            except Exception as e:
                print(f"Attention: Impossible de sauvegarder {png_filename} en PNG: {e}")
        
        return saved_files


class BiasAnalyzer:
    """Analyseur principal pour la détection de biais algorithmique"""
    
    def __init__(self, config: BiasAnalysisConfig = None):
        """
        Initialise l'analyseur de biais
        
        Args:
            config: Configuration d'analyse de biais
        """
        self.config = config or BiasAnalysisConfig()
        self.fairness_calculator = FairnessMetrics()
        self.visualizer = BiasVisualizer(self.config)
        self.statistical_tester = StatisticalTester(self.config.statistical_significance_level)
        self.analysis_results = {}
    
    def calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, protected_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calcule toutes les métriques d'équité pour les attributs protégés
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            y_prob: Probabilités prédites
            protected_attrs: Dictionnaire des attributs protégés
            
        Returns:
            Dictionnaire complet des métriques d'équité
        """
        fairness_results = {}
        
        for attr_name, attr_values in protected_attrs.items():
            print(f"Calcul des métriques d'équité pour: {attr_name}")
            
            attr_results = {}
            
            # Parité démographique
            attr_results['demographic_parity'] = self.fairness_calculator.demographic_parity(
                y_pred, attr_values
            )
            
            # Égalité des chances
            attr_results['equalized_odds'] = self.fairness_calculator.equalized_odds(
                y_true, y_pred, attr_values
            )
            
            # Égalité d'opportunité
            attr_results['equal_opportunity'] = self.fairness_calculator.equal_opportunity(
                y_true, y_pred, attr_values
            )
            
            # Métriques de calibration
            attr_results['calibration'] = self.fairness_calculator.calibration_metrics(
                y_true, y_prob, attr_values
            )
            
            fairness_results[attr_name] = attr_results
        
        return fairness_results
    
    def detect_bias_patterns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray, protected_attrs: Dict[str, np.ndarray], 
                           feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Détecte les patterns de biais dans les prédictions
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            y_prob: Probabilités prédites
            protected_attrs: Dictionnaire des attributs protégés
            feature_names: Noms des caractéristiques
            
        Returns:
            Dictionnaire des patterns de biais détectés
        """
        bias_patterns = {
            'overall_bias_detected': False,
            'bias_by_attribute': {},
            'intersectional_bias': {},
            'statistical_tests': {}
        }
        
        # Analyse par attribut protégé
        for attr_name, attr_values in protected_attrs.items():
            attr_bias = self._analyze_attribute_bias(
                y_true, y_pred, y_prob, attr_values, attr_name
            )
            bias_patterns['bias_by_attribute'][attr_name] = attr_bias
            
            if attr_bias['bias_detected']:
                bias_patterns['overall_bias_detected'] = True
        
        # Analyse intersectionnelle (race × sexe)
        if 'race' in protected_attrs and 'sex' in protected_attrs:
            intersectional_bias = self._analyze_intersectional_bias(
                y_true, y_pred, y_prob, 
                protected_attrs['race'], protected_attrs['sex']
            )
            bias_patterns['intersectional_bias']['race_sex'] = intersectional_bias
        
        return bias_patterns
    
    def analyze_disparate_impact(self, y_pred: np.ndarray, 
                               protected_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyse l'impact disparate selon la règle des 80%
        
        Args:
            y_pred: Prédictions binaires
            protected_attrs: Dictionnaire des attributs protégés
            
        Returns:
            Dictionnaire des résultats d'impact disparate
        """
        disparate_impact_results = {}
        
        for attr_name, attr_values in protected_attrs.items():
            # Calcul des taux de sélection par groupe
            unique_groups = np.unique(attr_values)
            selection_rates = {}
            
            for group in unique_groups:
                mask = attr_values == group
                selection_rate = np.mean(y_pred[mask])
                selection_rates[str(group)] = selection_rate
            
            # Calcul de l'impact disparate (ratio min/max)
            rates_values = list(selection_rates.values())
            if max(rates_values) > 0:
                disparate_impact = min(rates_values) / max(rates_values)
            else:
                disparate_impact = 1.0
            
            # Test de la règle des 80%
            passes_80_rule = disparate_impact >= self.config.fairness_threshold
            
            # Test statistique de différence de proportions
            groups = list(selection_rates.keys())
            if len(groups) >= 2:
                group1_mask = attr_values == groups[0]
                group2_mask = attr_values == groups[1]
                
                count1 = np.sum(y_pred[group1_mask])
                count2 = np.sum(y_pred[group2_mask])
                nobs1 = np.sum(group1_mask)
                nobs2 = np.sum(group2_mask)
                
                stat_test = self.statistical_tester.proportion_test(
                    count1, nobs1, count2, nobs2
                )
            else:
                stat_test = {'is_significant': False, 'p_value': 1.0}
            
            disparate_impact_results[attr_name] = {
                'selection_rates': selection_rates,
                'disparate_impact_ratio': disparate_impact,
                'passes_80_rule': passes_80_rule,
                'statistical_test': stat_test,
                'interpretation': self._interpret_disparate_impact(
                    disparate_impact, passes_80_rule, stat_test['is_significant']
                )
            }
        
        return disparate_impact_results
    
    def compare_group_outcomes(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray, protected_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare les résultats entre groupes démographiques
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            y_prob: Probabilités prédites
            protected_attrs: Dictionnaire des attributs protégés
            
        Returns:
            Dictionnaire de comparaison des résultats par groupe
        """
        group_comparisons = {}
        
        for attr_name, attr_values in protected_attrs.items():
            unique_groups = np.unique(attr_values)
            group_outcomes = {}
            
            for group in unique_groups:
                mask = attr_values == group
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                group_y_prob = y_prob[mask]
                
                # Métriques de performance
                if len(np.unique(group_y_true)) > 1:
                    tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
                    
                    outcomes = {
                        'sample_size': len(group_y_true),
                        'base_rate': np.mean(group_y_true),
                        'prediction_rate': np.mean(group_y_pred),
                        'accuracy': (tp + tn) / (tp + tn + fp + fn),
                        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
                        'mean_predicted_prob': np.mean(group_y_prob)
                    }
                    
                    # F1-score
                    if outcomes['precision'] + outcomes['recall'] > 0:
                        outcomes['f1_score'] = 2 * (outcomes['precision'] * outcomes['recall']) / \
                                             (outcomes['precision'] + outcomes['recall'])
                    else:
                        outcomes['f1_score'] = 0
                else:
                    outcomes = {
                        'sample_size': len(group_y_true),
                        'base_rate': np.mean(group_y_true),
                        'prediction_rate': np.mean(group_y_pred),
                        'mean_predicted_prob': np.mean(group_y_prob)
                    }
                
                group_outcomes[str(group)] = outcomes
            
            # Calcul des différences entre groupes
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'fpr', 'fnr']
            group_differences = {}
            
            for metric in metrics_to_compare:
                values = [outcomes.get(metric, 0) for outcomes in group_outcomes.values() 
                         if metric in outcomes]
                if values:
                    group_differences[f'{metric}_difference'] = max(values) - min(values)
            
            group_comparisons[attr_name] = {
                'group_outcomes': group_outcomes,
                'group_differences': group_differences
            }
        
        return group_comparisons
    
    def generate_bias_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray, protected_attrs: Dict[str, np.ndarray],
                           model_name: str = "COMPAS", 
                           dataset_description: str = "Dataset COMPAS") -> Dict[str, Any]:
        """
        Génère un rapport complet d'analyse de biais
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions binaires
            y_prob: Probabilités prédites
            protected_attrs: Dictionnaire des attributs protégés
            model_name: Nom du modèle analysé
            dataset_description: Description du dataset
            
        Returns:
            Rapport complet d'analyse de biais
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Génération du rapport d'analyse de biais...")
        print("=" * 60)
        
        # Calcul de toutes les métriques
        fairness_metrics = self.calculate_fairness_metrics(
            y_true, y_pred, y_prob, protected_attrs
        )
        
        bias_patterns = self.detect_bias_patterns(
            y_true, y_pred, y_prob, protected_attrs
        )
        
        disparate_impact = self.analyze_disparate_impact(y_pred, protected_attrs)
        
        group_comparisons = self.compare_group_outcomes(
            y_true, y_pred, y_prob, protected_attrs
        )
        
        # Génération des visualisations
        visualizations = self._generate_visualizations(
            y_true, y_pred, y_prob, protected_attrs, fairness_metrics, timestamp
        )
        
        # Recommandations de mitigation
        mitigation_recommendations = self._generate_mitigation_recommendations(
            fairness_metrics, bias_patterns, disparate_impact
        )
        
        # Benchmark avec les résultats ProPublica
        propublica_comparison = self._benchmark_against_propublica(
            fairness_metrics, bias_patterns
        )
        
        # Compilation du rapport final
        bias_report = {
            'metadata': {
                'model_name': model_name,
                'dataset_description': dataset_description,
                'analysis_timestamp': timestamp,
                'sample_size': len(y_true),
                'protected_attributes_analyzed': list(protected_attrs.keys()),
                'configuration': {
                    'fairness_threshold': self.config.fairness_threshold,
                    'significance_level': self.config.statistical_significance_level
                }
            },
            'executive_summary': self._generate_executive_summary(
                bias_patterns, disparate_impact, fairness_metrics
            ),
            'fairness_metrics': fairness_metrics,
            'bias_patterns': bias_patterns,
            'disparate_impact_analysis': disparate_impact,
            'group_comparisons': group_comparisons,
            'propublica_benchmark': propublica_comparison,
            'mitigation_recommendations': mitigation_recommendations,
            'visualizations': visualizations,
            'detailed_findings': self._generate_detailed_findings(
                fairness_metrics, bias_patterns, disparate_impact
            )
        }
        
        # Sauvegarde du rapport
        self._save_bias_report(bias_report, timestamp)
        
        return bias_report
    
    def visualize_bias_metrics(self, bias_report: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Crée toutes les visualisations de métriques de biais
        
        Args:
            bias_report: Rapport d'analyse de biais
            
        Returns:
            Dictionnaire des figures de visualisation
        """
        figures = {}
        
        # Extraction des données du rapport
        fairness_metrics = bias_report['fairness_metrics']
        protected_attrs_names = list(fairness_metrics.keys())
        
        # Dashboard des métriques d'équité pour chaque attribut
        for attr_name in protected_attrs_names:
            dashboard_fig = self.visualizer.create_fairness_metrics_dashboard(
                fairness_metrics[attr_name], attr_name
            )
            figures[f'fairness_dashboard_{attr_name}'] = dashboard_fig
        
        # Carte de chaleur des biais
        heatmap_fig = self.visualizer.create_bias_heatmap(fairness_metrics)
        figures['bias_heatmap'] = heatmap_fig
        
        # Visualisation de l'impact disparate
        disparate_impact_fig = self.visualizer.create_disparate_impact_visualization(
            bias_report['disparate_impact_analysis']
        )
        figures['disparate_impact'] = disparate_impact_fig
        
        return figures
    
    def _analyze_attribute_bias(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: np.ndarray, attr_values: np.ndarray, 
                              attr_name: str) -> Dict[str, Any]:
        """Analyse le biais pour un attribut protégé spécifique"""
        bias_indicators = {
            'bias_detected': False,
            'bias_level': 'none',  # none, low, moderate, high
            'bias_sources': [],
            'statistical_tests': {}
        }
        
        # Test de parité démographique
        dp_result = self.fairness_calculator.demographic_parity(y_pred, attr_values)
        if dp_result['disparate_impact'] < 0.8:
            bias_indicators['bias_detected'] = True
            bias_indicators['bias_sources'].append('demographic_parity')
        
        # Test d'égalité des chances
        eo_result = self.fairness_calculator.equalized_odds(y_true, y_pred, attr_values)
        if eo_result['equalized_odds_difference'] > 0.1:
            bias_indicators['bias_detected'] = True
            bias_indicators['bias_sources'].append('equalized_odds')
        
        # Niveau de biais global
        bias_score = 0
        if dp_result['disparate_impact'] < 0.5:
            bias_score += 3
        elif dp_result['disparate_impact'] < 0.8:
            bias_score += 1
        
        if eo_result['equalized_odds_difference'] > 0.2:
            bias_score += 3
        elif eo_result['equalized_odds_difference'] > 0.1:
            bias_score += 1
        
        if bias_score >= 4:
            bias_indicators['bias_level'] = 'high'
        elif bias_score >= 2:
            bias_indicators['bias_level'] = 'moderate'
        elif bias_score >= 1:
            bias_indicators['bias_level'] = 'low'
        
        return bias_indicators
    
    def _analyze_intersectional_bias(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: np.ndarray, race_attr: np.ndarray, 
                                   sex_attr: np.ndarray) -> Dict[str, Any]:
        """Analyse le biais intersectionnel race × sexe"""
        # Création de l'attribut intersectionnel
        unique_races = np.unique(race_attr)
        unique_sexes = np.unique(sex_attr)
        
        intersectional_outcomes = {}
        
        for race in unique_races:
            for sex in unique_sexes:
                mask = (race_attr == race) & (sex_attr == sex)
                if np.sum(mask) > 10:  # Seuil minimum pour l'analyse
                    group_name = f"{race}_{sex}"
                    
                    group_y_true = y_true[mask]
                    group_y_pred = y_pred[mask]
                    
                    intersectional_outcomes[group_name] = {
                        'sample_size': np.sum(mask),
                        'prediction_rate': np.mean(group_y_pred),
                        'base_rate': np.mean(group_y_true)
                    }
        
        # Calcul de la variance intersectionnelle
        prediction_rates = [outcome['prediction_rate'] 
                          for outcome in intersectional_outcomes.values()]
        variance_intersectional = np.var(prediction_rates) if prediction_rates else 0
        
        return {
            'intersectional_outcomes': intersectional_outcomes,
            'prediction_rate_variance': variance_intersectional,
            'high_variance_detected': variance_intersectional > 0.01
        }
    
    def _generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray, protected_attrs: Dict[str, np.ndarray],
                               fairness_metrics: Dict[str, Any], timestamp: str) -> Dict[str, str]:
        """Génère et sauvegarde toutes les visualisations"""
        visualizations = {}
        
        try:
            # Courbes ROC par groupe
            for attr_name, attr_values in protected_attrs.items():
                roc_fig = self.visualizer.create_roc_curves_by_group(
                    y_true, y_prob, attr_values, attr_name
                )
                
                # Sauvegarde
                filename = f"roc_curves_{attr_name}_{timestamp}"
                html_path = os.path.join(self.config.results_dir, f"{filename}.html")
                roc_fig.write_html(html_path)
                visualizations[f'roc_curves_{attr_name}'] = html_path
            
            # Graphiques de calibration
            for attr_name, attr_values in protected_attrs.items():
                cal_fig = self.visualizer.create_calibration_plots(
                    y_true, y_prob, attr_values, attr_name
                )
                
                filename = f"calibration_plots_{attr_name}_{timestamp}"
                html_path = os.path.join(self.config.results_dir, f"{filename}.html")
                cal_fig.write_html(html_path)
                visualizations[f'calibration_plots_{attr_name}'] = html_path
            
            # Autres visualisations via le rapport principal
            figures = self.visualize_bias_metrics({
                'fairness_metrics': fairness_metrics,
                'disparate_impact_analysis': self.analyze_disparate_impact(y_pred, protected_attrs)
            })
            
            saved_files = self.visualizer.save_all_visualizations(figures, timestamp)
            for i, filepath in enumerate(saved_files):
                visualizations[f'figure_{i}'] = filepath
                
        except Exception as e:
            print(f"Erreur lors de la génération des visualisations: {e}")
        
        return visualizations
    
    def _generate_mitigation_recommendations(self, fairness_metrics: Dict[str, Any], 
                                          bias_patterns: Dict[str, Any],
                                          disparate_impact: Dict[str, Any]) -> List[Dict[str, str]]:
        """Génère des recommandations de mitigation des biais"""
        recommendations = []
        
        # Recommandations basées sur l'impact disparate
        for attr_name, impact_data in disparate_impact.items():
            if not impact_data['passes_80_rule']:
                recommendations.append({
                    'type': 'Impact Disparate',
                    'attribute': attr_name,
                    'severity': 'Haute' if impact_data['disparate_impact_ratio'] < 0.5 else 'Modérée',
                    'recommendation': f"Réviser les critères de sélection pour {attr_name}. "
                                    f"Le ratio d'impact disparate ({impact_data['disparate_impact_ratio']:.3f}) "
                                    f"est en dessous du seuil de 80%. Considérer l'utilisation de techniques "
                                    f"de pré-traitement ou post-traitement pour équilibrer les taux de sélection.",
                    'technical_approach': [
                        "Pré-traitement: Rééchantillonnage, suppression d'attributs corrélés",
                        "In-processing: Contraintes d'équité dans l'optimisation du modèle",
                        "Post-traitement: Ajustement des seuils de décision par groupe"
                    ]
                })
        
        # Recommandations basées sur l'égalité des chances
        for attr_name, attr_metrics in fairness_metrics.items():
            if 'equalized_odds' in attr_metrics:
                eo_data = attr_metrics['equalized_odds']
                if eo_data['equalized_odds_difference'] > 0.1:
                    recommendations.append({
                        'type': 'Égalité des Chances',
                        'attribute': attr_name,
                        'severity': 'Haute' if eo_data['equalized_odds_difference'] > 0.2 else 'Modérée',
                        'recommendation': f"Différence significative dans l'égalité des chances pour {attr_name} "
                                        f"({eo_data['equalized_odds_difference']:.3f}). Implémenter des "
                                        f"techniques de calibration équitable.",
                        'technical_approach': [
                            "Calibration post-hoc par groupe démographique",
                            "Entraînement multi-objectif avec contraintes d'équité",
                            "Utilisation d'algorithmes d'équité comme FairLearn"
                        ]
                    })
        
        # Recommandations générales si biais détecté
        if bias_patterns['overall_bias_detected']:
            recommendations.append({
                'type': 'Recommandation Générale',
                'attribute': 'Tous',
                'severity': 'Critique',
                'recommendation': "Biais algorithmique détecté dans le système COMPAS. "
                                "Mise en place urgente d'un framework de gouvernance algorithmique.",
                'technical_approach': [
                    "Audit algorithmique régulier avec métriques d'équité",
                    "Formation des utilisateurs sur les biais algorithmiques",
                    "Mise en place de systèmes d'alerte automatique",
                    "Documentation transparente des limitations du modèle"
                ]
            })
        
        return recommendations
    
    def _benchmark_against_propublica(self, fairness_metrics: Dict[str, Any], 
                                    bias_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Compare les résultats avec les découvertes de ProPublica"""
        
        # Données de référence ProPublica (approximatives basées sur leur enquête)
        propublica_reference = {
            'demographic_parity': {
                'African-American_vs_Caucasian_disparate_impact': 0.55,  # Très biaisé
                'finding': "Les défendeurs afro-américains étaient presque deux fois plus "
                          "susceptibles d'être étiquetés à haut risque que les défendeurs blancs"
            },
            'predictive_parity': {
                'African-American_accuracy': 0.59,
                'Caucasian_accuracy': 0.66,
                'finding': "COMPAS était plus précis pour prédire la récidive chez les défendeurs blancs"
            },
            'false_positive_rates': {
                'African-American_fpr': 0.45,
                'Caucasian_fpr': 0.23,
                'finding': "Taux de faux positifs presque deux fois plus élevé pour les Afro-Américains"
            }
        }
        
        comparison_results = {
            'comparison_with_propublica': {},
            'consistency_with_findings': 'unknown',
            'discrepancies': []
        }
        
        # Comparaison de la parité démographique
        if 'race' in fairness_metrics:
            race_metrics = fairness_metrics['race']
            if 'demographic_parity' in race_metrics:
                dp_data = race_metrics['demographic_parity']
                our_disparate_impact = dp_data.get('disparate_impact', 1.0)
                
                comparison_results['comparison_with_propublica']['disparate_impact'] = {
                    'our_finding': our_disparate_impact,
                    'propublica_reference': propublica_reference['demographic_parity']['African-American_vs_Caucasian_disparate_impact'],
                    'consistent': our_disparate_impact < 0.8,
                    'interpretation': "Cohérent avec ProPublica" if our_disparate_impact < 0.8 
                                   else "Différent des conclusions de ProPublica"
                }
        
        # Évaluation de la cohérence globale
        consistent_findings = 0
        total_comparisons = 0
        
        for comparison in comparison_results['comparison_with_propublica'].values():
            if isinstance(comparison, dict) and 'consistent' in comparison:
                total_comparisons += 1
                if comparison['consistent']:
                    consistent_findings += 1
        
        if total_comparisons > 0:
            consistency_ratio = consistent_findings / total_comparisons
            if consistency_ratio >= 0.7:
                comparison_results['consistency_with_findings'] = 'high'
            elif consistency_ratio >= 0.3:
                comparison_results['consistency_with_findings'] = 'moderate'
            else:
                comparison_results['consistency_with_findings'] = 'low'
        
        return comparison_results
    
    def _generate_executive_summary(self, bias_patterns: Dict[str, Any], 
                                  disparate_impact: Dict[str, Any],
                                  fairness_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Génère un résumé exécutif de l'analyse de biais"""
        
        # Comptage des violations
        total_violations = 0
        high_severity_violations = 0
        
        for attr_name, impact_data in disparate_impact.items():
            if not impact_data['passes_80_rule']:
                total_violations += 1
                if impact_data['disparate_impact_ratio'] < 0.5:
                    high_severity_violations += 1
        
        # Détermination du niveau de risque global
        if high_severity_violations > 0:
            risk_level = "CRITIQUE"
            risk_color = "rouge"
        elif total_violations > 0:
            risk_level = "ÉLEVÉ"
            risk_color = "orange"
        elif bias_patterns['overall_bias_detected']:
            risk_level = "MODÉRÉ"
            risk_color = "jaune"
        else:
            risk_level = "FAIBLE"
            risk_color = "vert"
        
        executive_summary = {
            'risk_level': risk_level,
            'key_findings': f"Analyse de {len(fairness_metrics)} attributs protégés révèle "
                          f"{total_violations} violation(s) de la règle d'impact disparate des 80%. "
                          f"{high_severity_violations} violation(s) de haute sévérité détectée(s).",
            
            'main_concerns': [
                "Disparités significatives dans les taux de prédictions positives entre groupes raciaux" 
                if any(not d['passes_80_rule'] for d in disparate_impact.values()),
                "Violations de l'égalité des chances détectées",
                "Biais systémique confirmé, cohérent avec les découvertes de ProPublica"
            ][0:2],  # Limiter aux 2 principales préoccupations
            
            'immediate_actions': [
                f"Suspension recommandée de l'utilisation du système pour les décisions critiques" 
                if risk_level == "CRITIQUE" else "Audit approfondi recommandé",
                "Mise en place immédiate de mesures de mitigation des biais",
                "Formation obligatoire du personnel sur les biais algorithmiques"
            ],
            
            'business_impact': f"Risque {risk_level.lower()} de discrimination algorithmique. "
                             f"Impact potentiel sur la conformité réglementaire et la réputation.",
            
            'recommendation_priority': "URGENTE" if risk_level in ["CRITIQUE", "ÉLEVÉ"] else "NORMALE"
        }
        
        return executive_summary
    
    def _generate_detailed_findings(self, fairness_metrics: Dict[str, Any], 
                                  bias_patterns: Dict[str, Any],
                                  disparate_impact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère des conclusions détaillées pour chaque attribut"""
        detailed_findings = []
        
        for attr_name in fairness_metrics.keys():
            finding = {
                'attribute': attr_name,
                'summary': "",
                'metrics_analysis': {},
                'bias_detected': False,
                'severity': "none",
                'specific_issues': [],
                'technical_details': {}
            }
            
            # Analyse de l'impact disparate
            if attr_name in disparate_impact:
                impact_data = disparate_impact[attr_name]
                finding['metrics_analysis']['disparate_impact'] = {
                    'ratio': impact_data['disparate_impact_ratio'],
                    'passes_80_rule': impact_data['passes_80_rule'],
                    'selection_rates': impact_data['selection_rates']
                }
                
                if not impact_data['passes_80_rule']:
                    finding['bias_detected'] = True
                    finding['specific_issues'].append(
                        f"Violation de la règle d'impact disparate (ratio: {impact_data['disparate_impact_ratio']:.3f})"
                    )
            
            # Analyse des métriques d'équité
            attr_fairness = fairness_metrics[attr_name]
            
            if 'equalized_odds' in attr_fairness:
                eo_data = attr_fairness['equalized_odds']
                finding['metrics_analysis']['equalized_odds'] = {
                    'difference': eo_data['equalized_odds_difference'],
                    'tpr_difference': eo_data['tpr_difference'],
                    'fpr_difference': eo_data['fpr_difference']
                }
                
                if eo_data['equalized_odds_difference'] > 0.1:
                    finding['bias_detected'] = True
                    finding['specific_issues'].append(
                        f"Violation de l'égalité des chances (diff: {eo_data['equalized_odds_difference']:.3f})"
                    )
            
            # Détermination de la sévérité
            if finding['bias_detected']:
                violation_count = len(finding['specific_issues'])
                if violation_count >= 2:
                    finding['severity'] = "high"
                else:
                    finding['severity'] = "moderate"
            
            # Génération du résumé
            if finding['bias_detected']:
                finding['summary'] = f"Biais détecté pour {attr_name} avec {len(finding['specific_issues'])} violation(s) d'équité"
            else:
                finding['summary'] = f"Aucun biais significatif détecté pour {attr_name}"
            
            detailed_findings.append(finding)
        
        return detailed_findings
    
    def _interpret_disparate_impact(self, ratio: float, passes_80_rule: bool, 
                                  is_statistically_significant: bool) -> str:
        """Interprète les résultats d'impact disparate"""
        interpretation = []
        
        if passes_80_rule:
            interpretation.append("Respect de la règle d'impact disparate des 80%")
        else:
            interpretation.append(f"Violation de la règle des 80% (ratio: {ratio:.3f})")
        
        if is_statistically_significant:
            interpretation.append("Différence statistiquement significative détectée")
        else:
            interpretation.append("Différence non statistiquement significative")
        
        # Niveau de sévérité
        if ratio < 0.5:
            interpretation.append("SÉVÉRITÉ CRITIQUE - Disparité extrême")
        elif ratio < 0.8:
            interpretation.append("SÉVÉRITÉ ÉLEVÉE - Disparité importante")
        else:
            interpretation.append("Disparité acceptable")
        
        return " | ".join(interpretation)
    
    def _save_bias_report(self, bias_report: Dict[str, Any], timestamp: str) -> str:
        """Sauvegarde le rapport d'analyse de biais"""
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Sauvegarde JSON
        json_filename = f"bias_analysis_report_{timestamp}.json"
        json_path = os.path.join(self.config.results_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bias_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Rapport d'analyse de biais sauvegardé: {json_path}")
        return json_path


# Fonctions utilitaires pour l'intégration avec le pipeline principal

def prepare_protected_attributes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Prépare les attributs protégés à partir du DataFrame COMPAS
    
    Args:
        df: DataFrame COMPAS avec les données processées
        
    Returns:
        Dictionnaire des attributs protégés
    """
    protected_attrs = {}
    
    # Attribut race
    if 'race_African-American' in df.columns:
        # Reconstruction de l'attribut race à partir des colonnes one-hot
        race_columns = [col for col in df.columns if col.startswith('race_')]
        race_values = []
        
        for idx, row in df.iterrows():
            for col in race_columns:
                if row[col] == 1 or row[col] == True:
                    race_values.append(col.replace('race_', ''))
                    break
            else:
                race_values.append('Unknown')
        
        protected_attrs['race'] = np.array(race_values)
    
    # Attribut sexe
    if 'sex_Male' in df.columns:
        sex_values = ['Male' if row['sex_Male'] else 'Female' for _, row in df.iterrows()]
        protected_attrs['sex'] = np.array(sex_values)
    
    # Attribut âge (basé sur age_cat ou création de groupes)
    if 'age_cat' in df.columns:
        protected_attrs['age_group'] = df['age_cat'].values
    elif 'age' in df.columns:
        # Création de groupes d'âge
        age_groups = pd.cut(df['age'], bins=[0, 25, 35, 45, 100], 
                           labels=['18-25', '26-35', '36-45', '46+'])
        protected_attrs['age_group'] = age_groups.values
    
    return protected_attrs


def run_comprehensive_bias_analysis(df: pd.DataFrame, 
                                   y_true_col: str = 'two_year_recid',
                                   y_pred_col: str = None,
                                   y_prob_col: str = None,
                                   model_predictions: np.ndarray = None,
                                   model_probabilities: np.ndarray = None) -> Dict[str, Any]:
    """
    Lance une analyse complète de biais sur les données COMPAS
    
    Args:
        df: DataFrame avec les données COMPAS
        y_true_col: Nom de la colonne des vraies étiquettes
        y_pred_col: Nom de la colonne des prédictions (optionnel)
        y_prob_col: Nom de la colonne des probabilités (optionnel)
        model_predictions: Prédictions du modèle (optionnel)
        model_probabilities: Probabilités du modèle (optionnel)
        
    Returns:
        Rapport complet d'analyse de biais
    """
    # Préparation des données
    y_true = df[y_true_col].values
    
    # Utilisation des prédictions fournies ou des colonnes du DataFrame
    if model_predictions is not None:
        y_pred = model_predictions
    elif y_pred_col and y_pred_col in df.columns:
        y_pred = df[y_pred_col].values
    else:
        # Utilisation du score COMPAS comme prédiction de référence
        y_pred = (df['decile_score'] >= 5).astype(int).values  # Seuil à 5
    
    if model_probabilities is not None:
        y_prob = model_probabilities
    elif y_prob_col and y_prob_col in df.columns:
        y_prob = df[y_prob_col].values
    else:
        # Conversion du score COMPAS en probabilité approximative
        y_prob = df['decile_score'].values / 10.0
    
    # Préparation des attributs protégés
    protected_attrs = prepare_protected_attributes(df)
    
    # Configuration de l'analyse
    config = BiasAnalysisConfig(
        protected_attributes=list(protected_attrs.keys()),
        fairness_threshold=0.8,
        statistical_significance_level=0.05,
        save_visualizations=True,
        results_dir="data/results/bias_analysis"
    )
    
    # Lancement de l'analyse
    analyzer = BiasAnalyzer(config)
    
    bias_report = analyzer.generate_bias_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        protected_attrs=protected_attrs,
        model_name="COMPAS Risk Assessment",
        dataset_description="Dataset COMPAS - Analyse de biais algorithmique"
    )
    
    return bias_report


# Exemple d'utilisation
if __name__ == "__main__":
    print("Framework d'Analyse de Biais COMPAS")
    print("=" * 50)
    print("Ce module fournit une suite complète d'outils pour l'analyse de biais algorithmique.")
    print("Utilisation recommandée:")
    print("1. Charger les données COMPAS")
    print("2. Appeler run_comprehensive_bias_analysis()")
    print("3. Examiner le rapport généré")
    print("4. Implémenter les recommandations de mitigation")