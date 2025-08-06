"""
Module d'évaluation de l'équité pour le projet COMPAS

Ce module fournit des outils pour évaluer l'efficacité des stratégies de mitigation
des biais et comparer les performances avant/après mitigation.

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairnessEvaluator:
    """
    Évaluateur d'équité pour comparer les performances avant/après mitigation des biais.
    
    Cette classe permet d'évaluer l'efficacité des stratégies de mitigation et de
    fournir des recommandations pour l'amélioration de l'équité des modèles COMPAS.
    """
    
    def __init__(self, results_dir: str = "data/results/fairness_evaluation"):
        """
        Initialise l'évaluateur d'équité.
        
        Args:
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        self.baseline_results = {}
        self.mitigated_results = {}
        self.evaluation_results = {}
        
        # Créer le répertoire de résultats
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Évaluateur d'équité initialisé - Résultats: {self.results_dir}")
    
    def load_baseline_results(self, bias_analyzer_results: Dict[str, Any]) -> None:
        """
        Charge les résultats d'analyse de biais avant mitigation (baseline).
        
        Args:
            bias_analyzer_results: Résultats de CompasBiasAnalyzer
        """
        self.baseline_results = {
            'fairness_metrics': bias_analyzer_results.get('fairness_metrics', {}),
            'bias_patterns': bias_analyzer_results.get('bias_patterns', {}),
            'group_comparison': bias_analyzer_results.get('group_comparison', pd.DataFrame())
        }
        
        logger.info("Résultats baseline chargés pour évaluation d'équité")
    
    def load_mitigated_results(self, mitigated_bias_results: Dict[str, Any]) -> None:
        """
        Charge les résultats après mitigation des biais.
        
        Args:
            mitigated_bias_results: Résultats après mitigation
        """
        self.mitigated_results = {
            'fairness_metrics': mitigated_bias_results.get('fairness_metrics', {}),
            'bias_patterns': mitigated_bias_results.get('bias_patterns', {}),
            'group_comparison': mitigated_bias_results.get('group_comparison', pd.DataFrame())
        }
        
        logger.info("Résultats après mitigation chargés pour évaluation d'équité")
    
    def evaluate_mitigation_effectiveness(self, protected_attribute: str = 'race') -> Dict[str, Any]:
        """
        Évalue l'efficacité des stratégies de mitigation des biais.
        
        Args:
            protected_attribute: Attribut protégé évalué
            
        Returns:
            Dictionnaire avec l'évaluation détaillée
        """
        if not self.baseline_results or not self.mitigated_results:
            raise ValueError("Résultats baseline et mitigated non chargés.")
        
        effectiveness_results = {}
        
        # Comparer les métriques d'équité
        fairness_comparison = self._compare_fairness_metrics(protected_attribute)
        effectiveness_results['fairness_improvement'] = fairness_comparison
        
        # Comparer les performances des modèles
        performance_comparison = self._compare_model_performance(protected_attribute)
        effectiveness_results['performance_impact'] = performance_comparison
        
        # Analyser les trade-offs performance vs équité
        tradeoff_analysis = self._analyze_performance_fairness_tradeoff(protected_attribute)
        effectiveness_results['tradeoff_analysis'] = tradeoff_analysis
        
        # Évaluer la réduction des patterns de biais
        bias_reduction = self._evaluate_bias_pattern_reduction(protected_attribute)
        effectiveness_results['bias_reduction'] = bias_reduction
        
        # Score global d'efficacité
        effectiveness_score = self._calculate_effectiveness_score(effectiveness_results)
        effectiveness_results['effectiveness_score'] = effectiveness_score
        
        # Recommandations
        recommendations = self._generate_recommendations(effectiveness_results)
        effectiveness_results['recommendations'] = recommendations
        
        self.evaluation_results[protected_attribute] = effectiveness_results
        
        # Sauvegarder les résultats
        self._save_evaluation_results(protected_attribute)
        
        return effectiveness_results
    
    def _compare_fairness_metrics(self, protected_attribute: str) -> Dict[str, Any]:
        """Compare les métriques d'équité avant/après mitigation."""
        baseline_metrics = self.baseline_results['fairness_metrics'].get(protected_attribute, {})
        mitigated_metrics = self.mitigated_results['fairness_metrics'].get(protected_attribute, {})
        
        comparison_results = {}
        
        # Métriques clés à comparer
        key_metrics = [
            'demographic_parity_difference',
            'equal_opportunity_difference',
            'equalized_odds_difference',
            'disparate_impact_ratio',
            'calibration_difference'
        ]
        
        for model_name in baseline_metrics.keys():
            if model_name not in mitigated_metrics:
                continue
            
            model_comparison = {}
            baseline_model = baseline_metrics[model_name]
            mitigated_model = mitigated_metrics[model_name]
            
            for metric in key_metrics:
                baseline_value = baseline_model.get(metric, 0)
                mitigated_value = mitigated_model.get(metric, 0)
                
                # Calculer l'amélioration
                if 'ratio' in metric:
                    # Pour les ratios, calculer la réduction de l'écart à 1
                    baseline_deviation = abs(baseline_value - 1.0)
                    mitigated_deviation = abs(mitigated_value - 1.0)
                    improvement = baseline_deviation - mitigated_deviation
                    improvement_percent = (improvement / (baseline_deviation + 1e-8)) * 100
                else:
                    # Pour les différences, calculer la réduction
                    improvement = abs(baseline_value) - abs(mitigated_value)
                    improvement_percent = (improvement / (abs(baseline_value) + 1e-8)) * 100
                
                model_comparison[metric] = {
                    'baseline': baseline_value,
                    'mitigated': mitigated_value,
                    'improvement': improvement,
                    'improvement_percent': improvement_percent,
                    'improved': improvement > 0
                }
            
            # Score d'amélioration global pour ce modèle
            improvements = [v['improvement_percent'] for v in model_comparison.values()]
            model_comparison['overall_fairness_improvement'] = np.mean(improvements)
            
            comparison_results[model_name] = model_comparison
        
        return comparison_results
    
    def _compare_model_performance(self, protected_attribute: str) -> Dict[str, Any]:
        """Compare les performances des modèles avant/après mitigation."""
        baseline_comparison = self.baseline_results.get('group_comparison', pd.DataFrame())
        mitigated_comparison = self.mitigated_results.get('group_comparison', pd.DataFrame())
        
        if baseline_comparison.empty or mitigated_comparison.empty:
            return {}
        
        performance_comparison = {}
        
        # Métriques de performance à comparer
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        models = baseline_comparison['model'].unique()
        groups = baseline_comparison['group'].unique()
        
        for model in models:
            model_results = {}
            
            for group in groups:
                group_results = {}
                
                # Données baseline et mitigated pour ce modèle et groupe
                baseline_group = baseline_comparison[
                    (baseline_comparison['model'] == model) & 
                    (baseline_comparison['group'] == group)
                ]
                mitigated_group = mitigated_comparison[
                    (mitigated_comparison['model'] == model) & 
                    (mitigated_comparison['group'] == group)
                ]
                
                if baseline_group.empty or mitigated_group.empty:
                    continue
                
                for metric in performance_metrics:
                    if metric not in baseline_group.columns:
                        continue
                    
                    baseline_value = baseline_group[metric].iloc[0]
                    mitigated_value = mitigated_group[metric].iloc[0]
                    
                    change = mitigated_value - baseline_value
                    change_percent = (change / (baseline_value + 1e-8)) * 100
                    
                    group_results[metric] = {
                        'baseline': baseline_value,
                        'mitigated': mitigated_value,
                        'change': change,
                        'change_percent': change_percent
                    }
                
                model_results[group] = group_results
            
            # Performance globale du modèle (moyenne des groupes)
            if model_results:
                global_performance = {}
                for metric in performance_metrics:
                    changes = []
                    for group_data in model_results.values():
                        if metric in group_data:
                            changes.append(group_data[metric]['change_percent'])
                    
                    if changes:
                        global_performance[metric] = {
                            'avg_change_percent': np.mean(changes),
                            'performance_degradation': np.mean(changes) < -5  # >5% dégradation
                        }
                
                model_results['global_performance'] = global_performance
            
            performance_comparison[model] = model_results
        
        return performance_comparison
    
    def _analyze_performance_fairness_tradeoff(self, protected_attribute: str) -> Dict[str, Any]:
        """Analyse les trade-offs entre performance et équité."""
        fairness_improvement = self.evaluation_results.get(protected_attribute, {}).get('fairness_improvement', {})
        performance_impact = self.evaluation_results.get(protected_attribute, {}).get('performance_impact', {})
        
        tradeoff_analysis = {}
        
        for model_name in fairness_improvement.keys():
            if model_name not in performance_impact:
                continue
            
            # Score d'amélioration de l'équité
            fairness_score = fairness_improvement[model_name].get('overall_fairness_improvement', 0)
            
            # Score d'impact sur la performance
            performance_data = performance_impact[model_name].get('global_performance', {})
            performance_changes = []
            
            for metric_data in performance_data.values():
                if isinstance(metric_data, dict) and 'avg_change_percent' in metric_data:
                    performance_changes.append(metric_data['avg_change_percent'])
            
            performance_score = np.mean(performance_changes) if performance_changes else 0
            
            # Analyser le trade-off
            tradeoff_ratio = fairness_score / (abs(performance_score) + 1e-8)
            
            tradeoff_analysis[model_name] = {
                'fairness_improvement': fairness_score,
                'performance_impact': performance_score,
                'tradeoff_ratio': tradeoff_ratio,
                'tradeoff_quality': self._classify_tradeoff_quality(fairness_score, performance_score),
                'acceptable_tradeoff': fairness_score > 5 and performance_score > -10  # Seuils configables
            }
        
        return tradeoff_analysis
    
    def _classify_tradeoff_quality(self, fairness_improvement: float, performance_impact: float) -> str:
        """Classifie la qualité du trade-off performance vs équité."""
        if fairness_improvement > 10 and performance_impact > -2:
            return "Excellent"
        elif fairness_improvement > 5 and performance_impact > -5:
            return "Bon"
        elif fairness_improvement > 0 and performance_impact > -10:
            return "Acceptable"
        elif fairness_improvement > 0:
            return "Problématique"
        else:
            return "Inefficace"
    
    def _evaluate_bias_pattern_reduction(self, protected_attribute: str) -> Dict[str, Any]:
        """Évalue la réduction des patterns de biais."""
        baseline_patterns = self.baseline_results.get('bias_patterns', {})
        mitigated_patterns = self.mitigated_results.get('bias_patterns', {})
        
        reduction_analysis = {}
        
        for model_name in baseline_patterns.keys():
            if model_name not in mitigated_patterns:
                continue
            
            baseline_model = baseline_patterns[model_name]
            mitigated_model = mitigated_patterns[model_name]
            
            # Comparer les scores de biais
            baseline_score = baseline_model.get('bias_score', 0)
            mitigated_score = mitigated_model.get('bias_score', 0)
            
            score_reduction = baseline_score - mitigated_score
            score_reduction_percent = (score_reduction / (baseline_score + 1e-8)) * 100
            
            # Comparer les niveaux de biais
            baseline_level = baseline_model.get('bias_level', 'Acceptable')
            mitigated_level = mitigated_model.get('bias_level', 'Acceptable')
            
            level_improved = self._compare_bias_levels(baseline_level, mitigated_level)
            
            # Comparer le nombre de biais par catégorie
            categories = ['severe_bias', 'moderate_bias', 'potential_bias']
            category_reductions = {}
            
            for category in categories:
                baseline_count = len(baseline_model.get(category, []))
                mitigated_count = len(mitigated_model.get(category, []))
                reduction = baseline_count - mitigated_count
                category_reductions[category] = {
                    'baseline_count': baseline_count,
                    'mitigated_count': mitigated_count,
                    'reduction': reduction
                }
            
            reduction_analysis[model_name] = {
                'bias_score_reduction': score_reduction,
                'bias_score_reduction_percent': score_reduction_percent,
                'bias_level_baseline': baseline_level,
                'bias_level_mitigated': mitigated_level,
                'bias_level_improved': level_improved,
                'category_reductions': category_reductions
            }
        
        return reduction_analysis
    
    def _compare_bias_levels(self, baseline_level: str, mitigated_level: str) -> bool:
        """Compare les niveaux de biais et détermine s'il y a amélioration."""
        level_hierarchy = {
            'Biais Sévère': 4,
            'Biais Modéré': 3,
            'Biais Potentiel': 2,
            'Acceptable': 1
        }
        
        baseline_rank = level_hierarchy.get(baseline_level, 1)
        mitigated_rank = level_hierarchy.get(mitigated_level, 1)
        
        return mitigated_rank < baseline_rank
    
    def _calculate_effectiveness_score(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calcule un score global d'efficacité de la mitigation."""
        fairness_scores = []
        performance_scores = []
        
        # Collecter les scores d'amélioration de l'équité
        fairness_improvement = evaluation_results.get('fairness_improvement', {})
        for model_data in fairness_improvement.values():
            if 'overall_fairness_improvement' in model_data:
                fairness_scores.append(model_data['overall_fairness_improvement'])
        
        # Collecter les scores d'impact sur la performance
        tradeoff_analysis = evaluation_results.get('tradeoff_analysis', {})
        for model_data in tradeoff_analysis.values():
            if 'performance_impact' in model_data:
                performance_scores.append(model_data['performance_impact'])
        
        # Calculer les scores moyens
        avg_fairness_improvement = np.mean(fairness_scores) if fairness_scores else 0
        avg_performance_impact = np.mean(performance_scores) if performance_scores else 0
        
        # Score composite (pondéré: 70% équité, 30% performance)
        composite_score = (0.7 * avg_fairness_improvement) + (0.3 * avg_performance_impact)
        
        return {
            'average_fairness_improvement': avg_fairness_improvement,
            'average_performance_impact': avg_performance_impact,
            'composite_effectiveness_score': composite_score,
            'effectiveness_level': self._classify_effectiveness_level(composite_score)
        }
    
    def _classify_effectiveness_level(self, composite_score: float) -> str:
        """Classifie le niveau d'efficacité selon le score composite."""
        if composite_score >= 15:
            return "Très Efficace"
        elif composite_score >= 8:
            return "Efficace"
        elif composite_score >= 3:
            return "Modérément Efficace"
        elif composite_score >= 0:
            return "Peu Efficace"
        else:
            return "Inefficace"
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur l'évaluation."""
        recommendations = []
        
        effectiveness_score = evaluation_results.get('effectiveness_score', {})
        composite_score = effectiveness_score.get('composite_effectiveness_score', 0)
        
        tradeoff_analysis = evaluation_results.get('tradeoff_analysis', {})
        
        # Recommandations selon l'efficacité globale
        if composite_score >= 15:
            recommendations.append("✅ Excellent résultat de mitigation - Déployer en production")
            recommendations.append("📊 Maintenir un monitoring continu des métriques d'équité")
        elif composite_score >= 8:
            recommendations.append("👍 Mitigation efficace - Valider avec des tests supplémentaires")
            recommendations.append("🔍 Surveiller les performances sur des données réelles")
        elif composite_score >= 3:
            recommendations.append("⚠️ Mitigation modérée - Considérer des techniques supplémentaires")
            recommendations.append("🔧 Ajuster les hyperparamètres de mitigation")
        else:
            recommendations.append("❌ Mitigation insuffisante - Revoir la stratégie")
            recommendations.append("🔄 Essayer d'autres techniques de mitigation")
        
        # Recommandations spécifiques aux trade-offs
        problematic_models = []
        for model_name, tradeoff_data in tradeoff_analysis.items():
            if not tradeoff_data.get('acceptable_tradeoff', True):
                problematic_models.append(model_name)
        
        if problematic_models:
            recommendations.append(f"⚖️ Trade-off problématique pour: {', '.join(problematic_models)}")
            recommendations.append("🎯 Considérer des techniques de mitigation moins agressives")
        
        # Recommandations pour l'amélioration continue
        recommendations.extend([
            "📈 Évaluer sur des métriques d'équité supplémentaires",
            "🧪 Tester avec différents seuils de décision",
            "📋 Documenter toutes les modifications pour la traçabilité",
            "👥 Impliquer les parties prenantes dans l'évaluation des résultats"
        ])
        
        return recommendations
    
    def create_evaluation_dashboard(self, protected_attribute: str = 'race') -> str:
        """
        Crée un dashboard interactif d'évaluation de l'équité.
        
        Args:
            protected_attribute: Attribut protégé analysé
            
        Returns:
            Chemin du fichier HTML du dashboard
        """
        if protected_attribute not in self.evaluation_results:
            self.evaluate_mitigation_effectiveness(protected_attribute)
        
        eval_results = self.evaluation_results[protected_attribute]
        
        # Créer le dashboard avec Plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Amélioration des Métriques d\'Équité',
                'Impact sur les Performances',
                'Analyse Trade-off Performance vs Équité',
                'Réduction des Patterns de Biais',
                'Scores d\'Efficacité par Modèle',
                'Comparaison Avant/Après Mitigation'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Données pour les visualisations
        fairness_improvement = eval_results.get('fairness_improvement', {})
        tradeoff_analysis = eval_results.get('tradeoff_analysis', {})
        bias_reduction = eval_results.get('bias_reduction', {})
        
        models = list(fairness_improvement.keys())
        
        # 1. Amélioration des métriques d'équité
        metrics_to_show = ['demographic_parity_difference', 'equal_opportunity_difference', 'disparate_impact_ratio']
        
        for metric in metrics_to_show:
            improvements = []
            for model in models:
                model_data = fairness_improvement.get(model, {})
                improvement = model_data.get(metric, {}).get('improvement_percent', 0)
                improvements.append(improvement)
            
            fig.add_trace(
                go.Bar(name=metric.replace('_', ' ').title(), x=models, y=improvements),
                row=1, col=1
            )
        
        # 2. Impact sur les performances
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in performance_metrics:
            impacts = []
            for model in models:
                # Moyenner l'impact sur tous les groupes
                model_tradeoff = tradeoff_analysis.get(model, {})
                impact = model_tradeoff.get('performance_impact', 0)
                impacts.append(impact)
            
            fig.add_trace(
                go.Bar(name=metric.title(), x=models, y=impacts),
                row=1, col=2
            )
        
        # 3. Trade-off scatter plot
        fairness_scores = []
        performance_scores = []
        model_names = []
        
        for model, tradeoff_data in tradeoff_analysis.items():
            fairness_scores.append(tradeoff_data.get('fairness_improvement', 0))
            performance_scores.append(tradeoff_data.get('performance_impact', 0))
            model_names.append(model)
        
        fig.add_trace(
            go.Scatter(
                x=performance_scores,
                y=fairness_scores,
                mode='markers+text',
                text=model_names,
                textposition='top center',
                name='Trade-off Points',
                marker=dict(size=10)
            ),
            row=2, col=1
        )
        
        # 4. Réduction des patterns de biais
        bias_score_reductions = []
        for model in models:
            reduction = bias_reduction.get(model, {}).get('bias_score_reduction_percent', 0)
            bias_score_reductions.append(reduction)
        
        fig.add_trace(
            go.Bar(x=models, y=bias_score_reductions, name='Réduction Score Biais'),
            row=2, col=2
        )
        
        # 5. Scores d'efficacité
        effectiveness_score = eval_results.get('effectiveness_score', {})
        composite_score = effectiveness_score.get('composite_effectiveness_score', 0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=composite_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score d'Efficacité Global"},
                gauge={
                    'axis': {'range': [-10, 30]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-10, 0], 'color': "red"},
                        {'range': [0, 8], 'color': "yellow"},
                        {'range': [8, 30], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ),
            row=3, col=1
        )
        
        # 6. Comparaison avant/après (exemple avec parité démographique)
        baseline_dp = []
        mitigated_dp = []
        
        for model in models:
            baseline_val = 0
            mitigated_val = 0
            
            if model in fairness_improvement:
                dp_data = fairness_improvement[model].get('demographic_parity_difference', {})
                baseline_val = abs(dp_data.get('baseline', 0))
                mitigated_val = abs(dp_data.get('mitigated', 0))
            
            baseline_dp.append(baseline_val)
            mitigated_dp.append(mitigated_val)
        
        fig.add_trace(
            go.Bar(name='Avant Mitigation', x=models, y=baseline_dp),
            row=3, col=2
        )
        fig.add_trace(
            go.Bar(name='Après Mitigation', x=models, y=mitigated_dp),
            row=3, col=2
        )
        
        # Mise en forme du dashboard
        fig.update_layout(
            height=1400,
            showlegend=True,
            title_text=f"Dashboard d'Évaluation de l'Équité - {protected_attribute.title()}",
            title_x=0.5
        )
        
        # Ajouter une ligne de référence pour le trade-off
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Sauvegarder le dashboard
        html_path = os.path.join(self.results_dir, f"fairness_evaluation_dashboard_{protected_attribute}.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        
        logger.info(f"Dashboard d'évaluation d'équité sauvegardé: {html_path}")
        return html_path
    
    def generate_evaluation_report(self, protected_attribute: str = 'race') -> str:
        """
        Génère un rapport complet d'évaluation de l'équité.
        
        Args:
            protected_attribute: Attribut protégé analysé
            
        Returns:
            Chemin du fichier de rapport généré
        """
        if protected_attribute not in self.evaluation_results:
            self.evaluate_mitigation_effectiveness(protected_attribute)
        
        eval_results = self.evaluation_results[protected_attribute]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Données du rapport
        effectiveness_score = eval_results.get('effectiveness_score', {})
        recommendations = eval_results.get('recommendations', [])
        fairness_improvement = eval_results.get('fairness_improvement', {})
        tradeoff_analysis = eval_results.get('tradeoff_analysis', {})
        
        # Contenu du rapport
        report_content = f"""# Rapport d'Évaluation de l'Équité - Projet COMPAS

## Résumé Exécutif

**Date d'évaluation**: {datetime.now().strftime("%d/%m/%Y %H:%M")}  
**Attribut protégé**: {protected_attribute.title()}  
**Modèles évalués**: {', '.join(fairness_improvement.keys())}  

### Résultats Globaux

- **Score d'efficacité composite**: {effectiveness_score.get('composite_effectiveness_score', 0):.2f}
- **Niveau d'efficacité**: {effectiveness_score.get('effectiveness_level', 'Non évalué')}
- **Amélioration moyenne de l'équité**: {effectiveness_score.get('average_fairness_improvement', 0):.2f}%
- **Impact moyen sur les performances**: {effectiveness_score.get('average_performance_impact', 0):.2f}%

## 1. Analyse Détaillée par Modèle

"""
        
        # Analyse par modèle
        for model_name in fairness_improvement.keys():
            model_fairness = fairness_improvement[model_name]
            model_tradeoff = tradeoff_analysis.get(model_name, {})
            
            report_content += f"### {model_name}\n\n"
            
            # Métriques d'équité
            report_content += "**Amélioration des Métriques d'Équité:**\n"
            
            key_metrics = ['demographic_parity_difference', 'equal_opportunity_difference', 'disparate_impact_ratio']
            for metric in key_metrics:
                if metric in model_fairness:
                    data = model_fairness[metric]
                    improvement = data.get('improvement_percent', 0)
                    status = "✅" if data.get('improved', False) else "❌"
                    
                    report_content += f"- {metric.replace('_', ' ').title()}: {improvement:+.2f}% {status}\n"
                    report_content += f"  - Avant: {data.get('baseline', 0):.4f}\n"
                    report_content += f"  - Après: {data.get('mitigated', 0):.4f}\n"
            
            # Trade-off performance vs équité
            report_content += f"\n**Trade-off Performance vs Équité:**\n"
            report_content += f"- Amélioration équité: {model_tradeoff.get('fairness_improvement', 0):.2f}%\n"
            report_content += f"- Impact performance: {model_tradeoff.get('performance_impact', 0):.2f}%\n"
            report_content += f"- Qualité trade-off: {model_tradeoff.get('tradeoff_quality', 'Non évalué')}\n"
            report_content += f"- Trade-off acceptable: {'✅' if model_tradeoff.get('acceptable_tradeoff', False) else '❌'}\n\n"
        
        report_content += "## 2. Synthèse des Résultats\n\n"
        
        # Tableau récapitulatif
        report_content += "| Modèle | Score Équité | Impact Performance | Qualité Trade-off | Recommandation |\n"
        report_content += "|--------|--------------|-------------------|------------------|---------------|\n"
        
        for model_name in fairness_improvement.keys():
            fairness_score = fairness_improvement[model_name].get('overall_fairness_improvement', 0)
            performance_impact = tradeoff_analysis.get(model_name, {}).get('performance_impact', 0)
            tradeoff_quality = tradeoff_analysis.get(model_name, {}).get('tradeoff_quality', 'Non évalué')
            
            if fairness_score > 10 and performance_impact > -5:
                recommendation = "Déployer"
            elif fairness_score > 5:
                recommendation = "Valider"
            else:
                recommendation = "Améliorer"
            
            report_content += f"| {model_name} | {fairness_score:+.1f}% | {performance_impact:+.1f}% | {tradeoff_quality} | {recommendation} |\n"
        
        report_content += "\n## 3. Recommandations\n\n"
        
        for i, recommendation in enumerate(recommendations, 1):
            report_content += f"{i}. {recommendation}\n"
        
        report_content += """

## 4. Prochaines Étapes

### Actions Immédiates
1. **Validation Extended**: Tester les modèles sur des données de validation externes
2. **Monitoring Setup**: Mettre en place un système de surveillance des métriques d'équité
3. **Stakeholder Review**: Présenter les résultats aux parties prenantes métier

### Actions à Moyen Terme
1. **Continuous Improvement**: Itérer sur les techniques de mitigation
2. **Production Deployment**: Déployer les modèles validés avec monitoring continu
3. **Documentation**: Finaliser la documentation des processus de mitigation

### Actions à Long Terme
1. **Regular Audits**: Programmer des audits réguliers d'équité
2. **Model Retraining**: Planifier la réentraînement périodique avec nouvelles techniques
3. **Best Practices**: Développer un framework de bonnes pratiques pour l'équité

## 5. Conclusion

"""
        
        # Conclusion basée sur le score d'efficacité
        effectiveness_level = effectiveness_score.get('effectiveness_level', 'Non évalué')
        
        if effectiveness_level == "Très Efficace":
            report_content += """Les stratégies de mitigation des biais se sont révélées très efficaces, permettant une amélioration significative de l'équité avec un impact minimal sur les performances. Les modèles sont prêts pour un déploiement en production avec un monitoring continu."""
        elif effectiveness_level == "Efficace":
            report_content += """Les techniques de mitigation ont donné des résultats positifs avec une amélioration notable de l'équité. Une validation supplémentaire est recommandée avant le déploiement en production."""
        elif effectiveness_level == "Modérément Efficace":
            report_content += """La mitigation a apporté des améliorations modérées. Il est recommandé d'explorer des techniques supplémentaires ou d'ajuster les paramètres pour obtenir de meilleurs résultats."""
        else:
            report_content += """Les résultats de mitigation sont insuffisants ou problématiques. Une révision complète de la stratégie de mitigation est nécessaire."""
        
        report_content += """

---
*Rapport généré automatiquement par le module d'évaluation de l'équité COMPAS*
*Conforme aux standards d'équité algorithmique et d'évaluation de la mitigation des biais*
"""
        
        # Sauvegarder le rapport
        report_path = os.path.join(self.results_dir, f"fairness_evaluation_report_{protected_attribute}_{timestamp}.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Rapport d'évaluation d'équité généré: {report_path}")
        return report_path
    
    def _save_evaluation_results(self, protected_attribute: str) -> None:
        """Sauvegarde les résultats d'évaluation."""
        results_path = os.path.join(self.results_dir, f"evaluation_results_{protected_attribute}.json")
        
        # Sérialiser les résultats (en évitant les objets pandas)
        serializable_results = {}
        for key, value in self.evaluation_results[protected_attribute].items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict('records')
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Résultats d'évaluation sauvegardés: {results_path}")


# Fonctions utilitaires
def create_sample_evaluation() -> FairnessEvaluator:
    """
    Crée une évaluation d'équité d'exemple avec des données simulées.
    
    Returns:
        Évaluateur d'équité configuré avec des données d'exemple
    """
    evaluator = FairnessEvaluator()
    
    # Simuler des résultats baseline (avec biais)
    baseline_results = {
        'fairness_metrics': {
            'race': {
                'BiasedModel': {
                    'demographic_parity_difference': 0.25,
                    'equal_opportunity_difference': 0.20,
                    'disparate_impact_ratio': 1.4,
                    'passes_80_rule': False
                },
                'LessBiasedModel': {
                    'demographic_parity_difference': 0.15,
                    'equal_opportunity_difference': 0.12,
                    'disparate_impact_ratio': 1.2,
                    'passes_80_rule': False
                }
            }
        },
        'group_comparison': pd.DataFrame({
            'model': ['BiasedModel', 'BiasedModel', 'LessBiasedModel', 'LessBiasedModel'],
            'group': ['African-American', 'Caucasian', 'African-American', 'Caucasian'],
            'accuracy': [0.72, 0.78, 0.74, 0.76],
            'precision': [0.68, 0.75, 0.70, 0.73],
            'recall': [0.70, 0.72, 0.72, 0.71],
            'f1_score': [0.69, 0.73, 0.71, 0.72],
            'auc': [0.74, 0.80, 0.76, 0.78]
        })
    }
    
    # Simuler des résultats après mitigation (améliorés)
    mitigated_results = {
        'fairness_metrics': {
            'race': {
                'BiasedModel': {
                    'demographic_parity_difference': 0.08,
                    'equal_opportunity_difference': 0.06,
                    'disparate_impact_ratio': 1.1,
                    'passes_80_rule': True
                },
                'LessBiasedModel': {
                    'demographic_parity_difference': 0.05,
                    'equal_opportunity_difference': 0.04,
                    'disparate_impact_ratio': 1.05,
                    'passes_80_rule': True
                }
            }
        },
        'group_comparison': pd.DataFrame({
            'model': ['BiasedModel', 'BiasedModel', 'LessBiasedModel', 'LessBiasedModel'],
            'group': ['African-American', 'Caucasian', 'African-American', 'Caucasian'],
            'accuracy': [0.70, 0.76, 0.73, 0.75],
            'precision': [0.67, 0.73, 0.69, 0.72],
            'recall': [0.69, 0.70, 0.71, 0.70],
            'f1_score': [0.68, 0.71, 0.70, 0.71],
            'auc': [0.72, 0.78, 0.75, 0.77]
        })
    }
    
    evaluator.load_baseline_results(baseline_results)
    evaluator.load_mitigated_results(mitigated_results)
    
    return evaluator


def main():
    """Fonction principale pour démonstration."""
    print("⚖️ Démonstration du module d'évaluation de l'équité COMPAS")
    
    # Créer une évaluation d'exemple
    evaluator = create_sample_evaluation()
    
    # Évaluer l'efficacité de la mitigation
    print("\n📊 Évaluation de l'efficacité de la mitigation...")
    effectiveness_results = evaluator.evaluate_mitigation_effectiveness('race')
    
    # Afficher les résultats
    effectiveness_score = effectiveness_results.get('effectiveness_score', {})
    print(f"\nScore d'efficacité composite: {effectiveness_score.get('composite_effectiveness_score', 0):.2f}")
    print(f"Niveau d'efficacité: {effectiveness_score.get('effectiveness_level', 'Non évalué')}")
    
    # Afficher les recommandations
    recommendations = effectiveness_results.get('recommendations', [])
    print(f"\nRecommandations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")
    
    # Créer le dashboard
    print("\n📈 Génération du dashboard d'évaluation...")
    dashboard_path = evaluator.create_evaluation_dashboard('race')
    print(f"Dashboard sauvegardé: {dashboard_path}")
    
    # Générer le rapport
    print("\n📄 Génération du rapport d'évaluation...")
    report_path = evaluator.generate_evaluation_report('race')
    print(f"Rapport généré: {report_path}")
    
    print("\n✅ Évaluation de l'équité terminée avec succès!")
    print(f"Résultats disponibles dans: {evaluator.results_dir}")


if __name__ == "__main__":
    main()