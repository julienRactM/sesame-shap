"""
Module de Stratégies de Mitigation des Biais pour le Projet COMPAS
================================================================

Ce module implémente des stratégies complètes de mitigation des biais pour
les modèles de prédiction de récidive COMPAS, avec optimisations spécifiques
pour Mac M4 Pro et Apple Silicon.

Le module couvre trois approches principales:
- Pré-traitement: Échantillonnage, suppression de features, augmentation de données
- Traitement: Entraînement contraint par équité, débiaisage adversarial
- Post-traitement: Optimisation de seuils, calibration de sortie

Auteur: Système d'IA Claude - Expert ML Apple Silicon
Date: 2025-08-05
Optimisé pour: Mac M4 Pro avec Apple Silicon
"""

import os
import sys
import warnings
import logging
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.inspection import permutation_importance

# Bibliothèques de rééchantillonnage
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

# Métriques d'équité
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio,
    false_positive_rate, false_negative_rate,
    MetricFrame
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# Optimisation et parallélisation pour Mac M4 Pro
from scipy.optimize import minimize, differential_evolution
import xgboost as xgb
import lightgbm as lgb

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration pour Apple Silicon et Mac M4 Pro
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Configuration des warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/julienrm/Workspace/M2/sesame-shap/data/results/bias_mitigation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BiasMitigationFramework:
    """
    Framework principal pour la mitigation des biais dans les modèles COMPAS.
    
    Cette classe implémente des stratégies complètes de mitigation des biais
    avec optimisations spécifiques pour Mac M4 Pro.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialise le framework de mitigation des biais.
        
        Args:
            random_state: Graine aléatoire pour la reproductibilité
            n_jobs: Nombre de processus parallèles (-1 pour tous les cœurs)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        
        # Storage pour les résultats et modèles
        self.original_results = {}
        self.mitigated_results = {}
        self.mitigation_strategies = {}
        self.comparison_results = {}
        
        # Configuration des métriques d'équité
        self.fairness_metrics = [
            'demographic_parity_difference',
            'demographic_parity_ratio',
            'equalized_odds_difference',
            'equalized_odds_ratio'
        ]
        
        # Attributs sensibles par défaut pour COMPAS
        self.sensitive_attributes = ['race', 'sex']
        
        # Création des répertoires nécessaires
        self._create_directories()
        
        logger.info(f"BiasMitigationFramework initialisé avec {self.n_jobs} processus")
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour sauvegarder les résultats."""
        directories = [
            '/Users/julienrm/Workspace/M2/sesame-shap/data/models/bias_mitigation',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/bias_mitigation',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/processed/bias_mitigation'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Répertoires de mitigation des biais créés")

    def remove_sensitive_features(self, X: pd.DataFrame, y: pd.Series,
                                sensitive_features: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Supprime les features sensibles pour la mitigation des biais.
        
        Args:
            X: Features d'entrée
            y: Variable cible
            sensitive_features: Liste des features sensibles à supprimer
            
        Returns:
            Tuple contenant les features nettoyées et un rapport de suppression
        """
        logger.info("Suppression des features sensibles")
        
        if sensitive_features is None:
            sensitive_features = self.sensitive_attributes
        
        X_clean = X.copy()
        removal_report = {
            'method': 'feature_removal',
            'sensitive_features_target': sensitive_features,
            'features_removed': [],
            'features_remaining': [],
            'original_shape': X.shape,
            'new_shape': None
        }
        
        # Identifier toutes les colonnes liées aux features sensibles
        features_to_remove = []
        
        for sensitive_attr in sensitive_features:
            # Colonnes directes
            if sensitive_attr in X_clean.columns:
                features_to_remove.append(sensitive_attr)
            
            # Colonnes encodées (one-hot, etc.)
            for col in X_clean.columns:
                if sensitive_attr in col.lower():
                    features_to_remove.append(col)
        
        # Supprimer les doublons
        features_to_remove = list(set(features_to_remove))
        
        # Effectuer la suppression
        X_clean = X_clean.drop(columns=features_to_remove, errors='ignore')
        
        removal_report.update({
            'features_removed': features_to_remove,
            'features_remaining': list(X_clean.columns),
            'new_shape': X_clean.shape,
            'reduction_percentage': ((X.shape[1] - X_clean.shape[1]) / X.shape[1]) * 100
        })
        
        logger.info(f"Features supprimées: {len(features_to_remove)} / {X.shape[1]} "
                   f"({removal_report['reduction_percentage']:.1f}%)")
        
        return X_clean, removal_report

    def apply_fairness_sampling(self, X: pd.DataFrame, y: pd.Series,
                              sensitive_attr: pd.Series, strategy: str = 'smote_fairness',
                              target_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Applique des stratégies de rééchantillonnage conscientes de l'équité.
        
        Args:
            X: Features d'entrée
            y: Variable cible
            sensitive_attr: Attribut sensible pour la stratification
            strategy: Stratégie de rééchantillonnage
            target_ratio: Ratio cible pour l'équilibrage
            
        Returns:
            Tuple contenant les données rééchantillonnées et un rapport
        """
        logger.info(f"Application du rééchantillonnage avec stratégie: {strategy}")
        
        sampling_report = {
            'method': f'fairness_sampling_{strategy}',
            'original_distribution': {},
            'resampled_distribution': {},
            'original_size': len(X),
            'resampled_size': None,
            'fairness_improvement': {}
        }
        
        # Analyse de la distribution originale
        original_dist = self._analyze_group_distribution(y, sensitive_attr)
        sampling_report['original_distribution'] = original_dist
        
        # Application de la stratégie choisie
        if strategy == 'smote_fairness':
            X_resampled, y_resampled, sensitive_resampled = self._smote_fairness_aware(
                X, y, sensitive_attr, target_ratio
            )
        elif strategy == 'group_undersampling':
            X_resampled, y_resampled, sensitive_resampled = self._group_aware_undersampling(
                X, y, sensitive_attr, target_ratio
            )
        elif strategy == 'group_oversampling':
            X_resampled, y_resampled, sensitive_resampled = self._group_aware_oversampling(
                X, y, sensitive_attr, target_ratio
            )
        elif strategy == 'combined_sampling':
            X_resampled, y_resampled, sensitive_resampled = self._combined_fairness_sampling(
                X, y, sensitive_attr, target_ratio
            )
        else:
            raise ValueError(f"Stratégie de rééchantillonnage inconnue: {strategy}")
        
        # Analyse de la distribution après rééchantillonnage
        resampled_dist = self._analyze_group_distribution(y_resampled, sensitive_resampled)
        sampling_report.update({
            'resampled_distribution': resampled_dist,
            'resampled_size': len(X_resampled),
            'size_change_percentage': ((len(X_resampled) - len(X)) / len(X)) * 100
        })
        
        logger.info(f"Rééchantillonnage terminé: {len(X)} -> {len(X_resampled)} échantillons")
        
        return X_resampled, y_resampled, sampling_report

    def train_fairness_constrained_models(self, X: pd.DataFrame, y: pd.Series,
                                        sensitive_attr: pd.Series,
                                        constraint_type: str = 'demographic_parity',
                                        models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Entraîne des modèles avec contraintes d'équité.
        
        Args:
            X: Features d'entrée
            y: Variable cible
            sensitive_attr: Attribut sensible
            constraint_type: Type de contrainte d'équité
            models_to_train: Liste des modèles à entraîner
            
        Returns:
            Dictionnaire contenant les modèles entraînés et leurs métriques
        """
        logger.info(f"Entraînement de modèles avec contraintes d'équité: {constraint_type}")
        
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
        
        # Configuration des contraintes
        if constraint_type == 'demographic_parity':
            constraint = DemographicParity()
        elif constraint_type == 'equalized_odds':
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Type de contrainte non supporté: {constraint_type}")
        
        constrained_models = {}
        training_results = {}
        
        for model_name in models_to_train:
            logger.info(f"Entraînement du modèle contraint: {model_name}")
            
            try:
                # Création du modèle de base
                base_model = self._create_base_model(model_name)
                
                # Application de la contrainte avec ExponentiatedGradient
                constrained_model = ExponentiatedGradient(
                    estimator=base_model,
                    constraints=constraint,
                    eps=0.01,  # Tolerance pour la contrainte
                    max_iter=50,
                    nu=1e-6,
                    eta_mul=2.0
                )
                
                # Entraînement
                constrained_model.fit(X, y, sensitive_features=sensitive_attr)
                
                # Évaluation
                y_pred = constrained_model.predict(X)
                y_pred_proba = constrained_model.predict_proba(X)[:, 1] if hasattr(constrained_model, 'predict_proba') else y_pred
                
                # Métriques de performance
                performance_metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred),
                    'f1': f1_score(y, y_pred),
                    'auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5
                }
                
                # Métriques d'équité
                fairness_metrics = self._calculate_detailed_fairness_metrics(
                    y, y_pred, sensitive_attr
                )
                
                constrained_models[model_name] = constrained_model
                training_results[model_name] = {
                    'performance': performance_metrics,
                    'fairness': fairness_metrics,
                    'constraint_type': constraint_type,
                    'training_success': True
                }
                
                logger.info(f"{model_name} - Accuracy: {performance_metrics['accuracy']:.4f}, "
                           f"F1: {performance_metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {str(e)}")
                training_results[model_name] = {
                    'training_success': False,
                    'error': str(e)
                }
                continue
        
        return {
            'models': constrained_models,
            'results': training_results,
            'constraint_type': constraint_type
        }

    def calibrate_outputs_for_fairness(self, model, X: pd.DataFrame, y: pd.Series,
                                     sensitive_attr: pd.Series,
                                     calibration_method: str = 'platt') -> Tuple[Any, Dict]:
        """
        Calibre les sorties du modèle pour améliorer l'équité.
        
        Args:
            model: Modèle à calibrer
            X: Features d'entrée
            y: Variable cible
            sensitive_attr: Attribut sensible
            calibration_method: Méthode de calibration
            
        Returns:
            Tuple contenant le modèle calibré et un rapport
        """
        logger.info(f"Calibration du modèle pour l'équité avec méthode: {calibration_method}")
        
        calibration_report = {
            'method': f'fairness_calibration_{calibration_method}',
            'original_performance': {},
            'calibrated_performance': {},
            'fairness_improvement': {}
        }
        
        # Évaluation originale
        y_pred_original = model.predict(X)
        y_pred_proba_original = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred_original
        
        original_metrics = {
            'accuracy': accuracy_score(y, y_pred_original),
            'f1': f1_score(y, y_pred_original),
            'auc': roc_auc_score(y, y_pred_proba_original) if len(np.unique(y)) > 1 else 0.5
        }
        original_fairness = self._calculate_detailed_fairness_metrics(
            y, y_pred_original, sensitive_attr
        )
        
        calibration_report['original_performance'] = original_metrics
        calibration_report['original_fairness'] = original_fairness
        
        # Calibration par groupe pour améliorer l'équité
        calibrated_models = {}
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            X_group = X[group_mask]
            y_group = y[group_mask]
            
            if len(y_group) > 10:  # Minimum d'échantillons pour calibration
                if calibration_method == 'platt':
                    group_calibrator = CalibratedClassifierCV(
                        model, method='sigmoid', cv=3
                    )
                elif calibration_method == 'isotonic':
                    group_calibrator = CalibratedClassifierCV(
                        model, method='isotonic', cv=3
                    )
                else:
                    raise ValueError(f"Méthode de calibration inconnue: {calibration_method}")
                
                group_calibrator.fit(X_group, y_group)
                calibrated_models[str(group)] = group_calibrator
        
        # Création d'un modèle de calibration global adaptatif
        calibrated_model = FairnessAwareCalibrator(
            base_model=model,
            group_calibrators=calibrated_models,
            sensitive_attr_name='sensitive_group'
        )
        
        # Évaluation après calibration
        y_pred_calibrated = calibrated_model.predict(X, sensitive_attr)
        y_pred_proba_calibrated = calibrated_model.predict_proba(X, sensitive_attr)
        
        calibrated_metrics = {
            'accuracy': accuracy_score(y, y_pred_calibrated),
            'f1': f1_score(y, y_pred_calibrated),
            'auc': roc_auc_score(y, y_pred_proba_calibrated) if len(np.unique(y)) > 1 else 0.5
        }
        calibrated_fairness = self._calculate_detailed_fairness_metrics(
            y, y_pred_calibrated, sensitive_attr
        )
        
        calibration_report.update({
            'calibrated_performance': calibrated_metrics,
            'calibrated_fairness': calibrated_fairness,
            'performance_change': {
                metric: calibrated_metrics[metric] - original_metrics[metric]
                for metric in original_metrics.keys()
            }
        })
        
        logger.info("Calibration pour l'équité terminée")
        
        return calibrated_model, calibration_report

    def optimize_decision_thresholds(self, model, X: pd.DataFrame, y: pd.Series,
                                   sensitive_attr: pd.Series,
                                   constraint: str = 'demographic_parity') -> Tuple[Dict, Dict]:
        """
        Optimise les seuils de décision pour améliorer l'équité.
        
        Args:
            model: Modèle entraîné
            X: Features d'entrée
            y: Variable cible
            sensitive_attr: Attribut sensible
            constraint: Contrainte d'équité à optimiser
            
        Returns:
            Tuple contenant les seuils optimisés et un rapport
        """
        logger.info(f"Optimisation des seuils de décision pour contrainte: {constraint}")
        
        # Obtenir les probabilités prédites
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            # Fallback pour les modèles sans predict_proba
            y_scores = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict(X)
            y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        
        # Utilisation de ThresholdOptimizer de fairlearn
        if constraint == 'demographic_parity':
            fairness_constraint = DemographicParity()
        elif constraint == 'equalized_odds':
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError(f"Contrainte non supportée: {constraint}")
        
        threshold_optimizer = ThresholdOptimizer(
            estimator=model,
            constraints=fairness_constraint,
            prefit=True
        )
        
        # Entraînement de l'optimiseur de seuils
        threshold_optimizer.fit(X, y, sensitive_features=sensitive_attr)
        
        # Prédictions avec seuils optimisés
        y_pred_optimized = threshold_optimizer.predict(
            X, sensitive_features=sensitive_attr
        )
        
        # Analyse des seuils par groupe
        thresholds_by_group = {}
        optimization_report = {
            'method': f'threshold_optimization_{constraint}',
            'original_performance': {},
            'optimized_performance': {},
            'thresholds_by_group': {},
            'fairness_improvement': {}
        }
        
        # Performance originale
        y_pred_original = (y_proba >= 0.5).astype(int)
        original_metrics = {
            'accuracy': accuracy_score(y, y_pred_original),
            'precision': precision_score(y, y_pred_original),
            'recall': recall_score(y, y_pred_original),
            'f1': f1_score(y, y_pred_original)
        }
        
        # Performance optimisée
        optimized_metrics = {
            'accuracy': accuracy_score(y, y_pred_optimized),
            'precision': precision_score(y, y_pred_optimized),
            'recall': recall_score(y, y_pred_optimized),
            'f1': f1_score(y, y_pred_optimized)
        }
        
        # Calcul des seuils optimaux par groupe
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            group_proba = y_proba[group_mask]
            group_y_true = y[group_mask]
            group_y_pred_optimized = y_pred_optimized[group_mask]
            
            # Estimation du seuil optimal pour ce groupe
            optimal_threshold = self._estimate_group_threshold(
                group_proba, group_y_true, group_y_pred_optimized
            )
            thresholds_by_group[str(group)] = optimal_threshold
        
        optimization_report.update({
            'original_performance': original_metrics,
            'optimized_performance': optimized_metrics,
            'thresholds_by_group': thresholds_by_group,
            'performance_change': {
                metric: optimized_metrics[metric] - original_metrics[metric]
                for metric in original_metrics.keys()
            }
        })
        
        # Métriques d'équité
        original_fairness = self._calculate_detailed_fairness_metrics(
            y, y_pred_original, sensitive_attr
        )
        optimized_fairness = self._calculate_detailed_fairness_metrics(
            y, y_pred_optimized, sensitive_attr
        )
        
        optimization_report['fairness_improvement'] = {
            'original': original_fairness,
            'optimized': optimized_fairness
        }
        
        logger.info("Optimisation des seuils terminée")
        
        return thresholds_by_group, optimization_report

    def evaluate_mitigation_effectiveness(self, original_model, mitigated_model,
                                        X_test: pd.DataFrame, y_test: pd.Series,
                                        sensitive_attr_test: pd.Series) -> Dict[str, Any]:
        """
        Évalue l'efficacité des stratégies de mitigation des biais.
        
        Args:
            original_model: Modèle original (sans mitigation)
            mitigated_model: Modèle après mitigation
            X_test: Features de test
            y_test: Variable cible de test
            sensitive_attr_test: Attribut sensible de test
            
        Returns:
            Dictionnaire contenant l'évaluation comparative détaillée
        """
        logger.info("Évaluation de l'efficacité de la mitigation des biais")
        
        evaluation_report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_set_size': len(X_test),
            'original_model_results': {},
            'mitigated_model_results': {},
            'improvement_analysis': {},
            'trade_off_analysis': {}
        }
        
        # Évaluation du modèle original
        try:
            y_pred_original = original_model.predict(X_test)
            y_proba_original = original_model.predict_proba(X_test)[:, 1] if hasattr(original_model, 'predict_proba') else y_pred_original
            
            original_performance = {
                'accuracy': accuracy_score(y_test, y_pred_original),
                'precision': precision_score(y_test, y_pred_original),
                'recall': recall_score(y_test, y_pred_original),
                'f1': f1_score(y_test, y_pred_original),
                'auc': roc_auc_score(y_test, y_proba_original) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            original_fairness = self._calculate_detailed_fairness_metrics(
                y_test, y_pred_original, sensitive_attr_test
            )
            
            evaluation_report['original_model_results'] = {
                'performance': original_performance,
                'fairness': original_fairness
            }
            
        except Exception as e:
            logger.error(f"Erreur évaluation modèle original: {str(e)}")
            evaluation_report['original_model_results'] = {'error': str(e)}
        
        # Évaluation du modèle mitigé
        try:
            if hasattr(mitigated_model, 'predict_proba'):
                y_pred_mitigated = mitigated_model.predict(X_test)
                y_proba_mitigated = mitigated_model.predict_proba(X_test)[:, 1]
            elif hasattr(mitigated_model, 'predict') and hasattr(mitigated_model, 'predict_proba'):
                # Cas spécial pour les modèles avec calibration personnalisée
                y_pred_mitigated = mitigated_model.predict(X_test, sensitive_attr_test)
                y_proba_mitigated = mitigated_model.predict_proba(X_test, sensitive_attr_test)
            else:
                y_pred_mitigated = mitigated_model.predict(X_test)
                y_proba_mitigated = y_pred_mitigated
            
            mitigated_performance = {
                'accuracy': accuracy_score(y_test, y_pred_mitigated),
                'precision': precision_score(y_test, y_pred_mitigated),
                'recall': recall_score(y_test, y_pred_mitigated),
                'f1': f1_score(y_test, y_pred_mitigated),
                'auc': roc_auc_score(y_test, y_proba_mitigated) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            mitigated_fairness = self._calculate_detailed_fairness_metrics(
                y_test, y_pred_mitigated, sensitive_attr_test
            )
            
            evaluation_report['mitigated_model_results'] = {
                'performance': mitigated_performance,
                'fairness': mitigated_fairness
            }
            
            # Analyse des améliorations
            if 'original_model_results' in evaluation_report and 'error' not in evaluation_report['original_model_results']:
                performance_changes = {
                    metric: mitigated_performance[metric] - original_performance[metric]
                    for metric in original_performance.keys()
                }
                
                fairness_changes = self._calculate_fairness_improvements(
                    original_fairness, mitigated_fairness
                )
                
                evaluation_report['improvement_analysis'] = {
                    'performance_changes': performance_changes,
                    'fairness_improvements': fairness_changes,
                    'overall_bias_reduction': self._calculate_overall_bias_reduction(
                        original_fairness, mitigated_fairness
                    )
                }
                
                # Analyse du trade-off performance vs équité
                evaluation_report['trade_off_analysis'] = self._analyze_performance_fairness_tradeoff(
                    original_performance, mitigated_performance,
                    original_fairness, mitigated_fairness
                )
            
        except Exception as e:
            logger.error(f"Erreur évaluation modèle mitigé: {str(e)}")
            evaluation_report['mitigated_model_results'] = {'error': str(e)}
        
        logger.info("Évaluation de l'efficacité terminée")
        
        return evaluation_report

    def create_mitigation_comparison_dashboard(self, evaluation_results: List[Dict],
                                             strategy_names: List[str],
                                             save_path: str = None) -> str:
        """
        Crée un dashboard de comparaison des stratégies de mitigation.
        
        Args:
            evaluation_results: Liste des résultats d'évaluation
            strategy_names: Noms des stratégies comparées
            save_path: Chemin de sauvegarde du dashboard
            
        Returns:
            Chemin vers le dashboard généré
        """
        logger.info("Création du dashboard de comparaison des stratégies")
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/bias_mitigation/dashboard_{timestamp}.html"
        
        # Création du dashboard interactif avec Plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Performance: Accuracy vs F1-Score',
                'Équité: Parité Démographique',
                'Équité: Égalité des Chances',
                'Trade-off Performance vs Équité',
                'Amélioration du Biais par Stratégie',
                'Métriques Détaillées par Groupe'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # Extraction des données pour visualisation
        performance_data = []
        fairness_data = []
        
        for i, (result, strategy) in enumerate(zip(evaluation_results, strategy_names)):
            if 'mitigated_model_results' in result and 'error' not in result['mitigated_model_results']:
                perf = result['mitigated_model_results']['performance']
                fairness = result['mitigated_model_results']['fairness']
                
                performance_data.append({
                    'strategy': strategy,
                    'accuracy': perf.get('accuracy', 0),
                    'f1': perf.get('f1', 0),
                    'precision': perf.get('precision', 0),
                    'recall': perf.get('recall', 0),
                    'auc': perf.get('auc', 0)
                })
                
                # Extraction des métriques d'équité
                for attr, attr_fairness in fairness.items():
                    if isinstance(attr_fairness, dict) and 'demographic_parity_difference' in attr_fairness:
                        fairness_data.append({
                            'strategy': strategy,
                            'attribute': attr,
                            'dp_diff': abs(attr_fairness.get('demographic_parity_difference', 0)),
                            'eo_diff': abs(attr_fairness.get('equalized_odds_difference', 0)),
                            'dp_ratio': attr_fairness.get('demographic_parity_ratio', 1),
                            'eo_ratio': attr_fairness.get('equalized_odds_ratio', 1)
                        })
        
        # Graphique 1: Performance Accuracy vs F1
        if performance_data:
            df_perf = pd.DataFrame(performance_data)
            fig.add_trace(
                go.Scatter(
                    x=df_perf['accuracy'],
                    y=df_perf['f1'],
                    mode='markers+text',
                    text=df_perf['strategy'],
                    textposition="top center",
                    marker=dict(size=12, color=range(len(df_perf)), colorscale='viridis'),
                    name='Stratégies'
                ),
                row=1, col=1
            )
        
        # Graphique 2: Parité démographique
        if fairness_data:
            df_fair = pd.DataFrame(fairness_data)
            fig.add_trace(
                go.Bar(
                    x=df_fair['strategy'],
                    y=df_fair['dp_diff'],
                    name='Différence Parité Démographique',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # Graphique 3: Égalité des chances
        if fairness_data:
            fig.add_trace(
                go.Bar(
                    x=df_fair['strategy'],
                    y=df_fair['eo_diff'],
                    name='Différence Égalité des Chances',
                    marker_color='lightcoral'
                ),
                row=2, col=1
            )
        
        # Graphique 4: Trade-off
        if performance_data and fairness_data:
            df_combined = df_perf.merge(df_fair.groupby('strategy')[['dp_diff', 'eo_diff']].mean().reset_index(), on='strategy')
            df_combined['bias_score'] = (df_combined['dp_diff'] + df_combined['eo_diff']) / 2
            
            fig.add_trace(
                go.Scatter(
                    x=df_combined['accuracy'],
                    y=df_combined['bias_score'],
                    mode='markers+text',
                    text=df_combined['strategy'],
                    textposition="top center",
                    marker=dict(size=12, color='red'),
                    name='Trade-off Performance-Équité'
                ),
                row=2, col=2
            )
        
        # Graphique 5: Amélioration du biais
        if 'improvement_analysis' in evaluation_results[0]:
            bias_improvements = []
            for result, strategy in zip(evaluation_results, strategy_names):
                if 'improvement_analysis' in result:
                    bias_reduction = result['improvement_analysis'].get('overall_bias_reduction', 0)
                    bias_improvements.append({'strategy': strategy, 'improvement': bias_reduction})
            
            if bias_improvements:
                df_imp = pd.DataFrame(bias_improvements)
                fig.add_trace(
                    go.Bar(
                        x=df_imp['strategy'],
                        y=df_imp['improvement'],
                        name='Réduction du Biais (%)',
                        marker_color='green'
                    ),
                    row=3, col=1
                )
        
        # Configuration du layout
        fig.update_layout(
            title="Dashboard de Comparaison des Stratégies de Mitigation des Biais",
            height=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        # Mise à jour des axes
        fig.update_xaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="F1-Score", row=1, col=1)
        fig.update_yaxes(title_text="Différence", row=1, col=2)
        fig.update_yaxes(title_text="Différence", row=2, col=1)
        fig.update_xaxes(title_text="Accuracy", row=2, col=2)
        fig.update_yaxes(title_text="Score de Biais", row=2, col=2)
        fig.update_yaxes(title_text="Amélioration (%)", row=3, col=1)
        
        # Sauvegarde du dashboard
        fig.write_html(save_path)
        
        logger.info(f"Dashboard de comparaison sauvegardé: {save_path}")
        
        return save_path

    def generate_mitigation_report(self, evaluation_results: Dict,
                                 strategy_name: str = "Unknown",
                                 language: str = "fr") -> str:
        """
        Génère un rapport détaillé de mitigation des biais en français.
        
        Args:
            evaluation_results: Résultats d'évaluation de la mitigation
            strategy_name: Nom de la stratégie évaluée
            language: Langue du rapport
            
        Returns:
            Chemin vers le rapport généré
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/bias_mitigation/rapport_mitigation_{strategy_name}_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Rapport de Mitigation des Biais - {strategy_name}\n\n")
            f.write(f"**Date de génération:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Stratégie évaluée:** {strategy_name}\n")
            f.write(f"**Optimisé pour:** Mac M4 Pro avec Apple Silicon\n\n")
            
            # Résumé exécutif
            f.write("## Résumé Exécutif\n\n")
            
            if 'improvement_analysis' in evaluation_results:
                improvement = evaluation_results['improvement_analysis']
                if 'overall_bias_reduction' in improvement:
                    bias_reduction = improvement['overall_bias_reduction']
                    f.write(f"**Réduction globale du biais:** {bias_reduction:.1f}%\n\n")
                
                if 'performance_changes' in improvement:
                    perf_changes = improvement['performance_changes']
                    f.write("**Impact sur les performances:**\n")
                    for metric, change in perf_changes.items():
                        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        f.write(f"- {metric.title()}: {direction} {change:+.4f}\n")
                    f.write("\n")
            
            # Métriques détaillées
            f.write("## Métriques Détaillées\n\n")
            
            if 'original_model_results' in evaluation_results:
                f.write("### Modèle Original (Avant Mitigation)\n\n")
                original = evaluation_results['original_model_results']
                
                if 'performance' in original:
                    f.write("**Performance:**\n")
                    for metric, value in original['performance'].items():
                        f.write(f"- {metric.title()}: {value:.4f}\n")
                    f.write("\n")
                
                if 'fairness' in original:
                    f.write("**Métriques d'Équité:**\n")
                    for attr, fairness_data in original['fairness'].items():
                        if isinstance(fairness_data, dict) and 'demographic_parity_difference' in fairness_data:
                            f.write(f"- **{attr}:**\n")
                            f.write(f"  - Différence parité démographique: {fairness_data.get('demographic_parity_difference', 'N/A'):.4f}\n")
                            f.write(f"  - Ratio parité démographique: {fairness_data.get('demographic_parity_ratio', 'N/A'):.4f}\n")
                            f.write(f"  - Différence égalité des chances: {fairness_data.get('equalized_odds_difference', 'N/A'):.4f}\n")
                            f.write(f"  - Ratio égalité des chances: {fairness_data.get('equalized_odds_ratio', 'N/A'):.4f}\n")
                    f.write("\n")
            
            if 'mitigated_model_results' in evaluation_results:
                f.write("### Modèle Mitigé (Après Mitigation)\n\n")
                mitigated = evaluation_results['mitigated_model_results']
                
                if 'performance' in mitigated:
                    f.write("**Performance:**\n")
                    for metric, value in mitigated['performance'].items():
                        f.write(f"- {metric.title()}: {value:.4f}\n")
                    f.write("\n")
                
                if 'fairness' in mitigated:
                    f.write("**Métriques d'Équité:**\n")
                    for attr, fairness_data in mitigated['fairness'].items():
                        if isinstance(fairness_data, dict) and 'demographic_parity_difference' in fairness_data:
                            f.write(f"- **{attr}:**\n")
                            f.write(f"  - Différence parité démographique: {fairness_data.get('demographic_parity_difference', 'N/A'):.4f}\n")
                            f.write(f"  - Ratio parité démographique: {fairness_data.get('demographic_parity_ratio', 'N/A'):.4f}\n")
                            f.write(f"  - Différence égalité des chances: {fairness_data.get('equalized_odds_difference', 'N/A'):.4f}\n")
                            f.write(f"  - Ratio égalité des chances: {fairness_data.get('equalized_odds_ratio', 'N/A'):.4f}\n")
                    f.write("\n")
            
            # Analyse du trade-off
            if 'trade_off_analysis' in evaluation_results:
                f.write("## Analyse du Trade-off Performance vs Équité\n\n")
                tradeoff = evaluation_results['trade_off_analysis']
                
                f.write("Cette analyse évalue l'équilibre entre la performance prédictive et l'équité:\n\n")
                
                if 'performance_cost' in tradeoff:
                    cost = tradeoff['performance_cost']
                    f.write(f"**Coût en performance:** {cost:.4f} points\n")
                
                if 'fairness_gain' in tradeoff:
                    gain = tradeoff['fairness_gain']
                    f.write(f"**Gain en équité:** {gain:.4f} points\n")
                
                if 'efficiency_ratio' in tradeoff:
                    ratio = tradeoff['efficiency_ratio']
                    f.write(f"**Ratio d'efficacité:** {ratio:.4f} (gain équité / coût performance)\n\n")
            
            # Recommandations
            f.write("## Recommandations\n\n")
            
            if 'improvement_analysis' in evaluation_results:
                improvement = evaluation_results['improvement_analysis']
                bias_reduction = improvement.get('overall_bias_reduction', 0)
                
                if bias_reduction > 20:
                    f.write("✅ **Recommandation: ADOPTER** - Réduction significative du biais\n\n")
                    f.write("Cette stratégie montre une amélioration substantielle de l'équité. Les bénéfices justifient les coûts potentiels en performance.\n\n")
                elif bias_reduction > 10:
                    f.write("⚠️ **Recommandation: CONSIDÉRER** - Amélioration modérée\n\n")
                    f.write("Cette stratégie apporte des améliorations d'équité notables. Évaluez soigneusement le trade-off avec la performance.\n\n")
                else:
                    f.write("❌ **Recommandation: RÉVISER** - Amélioration limitée\n\n")
                    f.write("Cette stratégie n'apporte que des améliorations mineures. Considérez d'autres approches ou des paramètres différents.\n\n")
            
            # Prochaines étapes
            f.write("### Prochaines Étapes Recommandées\n\n")
            f.write("1. **Validation croisée** sur d'autres datasets pour confirmer les résultats\n")
            f.write("2. **Tests de robustesse** avec différents paramètres\n")
            f.write("3. **Analyse des sous-groupes** pour identifier des biais résiduels\n")
            f.write("4. **Déploiement pilote** avec monitoring continu de l'équité\n")
            f.write("5. **Évaluation longitudinale** des impacts à long terme\n\n")
            
            # Annexes techniques
            f.write("## Annexes Techniques\n\n")
            f.write(f"**Taille du jeu de test:** {evaluation_results.get('test_set_size', 'N/A')} échantillons\n")
            f.write(f"**Timestamp d'évaluation:** {evaluation_results.get('evaluation_timestamp', 'N/A')}\n")
            f.write(f"**Configuration système:** Mac M4 Pro, {os.cpu_count()} cœurs\n")
            f.write(f"**Framework utilisé:** BiasMitigationFramework v1.0\n\n")
        
        logger.info(f"Rapport de mitigation généré: {report_path}")
        return report_path

    # Méthodes auxiliaires privées
    
    def _create_base_model(self, model_name: str):
        """Crée un modèle de base pour l'entraînement contraint."""
        n_cores = self.n_jobs
        
        if model_name == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=1  # fairlearn ne supporte pas n_jobs > 1
            )
        elif model_name == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                n_jobs=1  # fairlearn ne supporte pas n_jobs > 1
            )
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                n_jobs=1,  # fairlearn ne supporte pas n_jobs > 1
                tree_method='hist',
                objective='binary:logistic'
            )
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")

    def _smote_fairness_aware(self, X: pd.DataFrame, y: pd.Series,
                            sensitive_attr: pd.Series, target_ratio: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Applique SMOTE avec conscience d'équité."""
        
        # Stratification par groupe sensible ET classe cible
        stratify_groups = y.astype(str) + "_" + sensitive_attr.astype(str)
        
        # Configuration SMOTE
        smote = SMOTE(
            sampling_strategy=target_ratio,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Application SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Reconstruction de l'attribut sensible pour les nouvelles observations
        # Stratégie: assigner l'attribut basé sur les k plus proches voisins
        from sklearn.neighbors import KNeighborsClassifier
        
        knn_sensitive = KNeighborsClassifier(n_neighbors=5, n_jobs=self.n_jobs)
        knn_sensitive.fit(X, sensitive_attr)
        sensitive_resampled = pd.Series(
            knn_sensitive.predict(X_resampled),
            index=X_resampled.index
        )
        
        return X_resampled, pd.Series(y_resampled, index=X_resampled.index), sensitive_resampled

    def _group_aware_undersampling(self, X: pd.DataFrame, y: pd.Series,
                                 sensitive_attr: pd.Series, target_ratio: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Applique un sous-échantillonnage conscient des groupes."""
        
        X_resampled_list = []
        y_resampled_list = []
        sensitive_resampled_list = []
        
        # Calculer la taille cible pour chaque groupe
        min_group_size = min([
            len(X[(sensitive_attr == group) & (y == class_val)])
            for group in sensitive_attr.unique()
            for class_val in y.unique()
        ])
        
        target_size_per_group_class = int(min_group_size * target_ratio)
        
        # Sous-échantillonner chaque combinaison groupe/classe
        for group in sensitive_attr.unique():
            for class_val in y.unique():
                mask = (sensitive_attr == group) & (y == class_val)
                group_data = X[mask]
                group_y = y[mask]
                group_sensitive = sensitive_attr[mask]
                
                if len(group_data) > target_size_per_group_class:
                    # Sous-échantillonnage
                    indices = np.random.choice(
                        group_data.index,
                        size=target_size_per_group_class,
                        replace=False
                    )
                    X_resampled_list.append(group_data.loc[indices])
                    y_resampled_list.append(group_y.loc[indices])
                    sensitive_resampled_list.append(group_sensitive.loc[indices])
                else:
                    # Garder tous les échantillons si le groupe est déjà petit
                    X_resampled_list.append(group_data)
                    y_resampled_list.append(group_y)
                    sensitive_resampled_list.append(group_sensitive)
        
        # Concaténation des résultats
        X_resampled = pd.concat(X_resampled_list, ignore_index=True)
        y_resampled = pd.concat(y_resampled_list, ignore_index=True)
        sensitive_resampled = pd.concat(sensitive_resampled_list, ignore_index=True)
        
        return X_resampled, y_resampled, sensitive_resampled

    def _group_aware_oversampling(self, X: pd.DataFrame, y: pd.Series,
                                sensitive_attr: pd.Series, target_ratio: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Applique un sur-échantillonnage conscient des groupes."""
        
        X_resampled_list = []
        y_resampled_list = []
        sensitive_resampled_list = []
        
        # Calculer la taille cible pour chaque groupe
        max_group_size = max([
            len(X[(sensitive_attr == group) & (y == class_val)])
            for group in sensitive_attr.unique()
            for class_val in y.unique()
        ])
        
        target_size_per_group_class = int(max_group_size * target_ratio)
        
        # Sur-échantillonner chaque combinaison groupe/classe
        for group in sensitive_attr.unique():
            for class_val in y.unique():
                mask = (sensitive_attr == group) & (y == class_val)
                group_data = X[mask]
                group_y = y[mask]
                group_sensitive = sensitive_attr[mask]
                
                if len(group_data) < target_size_per_group_class:
                    # Sur-échantillonnage avec remplacement
                    n_additional = target_size_per_group_class - len(group_data)
                    additional_indices = np.random.choice(
                        group_data.index,
                        size=n_additional,
                        replace=True
                    )
                    
                    X_additional = group_data.loc[additional_indices]
                    y_additional = group_y.loc[additional_indices]
                    sensitive_additional = group_sensitive.loc[additional_indices]
                    
                    # Ajouter du bruit pour diversifier les échantillons dupliqués
                    noise_std = 0.01 * X_additional.std()
                    X_additional_noisy = X_additional + np.random.normal(0, noise_std, X_additional.shape)
                    
                    X_resampled_list.extend([group_data, X_additional_noisy])
                    y_resampled_list.extend([group_y, y_additional])
                    sensitive_resampled_list.extend([group_sensitive, sensitive_additional])
                else:
                    X_resampled_list.append(group_data)
                    y_resampled_list.append(group_y)
                    sensitive_resampled_list.append(group_sensitive)
        
        # Concaténation des résultats
        X_resampled = pd.concat(X_resampled_list, ignore_index=True)
        y_resampled = pd.concat(y_resampled_list, ignore_index=True)
        sensitive_resampled = pd.concat(sensitive_resampled_list, ignore_index=True)
        
        return X_resampled, y_resampled, sensitive_resampled

    def _combined_fairness_sampling(self, X: pd.DataFrame, y: pd.Series,
                                  sensitive_attr: pd.Series, target_ratio: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Combine plusieurs techniques de rééchantillonnage pour l'équité."""
        
        # Étape 1: SMOTE pour équilibrer les classes
        smote = SMOTE(random_state=self.random_state, n_jobs=self.n_jobs)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        # Reconstruction de l'attribut sensible
        from sklearn.neighbors import KNeighborsClassifier
        knn_sensitive = KNeighborsClassifier(n_neighbors=3, n_jobs=self.n_jobs)
        knn_sensitive.fit(X, sensitive_attr)
        sensitive_smote = pd.Series(
            knn_sensitive.predict(X_smote),
            index=range(len(X_smote))
        )
        
        # Étape 2: Rééquilibrage par groupe
        X_final, y_final, sensitive_final = self._group_aware_undersampling(
            pd.DataFrame(X_smote, columns=X.columns),
            pd.Series(y_smote),
            sensitive_smote,
            target_ratio
        )
        
        return X_final, y_final, sensitive_final

    def _analyze_group_distribution(self, y: pd.Series, sensitive_attr: pd.Series) -> Dict:
        """Analyse la distribution des classes par groupe."""
        distribution = {}
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            group_y = y[group_mask]
            
            distribution[str(group)] = {
                'total_count': len(group_y),
                'positive_count': sum(group_y),
                'negative_count': len(group_y) - sum(group_y),
                'positive_rate': sum(group_y) / len(group_y) if len(group_y) > 0 else 0
            }
        
        return distribution

    def _calculate_detailed_fairness_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                           sensitive_attr: pd.Series) -> Dict:
        """Calcule des métriques d'équité détaillées."""
        fairness_metrics = {}
        
        # Métriques par attribut sensible
        for attr_name in ['race', 'sex']:  # Attributs sensibles typiques de COMPAS
            if attr_name in sensitive_attr.name or len(sensitive_attr.unique()) > 1:
                try:
                    # Métriques globales d'équité
                    dp_diff = demographic_parity_difference(
                        y_true, y_pred, sensitive_features=sensitive_attr
                    )
                    dp_ratio = demographic_parity_ratio(
                        y_true, y_pred, sensitive_features=sensitive_attr
                    )
                    eo_diff = equalized_odds_difference(
                        y_true, y_pred, sensitive_features=sensitive_attr
                    )
                    eo_ratio = equalized_odds_ratio(
                        y_true, y_pred, sensitive_features=sensitive_attr
                    )
                    
                    # Métriques par groupe
                    group_metrics = {}
                    for group in sensitive_attr.unique():
                        mask = sensitive_attr == group
                        if np.sum(mask) > 0:
                            group_y_true = y_true[mask]
                            group_y_pred = y_pred[mask]
                            
                            group_metrics[str(group)] = {
                                'count': int(np.sum(mask)),
                                'accuracy': accuracy_score(group_y_true, group_y_pred),
                                'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                                'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                                'f1': f1_score(group_y_true, group_y_pred, zero_division=0),
                                'positive_rate': np.mean(group_y_pred),
                                'fpr': false_positive_rate(group_y_true, group_y_pred),
                                'fnr': false_negative_rate(group_y_true, group_y_pred)
                            }
                    
                    fairness_metrics[attr_name] = {
                        'demographic_parity_difference': float(dp_diff),
                        'demographic_parity_ratio': float(dp_ratio),
                        'equalized_odds_difference': float(eo_diff),
                        'equalized_odds_ratio': float(eo_ratio),
                        'group_metrics': group_metrics
                    }
                    
                except Exception as e:
                    fairness_metrics[attr_name] = {'error': str(e)}
        
        # Si aucun attribut sensible spécifique n'est trouvé, utiliser l'attribut fourni
        if not fairness_metrics and len(sensitive_attr.unique()) > 1:
            try:
                dp_diff = demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                dp_ratio = demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                eo_diff = equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                eo_ratio = equalized_odds_ratio(
                    y_true, y_pred, sensitive_features=sensitive_attr
                )
                
                fairness_metrics['sensitive_attribute'] = {
                    'demographic_parity_difference': float(dp_diff),
                    'demographic_parity_ratio': float(dp_ratio),
                    'equalized_odds_difference': float(eo_diff),
                    'equalized_odds_ratio': float(eo_ratio)
                }
                
            except Exception as e:
                fairness_metrics['sensitive_attribute'] = {'error': str(e)}
        
        return fairness_metrics

    def _estimate_group_threshold(self, group_proba: np.ndarray, group_y_true: np.ndarray,
                                group_y_pred_optimized: np.ndarray) -> float:
        """Estime le seuil optimal pour un groupe spécifique."""
        
        # Si nous avons les prédictions optimisées, essayer de retrouver le seuil
        if len(group_proba) == len(group_y_pred_optimized):
            # Trouver le seuil qui sépare le mieux les prédictions optimisées
            thresholds = np.unique(group_proba)
            best_threshold = 0.5
            best_agreement = 0
            
            for threshold in thresholds:
                pred_at_threshold = (group_proba >= threshold).astype(int)
                agreement = np.mean(pred_at_threshold == group_y_pred_optimized)
                
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_threshold = threshold
            
            return float(best_threshold)
        
        return 0.5  # Seuil par défaut

    def _calculate_fairness_improvements(self, original_fairness: Dict, mitigated_fairness: Dict) -> Dict:
        """Calcule les améliorations d'équité."""
        improvements = {}
        
        for attr in original_fairness.keys():
            if attr in mitigated_fairness and 'error' not in original_fairness[attr]:
                attr_improvements = {}
                
                original_attr = original_fairness[attr]
                mitigated_attr = mitigated_fairness[attr]
                
                # Calculer les améliorations pour chaque métrique
                for metric in ['demographic_parity_difference', 'equalized_odds_difference']:
                    if metric in original_attr and metric in mitigated_attr:
                        original_val = abs(original_attr[metric])
                        mitigated_val = abs(mitigated_attr[metric])
                        
                        # Amélioration = réduction de la différence
                        improvement = ((original_val - mitigated_val) / original_val) * 100 if original_val > 0 else 0
                        attr_improvements[f'{metric}_improvement'] = improvement
                
                improvements[attr] = attr_improvements
        
        return improvements

    def _calculate_overall_bias_reduction(self, original_fairness: Dict, mitigated_fairness: Dict) -> float:
        """Calcule la réduction globale du biais."""
        
        total_original_bias = 0
        total_mitigated_bias = 0
        metric_count = 0
        
        for attr in original_fairness.keys():
            if attr in mitigated_fairness and 'error' not in original_fairness[attr]:
                original_attr = original_fairness[attr]
                mitigated_attr = mitigated_fairness[attr]
                
                for metric in ['demographic_parity_difference', 'equalized_odds_difference']:
                    if metric in original_attr and metric in mitigated_attr:
                        total_original_bias += abs(original_attr[metric])
                        total_mitigated_bias += abs(mitigated_attr[metric])
                        metric_count += 1
        
        if metric_count > 0 and total_original_bias > 0:
            reduction_percentage = ((total_original_bias - total_mitigated_bias) / total_original_bias) * 100
            return max(0, reduction_percentage)  # Pas de valeurs négatives
        
        return 0

    def _analyze_performance_fairness_tradeoff(self, original_perf: Dict, mitigated_perf: Dict,
                                             original_fairness: Dict, mitigated_fairness: Dict) -> Dict:
        """Analyse le trade-off entre performance et équité."""
        
        # Calcul du coût en performance
        performance_changes = {
            metric: mitigated_perf[metric] - original_perf[metric]
            for metric in original_perf.keys()
        }
        
        # Score de performance composite (moyenne pondérée)
        performance_weights = {'accuracy': 0.3, 'f1': 0.4, 'auc': 0.3}
        performance_cost = -sum(
            performance_changes.get(metric, 0) * weight
            for metric, weight in performance_weights.items()
        )
        
        # Calcul du gain en équité
        fairness_gain = self._calculate_overall_bias_reduction(original_fairness, mitigated_fairness) / 100
        
        # Ratio d'efficacité
        efficiency_ratio = fairness_gain / max(performance_cost, 0.001)  # Éviter division par zéro
        
        return {
            'performance_cost': performance_cost,
            'fairness_gain': fairness_gain,
            'efficiency_ratio': efficiency_ratio,
            'performance_changes': performance_changes,
            'recommendation': self._get_tradeoff_recommendation(performance_cost, fairness_gain)
        }

    def _get_tradeoff_recommendation(self, performance_cost: float, fairness_gain: float) -> str:
        """Génère une recommandation basée sur le trade-off."""
        
        if fairness_gain > 0.2 and performance_cost < 0.05:
            return "EXCELLENT: Gain d'équité élevé avec coût de performance minimal"
        elif fairness_gain > 0.1 and performance_cost < 0.1:
            return "BON: Trade-off favorable entre équité et performance"
        elif fairness_gain > 0.05:
            return "ACCEPTABLE: Amélioration d'équité modérée"
        else:
            return "QUESTIONNABLE: Gain d'équité limité, considérer d'autres approches"


class FairnessAwareCalibrator:
    """
    Calibrateur personnalisé conscient de l'équité pour les prédictions par groupe.
    """
    
    def __init__(self, base_model, group_calibrators: Dict, sensitive_attr_name: str):
        self.base_model = base_model
        self.group_calibrators = group_calibrators
        self.sensitive_attr_name = sensitive_attr_name
    
    def predict(self, X: pd.DataFrame, sensitive_attr: pd.Series) -> np.ndarray:
        """Prédit avec calibration par groupe."""
        predictions = np.zeros(len(X))
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            
            if str(group) in self.group_calibrators:
                calibrator = self.group_calibrators[str(group)]
                group_predictions = calibrator.predict(X[group_mask])
            else:
                # Fallback vers le modèle de base
                group_predictions = self.base_model.predict(X[group_mask])
            
            predictions[group_mask] = group_predictions
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame, sensitive_attr: pd.Series) -> np.ndarray:
        """Prédit les probabilités avec calibration par groupe."""
        probabilities = np.zeros(len(X))
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            
            if str(group) in self.group_calibrators:
                calibrator = self.group_calibrators[str(group)]
                if hasattr(calibrator, 'predict_proba'):
                    group_probabilities = calibrator.predict_proba(X[group_mask])[:, 1]
                else:
                    group_probabilities = calibrator.predict(X[group_mask])
            else:
                # Fallback vers le modèle de base
                if hasattr(self.base_model, 'predict_proba'):
                    group_probabilities = self.base_model.predict_proba(X[group_mask])[:, 1]
                else:
                    group_probabilities = self.base_model.predict(X[group_mask])
            
            probabilities[group_mask] = group_probabilities
        
        return probabilities


def main():
    """Fonction principale pour démonstration du framework de mitigation des biais."""
    
    print("🛡️ Framework de Mitigation des Biais COMPAS - Mac M4 Pro")
    print("=" * 60)
    
    # Initialisation du framework
    framework = BiasMitigationFramework(random_state=42)
    
    # Création de données d'exemple pour démonstration
    print("\n📊 Génération de données COMPAS avec biais simulé...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(35, 10, n_samples).clip(18, 80)
    priors_count = np.random.poisson(3, n_samples)
    
    # Attributs sensibles avec biais
    race = np.random.choice(['African-American', 'Caucasian', 'Hispanic'], 
                           n_samples, p=[0.4, 0.4, 0.2])
    sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2])
    
    # Cible avec biais racial simulé
    race_bias = np.where(race == 'African-American', 0.2, 0.0)
    base_prob = 0.3 + 0.3 * (priors_count / 10) + 0.1 * (age < 25)
    biased_prob = np.clip(base_prob + race_bias, 0, 1)
    
    two_year_recid = np.random.binomial(1, biased_prob, n_samples)
    
    # Création des DataFrames
    X = pd.DataFrame({
        'age': age,
        'priors_count': priors_count,
        'charge_degree_F': np.random.binomial(1, 0.3, n_samples),
        'score_high': np.random.binomial(1, biased_prob, n_samples)
    })
    
    y = pd.Series(two_year_recid)
    sensitive_race = pd.Series(race)
    
    print(f"✅ Données créées: {len(X)} échantillons")
    print(f"📈 Taux de récidive par race:")
    for r in race:
        mask = sensitive_race == r
        print(f"  - {r}: {y[mask].mean():.3f}")
    
    # Démonstration des stratégies de mitigation
    strategies_to_test = [
        'remove_sensitive_features',
        'fairness_sampling',
        'constrained_training',
        'threshold_optimization'
    ]
    
    results = {}
    
    for strategy in strategies_to_test:
        print(f"\n🔧 Test de la stratégie: {strategy}")
        
        try:
            if strategy == 'remove_sensitive_features':
                X_clean, report = framework.remove_sensitive_features(
                    X, y, ['race', 'sex']
                )
                results[strategy] = {
                    'success': True,
                    'method': 'preprocessing',
                    'report': report
                }
                
            elif strategy == 'fairness_sampling':
                X_resampled, y_resampled, report = framework.apply_fairness_sampling(
                    X, y, sensitive_race, strategy='smote_fairness'
                )
                results[strategy] = {
                    'success': True,
                    'method': 'preprocessing',
                    'report': report
                }
                
            elif strategy == 'constrained_training':
                constrained_results = framework.train_fairness_constrained_models(
                    X, y, sensitive_race, constraint_type='demographic_parity',
                    models_to_train=['logistic_regression']
                )
                results[strategy] = {
                    'success': True,
                    'method': 'in_processing',
                    'report': constrained_results['results']
                }
                
            elif strategy == 'threshold_optimization':
                # Entraîner un modèle simple pour la démonstration
                from sklearn.linear_model import LogisticRegression
                simple_model = LogisticRegression(random_state=42)
                simple_model.fit(X, y)
                
                thresholds, report = framework.optimize_decision_thresholds(
                    simple_model, X, y, sensitive_race, constraint='demographic_parity'
                )
                results[strategy] = {
                    'success': True,
                    'method': 'postprocessing',
                    'report': report
                }
            
            print(f"  ✅ {strategy} exécuté avec succès")
            
        except Exception as e:
            print(f"  ❌ Erreur dans {strategy}: {str(e)}")
            results[strategy] = {
                'success': False,
                'error': str(e)
            }
    
    # Résumé des résultats
    print(f"\n📋 Résumé des tests de mitigation:")
    print(f"✅ Stratégies réussies: {sum(1 for r in results.values() if r['success'])}/{len(results)}")
    
    successful_strategies = [name for name, result in results.items() if result['success']]
    if successful_strategies:
        print(f"🎯 Stratégies disponibles: {', '.join(successful_strategies)}")
    
    # Génération d'un rapport de démonstration
    print(f"\n📄 Génération du rapport de démonstration...")
    
    demo_evaluation = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_set_size': len(X),
        'improvement_analysis': {
            'overall_bias_reduction': 15.5,
            'performance_changes': {
                'accuracy': -0.02,
                'f1': -0.01,
                'precision': 0.01,
                'recall': -0.03
            }
        },
        'trade_off_analysis': {
            'performance_cost': 0.02,
            'fairness_gain': 0.155,
            'efficiency_ratio': 7.75
        }
    }
    
    report_path = framework.generate_mitigation_report(
        demo_evaluation, 
        strategy_name="Démonstration_Complète"
    )
    
    print(f"✅ Rapport généré: {report_path}")
    print(f"\n🎉 Démonstration du framework terminée!")
    print(f"🔍 Framework prêt pour l'analyse de biais COMPAS en production")


if __name__ == "__main__":
    main()