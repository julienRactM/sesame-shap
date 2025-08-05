"""
Framework d'entraînement de modèles de machine learning optimisé pour Mac M4 Pro
pour la prédiction de récidive COMPAS.

Ce module implémente plusieurs modèles de classification binaire avec optimisations
spécifiques à l'architecture Apple Silicon, métriques d'équité et préparation
pour l'analyse SHAP.

Auteur: Système d'IA Claude
Date: 2025-08-05
Optimisé pour: Mac M4 Pro avec Apple Silicon
"""

import os
import sys
import warnings
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    calibration_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Métriques d'équité
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio,
    false_positive_rate, false_negative_rate
)

# Configuration pour Apple Silicon et Mac M4 Pro
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Configuration spécifique pour XGBoost sur Mac M4 Pro
xgb.set_config(verbosity=1)

# Configuration des warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/julienrm/Workspace/M2/sesame-shap/data/results/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CompasModelTrainer:
    """
    Classe principale pour l'entraînement et l'évaluation de modèles ML
    optimisée pour Mac M4 Pro et l'analyse d'équité.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialise le trainer de modèles COMPAS.
        
        Args:
            data_path: Chemin vers les données COMPAS
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.random_state = random_state
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.fairness_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_train = None
        self.sensitive_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Configuration des modèles optimisée pour Apple Silicon
        self._configure_models()
        
        # Création des répertoires nécessaires
        self._create_directories()
        
        logger.info("CompasModelTrainer initialisé avec optimisations Mac M4 Pro")
    
    def _configure_models(self):
        """Configure les modèles avec optimisations Apple Silicon."""
        
        # Détection du nombre de cœurs pour optimisation
        n_cores = os.cpu_count()
        
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    n_jobs=n_cores
                ),
                'param_grid': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.15, 0.5, 0.7]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=n_cores
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=n_cores,
                    tree_method='hist',  # Optimisé pour Apple Silicon
                    objective='binary:logistic',
                    eval_metric='logloss'
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [1, 1.5, 2]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=n_cores,
                    objective='binary',
                    metric='binary_logloss',
                    device='cpu',  # Optimisé pour Mac M4 Pro CPU
                    verbose=-1
                ),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 1],
                    'reg_lambda': [0, 0.1, 1]
                }
            },
            'svm': {
                'model': SVC(
                    random_state=self.random_state,
                    probability=True
                ),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            },
            'neural_network': {
                'model': MLPClassifier(
                    random_state=self.random_state,
                    max_iter=1000
                ),
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam', 'lbfgs'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour sauvegarder les résultats."""
        directories = [
            '/Users/julienrm/Workspace/M2/sesame-shap/data/models',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/processed'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Répertoires créés avec succès")
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'two_year_recid',
                            sensitive_attributes: List[str] = ['race', 'sex']) -> None:
        """
        Prépare les données pour l'entraînement des modèles ML.
        
        Args:
            df: DataFrame contenant les données COMPAS
            target_column: Nom de la colonne cible
            sensitive_attributes: Liste des attributs sensibles pour l'analyse d'équité
        """
        logger.info("Début de la préparation des données d'entraînement")
        
        # Sauvegarde des noms de colonnes
        self.feature_names = [col for col in df.columns if col != target_column 
                             and col not in sensitive_attributes]
        
        # Préparation des features et de la cible
        X = df[self.feature_names].copy()
        y = df[target_column].copy()
        sensitive = df[sensitive_attributes].copy()
        
        # Gestion des valeurs manquantes
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(X.mode().iloc[0])
        
        # Encodage des variables catégorielles
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Stratification par cible et attributs sensibles pour la division
        # Création d'un groupe de stratification combiné
        stratify_group = y.astype(str)
        for attr in sensitive_attributes:
            stratify_group = stratify_group + "_" + sensitive[attr].astype(str)
        
        # Division des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
            stratify=stratify_group
        )
        
        self.sensitive_train, self.sensitive_test = train_test_split(
            sensitive, test_size=0.2, random_state=self.random_state,
            stratify=stratify_group
        )
        
        # Normalisation des features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Conversion en DataFrame pour maintenir les noms de colonnes
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.feature_names,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        logger.info(f"Données préparées: {len(self.X_train)} échantillons d'entraînement, "
                   f"{len(self.X_test)} échantillons de test")
        logger.info(f"Nombre de features: {len(self.feature_names)}")
        logger.info(f"Distribution de la cible - Train: {dict(self.y_train.value_counts())}")
        logger.info(f"Distribution de la cible - Test: {dict(self.y_test.value_counts())}")
    
    def train_multiple_models(self, use_hyperparameter_tuning: bool = True,
                            cv_folds: int = 5, n_jobs: int = -1) -> Dict[str, Any]:
        """
        Entraîne plusieurs modèles avec validation croisée.
        
        Args:
            use_hyperparameter_tuning: Utiliser l'optimisation d'hyperparamètres
            cv_folds: Nombre de plis pour la validation croisée
            n_jobs: Nombre de processus parallèles (-1 pour utiliser tous les cœurs)
        
        Returns:
            Dictionnaire contenant les modèles entraînés et leurs scores
        """
        logger.info("Début de l'entraînement des modèles multiples")
        
        if self.X_train is None:
            raise ValueError("Les données d'entraînement ne sont pas préparées. "
                           "Appelez prepare_training_data() d'abord.")
        
        # Configuration de la validation croisée stratifiée
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        trained_models = {}
        cv_scores = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Entraînement du modèle: {model_name}")
            
            try:
                model = config['model']
                
                # Choix des données (normalisées pour SVM et NN, originales pour les autres)
                if model_name in ['svm', 'neural_network']:
                    X_train_current = self.X_train_scaled
                    X_test_current = self.X_test_scaled
                else:
                    X_train_current = self.X_train
                    X_test_current = self.X_test
                
                if use_hyperparameter_tuning and model_name != 'svm':
                    # Optimisation d'hyperparamètres (évitée pour SVM à cause du temps)
                    logger.info(f"Optimisation d'hyperparamètres pour {model_name}")
                    
                    # Utilisation de RandomizedSearchCV pour efficacité sur Mac M4 Pro
                    search = RandomizedSearchCV(
                        model, 
                        config['param_grid'],
                        n_iter=20,  # Limité pour performance
                        cv=cv_strategy,
                        scoring='roc_auc',
                        n_jobs=n_jobs if n_jobs != -1 else os.cpu_count(),
                        random_state=self.random_state,
                        verbose=1
                    )
                    
                    search.fit(X_train_current, self.y_train)
                    best_model = search.best_estimator_
                    
                    logger.info(f"Meilleurs paramètres pour {model_name}: {search.best_params_}")
                    logger.info(f"Meilleur score CV: {search.best_score_:.4f}")
                
                else:
                    # Entraînement avec paramètres par défaut
                    best_model = model
                    best_model.fit(X_train_current, self.y_train)
                
                # Validation croisée pour évaluation robuste
                cv_score = cross_val_score(
                    best_model, X_train_current, self.y_train,
                    cv=cv_strategy, scoring='roc_auc', n_jobs=n_jobs
                )
                
                trained_models[model_name] = best_model
                cv_scores[model_name] = {
                    'mean': cv_score.mean(),
                    'std': cv_score.std(),
                    'scores': cv_score.tolist()
                }
                
                logger.info(f"{model_name} - Score CV moyen: {cv_score.mean():.4f} "
                           f"(±{cv_score.std():.4f})")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.cv_scores = cv_scores
        
        logger.info(f"Entraînement terminé. {len(trained_models)} modèles entraînés avec succès.")
        
        return {
            'models': trained_models,
            'cv_scores': cv_scores
        }
    
    def evaluate_models(self, include_fairness: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Évalue tous les modèles entraînés avec métriques complètes.
        
        Args:
            include_fairness: Inclure les métriques d'équité
        
        Returns:
            Dictionnaire contenant toutes les métriques d'évaluation
        """
        logger.info("Début de l'évaluation des modèles")
        
        if not self.models:
            raise ValueError("Aucun modèle entraîné. Appelez train_multiple_models() d'abord.")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Évaluation du modèle: {model_name}")
            
            # Choix des données appropriées
            if model_name in ['svm', 'neural_network']:
                X_test_current = self.X_test_scaled
            else:
                X_test_current = self.X_test
            
            # Prédictions
            y_pred = model.predict(X_test_current)
            y_pred_proba = model.predict_proba(X_test_current)[:, 1]
            
            # Métriques de performance standard
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
                'brier_score': brier_score_loss(self.y_test, y_pred_proba)
            }
            
            # Matrice de confusion
            cm = confusion_matrix(self.y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Rapport de classification
            class_report = classification_report(
                self.y_test, y_pred, output_dict=True
            )
            metrics['classification_report'] = class_report
            
            # Courbes ROC et Precision-Recall
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(
                self.y_test, y_pred_proba
            )
            
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist()
            }
            
            # Analyse de calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, y_pred_proba, n_bins=10
            )
            metrics['calibration'] = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
            
            # Métriques d'équité
            if include_fairness and self.sensitive_test is not None:
                fairness_metrics = self._calculate_fairness_metrics(
                    y_pred, y_pred_proba, model_name
                )
                metrics['fairness'] = fairness_metrics
            
            evaluation_results[model_name] = metrics
            
            logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")
        
        self.results = evaluation_results
        logger.info("Évaluation des modèles terminée")
        
        return evaluation_results
    
    def _calculate_fairness_metrics(self, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                                  model_name: str) -> Dict[str, Any]:
        """
        Calcule les métriques d'équité pour un modèle donné.
        
        Args:
            y_pred: Prédictions binaires
            y_pred_proba: Probabilités prédites
            model_name: Nom du modèle
        
        Returns:
            Dictionnaire des métriques d'équité
        """
        fairness_metrics = {}
        
        for sensitive_attr in self.sensitive_test.columns:
            sensitive_values = self.sensitive_test[sensitive_attr]
            
            # Métriques par groupe
            group_metrics = {}
            for group in sensitive_values.unique():
                mask = sensitive_values == group
                group_y_true = self.y_test[mask]
                group_y_pred = y_pred[mask]
                group_y_proba = y_pred_proba[mask]
                
                if len(group_y_true) > 0:
                    group_metrics[str(group)] = {
                        'count': len(group_y_true),
                        'accuracy': accuracy_score(group_y_true, group_y_pred),
                        'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                        'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                        'f1': f1_score(group_y_true, group_y_pred, zero_division=0),
                        'auc': roc_auc_score(group_y_true, group_y_proba) if len(np.unique(group_y_true)) > 1 else 0,
                        'positive_rate': np.mean(group_y_pred),
                        'fpr': false_positive_rate(group_y_true, group_y_pred),
                        'fnr': false_negative_rate(group_y_true, group_y_pred)
                    }
            
            # Métriques d'équité globales
            try:
                dp_diff = demographic_parity_difference(
                    self.y_test, y_pred, sensitive_features=sensitive_values
                )
                dp_ratio = demographic_parity_ratio(
                    self.y_test, y_pred, sensitive_features=sensitive_values
                )
                eo_diff = equalized_odds_difference(
                    self.y_test, y_pred, sensitive_features=sensitive_values
                )
                eo_ratio = equalized_odds_ratio(
                    self.y_test, y_pred, sensitive_features=sensitive_values
                )
                
                fairness_metrics[sensitive_attr] = {
                    'group_metrics': group_metrics,
                    'demographic_parity_difference': dp_diff,
                    'demographic_parity_ratio': dp_ratio,
                    'equalized_odds_difference': eo_diff,
                    'equalized_odds_ratio': eo_ratio
                }
                
            except Exception as e:
                logger.warning(f"Erreur calcul métriques d'équité pour {sensitive_attr}: {e}")
                fairness_metrics[sensitive_attr] = {
                    'group_metrics': group_metrics,
                    'error': str(e)
                }
        
        return fairness_metrics
    
    def compare_model_performance(self, save_plots: bool = True) -> pd.DataFrame:
        """
        Compare les performances de tous les modèles côte à côte.
        
        Args:
            save_plots: Sauvegarder les graphiques de comparaison
        
        Returns:
            DataFrame avec comparaison des métriques
        """
        logger.info("Début de la comparaison des performances des modèles")
        
        if not self.results:
            raise ValueError("Aucun résultat d'évaluation disponible. "
                           "Appelez evaluate_models() d'abord.")
        
        # Création du DataFrame de comparaison
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Modèle': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc'],
                'Brier Score': metrics['brier_score']
            }
            
            # Ajout des scores de validation croisée si disponibles
            if hasattr(self, 'cv_scores') and model_name in self.cv_scores:
                row['CV Score (mean)'] = self.cv_scores[model_name]['mean']
                row['CV Score (std)'] = self.cv_scores[model_name]['std']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
        
        # Sauvegarde du tableau de comparaison
        comparison_df.to_csv(
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/model_comparison.csv',
            index=False
        )
        
        if save_plots:
            self._create_comparison_plots(comparison_df)
        
        logger.info("Comparaison des modèles terminée")
        logger.info(f"Meilleur modèle (AUC): {comparison_df.iloc[0]['Modèle']} "
                   f"({comparison_df.iloc[0]['AUC-ROC']:.4f})")
        
        return comparison_df
    
    def _create_comparison_plots(self, comparison_df: pd.DataFrame):
        """Crée des graphiques de comparaison des modèles."""
        
        # Configuration du style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Graphique en barres des métriques principales
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            sns.barplot(data=comparison_df, x='Modèle', y=metric, ax=ax)
            ax.set_title(f'{metric}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            # Ajout des valeurs sur les barres
            for j, v in enumerate(comparison_df[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/julienrm/Workspace/M2/sesame-shap/data/results/model_comparison_metrics.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Graphique radar des performances
        fig = go.Figure()
        
        metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for _, row in comparison_df.iterrows():
            values = [row[metric] for metric in metrics_radar]
            values.append(values[0])  # Fermer le polygone
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_radar + [metrics_radar[0]],
                fill='toself',
                name=row['Modèle'],
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Graphique Radar - Comparaison des Modèles",
            showlegend=True
        )
        
        fig.write_html('/Users/julienrm/Workspace/M2/sesame-shap/data/results/model_comparison_radar.html')
        
        # 3. Courbes ROC combinées
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in self.results.items():
            if 'roc_curve' in metrics:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                auc = metrics['auc_roc']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Chance aléatoire')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbes ROC - Comparaison des Modèles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('/Users/julienrm/Workspace/M2/sesame-shap/data/results/roc_curves_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Graphiques de comparaison sauvegardés")
    
    def save_trained_models(self, save_path: str = None) -> Dict[str, str]:
        """
        Sauvegarde tous les modèles entraînés.
        
        Args:
            save_path: Chemin de sauvegarde personnalisé
        
        Returns:
            Dictionnaire des chemins de sauvegarde
        """
        if save_path is None:
            save_path = '/Users/julienrm/Workspace/M2/sesame-shap/data/models'
        
        if not self.models:
            raise ValueError("Aucun modèle à sauvegarder. Entraînez des modèles d'abord.")
        
        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            # Sauvegarde du modèle
            model_path = f"{save_path}/compas_{model_name}_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            saved_paths[f'{model_name}_model'] = model_path
            
            logger.info(f"Modèle {model_name} sauvegardé: {model_path}")
        
        # Sauvegarde du scaler
        scaler_path = f"{save_path}/compas_scaler_{timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        saved_paths['scaler'] = scaler_path
        
        # Sauvegarde des métadonnées
        metadata = {
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'model_names': list(self.models.keys()),
            'random_state': self.random_state,
            'cv_scores': getattr(self, 'cv_scores', {}),
            'evaluation_results': self.results
        }
        
        metadata_path = f"{save_path}/compas_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_paths['metadata'] = metadata_path
        
        logger.info(f"Tous les modèles et métadonnées sauvegardés avec timestamp: {timestamp}")
        
        return saved_paths
    
    def generate_training_report(self) -> str:
        """
        Génère un rapport complet de l'entraînement et de l'évaluation.
        
        Returns:
            Chemin vers le rapport généré
        """
        if not self.results:
            raise ValueError("Aucun résultat à rapporter. Évaluez les modèles d'abord.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/training_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Rapport d'Entraînement des Modèles COMPAS\n\n")
            f.write(f"**Date de génération:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Optimisé pour:** Mac M4 Pro avec Apple Silicon\n\n")
            
            # Résumé des données
            f.write("## Résumé des Données\n\n")
            f.write(f"- **Échantillons d'entraînement:** {len(self.X_train)}\n")
            f.write(f"- **Échantillons de test:** {len(self.X_test)}\n")
            f.write(f"- **Nombre de features:** {len(self.feature_names)}\n")
            f.write(f"- **Distribution cible (train):** {dict(self.y_train.value_counts())}\n")
            f.write(f"- **Distribution cible (test):** {dict(self.y_test.value_counts())}\n\n")
            
            # Performances des modèles
            f.write("## Performances des Modèles\n\n")
            comparison_df = self.compare_model_performance(save_plots=False)
            f.write(comparison_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            
            # Métriques détaillées par modèle
            f.write("## Métriques Détaillées par Modèle\n\n")
            for model_name, metrics in self.results.items():
                f.write(f"### {model_name.title()}\n\n")
                f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
                f.write(f"- **Precision:** {metrics['precision']:.4f}\n")
                f.write(f"- **Recall:** {metrics['recall']:.4f}\n")
                f.write(f"- **F1-Score:** {metrics['f1']:.4f}\n")
                f.write(f"- **AUC-ROC:** {metrics['auc_roc']:.4f}\n")
                f.write(f"- **Brier Score:** {metrics['brier_score']:.4f}\n\n")
                
                # Métriques d'équité si disponibles
                if 'fairness' in metrics:
                    f.write("#### Métriques d'Équité\n\n")
                    for attr, fairness_data in metrics['fairness'].items():
                        if 'error' not in fairness_data:
                            f.write(f"**{attr}:**\n")
                            f.write(f"- Différence de parité démographique: {fairness_data.get('demographic_parity_difference', 'N/A'):.4f}\n")
                            f.write(f"- Ratio de parité démographique: {fairness_data.get('demographic_parity_ratio', 'N/A'):.4f}\n")
                            f.write(f"- Différence d'égalité des chances: {fairness_data.get('equalized_odds_difference', 'N/A'):.4f}\n")
                            f.write(f"- Ratio d'égalité des chances: {fairness_data.get('equalized_odds_ratio', 'N/A'):.4f}\n\n")
            
            # Recommandations
            f.write("## Recommandations\n\n")
            best_model = comparison_df.iloc[0]['Modèle']
            best_auc = comparison_df.iloc[0]['AUC-ROC']
            
            f.write(f"**Meilleur modèle:** {best_model} (AUC = {best_auc:.4f})\n\n")
            f.write("### Prochaines étapes recommandées:\n\n")
            f.write("1. **Analyse SHAP** du meilleur modèle pour l'interprétabilité\n")
            f.write("2. **Analyse approfondie d'équité** avec métriques supplémentaires\n")
            f.write("3. **Calibration du modèle** si nécessaire\n")
            f.write("4. **Tests sur données nouvelles** pour validation\n")
            f.write("5. **Optimisation supplémentaire** des hyperparamètres\n\n")
            
            # Configuration technique
            f.write("## Configuration Technique\n\n")
            f.write(f"- **Plateforme:** macOS avec Apple Silicon M4 Pro\n")
            f.write(f"- **Cœurs utilisés:** {os.cpu_count()}\n")
            f.write(f"- **Graine aléatoire:** {self.random_state}\n")
            f.write(f"- **Validation croisée:** Stratifiée, 5 plis\n")
            f.write(f"- **Optimisation hyperparamètres:** RandomizedSearchCV\n\n")
        
        logger.info(f"Rapport de formation généré: {report_path}")
        return report_path


def main():
    """Fonction principale pour démonstration et tests."""
    
    print("🍎 Framework d'Entraînement ML COMPAS - Optimisé Mac M4 Pro")
    print("=" * 60)
    
    # Initialisation du trainer
    trainer = CompasModelTrainer()
    
    # Exemple d'utilisation avec données fictives pour démonstration
    # Dans un cas réel, vous chargeriez les vraies données COMPAS
    print("\n📊 Génération de données d'exemple pour démonstration...")
    
    # Création de données d'exemple (remplacez par le chargement des vraies données)
    np.random.seed(42)
    n_samples = 1000
    
    # Features numériques
    age = np.random.normal(35, 10, n_samples)
    priors_count = np.random.poisson(3, n_samples)
    
    # Features catégorielles
    race = np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], n_samples)
    sex = np.random.choice(['Male', 'Female'], n_samples)
    charge_degree = np.random.choice(['F', 'M'], n_samples)
    
    # Cible (avec biais simulé pour l'analyse d'équité)
    bias_factor = np.where(race == 'African-American', 0.3, 0.0)
    two_year_recid = np.random.binomial(1, 0.4 + bias_factor, n_samples)
    
    # Création du DataFrame
    demo_data = pd.DataFrame({
        'age': age,
        'priors_count': priors_count,
        'race': race,
        'sex': sex,
        'charge_degree': charge_degree,
        'two_year_recid': two_year_recid
    })
    
    print(f"✅ Données d'exemple créées: {len(demo_data)} échantillons")
    print(f"📈 Distribution de récidive: {dict(demo_data['two_year_recid'].value_counts())}")
    
    # Préparation des données
    print("\n🔧 Préparation des données d'entraînement...")
    trainer.prepare_training_data(demo_data)
    
    # Entraînement des modèles
    print("\n🚀 Entraînement des modèles multiples...")
    print("⚡ Optimisations Apple Silicon activées")
    
    training_results = trainer.train_multiple_models(
        use_hyperparameter_tuning=True,
        cv_folds=5
    )
    
    print(f"✅ {len(training_results['models'])} modèles entraînés avec succès")
    
    # Évaluation des modèles
    print("\n📊 Évaluation complète des modèles...")
    evaluation_results = trainer.evaluate_models(include_fairness=True)
    
    # Comparaison des performances
    print("\n🏆 Comparaison des performances...")
    comparison_df = trainer.compare_model_performance()
    
    print("\n📈 Résultats de comparaison:")
    print(comparison_df[['Modèle', 'Accuracy', 'F1-Score', 'AUC-ROC']].to_string(index=False))
    
    # Sauvegarde des modèles
    print("\n💾 Sauvegarde des modèles entraînés...")
    saved_paths = trainer.save_trained_models()
    
    print("✅ Modèles sauvegardés:")
    for key, path in saved_paths.items():
        print(f"  • {key}: {path}")
    
    # Génération du rapport
    print("\n📋 Génération du rapport final...")
    report_path = trainer.generate_training_report()
    print(f"✅ Rapport généré: {report_path}")
    
    print("\n🎉 Entraînement terminé avec succès!")
    print("🔍 Prêt pour l'analyse SHAP et l'interprétabilité")
    

if __name__ == "__main__":
    main()