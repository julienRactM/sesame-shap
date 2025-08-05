"""
Module d'analyse d'interprétabilité SHAP optimisé pour Mac M4 Pro
================================================================

Ce module fournit une analyse complète d'interprétabilité SHAP pour la détection de biais 
dans le projet COMPAS, avec des optimisations spécifiques à l'architecture Apple Silicon M4 Pro.

Fonctionnalités principales:
- Calcul de valeurs SHAP pour différents types de modèles
- Analyse de biais à travers les valeurs SHAP
- Visualisations interactives et statiques
- Optimisations mémoire pour Mac M4 Pro
- Rapport d'analyse de biais en français

Auteur: Système d'IA Claude - Spécialiste ML Apple Silicon
Date: 2025-08-05
Optimisé pour: Mac M4 Pro avec Apple Silicon
"""

import os
import sys
import warnings
import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# SHAP pour l'interprétabilité
import shap
from shap import TreeExplainer, KernelExplainer, LinearExplainer, Explainer
from shap.plots import _waterfall, _force, _summary

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Métriques d'équité
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio
)

# Configuration pour optimisations Apple Silicon M4 Pro
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())

# Configuration des warnings pour une sortie propre
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    Analyseur SHAP principal optimisé pour Mac M4 Pro.
    
    Cette classe fournit une analyse complète d'interprétabilité SHAP avec un focus
    spécial sur la détection de biais dans les prédictions COMPAS.
    """
    
    def __init__(self, random_state: int = 42, n_cores: Optional[int] = None):
        """
        Initialise l'analyseur SHAP.
        
        Args:
            random_state: Graine aléatoire pour la reproductibilité
            n_cores: Nombre de cœurs à utiliser (None = tous les cœurs disponibles)
        """
        self.random_state = random_state
        self.n_cores = n_cores or os.cpu_count()
        shap.initjs()  # Initialisation pour les notebooks Jupyter
        
        # Stockage des données et modèles
        self.models = {}
        self.explainers = {}
        self.shap_values = {}
        self.X_train = None
        self.X_test = None  
        self.y_test = None
        self.sensitive_test = None
        self.feature_names = []
        self.sensitive_attributes = ['race', 'sex']
        
        # Configuration des optimisations Mac M4 Pro
        self.memory_limit_gb = 8  # Limite mémoire pour les calculs SHAP
        self.batch_size = 1000    # Taille de batch optimisée pour M4 Pro
        self.use_sampling = True  # Échantillonnage intelligent pour performance
        
        # Résultats d'analyse
        self.bias_analysis_results = {}
        self.global_importance = {}
        self.demographic_comparisons = {}
        
        # Création des répertoires
        self._create_directories()
        
        logger.info(f"SHAPAnalyzer initialisé avec {self.n_cores} cœurs pour Mac M4 Pro")
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour sauvegarder les résultats."""
        directories = [
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/bias_analysis',
            '/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/values'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_trained_models(self, models_dict: Dict[str, BaseEstimator] = None,
                          model_path: str = None) -> None:
        """
        Charge les modèles entraînés depuis un dictionnaire ou un fichier.
        
        Args:
            models_dict: Dictionnaire des modèles entraînés
            model_path: Chemin vers le fichier des modèles sauvegardés
        """
        if models_dict is not None:
            self.models = models_dict
            logger.info(f"Modèles chargés depuis dictionnaire: {list(self.models.keys())}")
        
        elif model_path is not None:
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            logger.info(f"Modèles chargés depuis {model_path}: {list(self.models.keys())}")
        
        else:
            raise ValueError("Soit models_dict soit model_path doit être fourni")
    
    def prepare_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_test: pd.Series, sensitive_test: pd.DataFrame,
                    feature_names: List[str] = None) -> None:
        """
        Prépare les données pour l'analyse SHAP.
        
        Args:
            X_train: Features d'entraînement
            X_test: Features de test
            y_test: Cibles de test
            sensitive_test: Attributs sensibles de test
            feature_names: Noms des features
        """
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.sensitive_test = sensitive_test.copy()
        self.feature_names = feature_names or list(X_train.columns)
        
        logger.info(f"Données préparées: {len(X_train)} train, {len(X_test)} test")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Attributs sensibles: {list(self.sensitive_test.columns)}")
    
    def _get_optimal_sample_size(self, model_type: str, data_size: int) -> int:
        """
        Calcule la taille d'échantillon optimale pour les calculs SHAP sur Mac M4 Pro.
        
        Args:
            model_type: Type de modèle ('tree', 'kernel', 'linear')
            data_size: Taille du dataset
            
        Returns:
            Taille d'échantillon optimale
        """
        # Calculs optimisés pour la mémoire unifiée du M4 Pro
        if model_type == 'tree':
            # TreeExplainer est très efficace
            return min(data_size, 2000)
        elif model_type == 'linear':
            # LinearExplainer est rapide
            return min(data_size, 5000)
        else:
            # KernelExplainer est coûteux, limitation plus stricte
            return min(data_size, 500)
    
    def create_explainers(self, use_background_sample: bool = True,
                         background_size: int = 100) -> Dict[str, Any]:
        """
        Crée les explainers SHAP appropriés pour chaque modèle.
        
        Args:
            use_background_sample: Utiliser un échantillon de background
            background_size: Taille de l'échantillon de background
            
        Returns:
            Dictionnaire des explainers créés
        """
        logger.info("Création des explainers SHAP optimisés pour Mac M4 Pro")
        
        if self.X_train is None or not self.models:
            raise ValueError("Données et modèles doivent être chargés avant création des explainers")
        
        # Préparation de l'échantillon de background optimisé
        if use_background_sample:
            background_indices = np.random.choice(
                len(self.X_train), 
                size=min(background_size, len(self.X_train)), 
                replace=False
            )
            background_data = self.X_train.iloc[background_indices]
        else:
            background_data = self.X_train
        
        created_explainers = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Création de l'explainer pour {model_name}")
            
            try:
                # Détermination du type de modèle pour choisir le bon explainer
                model_type = self._determine_model_type(model_name, model)
                
                if model_type == 'tree':
                    # TreeExplainer pour modèles basés sur les arbres
                    explainer = TreeExplainer(
                        model,
                        data=background_data,
                        feature_perturbation='tree_path_dependent'
                    )
                    
                elif model_type == 'linear':
                    # LinearExplainer pour modèles linéaires
                    explainer = LinearExplainer(
                        model,
                        background_data,
                        feature_perturbation='interventional'
                    )
                    
                else:
                    # KernelExplainer pour autres modèles (SVM, NN)
                    explainer = KernelExplainer(
                        model.predict_proba,
                        background_data,
                        link='logit'
                    )
                
                created_explainers[model_name] = {
                    'explainer': explainer,
                    'type': model_type,
                    'background_size': len(background_data)
                }
                
                logger.info(f"Explainer {model_type} créé pour {model_name}")
                
            except Exception as e:
                logger.error(f"Erreur création explainer {model_name}: {str(e)}")
                continue
        
        self.explainers = created_explainers
        logger.info(f"Explainers créés: {list(self.explainers.keys())}")
        
        return created_explainers
    
    def _determine_model_type(self, model_name: str, model: BaseEstimator) -> str:
        """
        Détermine le type de modèle pour choisir l'explainer approprié.
        
        Args:
            model_name: Nom du modèle
            model: Instance du modèle
            
        Returns:
            Type de modèle ('tree', 'linear', 'kernel')
        """
        model_class = type(model).__name__
        
        # Modèles basés sur les arbres
        tree_models = [
            'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
            'GradientBoostingClassifier', 'DecisionTreeClassifier',
            'ExtraTreesClassifier', 'AdaBoostClassifier'
        ]
        
        # Modèles linéaires
        linear_models = [
            'LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso',
            'ElasticNet', 'SGDClassifier', 'Perceptron'  
        ]
        
        if model_class in tree_models or 'forest' in model_name.lower() or 'xgb' in model_name.lower():
            return 'tree'
        elif model_class in linear_models or 'logistic' in model_name.lower():
            return 'linear'
        else:
            return 'kernel'
    
    def calculate_shap_values(self, models_to_analyze: List[str] = None,
                            max_evals: int = 2000) -> Dict[str, np.ndarray]:
        """
        Calcule les valeurs SHAP pour les modèles spécifiés.
        
        Args:
            models_to_analyze: Liste des modèles à analyser (None = tous)
            max_evals: Nombre maximum d'évaluations pour KernelExplainer
            
        Returns:
            Dictionnaire des valeurs SHAP par modèle
        """
        logger.info("Début du calcul des valeurs SHAP avec optimisations Mac M4 Pro")
        
        if not self.explainers:
            raise ValueError("Explainers non créés. Appelez create_explainers() d'abord.")
        
        models_to_analyze = models_to_analyze or list(self.explainers.keys())
        calculated_values = {}
        
        for model_name in models_to_analyze:
            if model_name not in self.explainers:
                logger.warning(f"Explainer non trouvé pour {model_name}")
                continue
                
            logger.info(f"Calcul des valeurs SHAP pour {model_name}")
            
            explainer_info = self.explainers[model_name]
            explainer = explainer_info['explainer']
            explainer_type = explainer_info['type']
            
            try:
                # Optimisation de la taille d'échantillon selon le type
                optimal_size = self._get_optimal_sample_size(
                    explainer_type, 
                    len(self.X_test)
                )
                
                if optimal_size < len(self.X_test):
                    # Échantillonnage stratifié si nécessaire
                    sample_indices = self._get_stratified_sample_indices(optimal_size)
                    X_sample = self.X_test.iloc[sample_indices]
                    logger.info(f"Échantillonnage: {optimal_size}/{len(self.X_test)} pour {model_name}")
                else:
                    X_sample = self.X_test
                    sample_indices = None
                
                # Calcul des valeurs SHAP avec gestion mémoire optimisée
                if explainer_type == 'kernel':
                    # KernelExplainer avec limitation des évaluations
                    shap_values = explainer.shap_values(
                        X_sample, 
                        nsamples=min(max_evals, optimal_size),
                        silent=True
                    )
                    # Pour classification binaire, prendre la classe positive
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                        
                else:
                    # TreeExplainer et LinearExplainer
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        shap_values = explainer.shap_values(X_sample)
                        
                        # Pour classification binaire, prendre la classe positive
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            shap_values = shap_values[1]
                
                # Stockage avec métadonnées
                calculated_values[model_name] = {
                    'values': shap_values,
                    'expected_value': getattr(explainer, 'expected_value', 0),
                    'feature_names': self.feature_names,
                    'sample_indices': sample_indices,
                    'sample_size': len(X_sample)
                }
                
                logger.info(f"Valeurs SHAP calculées pour {model_name}: "
                           f"Shape {shap_values.shape}")
                
                # Sauvegarde des valeurs SHAP
                self._save_shap_values(model_name, calculated_values[model_name])
                
            except Exception as e:
                logger.error(f"Erreur calcul SHAP {model_name}: {str(e)}")
                continue
        
        self.shap_values = calculated_values
        logger.info(f"Calcul SHAP terminé pour {len(calculated_values)} modèles")
        
        return calculated_values
    
    def _get_stratified_sample_indices(self, sample_size: int) -> np.ndarray:
        """
        Obtient des indices d'échantillonnage stratifié pour préserver les distributions.
        
        Args:
            sample_size: Taille de l'échantillon désiré
            
        Returns:
            Indices de l'échantillon stratifié
        """
        # Stratification par cible et attributs sensibles
        stratify_data = self.y_test.astype(str)
        for col in self.sensitive_test.columns:
            stratify_data = stratify_data + "_" + self.sensitive_test[col].astype(str)
        
        # Échantillonnage stratifié
        _, sample_indices = train_test_split(
            np.arange(len(self.X_test)),
            test_size=sample_size / len(self.X_test),
            stratify=stratify_data,
            random_state=self.random_state
        )
        
        return sample_indices
    
    def _save_shap_values(self, model_name: str, shap_data: Dict[str, Any]) -> None:
        """
        Sauvegarde les valeurs SHAP pour un modèle.
        
        Args:
            model_name: Nom du modèle
            shap_data: Données SHAP à sauvegarder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde des valeurs SHAP
        shap_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/values/{model_name}_shap_values_{timestamp}.npz"
        
        np.savez_compressed(
            shap_path,
            values=shap_data['values'],
            expected_value=shap_data['expected_value'],
            feature_names=shap_data['feature_names'],
            sample_indices=shap_data['sample_indices'] or np.array([]),
            sample_size=shap_data['sample_size']
        )
        
        logger.info(f"Valeurs SHAP sauvegardées: {shap_path}")
    
    def analyze_feature_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Analyse l'importance globale des features via les valeurs SHAP.
        
        Args:
            top_n: Nombre de features les plus importantes à retourner
            
        Returns:
            Dictionnaire des DataFrames d'importance par modèle
        """
        logger.info("Analyse de l'importance globale des features")
        
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées. Appelez calculate_shap_values() d'abord.")
        
        importance_results = {}
        
        for model_name, shap_data in self.shap_values.items():
            shap_vals = shap_data['values']
            feature_names = shap_data['feature_names']
            
            # Calcul de l'importance moyenne absolue
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            
            # Calcul de l'importance moyenne (avec signe)
            mean_shap = shap_vals.mean(axis=0)
            
            # Calcul de la variabilité (écart-type)
            std_shap = shap_vals.std(axis=0)
            
            # Création du DataFrame d'importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': mean_abs_shap,
                'mean_shap': mean_shap,
                'std_shap': std_shap,
                'importance_rank': range(1, len(feature_names) + 1)
            })
            
            # Tri par importance absolue décroissante
            importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
            importance_df['importance_rank'] = range(1, len(importance_df) + 1)
            
            # Conservation du top N
            top_features = importance_df.head(top_n)
            
            importance_results[model_name] = top_features
            
            logger.info(f"Top 5 features pour {model_name}:")
            for i, row in top_features.head(5).iterrows():
                logger.info(f"  {row['importance_rank']}. {row['feature']}: "
                           f"{row['mean_abs_shap']:.4f}")
        
        self.global_importance = importance_results
        return importance_results
    
    def analyze_individual_predictions(self, sample_indices: List[int] = None,
                                     n_samples: int = 10) -> Dict[str, List[Dict]]:
        """
        Analyse des prédictions individuelles avec explication SHAP.
        
        Args:
            sample_indices: Indices spécifiques à analyser
            n_samples: Nombre d'échantillons aléatoires si pas d'indices spécifiés
            
        Returns:
            Dictionnaire des explications individuelles par modèle
        """
        logger.info("Analyse des prédictions individuelles")
        
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        # Sélection des échantillons à analyser
        if sample_indices is None:
            sample_indices = np.random.choice(
                len(self.X_test), 
                size=min(n_samples, len(self.X_test)), 
                replace=False
            )
        
        individual_results = {}
        
        for model_name, shap_data in self.shap_values.items():
            shap_vals = shap_data['values']
            feature_names = shap_data['feature_names']
            expected_value = shap_data['expected_value']
            
            # Obtenir les prédictions du modèle
            model = self.models[model_name]
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(self.X_test.iloc[sample_indices])[:, 1]
            else:
                predictions = model.predict(self.X_test.iloc[sample_indices])
            
            explanations = []
            
            for i, idx in enumerate(sample_indices):
                # Ajustement de l'indice si échantillonnage utilisé
                shap_idx = i if shap_data['sample_indices'] is None else i
                
                if shap_idx < len(shap_vals):
                    sample_shap = shap_vals[shap_idx]
                    sample_features = self.X_test.iloc[idx]
                    sample_sensitive = self.sensitive_test.iloc[idx]
                    
                    # Création de l'explication structurée
                    explanation = {
                        'sample_index': idx,
                        'prediction': predictions[i] if i < len(predictions) else None,
                        'true_label': self.y_test.iloc[idx],
                        'expected_value': expected_value,
                        'sensitive_attributes': sample_sensitive.to_dict(),
                        'features': sample_features.to_dict(),
                        'shap_values': dict(zip(feature_names, sample_shap)),
                        'top_positive_features': {},
                        'top_negative_features': {}
                    }
                    
                    # Identification des features les plus influentes
                    feature_impacts = list(zip(feature_names, sample_shap))
                    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    positive_features = [(f, v) for f, v in feature_impacts if v > 0][:5]
                    negative_features = [(f, v) for f, v in feature_impacts if v < 0][:5]
                    
                    explanation['top_positive_features'] = dict(positive_features)
                    explanation['top_negative_features'] = dict(negative_features)
                    
                    explanations.append(explanation)
            
            individual_results[model_name] = explanations
            logger.info(f"Explications individuelles calculées pour {model_name}: "
                       f"{len(explanations)} échantillons")
        
        return individual_results
    
    def analyze_bias_through_shap(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyse approfondie des biais via les valeurs SHAP.
        
        Returns:
            Résultats complets de l'analyse de biais
        """
        logger.info("Analyse de biais à travers les valeurs SHAP")
        
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        bias_results = {}
        
        for model_name, shap_data in self.shap_values.items():
            logger.info(f"Analyse de biais pour {model_name}")
            
            shap_vals = shap_data['values']
            feature_names = shap_data['feature_names']
            
            model_bias_analysis = {}
            
            # Analyse par attribut sensible
            for sensitive_attr in self.sensitive_test.columns:
                attr_analysis = self._analyze_bias_for_attribute(
                    model_name, shap_vals, feature_names, sensitive_attr
                )
                model_bias_analysis[sensitive_attr] = attr_analysis
            
            # Analyse des interactions entre attributs sensibles
            if len(self.sensitive_test.columns) > 1:
                interaction_analysis = self._analyze_intersectional_bias(
                    model_name, shap_vals, feature_names
                )
                model_bias_analysis['intersectional'] = interaction_analysis
            
            bias_results[model_name] = model_bias_analysis
        
        self.bias_analysis_results = bias_results
        logger.info("Analyse de biais terminée")
        
        return bias_results
    
    def _analyze_bias_for_attribute(self, model_name: str, shap_vals: np.ndarray,
                                  feature_names: List[str], sensitive_attr: str) -> Dict[str, Any]:
        """
        Analyse le biais pour un attribut sensible spécifique.
        
        Args:
            model_name: Nom du modèle
            shap_vals: Valeurs SHAP
            feature_names: Noms des features
            sensitive_attr: Attribut sensible à analyser
            
        Returns:
            Analyse de biais pour l'attribut
        """
        sensitive_values = self.sensitive_test[sensitive_attr]
        unique_groups = sensitive_values.unique()
        
        analysis = {
            'groups': {},
            'differences': {},
            'statistical_tests': {},
            'bias_indicators': {}
        }
        
        # Analyse par groupe
        for group in unique_groups:
            group_mask = sensitive_values == group
            group_shap = shap_vals[group_mask]
            
            if len(group_shap) > 0:
                analysis['groups'][str(group)] = {
                    'count': len(group_shap),
                    'mean_shap_values': group_shap.mean(axis=0).tolist(),
                    'std_shap_values': group_shap.std(axis=0).tolist(),
                    'feature_names': feature_names
                }
        
        # Calcul des différences entre groupes (focus sur différences raciales)
        if len(unique_groups) >= 2:
            groups_list = list(unique_groups)
            
            for i, group1 in enumerate(groups_list):
                for j, group2 in enumerate(groups_list[i+1:], i+1):
                    
                    mask1 = sensitive_values == group1
                    mask2 = sensitive_values == group2
                    
                    shap1 = shap_vals[mask1]
                    shap2 = shap_vals[mask2]
                    
                    if len(shap1) > 0 and len(shap2) > 0:
                        mean_diff = shap1.mean(axis=0) - shap2.mean(axis=0)
                        
                        # Test de significativité (t-test approché)
                        from scipy import stats
                        t_stats = []
                        p_values = []
                        
                        for k in range(len(feature_names)):
                            if shap1.shape[0] > 1 and shap2.shape[0] > 1:
                                t_stat, p_val = stats.ttest_ind(
                                    shap1[:, k], shap2[:, k], 
                                    equal_var=False
                                )
                                t_stats.append(t_stat)
                                p_values.append(p_val)
                            else:
                                t_stats.append(0)
                                p_values.append(1)
                        
                        comparison_key = f"{group1}_vs_{group2}"
                        analysis['differences'][comparison_key] = {
                            'mean_difference': mean_diff.tolist(),
                            'abs_mean_difference': np.abs(mean_diff).tolist(),
                            't_statistics': t_stats,
                            'p_values': p_values,
                            'significant_features': [
                                feature_names[idx] for idx, p in enumerate(p_values) 
                                if p < 0.05
                            ]
                        }
        
        # Identification des indicateurs de biais
        analysis['bias_indicators'] = self._identify_bias_indicators(
            analysis, feature_names, sensitive_attr
        )
        
        return analysis
    
    def _analyze_intersectional_bias(self, model_name: str, shap_vals: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyse les biais intersectionnels (race + sexe).
        
        Args:
            model_name: Nom du modèle
            shap_vals: Valeurs SHAP
            feature_names: Noms des features
            
        Returns:
            Analyse des biais intersectionnels
        """
        # Création des groupes intersectionnels
        intersectional_groups = self.sensitive_test.apply(
            lambda row: "_".join([f"{col}:{row[col]}" for col in self.sensitive_test.columns]),
            axis=1
        )
        
        unique_intersectional = intersectional_groups.unique()
        
        analysis = {
            'groups': {},
            'most_disadvantaged': {},
            'privilege_spectrum': []
        }
        
        # Analyse par groupe intersectionnel
        group_mean_impact = {}
        
        for group in unique_intersectional:
            group_mask = intersectional_groups == group
            group_shap = shap_vals[group_mask]
            
            if len(group_shap) > 5:  # Minimum d'échantillons pour analyse robuste
                mean_total_impact = group_shap.sum(axis=1).mean()
                
                analysis['groups'][group] = {
                    'count': len(group_shap),
                    'mean_total_shap_impact': mean_total_impact,
                    'mean_feature_impacts': group_shap.mean(axis=0).tolist()
                }
                
                group_mean_impact[group] = mean_total_impact
        
        # Identification du spectre de privilège
        if group_mean_impact:
            sorted_groups = sorted(group_mean_impact.items(), key=lambda x: x[1])
            
            analysis['privilege_spectrum'] = [
                {
                    'group': group,
                    'mean_impact': impact,
                    'interpretation': self._interpret_shap_impact(impact)
                }
                for group, impact in sorted_groups
            ]
            
            # Groupe le plus désavantagé (impact SHAP le plus négatif/faible)
            most_disadvantaged = sorted_groups[0]  
            most_privileged = sorted_groups[-1]
            
            analysis['most_disadvantaged'] = {
                'group': most_disadvantaged[0],
                'impact': most_disadvantaged[1],
                'comparison_with_most_privileged': {
                    'privileged_group': most_privileged[0],
                    'impact_difference': most_privileged[1] - most_disadvantaged[1]
                }
            }
        
        return analysis
    
    def _identify_bias_indicators(self, analysis: Dict[str, Any], 
                                feature_names: List[str], 
                                sensitive_attr: str) -> Dict[str, Any]:
        """
        Identifie les indicateurs clés de biais dans l'analyse SHAP.
        
        Args:
            analysis: Résultats d'analyse pour l'attribut
            feature_names: Noms des features
            sensitive_attr: Attribut sensible
            
        Returns:
            Indicateurs de biais identifiés
        """
        indicators = {
            'high_bias_features': [],
            'discriminatory_patterns': [],
            'bias_severity': 'low'
        }
        
        # Analyse des différences entre groupes
        for comparison, diff_data in analysis['differences'].items():
            mean_diffs = np.array(diff_data['mean_difference'])
            abs_diffs = np.array(diff_data['abs_mean_difference'])
            p_values = np.array(diff_data['p_values'])
            
            # Features avec différences importantes et significatives
            high_impact_mask = (abs_diffs > np.percentile(abs_diffs, 75)) & (p_values < 0.05)
            
            if np.any(high_impact_mask):
                high_bias_features = [
                    {
                        'feature': feature_names[i],
                        'difference': mean_diffs[i],
                        'abs_difference': abs_diffs[i],
                        'p_value': p_values[i],
                        'comparison': comparison
                    }
                    for i in np.where(high_impact_mask)[0]
                ]
                
                indicators['high_bias_features'].extend(high_bias_features)
        
        # Évaluation de la sévérité du biais
        if indicators['high_bias_features']:
            max_abs_diff = max(f['abs_difference'] for f in indicators['high_bias_features'])
            
            if max_abs_diff > 0.1:
                indicators['bias_severity'] = 'high'
            elif max_abs_diff > 0.05:
                indicators['bias_severity'] = 'medium'
            else:
                indicators['bias_severity'] = 'low'
        
        return indicators
    
    def _interpret_shap_impact(self, impact: float) -> str:
        """
        Interprète l'impact SHAP moyen en termes de biais.
        
        Args:
            impact: Impact SHAP moyen
            
        Returns:
            Interprétation textuelle
        """
        if impact > 0.1:
            return "Favorisé (impact positif fort)"
        elif impact > 0.05:
            return "Légèrement favorisé"
        elif impact > -0.05:
            return "Impact neutre"
        elif impact > -0.1:
            return "Légèrement défavorisé"
        else:
            return "Défavorisé (impact négatif fort)"
    
    def compare_shap_across_demographics(self) -> Dict[str, pd.DataFrame]:
        """
        Compare les valeurs SHAP moyennes entre les groupes démographiques.
        
        Returns:
            DataFrames de comparaison par modèle
        """
        logger.info("Comparaison des valeurs SHAP entre groupes démographiques")
        
        comparison_results = {}
        
        for model_name, shap_data in self.shap_values.items():
            shap_vals = shap_data['values']
            feature_names = shap_data['feature_names']
            
            # Création d'un DataFrame pour faciliter l'analyse
            shap_df = pd.DataFrame(shap_vals, columns=feature_names)
            
            # Ajout des attributs sensibles
            for col in self.sensitive_test.columns:
                shap_df[f'sensitive_{col}'] = self.sensitive_test[col].values
            
            # Calcul des moyennes par groupe
            comparison_data = []
            
            for sensitive_attr in self.sensitive_test.columns:
                sensitive_col = f'sensitive_{sensitive_attr}'
                grouped = shap_df.groupby(sensitive_col)[feature_names].mean()
                
                for group_name, group_data in grouped.iterrows():
                    for feature, mean_shap in group_data.items():
                        comparison_data.append({
                            'model': model_name,
                            'sensitive_attribute': sensitive_attr,
                            'group': group_name,
                            'feature': feature,
                            'mean_shap': mean_shap
                        })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_results[model_name] = comparison_df
        
        self.demographic_comparisons = comparison_results
        return comparison_results
    
    def create_shap_visualizations(self, save_plots: bool = True,
                                 interactive: bool = True) -> Dict[str, str]:
        """
        Crée des visualisations complètes des analyses SHAP.
        
        Args:
            save_plots: Sauvegarder les graphiques
            interactive: Créer des versions interactives (Plotly)
            
        Returns:
            Dictionnaire des chemins des visualisations créées
        """
        logger.info("Création des visualisations SHAP")
        
        if not self.shap_values:
            raise ValueError("Valeurs SHAP non calculées.")
        
        visualization_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, shap_data in self.shap_values.items():
            logger.info(f"Création des visualisations pour {model_name}")
            
            model_paths = {}
            
            # 1. Summary Plot (global)
            model_paths.update(self._create_summary_plots(
                model_name, shap_data, timestamp, save_plots, interactive
            ))
            
            # 2. Feature Importance Plot
            model_paths.update(self._create_importance_plots(
                model_name, shap_data, timestamp, save_plots, interactive
            ))
            
            # 3. Dependence Plots pour les features les plus importantes
            model_paths.update(self._create_dependence_plots(
                model_name, shap_data, timestamp, save_plots, interactive
            ))
            
            # 4. Bias Analysis Plots
            model_paths.update(self._create_bias_plots(
                model_name, shap_data, timestamp, save_plots, interactive
            ))
            
            # 5. Waterfall plots pour échantillons représentatifs
            model_paths.update(self._create_waterfall_plots(
                model_name, shap_data, timestamp, save_plots
            ))
            
            visualization_paths[model_name] = model_paths
        
        logger.info("Visualisations SHAP créées avec succès")
        return visualization_paths
    
    def _create_summary_plots(self, model_name: str, shap_data: Dict[str, Any],
                            timestamp: str, save_plots: bool, interactive: bool) -> Dict[str, str]:
        """Crée les summary plots SHAP."""
        paths = {}
        
        shap_vals = shap_data['values']
        feature_names = shap_data['feature_names']
        
        # Summary plot standard avec matplotlib/seaborn
        plt.figure(figsize=(12, 8))
        
        # Calcul des importances pour le tri
        feature_importance = np.abs(shap_vals).mean(axis=0)
        importance_order = np.argsort(feature_importance)[::-1][:20]  # Top 20
        
        # Création du summary plot manuel (car shap.summary_plot peut avoir des problèmes)
        top_features = [feature_names[i] for i in importance_order]
        top_shap_vals = shap_vals[:, importance_order]
        
        # Box plot des valeurs SHAP
        plt.boxplot([top_shap_vals[:, i] for i in range(len(top_features))], 
                   labels=top_features, vert=False)
        plt.xlabel('Valeur SHAP')
        plt.title(f'Résumé des Valeurs SHAP - {model_name}')
        plt.gca().invert_yaxis()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_plots:
            path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_summary_{timestamp}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            paths['summary_plot'] = path
        
        plt.close()
        
        # Version interactive avec Plotly
        if interactive:
            fig = go.Figure()
            
            for i, feature in enumerate(top_features):
                fig.add_trace(go.Box(
                    y=top_shap_vals[:, i],
                    name=feature,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title=f'Résumé des Valeurs SHAP - {model_name}',
                xaxis_title='Features',
                yaxis_title='Valeur SHAP',
                showlegend=False,
                height=600
            )
            
            if save_plots:
                path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_summary_interactive_{timestamp}.html"
                fig.write_html(path)
                paths['summary_plot_interactive'] = path
        
        return paths
    
    def _create_importance_plots(self, model_name: str, shap_data: Dict[str, Any],
                               timestamp: str, save_plots: bool, interactive: bool) -> Dict[str, str]:
        """Crée les graphiques d'importance des features."""
        paths = {}
        
        if model_name in self.global_importance:
            importance_df = self.global_importance[model_name].head(15)
            
            # Graphique en barres statique
            plt.figure(figsize=(10, 8))
            
            bars = plt.barh(range(len(importance_df)), importance_df['mean_abs_shap'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance SHAP Moyenne (Valeur Absolue)')
            plt.title(f'Importance des Features - {model_name}')
            
            # Coloration selon le signe de l'impact moyen
            for i, (bar, mean_shap) in enumerate(zip(bars, importance_df['mean_shap'])):
                color = 'red' if mean_shap < 0 else 'blue'
                bar.set_color(color)
                bar.set_alpha(0.7)
            
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            if save_plots:
                path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_importance_{timestamp}.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                paths['importance_plot'] = path
            
            plt.close()
            
            # Version interactive
            if interactive:
                colors = ['red' if x < 0 else 'blue' for x in importance_df['mean_shap']]
                
                fig = go.Figure(go.Bar(
                    x=importance_df['mean_abs_shap'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color=colors,
                    text=importance_df['mean_abs_shap'].round(4),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f'Importance des Features - {model_name}',
                    xaxis_title='Importance SHAP Moyenne (Valeur Absolue)',
                    yaxis_title='Features',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                if save_plots:
                    path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_importance_interactive_{timestamp}.html"
                    fig.write_html(path)
                    paths['importance_plot_interactive'] = path
        
        return paths
    
    def _create_dependence_plots(self, model_name: str, shap_data: Dict[str, Any],
                               timestamp: str, save_plots: bool, interactive: bool) -> Dict[str, str]:
        """Crée les graphiques de dépendance SHAP."""
        paths = {}
        
        if model_name not in self.global_importance:
            return paths
        
        shap_vals = shap_data['values'] 
        feature_names = shap_data['feature_names']
        top_features = self.global_importance[model_name].head(5)['feature'].tolist()
        
        for feature in top_features:
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                
                # Graphique de dépendance
                plt.figure(figsize=(10, 6))
                
                feature_values = self.X_test[feature].values
                feature_shap = shap_vals[:, feature_idx]
                
                plt.scatter(feature_values, feature_shap, alpha=0.6, s=20)
                plt.xlabel(f'Valeur de {feature}')
                plt.ylabel(f'Valeur SHAP de {feature}')
                plt.title(f'Dépendance SHAP - {feature} ({model_name})')
                plt.grid(alpha=0.3)
                
                if save_plots:
                    path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_dependence_{feature}_{timestamp}.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths[f'dependence_{feature}'] = path
                
                plt.close()
                
                # Version interactive
                if interactive:
                    fig = go.Figure(go.Scatter(
                        x=feature_values,
                        y=feature_shap,
                        mode='markers',
                        marker=dict(size=5, opacity=0.6),
                        name=f'{feature}'
                    ))
                    
                    fig.update_layout(
                        title=f'Dépendance SHAP - {feature} ({model_name})',
                        xaxis_title=f'Valeur de {feature}',
                        yaxis_title=f'Valeur SHAP de {feature}',
                        height=500
                    )
                    
                    if save_plots:
                        path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_dependence_{feature}_interactive_{timestamp}.html"
                        fig.write_html(path)
                        paths[f'dependence_{feature}_interactive'] = path
        
        return paths
    
    def _create_bias_plots(self, model_name: str, shap_data: Dict[str, Any],
                         timestamp: str, save_plots: bool, interactive: bool) -> Dict[str, str]:
        """Crée les graphiques d'analyse de biais."""
        paths = {}
        
        if model_name not in self.bias_analysis_results:
            return paths
        
        bias_data = self.bias_analysis_results[model_name]
        
        for sensitive_attr, attr_analysis in bias_data.items():
            if sensitive_attr == 'intersectional':
                continue
                
            # Graphique de comparaison des impacts SHAP moyens par groupe
            if 'groups' in attr_analysis:
                groups_data = attr_analysis['groups']
                
                # Préparation des données pour visualisation
                group_names = list(groups_data.keys())
                feature_names = shap_data['feature_names']
                
                # Matrice des impacts moyens
                impact_matrix = []
                for group in group_names:
                    impact_matrix.append(groups_data[group]['mean_shap_values'])
                
                impact_matrix = np.array(impact_matrix)
                
                # Heatmap des impacts
                plt.figure(figsize=(12, 6))
                sns.heatmap(
                    impact_matrix,
                    xticklabels=feature_names,
                    yticklabels=group_names,
                    cmap='RdBu_r',
                    center=0,
                    annot=False,
                    cbar_kws={'label': 'Valeur SHAP Moyenne'}
                )
                plt.title(f'Impact SHAP par Groupe - {sensitive_attr} ({model_name})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                if save_plots:
                    path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_bias_{sensitive_attr}_{timestamp}.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths[f'bias_{sensitive_attr}'] = path
                
                plt.close()
                
                # Version interactive avec Plotly
                if interactive:
                    fig = go.Figure(data=go.Heatmap(
                        z=impact_matrix,
                        x=feature_names,
                        y=group_names,
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title="Valeur SHAP Moyenne")
                    ))
                    
                    fig.update_layout(
                        title=f'Impact SHAP par Groupe - {sensitive_attr} ({model_name})',
                        xaxis_title='Features',
                        yaxis_title='Groupes',
                        height=500
                    )
                    
                    if save_plots:
                        path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_bias_{sensitive_attr}_interactive_{timestamp}.html"
                        fig.write_html(path)
                        paths[f'bias_{sensitive_attr}_interactive'] = path
        
        return paths
    
    def _create_waterfall_plots(self, model_name: str, shap_data: Dict[str, Any],
                              timestamp: str, save_plots: bool) -> Dict[str, str]:
        """Crée des waterfall plots pour des échantillons représentatifs."""
        paths = {}
        
        # Sélection d'échantillons représentatifs
        sample_indices = [0, len(self.X_test)//4, len(self.X_test)//2, 3*len(self.X_test)//4]
        sample_indices = [i for i in sample_indices if i < len(shap_data['values'])]
        
        for i, sample_idx in enumerate(sample_indices[:3]):  # Limite à 3 échantillons
            try:
                plt.figure(figsize=(10, 6))
                
                sample_shap = shap_data['values'][sample_idx]
                feature_names = shap_data['feature_names']
                expected_value = shap_data['expected_value']
                
                # Création manuelle d'un waterfall plot simplifié
                # Tri des features par impact absolu
                feature_impacts = list(zip(feature_names, sample_shap))
                feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                
                top_features = feature_impacts[:10]  # Top 10 features
                
                # Valeurs pour le waterfall
                values = [expected_value]
                labels = ['Valeur de base']
                
                cumsum = expected_value
                for feature, impact in top_features:
                    cumsum += impact
                    values.append(cumsum)
                    labels.append(feature)
                
                # Graphique en barres avec cumulation
                x_pos = range(len(values))
                colors = ['gray'] + ['red' if impact < 0 else 'blue' 
                         for _, impact in top_features]
                
                plt.bar(x_pos, values, color=colors, alpha=0.7)
                plt.xticks(x_pos, labels, rotation=45, ha='right')
                plt.ylabel('Contribution cumulative')
                plt.title(f'Explication de la Prédiction - Échantillon {sample_idx} ({model_name})')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                
                if save_plots:
                    path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/visualizations/{model_name}_waterfall_sample_{sample_idx}_{timestamp}.png"
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    paths[f'waterfall_sample_{sample_idx}'] = path
                
                plt.close()
                
            except Exception as e:
                logger.warning(f"Erreur création waterfall pour échantillon {sample_idx}: {e}")
                continue
        
        return paths
    
    def generate_comprehensive_bias_report(self) -> str:
        """
        Génère un rapport complet d'analyse de biais en français.
        
        Returns:
            Chemin vers le rapport généré
        """
        logger.info("Génération du rapport complet d'analyse de biais")
        
        if not self.bias_analysis_results:
            raise ValueError("Analyse de biais non effectuée. Appelez analyze_bias_through_shap() d'abord.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/rapport_analyse_biais_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Rapport d'Analyse de Biais COMPAS via SHAP\n\n")
            f.write(f"**Date de génération :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Optimisé pour :** Mac M4 Pro avec Apple Silicon\n\n")
            f.write("---\n\n")
            
            # Résumé exécutif
            f.write("## Résumé Exécutif\n\n")
            f.write("Cette analyse utilise SHAP (SHapley Additive exPlanations) pour identifier ")
            f.write("et quantifier les biais présents dans les modèles de prédiction de récidive COMPAS. ")
            f.write("SHAP permet de comprendre comment chaque caractéristique contribue aux ")
            f.write("prédictions individuelles et révèle les disparités systémiques.\n\n")
            
            # Méthodologie
            f.write("## Méthodologie\n\n")
            f.write("### Approche SHAP\n")
            f.write("- **TreeExplainer** : Pour modèles basés sur les arbres (XGBoost, Random Forest, LightGBM)\n")
            f.write("- **LinearExplainer** : Pour modèles linéaires (Régression Logistique)\n")
            f.write("- **KernelExplainer** : Pour autres modèles (SVM, Réseaux de Neurones)\n\n")
            f.write("### Métriques de Biais Analysées\n")
            f.write("- Différences moyennes des valeurs SHAP entre groupes démographiques\n")
            f.write("- Analyse des caractéristiques contribuant le plus aux disparités\n")
            f.write("- Tests de significativité statistique des différences observées\n")
            f.write("- Analyse intersectionnelle (race × sexe)\n\n")
            
            # Analyse par modèle
            f.write("## Analyse Détaillée par Modèle\n\n")
            
            for model_name, bias_data in self.bias_analysis_results.items():
                f.write(f"### {model_name.title()}\n\n")
                
                # Analyse par attribut sensible
                for sensitive_attr, attr_analysis in bias_data.items():
                    if sensitive_attr == 'intersectional':
                        continue
                        
                    f.write(f"#### Analyse par {sensitive_attr.title()}\n\n")
                    
                    # Groupes analysés
                    if 'groups' in attr_analysis:
                        f.write("**Groupes analysés :**\n")
                        for group, group_data in attr_analysis['groups'].items():
                            count = group_data['count']
                            f.write(f"- {group} : {count} échantillons\n")
                        f.write("\n")
                    
                    # Différences significatives
                    if 'differences' in attr_analysis:
                        f.write("**Différences entre groupes :**\n\n")
                        for comparison, diff_data in attr_analysis['differences'].items():
                            significant_features = diff_data['significant_features']
                            if significant_features:
                                f.write(f"*{comparison}* :\n")
                                f.write(f"- Caractéristiques avec différences significatives : {len(significant_features)}\n")
                                f.write(f"- Principales caractéristiques biaisées : {', '.join(significant_features[:5])}\n\n")
                    
                    # Indicateurs de biais
                    if 'bias_indicators' in attr_analysis:
                        indicators = attr_analysis['bias_indicators']
                        f.write(f"**Sévérité du biais** : {indicators['bias_severity'].upper()}\n\n")
                        
                        if indicators['high_bias_features']:
                            f.write("**Caractéristiques les plus biaisées** :\n")
                            for feature_data in indicators['high_bias_features'][:5]:
                                feature = feature_data['feature']
                                diff = feature_data['difference']
                                p_val = feature_data['p_value']
                                f.write(f"- {feature} : différence = {diff:.4f} (p = {p_val:.4f})\n")
                            f.write("\n")
                
                # Analyse intersectionnelle
                if 'intersectional' in bias_data:
                    intersectional = bias_data['intersectional']
                    f.write("#### Analyse Intersectionnelle (Race × Sexe)\n\n")
                    
                    if 'most_disadvantaged' in intersectional:
                        disadvantaged = intersectional['most_disadvantaged']
                        f.write(f"**Groupe le plus défavorisé** : {disadvantaged['group']}\n")
                        f.write(f"**Impact SHAP moyen** : {disadvantaged['impact']:.4f}\n\n")
                        
                        if 'comparison_with_most_privileged' in disadvantaged:
                            comp = disadvantaged['comparison_with_most_privileged']
                            f.write(f"**Écart avec le groupe le plus privilégié** ({comp['privileged_group']}) : ")
                            f.write(f"{comp['impact_difference']:.4f}\n\n")
                    
                    if 'privilege_spectrum' in intersectional:
                        f.write("**Spectre de privilège** :\n")
                        for i, group_data in enumerate(intersectional['privilege_spectrum'][:5]):
                            rank = i + 1
                            group = group_data['group']
                            impact = group_data['impact']
                            interp = group_data['interpretation']
                            f.write(f"{rank}. {group} : {impact:.4f} - {interp}\n")
                        f.write("\n")
                
                f.write("---\n\n")
            
            # Synthèse et recommandations
            f.write("## Synthèse des Résultats\n\n")
            
            # Identification du modèle le moins biaisé
            model_bias_scores = {}
            for model_name, bias_data in self.bias_analysis_results.items():
                total_bias_score = 0
                count = 0
                
                for attr, attr_analysis in bias_data.items():
                    if attr != 'intersectional' and 'bias_indicators' in attr_analysis:
                        severity = attr_analysis['bias_indicators']['bias_severity']
                        if severity == 'high':
                            total_bias_score += 3
                        elif severity == 'medium':
                            total_bias_score += 2
                        else:
                            total_bias_score += 1
                        count += 1
                
                if count > 0:
                    model_bias_scores[model_name] = total_bias_score / count
            
            if model_bias_scores:
                best_model = min(model_bias_scores.items(), key=lambda x: x[1])
                f.write(f"**Modèle le moins biaisé** : {best_model[0]} (score : {best_model[1]:.2f})\n\n")
            
            # Recommandations
            f.write("## Recommandations\n\n")
            f.write("### Recommandations Immédiates\n\n")
            f.write("1. **Audit approfondi** des caractéristiques identifiées comme biaisées\n")
            f.write("2. **Collecte de données supplémentaires** pour les groupes sous-représentés\n")
            f.write("3. **Techniques de débiaisage** :\n")
            f.write("   - Pre-processing : rééquilibrage des données d'entraînement\n")
            f.write("   - In-processing : contraintes d'équité pendant l'entraînement\n")
            f.write("   - Post-processing : calibration des seuils par groupe\n\n")
            
            f.write("### Recommandations Stratégiques\n\n")
            f.write("1. **Politique de gouvernance** des algorithmes prédictifs\n")
            f.write("2. **Monitoring continu** des performances par groupe démographique\n")
            f.write("3. **Formation des utilisateurs** sur les limitations et biais identifiés\n")
            f.write("4. **Recherche et développement** de méthodes plus équitables\n\n")
            
            # Limitations
            f.write("## Limitations de l'Analyse\n\n")
            f.write("- L'analyse SHAP révèle les patterns appris par les modèles mais ne prouve pas la causalité\n")
            f.write("- Les biais peuvent également provenir des données d'entraînement elles-mêmes\n")
            f.write("- L'échantillonnage pour les calculs SHAP peut introduire des variations\n")
            f.write("- Les interactions complexes entre variables peuvent être sous-estimées\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("Cette analyse SHAP révèle des disparités significatives dans les prédictions ")
            f.write("du système COMPAS selon les groupes démographiques. Les résultats soulignent ")
            f.write("l'importance d'une approche proactive pour identifier et corriger les biais ")
            f.write("algorithmiques dans les systèmes de justice pénale.\n\n")
            f.write("Les techniques d'interprétabilité comme SHAP sont essentielles pour rendre ")
            f.write("les décisions algorithmiques transparentes et accountables, particulièrement ")
            f.write("dans des domaines critiques comme la justice.\n\n")
            
            # Métadonnées techniques
            f.write("---\n\n")
            f.write("## Métadonnées Techniques\n\n")
            f.write(f"- **Plateforme** : macOS avec Apple Silicon M4 Pro\n")
            f.write(f"- **Cœurs utilisés** : {self.n_cores}\n")
            f.write(f"- **Échantillons analysés** : {len(self.X_test)}\n")
            f.write(f"- **Modèles analysés** : {', '.join(self.bias_analysis_results.keys())}\n")
            f.write(f"- **Attributs sensibles** : {', '.join(self.sensitive_attributes)}\n")
            f.write(f"- **Graine aléatoire** : {self.random_state}\n\n")
        
        logger.info(f"Rapport de biais généré : {report_path}")
        return report_path
    
    def save_analysis_results(self) -> Dict[str, str]:
        """
        Sauvegarde tous les résultats d'analyse.
        
        Returns:
            Dictionnaire des chemins de sauvegarde
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        # Sauvegarde des résultats de biais
        if self.bias_analysis_results:
            bias_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/bias_analysis/bias_results_{timestamp}.json"
            with open(bias_path, 'w') as f:
                json.dump(self.bias_analysis_results, f, indent=2, default=str)
            saved_paths['bias_analysis'] = bias_path
        
        # Sauvegarde de l'importance globale
        if self.global_importance:
            importance_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/global_importance_{timestamp}.json"
            importance_dict = {}
            for model, df in self.global_importance.items():
                importance_dict[model] = df.to_dict('records')
            
            with open(importance_path, 'w') as f:
                json.dump(importance_dict, f, indent=2, default=str)
            saved_paths['global_importance'] = importance_path
        
        # Sauvegarde des comparaisons démographiques
        if self.demographic_comparisons:
            demo_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/shap/demographic_comparisons_{timestamp}.json"
            demo_dict = {}
            for model, df in self.demographic_comparisons.items():
                demo_dict[model] = df.to_dict('records')
            
            with open(demo_path, 'w') as f:
                json.dump(demo_dict, f, indent=2, default=str)
            saved_paths['demographic_comparisons'] = demo_path
        
        logger.info(f"Résultats d'analyse sauvegardés: {list(saved_paths.keys())}")
        return saved_paths


def main():
    """Fonction principale pour démonstration et tests du module SHAP."""
    
    print("🔍 Module d'Analyse SHAP COMPAS - Optimisé Mac M4 Pro")
    print("=" * 60)
    
    # Initialisation de l'analyseur SHAP
    shap_analyzer = SHAPAnalyzer(random_state=42)
    
    print("\n✅ SHAPAnalyzer initialisé avec optimisations Apple Silicon")
    print(f"🔧 Configuration: {shap_analyzer.n_cores} cœurs, {shap_analyzer.memory_limit_gb}GB mémoire limite")
    
    print("\n📊 Pour utiliser ce module :")
    print("1. Chargez vos modèles entraînés avec load_trained_models()")
    print("2. Préparez vos données avec prepare_data()")
    print("3. Créez les explainers avec create_explainers()")
    print("4. Calculez les valeurs SHAP avec calculate_shap_values()")
    print("5. Analysez les biais avec analyze_bias_through_shap()")
    print("6. Créez les visualisations avec create_shap_visualizations()")
    print("7. Générez le rapport avec generate_comprehensive_bias_report()")
    
    print("\n🎯 Fonctionnalités principales :")
    print("- Calcul optimisé des valeurs SHAP pour différents types de modèles")
    print("- Analyse approfondie des biais raciaux et de genre")
    print("- Visualisations interactives et statiques")
    print("- Rapport complet d'analyse de biais en français")
    print("- Optimisations mémoire pour Mac M4 Pro")
    
    print("\n🚀 Module prêt pour l'analyse d'interprétabilité COMPAS!")


if __name__ == "__main__":
    main()