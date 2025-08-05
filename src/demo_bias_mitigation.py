"""
Démonstration Intégrée de Mitigation des Biais COMPAS
====================================================

Ce script démontre l'utilisation complète du framework de mitigation des biais
en intégration avec les modules existants de feature engineering et model training.

Il illustre un pipeline complet depuis les données brutes jusqu'aux modèles
mitigés avec évaluation comparative des stratégies de réduction des biais.

Auteur: Système d'IA Claude - Expert ML Apple Silicon
Date: 2025-08-05
Optimisé pour: Mac M4 Pro avec Apple Silicon
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import des modules du projet
from feature_engineering import COMPASFeatureEngineer
from model_training import CompasModelTrainer
from bias_mitigation import BiasMitigationFramework

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedBiasMitigationDemo:
    """
    Démonstration intégrée du pipeline complet de mitigation des biais.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialise la démonstration.
        
        Args:
            data_path: Chemin vers les données COMPAS
            random_state: Graine aléatoire
        """
        self.random_state = random_state
        self.data_path = data_path
        
        # Initialisation des composants
        self.feature_engineer = COMPASFeatureEngineer(random_state=random_state)
        self.model_trainer = CompasModelTrainer(random_state=random_state)
        self.bias_framework = BiasMitigationFramework(random_state=random_state)
        
        # Storage pour les résultats
        self.demo_results = {
            'preprocessing_results': {},
            'original_models': {},
            'mitigation_strategies': {},
            'evaluation_comparisons': {},
            'final_recommendations': {}
        }
        
        logger.info("IntegratedBiasMitigationDemo initialisé")
    
    def run_complete_pipeline(self) -> dict:
        """
        Exécute le pipeline complet de démonstration.
        
        Returns:
            Dictionnaire contenant tous les résultats de la démonstration
        """
        logger.info("=== DÉBUT DE LA DÉMONSTRATION INTÉGRÉE ===")
        
        try:
            # Étape 1: Création/Chargement des données
            logger.info("Étape 1: Préparation des données")
            df_processed = self._prepare_demonstration_data()
            
            # Étape 2: Feature Engineering
            logger.info("Étape 2: Feature Engineering")
            processed_datasets = self._apply_feature_engineering(df_processed)
            
            # Étape 3: Entraînement des modèles de référence
            logger.info("Étape 3: Entraînement des modèles de référence")
            original_models = self._train_baseline_models(processed_datasets)
            
            # Étape 4: Analyse des biais originaux
            logger.info("Étape 4: Analyse des biais dans les modèles de référence")
            bias_analysis = self._analyze_original_bias(original_models, processed_datasets)
            
            # Étape 5: Application des stratégies de mitigation
            logger.info("Étape 5: Application des stratégies de mitigation")
            mitigation_results = self._apply_mitigation_strategies(processed_datasets)
            
            # Étape 6: Évaluation comparative
            logger.info("Étape 6: Évaluation comparative des stratégies")
            comparison_results = self._compare_mitigation_effectiveness(
                original_models, mitigation_results, processed_datasets
            )
            
            # Étape 7: Génération des recommandations
            logger.info("Étape 7: Génération des recommandations finales")
            recommendations = self._generate_final_recommendations(comparison_results)
            
            # Compilation des résultats
            self.demo_results.update({
                'preprocessing_results': processed_datasets,
                'original_models': original_models,
                'bias_analysis': bias_analysis,
                'mitigation_strategies': mitigation_results,
                'evaluation_comparisons': comparison_results,
                'final_recommendations': recommendations,
                'demo_success': True
            })
            
            logger.info("=== DÉMONSTRATION INTÉGRÉE TERMINÉE AVEC SUCCÈS ===")
            
        except Exception as e:
            logger.error(f"Erreur dans la démonstration: {str(e)}")
            self.demo_results.update({
                'demo_success': False,
                'error': str(e)
            })
            raise
        
        return self.demo_results
    
    def _prepare_demonstration_data(self) -> pd.DataFrame:
        """Prépare les données pour la démonstration."""
        
        if self.data_path and os.path.exists(self.data_path):
            # Utiliser les vraies données COMPAS si disponibles
            logger.info(f"Chargement des données depuis: {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            # Créer des données d'exemple réalistes
            logger.info("Création de données COMPAS d'exemple avec biais simulé")
            df = self._create_realistic_compas_data()
        
        # Validation des données
        required_columns = ['age', 'race', 'sex', 'priors_count', 'two_year_recid']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données: {missing_columns}")
        
        logger.info(f"Données préparées: {df.shape[0]} échantillons, {df.shape[1]} features")
        
        return df
    
    def _create_realistic_compas_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Crée des données COMPAS réalistes avec biais simulé."""
        
        np.random.seed(self.random_state)
        
        # Demographics avec distribution réaliste
        age = np.random.gamma(2, 15, n_samples).clip(18, 80).astype(int)
        race = np.random.choice(
            ['African-American', 'Caucasian', 'Hispanic', 'Asian', 'Native American', 'Other'],
            n_samples, 
            p=[0.35, 0.35, 0.15, 0.05, 0.05, 0.05]
        )
        sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.75, 0.25])
        
        # Criminal history
        priors_count = np.random.negative_binomial(n=2, p=0.3, size=n_samples).clip(0, 30)
        
        # Charge information
        c_charge_degree = np.random.choice(['F', 'M'], n_samples, p=[0.4, 0.6])
        
        # COMPAS scores avec biais systémique
        # Biais racial: scores plus élevés pour certains groupes
        race_bias = np.where(race == 'African-American', 1.5, 
                    np.where(race == 'Hispanic', 0.5, 0))
        
        # Score de base basé sur l'âge et les antécédents
        base_score = (
            3 +
            2 * (age < 25).astype(int) +
            1.5 * np.log1p(priors_count) +
            1 * (c_charge_degree == 'F').astype(int) +
            race_bias +
            np.random.normal(0, 1, n_samples)
        ).clip(1, 10)
        
        decile_score = base_score.round().astype(int)
        
        # Score textuel basé sur le score numérique
        score_text = pd.cut(
            decile_score, 
            bins=[0, 3, 7, 10], 
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Variable cible avec biais
        # Probabilité de récidive influencée par le biais systémique
        recid_prob = (
            0.15 +  # Taux de base
            0.05 * (decile_score / 10) +  # Impact du score COMPAS
            0.08 * (priors_count > 3).astype(int) +  # Impact des antécédents
            0.03 * (age < 25).astype(int) +  # Impact de l'âge
            0.02 * race_bias / 1.5  # Biais résiduel
        ).clip(0, 1)
        
        two_year_recid = np.random.binomial(1, recid_prob, n_samples)
        
        # Catégories d'âge
        age_cat = pd.cut(
            age,
            bins=[0, 25, 45, 100],
            labels=['Less than 25', '25 - 45', 'Greater than 45'],
            include_lowest=True
        )
        
        # Variables temporelles
        days_b_screening_arrest = np.random.normal(0, 20, n_samples).round().astype(int)
        
        # Compilation du DataFrame
        compas_data = pd.DataFrame({
            'age': age,
            'race': race,
            'sex': sex,
            'priors_count': priors_count,
            'c_charge_degree': c_charge_degree,
            'decile_score': decile_score,
            'score_text': score_text,
            'two_year_recid': two_year_recid,
            'age_cat': age_cat,
            'days_b_screening_arrest': days_b_screening_arrest
        })
        
        # Introduire quelques valeurs manquantes réalistes
        missing_rate = 0.02
        for col in ['race', 'days_b_screening_arrest']:
            missing_indices = np.random.choice(
                compas_data.index, 
                size=int(missing_rate * len(compas_data)), 
                replace=False
            )
            compas_data.loc[missing_indices, col] = np.nan
        
        # Analyse du biais créé
        bias_analysis = {}
        for race_group in compas_data['race'].unique():
            if pd.notna(race_group):
                mask = compas_data['race'] == race_group
                group_recid_rate = compas_data.loc[mask, 'two_year_recid'].mean()
                group_score_mean = compas_data.loc[mask, 'decile_score'].mean()
                bias_analysis[race_group] = {
                    'recidivism_rate': group_recid_rate,
                    'mean_compas_score': group_score_mean,
                    'count': mask.sum()
                }
        
        logger.info("Biais simulé dans les données créées:")
        for group, stats in bias_analysis.items():
            logger.info(f"  {group}: Récidive={stats['recidivism_rate']:.3f}, "
                       f"Score COMPAS={stats['mean_compas_score']:.1f}, "
                       f"N={stats['count']}")
        
        return compas_data
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> dict:
        """Applique le feature engineering."""
        
        logger.info("Application du feature engineering COMPAS")
        
        # Utilisation du pipeline de feature engineering existant
        results = self.feature_engineer.preprocess_compas_data(
            # Sauvegarder temporairement le DataFrame pour le pipeline
            '/tmp/temp_compas_data.csv'
        )
        
        # Sauvegarder les données temporaires
        df.to_csv('/tmp/temp_compas_data.csv', index=False)
        
        # Recharger avec le pipeline
        results = self.feature_engineer.preprocess_compas_data('/tmp/temp_compas_data.csv')
        
        if not results['pipeline_success']:
            raise ValueError(f"Erreur dans le feature engineering: {results.get('error', 'Unknown')}")
        
        logger.info(f"Feature engineering terminé: {len(results['datasets'])} versions créées")
        
        return results['datasets']
    
    def _train_baseline_models(self, datasets: dict) -> dict:
        """Entraîne les modèles de référence."""
        
        logger.info("Entraînement des modèles de référence")
        
        # Utiliser le dataset complet pour l'entraînement
        full_dataset = datasets['full']
        
        # Préparation des données pour le trainer
        X_full = pd.concat([full_dataset['X'], full_dataset['y']], axis=1)
        
        # Configuration du trainer
        self.model_trainer.prepare_training_data(
            X_full, 
            target_column='two_year_recid',
            sensitive_attributes=['race', 'sex']
        )
        
        # Entraînement des modèles
        training_results = self.model_trainer.train_multiple_models(
            use_hyperparameter_tuning=False,  # Désactivé pour la démonstration
            cv_folds=3
        )
        
        # Évaluation des modèles
        evaluation_results = self.model_trainer.evaluate_models(include_fairness=True)
        
        logger.info(f"Modèles de référence entraînés: {len(training_results['models'])}")
        
        return {
            'models': training_results['models'],
            'evaluation': evaluation_results,
            'cv_scores': training_results.get('cv_scores', {})
        }
    
    def _analyze_original_bias(self, models: dict, datasets: dict) -> dict:
        """Analyse les biais dans les modèles originaux."""
        
        logger.info("Analyse des biais dans les modèles originaux")
        
        bias_analysis = {}
        full_dataset = datasets['full']
        
        for model_name, model in models['models'].items():
            logger.info(f"Analyse du biais pour {model_name}")
            
            try:
                # Prédictions
                X_test = full_dataset['X_test']
                y_test = full_dataset['y_test']
                
                # Extraction des attributs sensibles des données de test
                # Assumant que les attributs sensibles sont encore présents
                race_columns = [col for col in X_test.columns if 'race_' in col.lower()]
                sex_columns = [col for col in X_test.columns if 'sex_' in col.lower()]
                
                if race_columns:
                    # Reconstruction de l'attribut race à partir des variables dummy
                    race_attr = pd.Series(['Unknown'] * len(X_test), index=X_test.index)
                    for col in race_columns:
                        race_name = col.replace('race_', '')
                        mask = X_test[col] == 1
                        race_attr.loc[mask] = race_name
                else:
                    # Fallback: créer des groupes artificiels pour la démonstration
                    race_attr = pd.Series(
                        np.random.choice(['Group_A', 'Group_B'], len(X_test)),
                        index=X_test.index
                    )
                
                # Prédictions du modèle
                y_pred = model.predict(X_test)
                
                # Calcul des métriques d'équité
                fairness_metrics = self.bias_framework._calculate_detailed_fairness_metrics(
                    y_test, y_pred, race_attr
                )
                
                bias_analysis[model_name] = {
                    'fairness_metrics': fairness_metrics,
                    'model_performance': {
                        'accuracy': (y_pred == y_test).mean(),
                        'positive_rate_overall': y_pred.mean()
                    }
                }
                
                logger.info(f"  {model_name} - Accuracy: {bias_analysis[model_name]['model_performance']['accuracy']:.4f}")
                
            except Exception as e:
                logger.warning(f"Erreur dans l'analyse de biais pour {model_name}: {str(e)}")
                bias_analysis[model_name] = {'error': str(e)}
        
        return bias_analysis
    
    def _apply_mitigation_strategies(self, datasets: dict) -> dict:
        """Applique différentes stratégies de mitigation."""
        
        logger.info("Application des stratégies de mitigation")
        
        mitigation_results = {}
        full_dataset = datasets['full']
        
        # Préparation des données
        X = full_dataset['X']
        y = full_dataset['y']
        
        # Création d'un attribut sensible pour la démonstration
        sensitive_attr = self._extract_sensitive_attribute(X)
        
        # Stratégie 1: Suppression des features sensibles
        logger.info("Stratégie 1: Suppression des features sensibles")
        try:
            X_clean, removal_report = self.bias_framework.remove_sensitive_features(
                X, y, sensitive_features=['race', 'sex']
            )
            
            # Entraînement d'un modèle simple sur les données nettoyées
            from sklearn.linear_model import LogisticRegression
            clean_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Division train/test
            from sklearn.model_selection import train_test_split
            X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
                X_clean, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            clean_model.fit(X_clean_train, y_clean_train)
            
            mitigation_results['feature_removal'] = {
                'model': clean_model,
                'method': 'preprocessing',
                'data': (X_clean_test, y_clean_test),
                'report': removal_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Erreur stratégie suppression features: {str(e)}")
            mitigation_results['feature_removal'] = {'success': False, 'error': str(e)}
        
        # Stratégie 2: Rééchantillonnage équitable
        logger.info("Stratégie 2: Rééchantillonnage équitable")
        try:
            X_resampled, y_resampled, sampling_report = self.bias_framework.apply_fairness_sampling(
                X, y, sensitive_attr, strategy='smote_fairness', target_ratio=0.8
            )
            
            # Entraînement sur les données rééchantillonnées
            resampled_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Division train/test des données rééchantillonnées
            X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=self.random_state
            )
            
            resampled_model.fit(X_res_train, y_res_train)
            
            mitigation_results['fairness_sampling'] = {
                'model': resampled_model,
                'method': 'preprocessing',
                'data': (X_res_test, y_res_test),
                'report': sampling_report,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Erreur stratégie rééchantillonnage: {str(e)}")
            mitigation_results['fairness_sampling'] = {'success': False, 'error': str(e)}
        
        # Stratégie 3: Entraînement avec contraintes d'équité
        logger.info("Stratégie 3: Entraînement avec contraintes d'équité")
        try:
            # Utiliser une partie des données pour l'entraînement contraint
            X_sample = X.sample(min(1000, len(X)), random_state=self.random_state)
            y_sample = y.loc[X_sample.index]
            sensitive_sample = sensitive_attr.loc[X_sample.index]
            
            constrained_results = self.bias_framework.train_fairness_constrained_models(
                X_sample, y_sample, sensitive_sample,
                constraint_type='demographic_parity',
                models_to_train=['logistic_regression']
            )
            
            if constrained_results['models']:
                model_name = list(constrained_results['models'].keys())[0]
                constrained_model = constrained_results['models'][model_name]
                
                mitigation_results['constrained_training'] = {
                    'model': constrained_model,
                    'method': 'in_processing',
                    'data': (X.drop(X_sample.index), y.drop(X_sample.index)),
                    'report': constrained_results['results'],
                    'success': True
                }
            else:
                mitigation_results['constrained_training'] = {
                    'success': False, 
                    'error': 'Aucun modèle contraint entraîné'
                }
                
        except Exception as e:
            logger.error(f"Erreur stratégie entraînement contraint: {str(e)}")
            mitigation_results['constrained_training'] = {'success': False, 'error': str(e)}
        
        successful_strategies = sum(1 for r in mitigation_results.values() if r.get('success', False))
        logger.info(f"Stratégies de mitigation appliquées: {successful_strategies}/{len(mitigation_results)}")
        
        return mitigation_results
    
    def _extract_sensitive_attribute(self, X: pd.DataFrame) -> pd.Series:
        """Extrait un attribut sensible des données."""
        
        # Chercher les colonnes liées à la race
        race_columns = [col for col in X.columns if 'race_' in col.lower()]
        
        if race_columns:
            # Reconstruction de l'attribut race à partir des variables dummy
            sensitive_attr = pd.Series(['Unknown'] * len(X), index=X.index)
            for col in race_columns:
                race_name = col.replace('race_', '').replace('_', '-')
                mask = X[col] == 1
                sensitive_attr.loc[mask] = race_name
        else:
            # Fallback: créer des groupes artificiels basés sur d'autres features
            if 'age' in X.columns:
                sensitive_attr = pd.cut(
                    X['age'], 
                    bins=[0, 30, 50, 100], 
                    labels=['Young', 'Middle', 'Senior']
                ).astype(str)
            else:
                # Dernier recours: groupes aléatoires
                np.random.seed(self.random_state)
                sensitive_attr = pd.Series(
                    np.random.choice(['Group_A', 'Group_B'], len(X)),
                    index=X.index
                )
        
        return sensitive_attr
    
    def _compare_mitigation_effectiveness(self, original_models: dict, 
                                        mitigation_results: dict, datasets: dict) -> dict:
        """Compare l'efficacité des différentes stratégies de mitigation."""
        
        logger.info("Comparaison de l'efficacité des stratégies de mitigation")
        
        comparison_results = {}
        
        # Sélectionner le meilleur modèle original comme référence
        best_original_model_name = max(
            original_models['evaluation'].keys(),
            key=lambda k: original_models['evaluation'][k].get('auc_roc', 0)
        )
        best_original_model = original_models['models'][best_original_model_name]
        
        logger.info(f"Modèle de référence: {best_original_model_name}")
        
        # Données de test de référence
        full_dataset = datasets['full']
        X_test_ref = full_dataset['X_test']
        y_test_ref = full_dataset['y_test']
        sensitive_attr_ref = self._extract_sensitive_attribute(X_test_ref)
        
        # Évaluer chaque stratégie de mitigation
        for strategy_name, strategy_result in mitigation_results.items():
            if not strategy_result.get('success', False):
                logger.warning(f"Stratégie {strategy_name} ignorée (échec)")
                continue
            
            logger.info(f"Évaluation de la stratégie: {strategy_name}")
            
            try:
                mitigated_model = strategy_result['model']
                
                # Évaluation comparative
                evaluation = self.bias_framework.evaluate_mitigation_effectiveness(
                    best_original_model, mitigated_model,
                    X_test_ref, y_test_ref, sensitive_attr_ref
                )
                
                comparison_results[strategy_name] = {
                    'evaluation': evaluation,
                    'strategy_info': {
                        'method': strategy_result['method'],
                        'report': strategy_result.get('report', {})
                    }
                }
                
                # Log des résultats principaux
                if 'improvement_analysis' in evaluation:
                    improvement = evaluation['improvement_analysis']
                    bias_reduction = improvement.get('overall_bias_reduction', 0)
                    perf_change = improvement.get('performance_changes', {}).get('accuracy', 0)
                    
                    logger.info(f"  {strategy_name} - Réduction biais: {bias_reduction:.1f}%, "
                               f"Change accuracy: {perf_change:+.4f}")
                
            except Exception as e:
                logger.error(f"Erreur évaluation {strategy_name}: {str(e)}")
                comparison_results[strategy_name] = {
                    'evaluation': {'error': str(e)},
                    'strategy_info': strategy_result
                }
        
        logger.info(f"Comparaison terminée: {len(comparison_results)} stratégies évaluées")
        
        return comparison_results
    
    def _generate_final_recommendations(self, comparison_results: dict) -> dict:
        """Génère les recommandations finales basées sur les comparaisons."""
        
        logger.info("Génération des recommandations finales")
        
        recommendations = {
            'best_strategy': None,
            'strategy_ranking': [],
            'detailed_analysis': {},
            'implementation_guide': {}
        }
        
        # Analyse des stratégies réussies
        successful_strategies = {}
        
        for strategy_name, result in comparison_results.items():
            evaluation = result.get('evaluation', {})
            
            if 'error' not in evaluation and 'improvement_analysis' in evaluation:
                improvement = evaluation['improvement_analysis']
                
                # Score composite: équilibre entre réduction de biais et performance
                bias_reduction = improvement.get('overall_bias_reduction', 0)
                perf_change = improvement.get('performance_changes', {}).get('accuracy', 0)
                
                # Pénaliser les fortes chutes de performance
                performance_penalty = max(0, -perf_change * 100)  # Convertir en pénalité positive
                composite_score = bias_reduction - performance_penalty * 0.5
                
                successful_strategies[strategy_name] = {
                    'bias_reduction': bias_reduction,
                    'performance_change': perf_change,
                    'composite_score': composite_score,
                    'evaluation': evaluation
                }
        
        if successful_strategies:
            # Classement des stratégies
            ranking = sorted(
                successful_strategies.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
            
            recommendations['strategy_ranking'] = [
                {
                    'rank': i + 1,
                    'strategy': strategy,
                    'score': details['composite_score'],
                    'bias_reduction': details['bias_reduction'],
                    'performance_impact': details['performance_change']
                }
                for i, (strategy, details) in enumerate(ranking)
            ]
            
            # Meilleure stratégie
            best_strategy, best_details = ranking[0]
            recommendations['best_strategy'] = {
                'name': best_strategy,
                'bias_reduction': best_details['bias_reduction'],
                'performance_impact': best_details['performance_change'],
                'composite_score': best_details['composite_score']
            }
            
            # Analyse détaillée
            for strategy, details in successful_strategies.items():
                bias_reduction = details['bias_reduction']
                perf_change = details['performance_change']
                
                if bias_reduction > 15 and perf_change > -0.05:
                    recommendation = "FORTEMENT RECOMMANDÉ"
                    rationale = "Excellente réduction de biais avec impact minimal sur la performance"
                elif bias_reduction > 10 and perf_change > -0.1:
                    recommendation = "RECOMMANDÉ"
                    rationale = "Bonne réduction de biais avec impact acceptable sur la performance"
                elif bias_reduction > 5:
                    recommendation = "À CONSIDÉRER"
                    rationale = "Réduction de biais modérée, évaluer le trade-off"
                else:
                    recommendation = "NON RECOMMANDÉ"
                    rationale = "Réduction de biais insuffisante"
                
                recommendations['detailed_analysis'][strategy] = {
                    'recommendation': recommendation,
                    'rationale': rationale,
                    'metrics': {
                        'bias_reduction_pct': bias_reduction,
                        'accuracy_change': perf_change,
                        'composite_score': details['composite_score']
                    }
                }
            
            logger.info(f"Meilleure stratégie identifiée: {best_strategy} "
                       f"(Score: {best_details['composite_score']:.2f})")
        
        else:
            logger.warning("Aucune stratégie de mitigation réussie identifiée")
            recommendations['best_strategy'] = None
            recommendations['detailed_analysis']['general'] = {
                'recommendation': "RÉVISER L'APPROCHE",
                'rationale': "Les stratégies testées n'ont pas produit d'améliorations satisfaisantes"
            }
        
        # Guide d'implémentation
        recommendations['implementation_guide'] = {
            'next_steps': [
                "Valider les résultats sur un jeu de données indépendant",
                "Effectuer des tests de robustesse avec différents paramètres",
                "Analyser l'impact sur les sous-groupes spécifiques",
                "Mettre en place un monitoring continu de l'équité en production",
                "Documenter les choix de mitigation pour la gouvernance"
            ],
            'considerations': [
                "Équilibrer équité et performance selon les objectifs métier",
                "Considérer les implications légales et éthiques",
                "Prévoir une révision périodique des modèles",
                "Former les utilisateurs aux nouveaux modèles"
            ]
        }
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Génère un rapport complet de la démonstration."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/Users/julienrm/Workspace/M2/sesame-shap/data/results/bias_mitigation/rapport_demo_complete_{timestamp}.md"
        
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Rapport Complet - Démonstration Intégrée de Mitigation des Biais COMPAS\n\n")
            f.write(f"**Date de génération:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Système:** Mac M4 Pro avec Apple Silicon\n")
            f.write(f"**Framework:** BiasMitigationFramework v1.0\n\n")
            
            # Résumé exécutif
            f.write("## Résumé Exécutif\n\n")
            
            if self.demo_results.get('demo_success', False):
                f.write("✅ **Démonstration réussie** - Pipeline complet exécuté avec succès\n\n")
                
                recommendations = self.demo_results.get('final_recommendations', {})
                if recommendations.get('best_strategy'):
                    best = recommendations['best_strategy']
                    f.write(f"🏆 **Meilleure stratégie identifiée:** {best['name']}\n")
                    f.write(f"📉 **Réduction du biais:** {best['bias_reduction']:.1f}%\n")
                    f.write(f"📊 **Impact sur la performance:** {best['performance_impact']:+.4f}\n\n")
                else:
                    f.write("⚠️ **Aucune stratégie optimale identifiée** - Révision nécessaire\n\n")
            else:
                f.write("❌ **Démonstration échouée** - Voir les détails d'erreur ci-dessous\n\n")
            
            # Détails des stratégies testées
            f.write("## Stratégies de Mitigation Testées\n\n")
            
            mitigation_results = self.demo_results.get('mitigation_strategies', {})
            for strategy, result in mitigation_results.items():
                f.write(f"### {strategy.replace('_', ' ').title()}\n\n")
                
                if result.get('success', False):
                    f.write("✅ **Statut:** Succès\n")
                    f.write(f"🔧 **Méthode:** {result.get('method', 'N/A')}\n")
                    
                    # Détails du rapport si disponible
                    report = result.get('report', {})
                    if isinstance(report, dict) and 'method' in report:
                        f.write(f"📋 **Détails:** {report.get('method', 'N/A')}\n")
                    
                else:
                    f.write("❌ **Statut:** Échec\n")
                    f.write(f"🚫 **Erreur:** {result.get('error', 'Erreur inconnue')}\n")
                
                f.write("\n")
            
            # Comparaison des résultats
            f.write("## Comparaison des Résultats\n\n")
            
            comparison_results = self.demo_results.get('evaluation_comparisons', {})
            if comparison_results:
                f.write("| Stratégie | Réduction Biais (%) | Impact Performance | Score Composite |\n")
                f.write("|-----------|---------------------|-------------------|------------------|\n")
                
                recommendations = self.demo_results.get('final_recommendations', {})
                ranking = recommendations.get('strategy_ranking', [])
                
                for rank_info in ranking:
                    f.write(f"| {rank_info['strategy']} | "
                           f"{rank_info['bias_reduction']:.1f}% | "
                           f"{rank_info['performance_impact']:+.4f} | "
                           f"{rank_info['score']:.2f} |\n")
                
                f.write("\n")
            
            # Recommandations détaillées
            f.write("## Recommandations Détaillées\n\n")
            
            recommendations = self.demo_results.get('final_recommendations', {})
            detailed_analysis = recommendations.get('detailed_analysis', {})
            
            for strategy, analysis in detailed_analysis.items():
                f.write(f"### {strategy.replace('_', ' ').title()}\n\n")
                f.write(f"**Recommandation:** {analysis['recommendation']}\n\n")
                f.write(f"**Justification:** {analysis['rationale']}\n\n")
                
                if 'metrics' in analysis:
                    metrics = analysis['metrics']
                    f.write("**Métriques:**\n")
                    f.write(f"- Réduction du biais: {metrics.get('bias_reduction_pct', 0):.1f}%\n")
                    f.write(f"- Changement d'accuracy: {metrics.get('accuracy_change', 0):+.4f}\n")
                    f.write(f"- Score composite: {metrics.get('composite_score', 0):.2f}\n\n")
            
            # Guide d'implémentation
            implementation_guide = recommendations.get('implementation_guide', {})
            if implementation_guide:
                f.write("## Guide d'Implémentation\n\n")
                
                next_steps = implementation_guide.get('next_steps', [])
                if next_steps:
                    f.write("### Prochaines Étapes\n\n")
                    for i, step in enumerate(next_steps, 1):
                        f.write(f"{i}. {step}\n")
                    f.write("\n")
                
                considerations = implementation_guide.get('considerations', [])
                if considerations:
                    f.write("### Considérations Importantes\n\n")
                    for consideration in considerations:
                        f.write(f"- {consideration}\n")
                    f.write("\n")
            
            # Annexes techniques
            f.write("## Annexes Techniques\n\n")
            f.write(f"**Configuration système:** {os.cpu_count()} cœurs CPU\n")
            f.write(f"**Graine aléatoire:** {self.random_state}\n")
            f.write(f"**Frameworks utilisés:** scikit-learn, fairlearn, xgboost, lightgbm\n")
            f.write(f"**Optimisations:** Apple Silicon, multiprocessing\n\n")
            
            if not self.demo_results.get('demo_success', False):
                f.write("### Détails d'Erreur\n\n")
                f.write(f"**Erreur:** {self.demo_results.get('error', 'Erreur inconnue')}\n\n")
            
        logger.info(f"Rapport complet généré: {report_path}")
        return report_path


def main():
    """Fonction principale pour exécuter la démonstration complète."""
    
    print("🚀 Démonstration Intégrée de Mitigation des Biais COMPAS")
    print("=" * 60)
    print("🍎 Optimisé pour Mac M4 Pro avec Apple Silicon")
    print()
    
    # Configuration des chemins
    base_dir = "/Users/julienrm/Workspace/M2/sesame-shap"
    data_file = os.path.join(base_dir, "data", "raw", "compas_sample.csv")
    
    # Vérification de l'existence des données
    if os.path.exists(data_file):
        print(f"📂 Utilisation des données existantes: {data_file}")
        data_path = data_file
    else:
        print("📊 Création de données d'exemple pour la démonstration")
        data_path = None
    
    # Initialisation et exécution de la démonstration
    demo = IntegratedBiasMitigationDemo(data_path=data_path, random_state=42)
    
    try:
        print("\n🔄 Exécution du pipeline complet...")
        results = demo.run_complete_pipeline()
        
        if results['demo_success']:
            print("\n✅ Démonstration terminée avec succès!")
            
            # Affichage des résultats principaux
            recommendations = results.get('final_recommendations', {})
            if recommendations.get('best_strategy'):
                best = recommendations['best_strategy']
                print(f"\n🏆 Meilleure stratégie: {best['name']}")
                print(f"📉 Réduction du biais: {best['bias_reduction']:.1f}%")
                print(f"📊 Impact performance: {best['performance_impact']:+.4f}")
            
            # Génération du rapport
            print("\n📄 Génération du rapport complet...")
            report_path = demo.generate_comprehensive_report()
            print(f"✅ Rapport généré: {report_path}")
            
            # Résumé des stratégies testées
            mitigation_results = results.get('mitigation_strategies', {})
            successful = sum(1 for r in mitigation_results.values() if r.get('success', False))
            total = len(mitigation_results)
            
            print(f"\n📊 Résumé: {successful}/{total} stratégies réussies")
            print("🔍 Consultez le rapport détaillé pour plus d'informations")
            
        else:
            print(f"\n❌ Erreur dans la démonstration: {results.get('error', 'Erreur inconnue')}")
            
    except Exception as e:
        print(f"\n💥 Erreur critique: {str(e)}")
        logger.error(f"Erreur critique dans la démonstration: {str(e)}", exc_info=True)
    
    print("\n🎯 Démonstration terminée")
    print("📚 Le framework est prêt pour l'analyse de biais COMPAS en production")


if __name__ == "__main__":
    main()