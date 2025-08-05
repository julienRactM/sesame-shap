"""
Module d'ingénierie des caractéristiques pour le dataset COMPAS
=============================================================

Ce module fournit un pipeline complet de préprocessing pour le dataset COMPAS,
avec une attention particulière aux biais et à la préparation des données pour
l'analyse d'équité et la modélisation ML.

Auteur: Data Engineering Pipeline
Date: 2025-08-05
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class COMPASFeatureEngineer:
    """
    Classe principale pour l'ingénierie des caractéristiques du dataset COMPAS.
    
    Cette classe gère le preprocessing complet du dataset COMPAS avec une approche
    consciente des biais, incluant le traitement des valeurs manquantes, l'encodage
    des variables catégorielles, la création de features dérivées, et la préparation
    de multiples versions du dataset.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le préprocesseur COMPAS.
        
        Args:
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.random_state = random_state
        self.feature_encoders = {}
        self.scalers = {}
        self.feature_descriptions = {}
        self.data_quality_report = {}
        self.processing_log = []
        
        # Attributs sensibles pour l'analyse de biais
        self.sensitive_attributes = ['race', 'sex', 'age_cat']
        
        # Configuration des seuils et paramètres
        self.age_group_thresholds = [25, 35, 45, 55]
        self.prior_count_thresholds = [0, 1, 3, 10]
        
        logger.info("COMPASFeatureEngineer initialisé avec random_state=%d", random_state)
    
    def load_compas_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données COMPAS depuis un fichier CSV.
        
        Args:
            file_path: Chemin vers le fichier CSV des données COMPAS
            
        Returns:
            DataFrame contenant les données brutes
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format des données n'est pas valide
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Données COMPAS chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Validation basique du format
            expected_columns = ['age', 'race', 'sex', 'priors_count', 'c_charge_degree', 
                              'score_text', 'decile_score', 'two_year_recid']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Colonnes manquantes détectées: {missing_columns}")
            
            self._log_processing_step("Chargement des données", df.shape)
            return df
            
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise ValueError(f"Impossible de charger les données: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Traite les valeurs manquantes avec des stratégies appropriées par type de variable.
        
        Args:
            df: DataFrame avec les données brutes
            
        Returns:
            DataFrame avec les valeurs manquantes traitées
        """
        logger.info("Début du traitement des valeurs manquantes")
        df_processed = df.copy()
        
        # Rapport sur les valeurs manquantes avant traitement
        missing_before = df_processed.isnull().sum()
        missing_report = {
            'before_treatment': missing_before[missing_before > 0].to_dict(),
            'strategies_applied': {}
        }
        
        # Stratégies par type de variable
        
        # 1. Variables numériques - imputation par la médiane pour éviter l'impact des outliers
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                median_value = df_processed[col].median()
                df_processed.loc[:, col] = df_processed[col].fillna(median_value)
                missing_report['strategies_applied'][col] = f'imputation_mediane_{median_value}'
                logger.info(f"Variable {col}: {missing_before[col]} valeurs manquantes imputées par la médiane ({median_value})")
        
        # 2. Variables catégorielles - imputation par le mode ou création d'une catégorie "Unknown"
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                # Pour les attributs sensibles, on crée une catégorie "Unknown"
                if col in self.sensitive_attributes:
                    df_processed.loc[:, col] = df_processed[col].fillna('Unknown')
                    missing_report['strategies_applied'][col] = 'categorie_unknown'
                    logger.info(f"Variable sensible {col}: {missing_before[col]} valeurs manquantes -> 'Unknown'")
                else:
                    # Pour les autres variables catégorielles, imputation par le mode
                    mode_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 'Unknown'
                    df_processed.loc[:, col] = df_processed[col].fillna(mode_value)
                    missing_report['strategies_applied'][col] = f'imputation_mode_{mode_value}'
                    logger.info(f"Variable {col}: {missing_before[col]} valeurs manquantes imputées par le mode ({mode_value})")
        
        # 3. Traitement spécial pour certaines variables COMPAS
        if 'days_b_screening_arrest' in df_processed.columns:
            # Cette variable peut avoir des valeurs extrêmes, on applique un seuil
            df_processed['days_b_screening_arrest'] = df_processed['days_b_screening_arrest'].clip(-30, 30)
            missing_report['strategies_applied']['days_b_screening_arrest'] = 'clipping_-30_30'
        
        # Rapport final
        missing_after = df_processed.isnull().sum()
        missing_report['after_treatment'] = missing_after[missing_after > 0].to_dict()
        missing_report['total_missing_before'] = missing_before.sum()
        missing_report['total_missing_after'] = missing_after.sum()
        
        self.data_quality_report['missing_values'] = missing_report
        self._log_processing_step("Traitement des valeurs manquantes", df_processed.shape)
        
        logger.info(f"Traitement des valeurs manquantes terminé. Avant: {missing_report['total_missing_before']}, Après: {missing_report['total_missing_after']}")
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode les variables catégorielles avec des stratégies appropriées.
        
        Args:
            df: DataFrame avec les données préprocessées
            
        Returns:
            Tuple contenant le DataFrame encodé et un dictionnaire des encoders utilisés
        """
        logger.info("Début de l'encodage des variables catégorielles")
        df_encoded = df.copy()
        encoders_used = {}
        
        # Variables à encoder avec One-Hot Encoding (nominales)
        onehot_variables = ['race', 'sex', 'c_charge_degree']
        
        # Variables à encoder avec Label Encoding (ordinales)
        label_variables = ['age_cat', 'score_text']
        
        # Variables catégorielles dérivées à encoder
        categorical_derived = []
        
        # 1. One-Hot Encoding pour les variables nominales
        for var in onehot_variables:
            if var in df_encoded.columns:
                # Création des variables dummy
                dummies = pd.get_dummies(df_encoded[var], prefix=var, drop_first=False)
                
                # Sauvegarde des colonnes créées pour référence future
                encoders_used[var] = {
                    'type': 'onehot',
                    'categories': list(dummies.columns),
                    'original_categories': df_encoded[var].unique().tolist()
                }
                
                # Ajout au DataFrame et suppression de l'originale
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(var, axis=1, inplace=True)
                
                logger.info(f"One-Hot Encoding appliqué à {var}: {len(dummies.columns)} nouvelles colonnes créées")
        
        # 2. Label Encoding pour les variables ordinales
        for var in label_variables:
            if var in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{var}_encoded'] = le.fit_transform(df_encoded[var].astype(str))
                
                encoders_used[var] = {
                    'type': 'label',
                    'encoder': le,
                    'classes': le.classes_.tolist()
                }
                
                # Garder l'originale pour référence
                logger.info(f"Label Encoding appliqué à {var}: {len(le.classes_)} classes")
        
        # 3. Encodage spécial pour les scores COMPAS
        if 'score_text' in df.columns:
            # Mapping ordinal pour les scores de risque
            risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            df_encoded['risk_score_ordinal'] = df_encoded['score_text'].map(risk_mapping)
            
            # Si des valeurs ne sont pas mappées, les remplacer par la médiane
            if df_encoded['risk_score_ordinal'].isnull().sum() > 0:
                median_risk = df_encoded['risk_score_ordinal'].median()
                df_encoded.loc[:, 'risk_score_ordinal'] = df_encoded['risk_score_ordinal'].fillna(median_risk)
                logger.warning(f"Valeurs de score_text non reconnues remplacées par la médiane: {median_risk}")
        
        # 4. Vérification finale - encoder toutes les variables catégorielles restantes
        remaining_categorical = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in remaining_categorical:
            if col not in ['age_group_detailed', 'priors_category', 'screening_delay_category']:  # Variables à traiter plus tard
                # Encoder avec LabelEncoder par défaut
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoders_used[f'{col}_final'] = {
                    'type': 'label_final',
                    'encoder': le,
                    'classes': le.classes_.tolist()
                }
                logger.info(f"Encodage final appliqué à {col}")
        
        self.feature_encoders = encoders_used
        self._log_processing_step("Encodage des variables catégorielles", df_encoded.shape)
        
        logger.info(f"Encodage terminé. {len(encoders_used)} variables encodées")
        return df_encoded, encoders_used
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features dérivées utiles pour l'analyse et la modélisation.
        
        Args:
            df: DataFrame avec les données encodées
            
        Returns:
            DataFrame avec les features dérivées ajoutées
        """
        logger.info("Début de la création des features dérivées")
        df_derived = df.copy()
        
        derived_features_created = []
        
        # 1. Groupes d'âge détaillés
        if 'age' in df_derived.columns:
            df_derived['age_group_detailed'] = pd.cut(
                df_derived['age'],
                bins=[0] + self.age_group_thresholds + [100],
                labels=['18-24', '25-34', '35-44', '45-54', '55+'],
                include_lowest=True
            )
            
            # Version numérique pour les modèles
            df_derived['age_group_numeric'] = pd.cut(
                df_derived['age'],
                bins=[0] + self.age_group_thresholds + [100],
                labels=range(5),
                include_lowest=True
            ).astype(int)
            
            derived_features_created.extend(['age_group_detailed', 'age_group_numeric'])
        
        # 2. Catégories de antécédents criminels
        if 'priors_count' in df_derived.columns:
            df_derived['priors_category'] = pd.cut(
                df_derived['priors_count'],
                bins=[-1] + self.prior_count_thresholds + [100],
                labels=[0, 1, 2, 3, 4],  # Utiliser des labels numériques
                include_lowest=True
            ).astype(int)
            
            # Variables binaires pour les seuils importants
            df_derived['has_priors'] = (df_derived['priors_count'] > 0).astype(int)
            df_derived['many_priors'] = (df_derived['priors_count'] > 3).astype(int)
            
            derived_features_created.extend(['priors_category', 'has_priors', 'many_priors'])
        
        # 3. Features liées aux scores COMPAS
        if 'decile_score' in df_derived.columns:
            df_derived['high_risk_score'] = (df_derived['decile_score'] >= 7).astype(int)
            df_derived['medium_risk_score'] = ((df_derived['decile_score'] >= 4) & 
                                              (df_derived['decile_score'] <= 6)).astype(int)
            df_derived['low_risk_score'] = (df_derived['decile_score'] <= 3).astype(int)
            
            derived_features_created.extend(['high_risk_score', 'medium_risk_score', 'low_risk_score'])
        
        # 4. Features d'interaction (potentiellement problématiques pour les biais)
        # Age × Antécédents
        if 'age' in df_derived.columns and 'priors_count' in df_derived.columns:
            df_derived['age_priors_interaction'] = df_derived['age'] * df_derived['priors_count']
            derived_features_created.append('age_priors_interaction')
        
        # 5. Features temporelles si disponibles
        if 'days_b_screening_arrest' in df_derived.columns:
            df_derived['screening_delay_abs'] = np.abs(df_derived['days_b_screening_arrest'])
            df_derived['screening_delay_category'] = pd.cut(
                df_derived['screening_delay_abs'],
                bins=[0, 1, 7, 30, 365],
                labels=[0, 1, 2, 3],  # Utiliser des labels numériques
                include_lowest=True
            ).astype(int)
            derived_features_created.extend(['screening_delay_abs', 'screening_delay_category'])
        
        # 6. Transformation logarithmique pour les variables avec distribution asymétrique
        skewed_vars = ['priors_count', 'age']
        for var in skewed_vars:
            if var in df_derived.columns:
                df_derived[f'{var}_log'] = np.log1p(df_derived[var])  # log(1+x) pour éviter log(0)
                derived_features_created.append(f'{var}_log')
        
        # 7. Variables binaires pour les seuils critiques
        if 'age' in df_derived.columns:
            df_derived['is_young'] = (df_derived['age'] < 25).astype(int)
            df_derived['is_elderly'] = (df_derived['age'] > 65).astype(int)
            derived_features_created.extend(['is_young', 'is_elderly'])
        
        # Documentation des features créées
        self.feature_descriptions.update({
            feature: f"Feature dérivée créée le {datetime.now().strftime('%Y-%m-%d')}"
            for feature in derived_features_created
        })
        
        # Encodage final des variables catégorielles restantes créées
        remaining_categorical = df_derived.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in remaining_categorical:
            if col != 'age_group_detailed':  # Garder cette version pour la lisibilité
                le = LabelEncoder()
                df_derived[f'{col}_numeric'] = le.fit_transform(df_derived[col].astype(str))
                derived_features_created.append(f'{col}_numeric')
                logger.info(f"Encodage numérique appliqué à la feature dérivée {col}")
        
        self._log_processing_step("Création des features dérivées", df_derived.shape)
        logger.info(f"Features dérivées créées: {len(derived_features_created)} nouvelles variables")
        
        return df_derived
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Effectue des contrôles de qualité complets sur les données.
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Dictionnaire contenant le rapport de qualité des données
        """
        logger.info("Début de la validation de la qualité des données")
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'data_types': df.dtypes.to_dict(),
            'missing_values': {},
            'duplicates': {},
            'outliers': {},
            'value_ranges': {},
            'categorical_distributions': {},
            'data_consistency': {},
            'warnings': []
        }
        
        # 1. Analyse des valeurs manquantes
        missing_counts = df.isnull().sum()
        quality_report['missing_values'] = {
            'total_missing': int(missing_counts.sum()),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (missing_counts / len(df) * 100).round(2).to_dict()
        }
        
        # 2. Détection des doublons
        duplicate_count = df.duplicated().sum()
        quality_report['duplicates'] = {
            'total_duplicates': int(duplicate_count),
            'duplicate_percentage': round(duplicate_count / len(df) * 100, 2)
        }
        
        if duplicate_count > 0:
            quality_report['warnings'].append(f"{duplicate_count} lignes dupliquées détectées")
        
        # 3. Analyse des outliers pour les variables numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                quality_report['outliers'][col] = {
                    'count': int(outliers),
                    'percentage': round(outliers / len(df) * 100, 2),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                }
        
        # 4. Analyse des plages de valeurs
        for col in numeric_columns:
            if col in df.columns:
                quality_report['value_ranges'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        
        # 5. Distribution des variables catégorielles
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col in df.columns:
                value_counts = df[col].value_counts()
                quality_report['categorical_distributions'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                    'distribution': value_counts.head(10).to_dict()
                }
        
        # 6. Contrôles de cohérence spécifiques au dataset COMPAS
        consistency_checks = []
        
        # Vérification des scores COMPAS
        if 'decile_score' in df.columns:
            invalid_scores = ((df['decile_score'] < 1) | (df['decile_score'] > 10)).sum()
            if invalid_scores > 0:
                consistency_checks.append(f"Scores COMPAS invalides: {invalid_scores}")
        
        # Vérification de la cohérence âge/catégorie d'âge
        if 'age' in df.columns and 'age_cat' in df.columns:
            # Cette vérification nécessiterait de connaître la correspondance exacte
            # On peut au moins vérifier les valeurs extrêmes
            if (df['age'] < 18).any():
                consistency_checks.append("Âges < 18 ans détectés")
            if (df['age'] > 100).any():
                consistency_checks.append("Âges > 100 ans détectés")
        
        quality_report['data_consistency']['checks_performed'] = consistency_checks
        
        # 7. Avertissements généraux
        if quality_report['missing_values']['total_missing'] > len(df) * 0.05:
            quality_report['warnings'].append("Plus de 5% de valeurs manquantes dans le dataset")
        
        if any(info['percentage'] > 10 for info in quality_report['outliers'].values() if isinstance(info, dict)):
            quality_report['warnings'].append("Variables avec plus de 10% d'outliers détectées")
        
        self.data_quality_report.update(quality_report)
        self._log_processing_step("Validation de la qualité des données", df.shape)
        
        logger.info(f"Validation terminée. {len(quality_report['warnings'])} avertissements générés")
        return quality_report
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, 
                                    target_column: str = 'two_year_recid',
                                    scale_features: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Prépare les features pour la modélisation ML avec différentes versions du dataset.
        
        Args:
            df: DataFrame avec toutes les features créées
            target_column: Nom de la variable cible
            scale_features: Si True, normalise les features numériques
            
        Returns:
            Dictionnaire contenant différentes versions du dataset préparées
        """
        logger.info("Début de la préparation des features pour la modélisation")
        
        if target_column not in df.columns:
            logger.error(f"Variable cible '{target_column}' non trouvée dans le dataset")
            raise ValueError(f"Variable cible '{target_column}' non trouvée")
        
        # Séparation des features et de la cible
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        prepared_datasets = {}
        
        # 1. Dataset complet (toutes les features)
        X_full = X.copy()
        
        # S'assurer que tous les datasets sont entièrement numériques
        X_full = self._ensure_numeric_dataset(X_full)
        
        # Identification des colonnes numériques pour la normalisation
        numeric_columns = X_full.select_dtypes(include=[np.number]).columns.tolist()
        
        if scale_features and numeric_columns:
            scaler_full = StandardScaler()
            X_full[numeric_columns] = scaler_full.fit_transform(X_full[numeric_columns])
            self.scalers['full_dataset'] = scaler_full
            logger.info(f"Normalisation appliquée à {len(numeric_columns)} features numériques")
        
        prepared_datasets['full'] = {
            'X': X_full,
            'y': y,
            'feature_names': X_full.columns.tolist(),
            'description': 'Dataset complet avec toutes les features'
        }
        
        # 2. Dataset sans attributs sensibles (pour mitigation des biais)
        sensitive_feature_patterns = ['race_', 'sex_', 'age_cat', 'age_group']
        sensitive_columns = []
        
        for pattern in sensitive_feature_patterns:
            sensitive_columns.extend([col for col in X_full.columns if pattern in col])
        
        # Ajouter les attributs sensibles directs s'ils existent encore
        direct_sensitive = [col for col in self.sensitive_attributes if col in X_full.columns]
        sensitive_columns.extend(direct_sensitive)
        
        # Supprimer les doublons
        sensitive_columns = list(set(sensitive_columns))
        
        X_no_sensitive = X_full.drop(columns=sensitive_columns, errors='ignore')
        X_no_sensitive = self._ensure_numeric_dataset(X_no_sensitive)
        
        prepared_datasets['no_sensitive'] = {
            'X': X_no_sensitive,
            'y': y,
            'feature_names': X_no_sensitive.columns.tolist(),
            'sensitive_features_removed': sensitive_columns,
            'description': 'Dataset sans attributs sensibles pour mitigation des biais'
        }
        
        # 3. Dataset simplifié (features principales pour l'interprétabilité)
        # Sélection des features les plus importantes/interprétables
        important_features = []
        
        # Features de base importantes
        base_features = ['age', 'priors_count', 'c_charge_degree', 'decile_score']
        important_features.extend([f for f in base_features if f in X_full.columns])
        
        # Features dérivées importantes
        derived_features = ['has_priors', 'high_risk_score', 'age_group_numeric']
        important_features.extend([f for f in derived_features if f in X_full.columns])
        
        # Features encodées importantes (garder les principales catégories)
        for col in ['c_charge_degree_F', 'c_charge_degree_M']:  # Felony, Misdemeanor
            if col in X_full.columns:
                important_features.append(col)
        
        X_simplified = X_full[important_features] if important_features else X_full.iloc[:, :10]
        X_simplified = self._ensure_numeric_dataset(X_simplified)
        
        prepared_datasets['simplified'] = {
            'X': X_simplified,
            'y': y,
            'feature_names': X_simplified.columns.tolist(),
            'description': 'Dataset simplifié avec features principales pour interprétabilité'
        }
        
        # 4. Création des splits train/test pour chaque version
        for dataset_name, dataset_info in prepared_datasets.items():
            X_data = dataset_info['X']
            y_data = dataset_info['y']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, 
                test_size=0.2, 
                random_state=self.random_state,
                stratify=y_data
            )
            
            dataset_info.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
        self._log_processing_step("Préparation pour la modélisation", 
                                {"versions": len(prepared_datasets)})
        
        logger.info(f"Préparation terminée. {len(prepared_datasets)} versions du dataset créées")
        return prepared_datasets
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Retourne les descriptions détaillées de toutes les features.
        
        Returns:
            Dictionnaire avec les descriptions de chaque feature
        """
        descriptions = {
            # Features originales COMPAS
            'age': 'Âge du défendeur au moment de l\'évaluation COMPAS',
            'race': 'Race/origine ethnique du défendeur',
            'sex': 'Sexe du défendeur',
            'priors_count': 'Nombre d\'antécédents criminels',
            'c_charge_degree': 'Degré de l\'accusation actuelle (Felony/Misdemeanor)',
            'score_text': 'Score de risque COMPAS textuel (Low/Medium/High)',
            'decile_score': 'Score de risque COMPAS numérique (1-10)',
            'two_year_recid': 'Variable cible: récidive dans les 2 ans (0/1)',
            
            # Features encodées
            'race_African-American': 'Indicateur binaire: race Afro-Américaine',
            'race_Caucasian': 'Indicateur binaire: race Caucasienne',
            'race_Hispanic': 'Indicateur binaire: race Hispanique',
            'race_Other': 'Indicateur binaire: autres races',
            'sex_Female': 'Indicateur binaire: sexe féminin',
            'sex_Male': 'Indicateur binaire: sexe masculin',
            'c_charge_degree_F': 'Indicateur binaire: accusation de type Felony',
            'c_charge_degree_M': 'Indicateur binaire: accusation de type Misdemeanor',
            
            # Features dérivées
            'age_group_detailed': 'Groupes d\'âge détaillés (catégoriel)',
            'age_group_numeric': 'Groupes d\'âge numériques (0-5)',
            'priors_category': 'Catégories d\'antécédents (None/Low/Medium/High/Very_High)',
            'has_priors': 'Indicateur binaire: présence d\'antécédents',
            'many_priors': 'Indicateur binaire: nombreux antécédents (>3)',
            'high_risk_score': 'Indicateur binaire: score COMPAS élevé (≥7)',
            'medium_risk_score': 'Indicateur binaire: score COMPAS moyen (4-6)',
            'low_risk_score': 'Indicateur binaire: score COMPAS faible (≤3)',
            'age_priors_interaction': 'Feature d\'interaction: âge × nombre d\'antécédents',
            'age_log': 'Transformation logarithmique de l\'âge',
            'priors_count_log': 'Transformation logarithmique du nombre d\'antécédents',
            'is_young': 'Indicateur binaire: jeune (<25 ans)',
            'is_elderly': 'Indicateur binaire: âgé (>65 ans)',
            
            # Features temporelles
            'days_b_screening_arrest': 'Jours entre arrestation et évaluation COMPAS',
            'screening_delay_abs': 'Valeur absolue du délai d\'évaluation',
            'screening_delay_category': 'Catégorie de délai d\'évaluation',
        }
        
        # Ajouter les descriptions personnalisées créées pendant le processing
        descriptions.update(self.feature_descriptions)
        
        return descriptions
    
    def save_processed_datasets(self, datasets: Dict, output_dir: str) -> Dict[str, str]:
        """
        Sauvegarde les datasets traités dans des fichiers CSV.
        
        Args:
            datasets: Dictionnaire des datasets préparés
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire des chemins des fichiers sauvegardés
        """
        logger.info(f"Sauvegarde des datasets dans {output_dir}")
        
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for dataset_name, dataset_info in datasets.items():
            # Sauvegarder le dataset complet
            full_data = pd.concat([dataset_info['X'], dataset_info['y']], axis=1)
            full_path = os.path.join(output_dir, f'compas_{dataset_name}_{timestamp}.csv')
            full_data.to_csv(full_path, index=False)
            saved_files[f'{dataset_name}_full'] = full_path
            
            # Sauvegarder les splits train/test séparément
            if 'X_train' in dataset_info:
                # Train set
                train_data = pd.concat([dataset_info['X_train'], dataset_info['y_train']], axis=1)
                train_path = os.path.join(output_dir, f'compas_{dataset_name}_train_{timestamp}.csv')
                train_data.to_csv(train_path, index=False)
                saved_files[f'{dataset_name}_train'] = train_path
                
                # Test set
                test_data = pd.concat([dataset_info['X_test'], dataset_info['y_test']], axis=1)
                test_path = os.path.join(output_dir, f'compas_{dataset_name}_test_{timestamp}.csv')
                test_data.to_csv(test_path, index=False)
                saved_files[f'{dataset_name}_test'] = test_path
        
        # Sauvegarder les métadonnées
        metadata = {
            'processing_timestamp': timestamp,
            'datasets_info': {
                name: {
                    'description': info['description'],
                    'feature_count': len(info['feature_names']),
                    'feature_names': info['feature_names'],
                    'sample_count': len(info['X'])
                }
                for name, info in datasets.items()
            },
            'feature_descriptions': self.get_feature_descriptions(),
            'processing_log': self.processing_log,
            'data_quality_report': self.data_quality_report
        }
        
        metadata_path = os.path.join(output_dir, f'compas_metadata_{timestamp}.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        saved_files['metadata'] = metadata_path
        
        self._log_processing_step("Sauvegarde des datasets", {"files_saved": len(saved_files)})
        logger.info(f"Sauvegarde terminée. {len(saved_files)} fichiers créés")
        
        return saved_files
    
    def preprocess_compas_data(self, file_path: str) -> Dict:
        """
        Pipeline complet de préprocessing des données COMPAS.
        
        Args:
            file_path: Chemin vers le fichier CSV des données COMPAS brutes
            
        Returns:
            Dictionnaire contenant tous les datasets préparés et les métadonnées
        """
        logger.info("=== DÉBUT DU PIPELINE DE PREPROCESSING COMPAS ===")
        
        try:
            # 1. Chargement des données
            df_raw = self.load_compas_data(file_path)
            
            # 2. Traitement des valeurs manquantes
            df_clean = self.handle_missing_values(df_raw)
            
            # 3. Encodage des variables catégorielles
            df_encoded, encoders = self.encode_categorical_features(df_clean)
            
            # 4. Création des features dérivées
            df_features = self.create_derived_features(df_encoded)
            
            # 5. Validation de la qualité des données
            quality_report = self.validate_data_quality(df_features)
            
            # 6. Préparation pour la modélisation
            prepared_datasets = self.prepare_features_for_modeling(df_features)
            
            # 7. Résumé final
            pipeline_results = {
                'datasets': prepared_datasets,
                'quality_report': quality_report,
                'feature_descriptions': self.get_feature_descriptions(),
                'encoders': self.feature_encoders,
                'scalers': self.scalers,
                'processing_log': self.processing_log,
                'pipeline_success': True
            }
            
            logger.info("=== PIPELINE DE PREPROCESSING TERMINÉ AVEC SUCCÈS ===")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de preprocessing: {str(e)}")
            pipeline_results = {
                'pipeline_success': False,
                'error': str(e),
                'processing_log': self.processing_log
            }
            return pipeline_results
    
    def _ensure_numeric_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        S'assure que le dataset est entièrement numérique pour la modélisation.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame entièrement numérique
        """
        df_numeric = df.copy()
        
        # Identifier les colonnes non numériques
        non_numeric_cols = df_numeric.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_cols:
            logger.info(f"Conversion de {len(non_numeric_cols)} colonnes non numériques: {non_numeric_cols}")
            
            for col in non_numeric_cols:
                try:
                    # Cas spécial pour les variables dummy (True/False)
                    if df_numeric[col].dtype == bool or set(df_numeric[col].unique()) <= {True, False, 0, 1}:
                        df_numeric.loc[:, col] = df_numeric[col].astype(int)
                        logger.info(f"Conversion booléenne vers entier pour {col}")
                        continue
                    
                    # Essayer de convertir en numérique directement
                    numeric_values = pd.to_numeric(df_numeric[col], errors='coerce')
                    
                    # Si pas de NaN créés, utiliser la conversion directe
                    if numeric_values.isnull().sum() == 0:
                        df_numeric.loc[:, col] = numeric_values
                        logger.info(f"Conversion numérique directe pour {col}")
                    else:
                        # Utiliser LabelEncoder
                        le = LabelEncoder()
                        df_numeric.loc[:, col] = le.fit_transform(df_numeric[col].astype(str))
                        logger.info(f"LabelEncoder appliqué à {col}")
                    
                except Exception as e:
                    # En dernier recours, utiliser LabelEncoder
                    le = LabelEncoder()
                    df_numeric.loc[:, col] = le.fit_transform(df_numeric[col].astype(str))
                    logger.warning(f"Conversion forcée avec LabelEncoder pour {col}: {str(e)}")
        
        # Vérification finale et nettoyage agressif si nécessaire
        final_non_numeric = df_numeric.select_dtypes(exclude=[np.number]).columns.tolist()
        if final_non_numeric:
            logger.warning(f"Nettoyage final pour {len(final_non_numeric)} colonnes restantes")
            for col in final_non_numeric:
                # Conversion la plus agressive possible
                try:
                    df_numeric.loc[:, col] = pd.Categorical(df_numeric[col]).codes
                    logger.info(f"Conversion par codes catégoriels pour {col}")
                except Exception as e:
                    # Supprimer la colonne si impossible à convertir
                    df_numeric.drop(col, axis=1, inplace=True)
                    logger.error(f"Colonne {col} supprimée car impossible à convertir: {str(e)}")
        
        return df_numeric
    
    def _log_processing_step(self, step_name: str, details: Union[Tuple, Dict]):
        """
        Enregistre une étape du processing dans le log.
        
        Args:
            step_name: Nom de l'étape
            details: Détails de l'étape (taille des données, etc.)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'details': details
        }
        self.processing_log.append(log_entry)


# Fonctions utilitaires pour l'utilisation du pipeline

def create_sample_compas_data(n_samples: int = 1000, output_path: str = None) -> pd.DataFrame:
    """
    Crée un dataset COMPAS d'exemple pour tester le pipeline.
    
    Args:
        n_samples: Nombre d'échantillons à créer
        output_path: Chemin de sauvegarde (optionnel)
        
    Returns:
        DataFrame contenant les données d'exemple
    """
    np.random.seed(42)
    
    # Simulation de données réalistes
    data = {
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], 
                                n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'sex': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
        'priors_count': np.random.poisson(2, n_samples).clip(0, 20),
        'c_charge_degree': np.random.choice(['F', 'M'], n_samples, p=[0.3, 0.7]),
        'decile_score': np.random.randint(1, 11, n_samples),
        'days_b_screening_arrest': np.random.normal(0, 15, n_samples).clip(-30, 30).astype(int)
    }
    
    # Score textuel basé sur le score numérique
    data['score_text'] = pd.cut(data['decile_score'], 
                               bins=[0, 3, 7, 10], 
                               labels=['Low', 'Medium', 'High'])
    
    # Catégorie d'âge
    data['age_cat'] = pd.cut(data['age'], 
                            bins=[0, 25, 45, 100], 
                            labels=['Less than 25', '25 - 45', 'Greater than 45'])
    
    # Variable cible (corrélée avec les features mais avec du bruit)
    prob_recid = (
        0.1 + 
        0.4 * (data['decile_score'] / 10) +
        0.2 * (data['priors_count'] > 2).astype(int) +
        0.1 * (data['age'] < 25).astype(int)
    )
    data['two_year_recid'] = np.random.binomial(1, prob_recid)
    
    df = pd.DataFrame(data)
    
    # Introduire quelques valeurs manquantes
    for col in ['race', 'priors_count', 'days_b_screening_arrest']:
        missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Données d'exemple sauvegardées dans {output_path}")
    
    return df


def main():
    """
    Fonction principale pour démontrer l'utilisation du pipeline.
    """
    # Configuration des chemins
    base_dir = "/Users/julienrm/Workspace/M2/sesame-shap"
    data_dir = os.path.join(base_dir, "data")
    raw_data_path = os.path.join(data_dir, "raw", "compas_sample.csv")
    processed_data_dir = os.path.join(data_dir, "processed")
    
    # Créer les répertoires nécessaires
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Créer des données d'exemple si elles n'existent pas
    if not os.path.exists(raw_data_path):
        logger.info("Création de données COMPAS d'exemple...")
        create_sample_compas_data(n_samples=1000, output_path=raw_data_path)
    
    # Initialiser le preprocesseur
    processor = COMPASFeatureEngineer(random_state=42)
    
    # Exécuter le pipeline complet
    logger.info("Exécution du pipeline de preprocessing...")
    results = processor.preprocess_compas_data(raw_data_path)
    
    if results['pipeline_success']:
        # Sauvegarder les datasets traités
        logger.info("Sauvegarde des datasets traités...")
        saved_files = processor.save_processed_datasets(
            results['datasets'], 
            processed_data_dir
        )
        
        # Afficher un résumé
        print("\n=== RÉSUMÉ DU PREPROCESSING ===")
        print(f"✅ Pipeline exécuté avec succès")
        print(f"📊 {len(results['datasets'])} versions du dataset créées:")
        
        for name, info in results['datasets'].items():
            print(f"  - {name}: {len(info['feature_names'])} features, {len(info['X'])} échantillons")
        
        print(f"💾 {len(saved_files)} fichiers sauvegardés dans {processed_data_dir}")
        print(f"📋 {len(results['feature_descriptions'])} features documentées")
        print(f"⚠️  {len(results['quality_report']['warnings'])} avertissements de qualité")
        
    else:
        print(f"❌ Erreur dans le pipeline: {results['error']}")


if __name__ == "__main__":
    main()