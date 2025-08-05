# Pipeline de Feature Engineering COMPAS

Ce module fournit un pipeline complet de préprocessing pour le dataset COMPAS avec une approche consciente des biais.

## Fonctionnalités Principales

### 🔧 Preprocessing Robuste
- **Traitement des valeurs manquantes** : Stratégies adaptées par type de variable
- **Encodage des variables catégorielles** : One-hot encoding et label encoding
- **Création de features dérivées** : Groupes d'âge, catégories d'antécédents, interactions
- **Normalisation** : StandardScaler pour les features numériques

### 🎯 Conscience des Biais
- **Séparation des attributs sensibles** : race, sexe, âge
- **Versions multiples du dataset** :
  - Complet (toutes les features)
  - Sans attributs sensibles (pour mitigation des biais)
  - Simplifié (pour interprétabilité)
- **Documentation des sources de biais potentielles**

### 📊 Contrôle Qualité
- **Validation des données** : Détection d'outliers, valeurs manquantes, doublons
- **Rapports détaillés** : Métadonnées complètes, logs de processing
- **Tests de cohérence** : Vérifications spécifiques au dataset COMPAS

## Utilisation

### Utilisation Basique

```python
from src.feature_engineering import COMPASFeatureEngineer

# Initialiser le preprocesseur
processor = COMPASFeatureEngineer(random_state=42)

# Exécuter le pipeline complet
results = processor.preprocess_compas_data('path/to/compas_data.csv')

# Sauvegarder les datasets traités
processor.save_processed_datasets(
    results['datasets'], 
    'data/processed/'
)
```

### Pipeline Détaillé

```python
# 1. Chargement des données
df = processor.load_compas_data('data.csv')

# 2. Traitement des valeurs manquantes
df_clean = processor.handle_missing_values(df)

# 3. Encodage des variables catégorielles
df_encoded, encoders = processor.encode_categorical_features(df_clean)

# 4. Création des features dérivées
df_features = processor.create_derived_features(df_encoded)

# 5. Validation de la qualité
quality_report = processor.validate_data_quality(df_features)

# 6. Préparation pour la modélisation
datasets = processor.prepare_features_for_modeling(df_features)
```

## Features Créées

### Features de Base (Encodées)
- **race_*** : Variables dummy pour les origines ethniques
- **sex_*** : Variables dummy pour le sexe
- **c_charge_degree_*** : Variables dummy pour le degré d'accusation
- **age_cat_encoded** : Catégorie d'âge encodée
- **score_text_encoded** : Score de risque textuel encodé
- **risk_score_ordinal** : Score de risque ordinal (0-2)

### Features Dérivées
- **age_group_detailed** : Groupes d'âge détaillés (18-24, 25-34, etc.)
- **age_group_numeric** : Version numérique des groupes d'âge
- **priors_category** : Catégories d'antécédents (None, Low, Medium, High, Very_High)
- **has_priors** : Indicateur binaire de présence d'antécédents
- **many_priors** : Indicateur binaire de nombreux antécédents (>3)
- **high_risk_score** : Indicateur binaire de score COMPAS élevé (≥7)
- **medium_risk_score** : Indicateur binaire de score COMPAS moyen (4-6)
- **low_risk_score** : Indicateur binaire de score COMPAS faible (≤3)

### Features d'Interaction
- **age_priors_interaction** : Interaction âge × nombre d'antécédents
- **age_log** : Transformation logarithmique de l'âge
- **priors_count_log** : Transformation logarithmique des antécédents

### Features Temporelles
- **screening_delay_abs** : Valeur absolue du délai d'évaluation
- **screening_delay_category** : Catégorie de délai (Same_Day, Within_Week, etc.)

### Features Binaires
- **is_young** : Indicateur d'âge < 25 ans
- **is_elderly** : Indicateur d'âge > 65 ans

## Versions des Datasets

### 1. Dataset Complet (`full`)
- **35 features** incluant toutes les variables et features dérivées
- Recommandé pour la modélisation initiale et l'exploration

### 2. Dataset Sans Attributs Sensibles (`no_sensitive`)
- **23 features** sans race, sexe, et catégories d'âge directes
- Recommandé pour l'analyse d'équité et la mitigation des biais

### 3. Dataset Simplifié (`simplified`)
- **8 features** principales sélectionnées pour l'interprétabilité
- Recommandé pour les modèles explicables et les présentations

## Rapport de Qualité

Le pipeline génère automatiquement :

- **Analyse des valeurs manquantes** : Avant/après traitement
- **Détection d'outliers** : Méthode IQR par variable
- **Distribution des variables** : Statistiques descriptives
- **Tests de cohérence** : Vérifications spécifiques COMPAS
- **Avertissements** : Signalement des problèmes potentiels

## Fichiers Générés

### Datasets
- `compas_full_YYYYMMDD_HHMMSS.csv` : Dataset complet
- `compas_no_sensitive_YYYYMMDD_HHMMSS.csv` : Sans attributs sensibles
- `compas_simplified_YYYYMMDD_HHMMSS.csv` : Version simplifiée
- `*_train.csv` et `*_test.csv` : Splits d'entraînement et de test

### Métadonnées
- `compas_metadata_YYYYMMDD_HHMMSS.json` : Informations complètes
  - Description des datasets
  - Liste des features
  - Rapport de qualité
  - Log de processing
  - Description des features

## Recommandations d'Usage

### Pour l'Analyse d'Équité
1. Utiliser le dataset **sans attributs sensibles**
2. Comparer les performances avec le dataset complet
3. Analyser l'importance des features potentiellement biaisées

### Pour l'Interprétabilité
1. Utiliser le dataset **simplifié**
2. Se concentrer sur les features principales
3. Éviter les interactions complexes

### Pour la Performance
1. Commencer avec le dataset **complet**
2. Utiliser la validation croisée stratifiée
3. Surveiller les métriques d'équité

## Exemple de Résultats

```
📊 3 versions du dataset créées:
  - full: 35 features, 1000 échantillons
  - no_sensitive: 23 features, 1000 échantillons  
  - simplified: 8 features, 1000 échantillons

📋 Impact de la suppression des attributs sensibles:
  - Performance avec attributs sensibles: 0.605
  - Performance sans attributs sensibles: 0.570
  - Différence: +0.035 (Impact modéré - À investiguer)

✅ Recommandations:
  - Le dataset simplifié offre des performances comparables
  - Le dataset sans attributs sensibles est disponible pour l'analyse d'équité
```

## Bonnes Pratiques

### Avant l'Utilisation
- Vérifier la qualité des données sources
- Examiner le rapport de qualité généré
- Comprendre les features créées

### Pendant la Modélisation
- Utiliser la validation croisée stratifiée
- Surveiller les métriques d'équité
- Comparer les performances entre versions

### Après la Modélisation
- Analyser l'importance des features
- Vérifier l'absence de biais dans les prédictions
- Documenter les choix de preprocessing

## Dépendances

- `pandas >= 1.5.0`
- `numpy >= 1.21.0`
- `scikit-learn >= 1.2.0`
- `matplotlib >= 3.6.0` (pour les visualisations)

## Contribution

Pour contribuer à ce pipeline :

1. Respecter les conventions de nommage
2. Ajouter des tests pour les nouvelles fonctionnalités
3. Documenter les nouvelles features créées
4. Maintenir la compatibilité avec les versions existantes

## Support

Pour toute question ou problème :

1. Vérifier les logs de processing
2. Examiner le rapport de qualité
3. Consulter la documentation des features
4. Vérifier les versions des dépendances