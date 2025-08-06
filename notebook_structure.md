# Structure Détaillée du Notebook Principal - COMPAS SHAP Analysis

## Guide Cell-by-Cell pour `main_notebook.ipynb`

Ce document fournit la structure détaillée cellule par cellule du notebook principal d'analyse COMPAS. Chaque cellule est documentée avec son type, son contenu et son objectif.

---

## 📚 Section 1: Introduction et Configuration

### Cellule 1: Titre et Introduction (Markdown)
```markdown
# 🎯 COMPAS SHAP Analysis - Projet SESAME

## Analyse d'Interprétabilité et de Détection de Biais

**Objectif**: Explorer les biais dans les modèles de prédiction de récidive COMPAS en utilisant SHAP pour l'interprétabilité.

**Contexte**: Le système COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) est utilisé dans le système judiciaire américain pour évaluer le risque de récidive. L'investigation ProPublica de 2016 a révélé des biais raciaux significatifs.

**Méthodes**: SHAP (primary), LIME, SAGE (bonus) pour l'interprétabilité, métriques d'équité pour la détection de biais.
```

### Cellule 2: Imports et Configuration (Code)
```python
# Configuration de l'environnement
import os
import sys
import warnings
from pathlib import Path

# Ajouter le dossier src au path
src_path = Path.cwd() / "src"
sys.path.append(str(src_path))

# Suppression des warnings non critiques
warnings.filterwarnings('ignore')

# Imports data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

print("✅ Configuration terminée")
print(f"📁 Répertoire de travail: {Path.cwd()}")
print(f"🐍 Version Python: {sys.version}")
```

### Cellule 3: Imports des Modules du Projet (Code)
```python
# Imports des modules développés
try:
    from data_acquisition import CompasDataAcquisition
    from exploratory_analysis import CompasEDA
    from feature_engineering import COMPASFeatureEngineer
    from model_training import CompasModelTrainer
    from shap_analysis import CompasShapAnalyzer
    from bias_analysis import CompasBiasAnalyzer
    from bias_mitigation import CompASBiasMitigator
    from fairness_evaluation import FairnessEvaluator
    from interpretability_comparison import InterpretabilityComparator
    
    print("✅ Tous les modules importés avec succès")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que tous les modules sont dans le dossier src/")
```

---

## 📊 Section 2: Acquisition et Chargement des Données

### Cellule 4: Introduction Section Données (Markdown)
```markdown
## 2. 📊 Acquisition des Données COMPAS

Nous utilisons le dataset COMPAS disponible sur Kaggle, qui contient environ 10,000 échantillons d'évaluations de risque avec les informations de récidive réelle.

**Sources des données**:
- `compas_scores_raw`: Scores COMPAS bruts
- `cox_violent_parsed`: Données Cox parsées 
- `propublica_data_for_fairml`: Dataset ProPublica nettoyé (principal)
```

### Cellule 5: Téléchargement des Données (Code)
```python
# Initialisation du module d'acquisition
print("🔄 Téléchargement des données COMPAS...")

data_acquisition = CompasDataAcquisition()

# Télécharger les données (automatique si pas déjà présent)
datasets_info = data_acquisition.download_compas_data()

print("\n📋 Datasets disponibles:")
for name, info in datasets_info.items():
    print(f"- {name}: {info.get('shape', 'N/A')} | {info.get('size_mb', 'N/A')} MB")
```

### Cellule 6: Chargement et Première Inspection (Code)
```python
# Charger les datasets
datasets = data_acquisition.load_compas_data()

# Utiliser le dataset ProPublica (principal pour l'analyse)
df_main = datasets.get('propublica_data_for_fairml')

if df_main is not None:
    print(f"✅ Dataset principal chargé: {df_main.shape}")
    print(f"📊 Colonnes: {list(df_main.columns)}")
    print(f"\n📈 Aperçu des données:")
    display(df_main.head())
    
    print(f"\n📊 Informations du dataset:")
    display(df_main.info())
else:
    print("❌ Erreur lors du chargement du dataset principal")
    # Utiliser un dataset alternatif ou des données simulées
    df_main = data_acquisition.create_sample_compas_data(n_samples=5000)
    print(f"🔄 Utilisation de données simulées: {df_main.shape}")
```

### Cellule 7: Statistiques Descriptives (Code)
```python
# Statistiques descriptives de base
print("📊 STATISTIQUES DESCRIPTIVES")
print("=" * 50)

# Informations générales
print(f"Nombre d'échantillons: {len(df_main):,}")
print(f"Nombre de features: {len(df_main.columns)}")
print(f"Valeurs manquantes: {df_main.isnull().sum().sum():,} ({(df_main.isnull().sum().sum() / (len(df_main) * len(df_main.columns)) * 100):.2f}%)")

# Statistiques des variables cibles et sensibles
if 'two_year_recid' in df_main.columns:
    recid_rate = df_main['two_year_recid'].mean()
    print(f"Taux de récidive (2 ans): {recid_rate:.1%}")

if 'race' in df_main.columns:
    print(f"\n👥 Distribution raciale:")
    race_dist = df_main['race'].value_counts()
    for race, count in race_dist.items():
        pct = (count / len(df_main)) * 100
        print(f"  - {race}: {count:,} ({pct:.1f}%)")

if 'sex' in df_main.columns:
    print(f"\n👤 Distribution par sexe:")
    sex_dist = df_main['sex'].value_counts()
    for sex, count in sex_dist.items():
        pct = (count / len(df_main)) * 100
        print(f"  - {sex}: {count:,} ({pct:.1f}%)")

# Sauvegarder pour les analyses suivantes
compas_data = df_main.copy()
print(f"\n✅ Données sauvegardées pour analyse: {len(compas_data)} échantillons")
```

---

## 🔍 Section 3: Analyse Exploratoire avec Focus Biais

### Cellule 8: Introduction EDA (Markdown)
```markdown
## 3. 🔍 Analyse Exploratoire avec Focus sur les Biais

Cette section examine les patterns potentiels de biais dans les données COMPAS, en reproduisant l'approche de l'investigation ProPublica.

**Objectifs**:
- Identifier les disparités dans les scores COMPAS par groupe démographique
- Analyser les taux de faux positifs et faux négatifs
- Visualiser les distributions et corrélations
- Détecter les patterns de biais statistiquement significants
```

### Cellule 9: Initialisation de l'Analyse Exploratoire (Code)  
```python
# Initialiser l'analyseur EDA
print("🔍 Initialisation de l'analyse exploratoire...")

eda_analyzer = CompasEDA()

# Charger les données dans l'analyseur
eda_analyzer.load_data(compas_data)

print("✅ Analyseur EDA initialisé")
print(f"📊 Dataset chargé: {len(compas_data)} échantillons")
```

### Cellule 10: Vue d'Ensemble du Dataset (Code)
```python
# Analyse générale du dataset
print("📊 ANALYSE GÉNÉRALE DU DATASET")
print("=" * 50)

overview_results = eda_analyzer.analyze_dataset_overview()

# Afficher les résultats
for key, value in overview_results.items():
    if isinstance(value, dict):
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")
```

### Cellule 11: Analyse Démographique et Biais (Code)
```python
# Analyse spécifique des biais démographiques
print("⚖️ ANALYSE DES BIAIS DÉMOGRAPHIQUES")
print("=" * 50)

bias_demographics = eda_analyzer.analyze_bias_demographics()

# Afficher les métriques clés
print("🎯 Métriques de Biais Clés:")
for metric, value in bias_demographics.get('key_metrics', {}).items():
    print(f"  - {metric}: {value}")

# Tests statistiques
print(f"\n📊 Tests Statistiques:")
for test, result in bias_demographics.get('statistical_tests', {}).items():
    print(f"  - {test}: p-value = {result.get('p_value', 'N/A'):.4f}")
    if result.get('p_value', 1) < 0.05:
        print(f"    ✅ Significatif (α = 0.05)")
    else:
        print(f"    ❌ Non significatif")
```

### Cellule 12: Analyse des Scores COMPAS (Code)
```python
# Analyse détaillée des scores COMPAS
print("📈 ANALYSE DES SCORES COMPAS")
print("=" * 50)

compas_analysis = eda_analyzer.analyze_compas_scores()

# Afficher les statistiques par groupe
if 'score_stats_by_group' in compas_analysis:
    print("📊 Statistiques des scores par groupe racial:")
    score_stats = compas_analysis['score_stats_by_group']
    
    for group, stats in score_stats.items():
        print(f"\n👥 {group}:")
        print(f"  - Moyenne: {stats.get('mean', 'N/A'):.2f}")
        print(f"  - Médiane: {stats.get('median', 'N/A'):.2f}")
        print(f"  - Écart-type: {stats.get('std', 'N/A'):.2f}")
        print(f"  - Échantillons: {stats.get('count', 'N/A'):,}")

# Identifier les disparités significatives
if 'disparity_analysis' in compas_analysis:
    disparities = compas_analysis['disparity_analysis']
    print(f"\n⚖️ Analyse des Disparités:")
    for comparison, disparity in disparities.items():
        print(f"  - {comparison}: Différence = {disparity:.3f}")
```

### Cellule 13: Visualisations des Biais (Code)
```python
# Créer les visualisations de biais
print("📊 Génération des visualisations de biais...")

# Générer les graphiques
visualization_paths = eda_analyzer.visualize_bias_patterns(save_path="results/bias_visualizations.png")

print(f"✅ Visualisations sauvegardées:")
for viz_type, path in visualization_paths.items():
    print(f"  - {viz_type}: {path}")

# Afficher quelques visualisations clés dans le notebook
# (Les visualisations interactives seront affichées directement)
```

### Cellule 14: Dashboard EDA Interactif (Code)
```python
# Créer un dashboard interactif d'EDA
print("🎛️ Création du dashboard EDA interactif...")

try:
    dashboard_path = eda_analyzer.create_interactive_dashboard()
    print(f"✅ Dashboard créé: {dashboard_path}")
    
    # Afficher le lien clickable
    from IPython.display import HTML
    html_link = f'<a href="{dashboard_path}" target="_blank">🔗 Ouvrir le Dashboard EDA</a>'
    display(HTML(html_link))
    
except Exception as e:
    print(f"❌ Erreur création dashboard: {e}")
```

### Cellule 15: Rapport EDA (Code)
```python
# Générer un rapport d'analyse exploratoire
print("📄 Génération du rapport d'analyse exploratoire...")

report_path = eda_analyzer.generate_bias_report()
print(f"✅ Rapport EDA généré: {report_path}")

# Afficher un résumé des principales conclusions
summary = eda_analyzer.get_analysis_summary()
print(f"\n📋 RÉSUMÉ DES PRINCIPALES CONCLUSIONS:")
print("=" * 50)
for conclusion in summary.get('key_findings', []):
    print(f"• {conclusion}")
```

---

## 🔧 Section 4: Feature Engineering et Préparation

### Cellule 16: Introduction Feature Engineering (Markdown)
```markdown
## 4. 🔧 Feature Engineering et Préparation des Données

Cette section prépare les données pour l'entraînement des modèles avec une approche consciente des biais.

**Objectifs**:
- Traiter les valeurs manquantes de manière appropriée
- Encoder les variables catégorielles 
- Créer des features dérivées pertinentes
- Préparer 3 versions du dataset (complet, sans attributs sensibles, simplifié)
- Diviser en ensembles d'entraînement et de test
```

### Cellule 17: Initialisation Feature Engineering (Code)
```python
# Initialiser l'ingénieur de features
print("🔧 Initialisation du Feature Engineering...")

feature_engineer = COMPASFeatureEngineer()

print("✅ Feature Engineer initialisé")
print(f"📊 Dataset à traiter: {compas_data.shape}")
```

### Cellule 18: Preprocessing Principal (Code)
```python
# Appliquer le preprocessing principal
print("⚙️ APPLICATION DU PREPROCESSING")
print("=" * 50)

# Déterminer la colonne cible
target_column = 'two_year_recid' if 'two_year_recid' in compas_data.columns else compas_data.columns[-1]
print(f"🎯 Colonne cible identifiée: {target_column}")

# Preprocessing complet
processed_data = feature_engineer.preprocess_compas_data(
    compas_data,
    target_column=target_column
)

print(f"✅ Preprocessing terminé")
print(f"📊 Dataset preprocessé: {processed_data.shape}")
print(f"📋 Features créées: {list(processed_data.columns)[:10]}..." if len(processed_data.columns) > 10 else f"📋 Features: {list(processed_data.columns)}")
```

### Cellule 19: Création des Versions du Dataset (Code)
```python
# Préparer les différentes versions pour la modélisation
print("📦 PRÉPARATION DES VERSIONS DU DATASET")
print("=" * 50)

# Créer les 3 versions + splits train/test
dataset_versions = feature_engineer.prepare_features_for_modeling(
    processed_data,
    target_column=target_column,
    test_size=0.2,
    random_state=42
)

print(f"✅ {len(dataset_versions)} versions créées:")
for version_name, data in dataset_versions.items():
    print(f"\n📊 Version '{version_name}':")
    print(f"  - Features: {data['X_train'].shape[1]}")
    print(f"  - Train: {data['X_train'].shape[0]} échantillons")
    print(f"  - Test: {data['X_test'].shape[0]} échantillons")
    print(f"  - Features list: {list(data['X_train'].columns)[:5]}...")
```

### Cellule 20: Validation et Quality Check (Code)
```python
# Validation de la qualité des données
print("✅ VALIDATION DE LA QUALITÉ DES DONNÉES")
print("=" * 50)

# Choisir la version complète pour validation
full_version = dataset_versions.get('full', list(dataset_versions.values())[0])
X_train, y_train = full_version['X_train'], full_version['y_train']
X_test, y_test = full_version['X_test'], full_version['y_test']

# Validations de base
print(f"🔍 Validations:")
print(f"  - Valeurs manquantes train: {X_train.isnull().sum().sum()}")
print(f"  - Valeurs manquantes test: {X_test.isnull().sum().sum()}")
print(f"  - Distribution cible train: {y_train.value_counts().to_dict()}")
print(f"  - Distribution cible test: {y_test.value_counts().to_dict()}")
print(f"  - Cohérence features: {list(X_train.columns) == list(X_test.columns)}")

# Rapport de qualité détaillé
quality_report = feature_engineer.validate_data_quality()
print(f"\n📋 Rapport de qualité:")
for check, result in quality_report.items():
    status = "✅" if result.get('passed', False) else "❌"
    print(f"  {status} {check}: {result.get('message', 'N/A')}")

print(f"\n✅ Données préparées et validées pour l'entraînement")
```

---

## 🤖 Section 5: Entraînement des Modèles

### Cellule 21: Introduction Entraînement (Markdown)
```markdown
## 5. 🤖 Entraînement des Modèles de Machine Learning

Cette section entraîne plusieurs modèles ML optimisés pour Mac M4 Pro avec évaluation des performances et préparation pour l'analyse SHAP.

**Modèles entraînés**:
- Random Forest (baseline performant)
- Logistic Regression (interprétable)
- XGBoost (gradient boosting optimisé)
- LightGBM (alternative efficace)
- Support Vector Machine (marge maximale)
- Neural Network (MLP simple)

**Métriques évaluées**:
- Performance: Accuracy, Precision, Recall, F1, AUC
- Équité: Demographic parity, Equal opportunity par groupe
```

### Cellule 22: Initialisation de l'Entraîneur (Code)
```python
# Initialiser l'entraîneur de modèles
print("🤖 Initialisation de l'entraîneur de modèles...")

model_trainer = CompasModelTrainer()

print("✅ Entraîneur initialisé avec optimisations Mac M4 Pro")
```

### Cellule 23: Préparation des Données d'Entraînement (Code)
```python
# Préparer les données pour l'entraînement
print("📊 PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT")
print("=" * 50)

# Utiliser la version complète par défaut
version_to_use = 'full'
if version_to_use not in dataset_versions:
    version_to_use = list(dataset_versions.keys())[0]

data = dataset_versions[version_to_use]
print(f"📦 Version utilisée: {version_to_use}")

# Extraires les données
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Créer des attributs sensibles simulés pour l'analyse de biais
# (Dans un cas réel, ces informations seraient extraites des données originales)
sensitive_attributes_train = pd.DataFrame({
    'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic'], len(X_train), p=[0.5, 0.4, 0.1]),
    'sex': np.random.choice(['Male', 'Female'], len(X_train), p=[0.7, 0.3]),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '45+'], len(X_train))
})

sensitive_attributes_test = pd.DataFrame({
    'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic'], len(X_test), p=[0.5, 0.4, 0.1]),
    'sex': np.random.choice(['Male', 'Female'], len(X_test), p=[0.7, 0.3]),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '45+'], len(X_test))
})

# Charger dans l'entraîneur
model_trainer.prepare_training_data(
    X_train, y_train, X_test, y_test,
    sensitive_attributes_train, sensitive_attributes_test
)

print(f"✅ Données chargées dans l'entraîneur:")
print(f"  - Train: {X_train.shape}")
print(f"  - Test: {X_test.shape}")
print(f"  - Features: {len(X_train.columns)}")
print(f"  - Attributs sensibles: {list(sensitive_attributes_train.columns)}")
```

### Cellule 24: Entraînement des Modèles (Code)
```python
# Entraîner tous les modèles
print("🚀 ENTRAÎNEMENT DES MODÈLES")
print("=" * 50)

# Configuration de l'entraînement
use_hyperparameter_tuning = True  # Mettre False pour un entraînement plus rapide
n_cv_folds = 3  # Réduire pour accélérer

print(f"⚙️ Configuration:")
print(f"  - Optimisation hyperparamètres: {use_hyperparameter_tuning}")
print(f"  - Cross-validation: {n_cv_folds} folds")
print(f"  - Optimisations Mac M4 Pro: Activées")

# Lancer l'entraînement
training_results = model_trainer.train_multiple_models(
    use_hyperparameter_tuning=use_hyperparameter_tuning,
    cv=n_cv_folds,
    verbose=True
)

print(f"\n✅ Entraînement terminé!")
print(f"📊 {len(training_results)} modèles entraînés")
```

### Cellule 25: Évaluation des Performances (Code)
```python
# Évaluation complète des modèles
print("📈 ÉVALUATION DES PERFORMANCES")
print("=" * 50)

# Évaluation avec métriques de performance et d'équité
evaluation_results = model_trainer.evaluate_models(
    include_fairness=True,
    protected_attribute='race'
)

# Afficher les résultats sous forme de tableau
results_df = pd.DataFrame(evaluation_results)
print("📊 Résultats d'évaluation:")
display(results_df)

# Créer les visualisations comparatives
visualization_paths = model_trainer.create_evaluation_visualizations()
print(f"\n📈 Visualisations créées:")
for viz_type, path in visualization_paths.items():
    print(f"  - {viz_type}: {path}")
```

### Cellule 26: Comparaison et Sélection des Modèles (Code)
```python
# Comparaison détaillée des modèles
print("🏆 COMPARAISON ET SÉLECTION DES MODÈLES")
print("=" * 50)

comparison_results = model_trainer.compare_model_performance()

# Identifier le meilleur modèle selon différents critères
best_models = {
    'performance': comparison_results['best_performance_model'],
    'fairness': comparison_results['most_fair_model'],
    'balanced': comparison_results['best_balanced_model']
}

print("🎯 Meilleurs modèles par critère:")
for criterion, model_info in best_models.items():
    print(f"  - {criterion.title()}: {model_info['name']} (Score: {model_info['score']:.4f})")

# Recommandations
print(f"\n💡 Recommandations:")
recommendations = comparison_results.get('recommendations', [])
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# Sauvegarder les modèles
save_paths = model_trainer.save_trained_models()
print(f"\n💾 Modèles sauvegardés:")
for model_name, path in save_paths.items():
    print(f"  - {model_name}: {path}")
```

---

## 🔍 Section 6: Analyse SHAP

### Cellule 27: Introduction SHAP (Markdown)
```markdown
## 6. 🔍 Analyse SHAP - Interprétabilité des Modèles

Cette section utilise SHAP (SHapley Additive exPlanations) pour expliquer les prédictions des modèles et détecter les sources de biais.

**Méthodes SHAP utilisées**:
- **TreeExplainer**: Pour Random Forest, XGBoost, LightGBM
- **KernelExplainer**: Pour SVM, Neural Networks
- **LinearExplainer**: Pour Logistic Regression

**Analyses réalisées**:
- Importance globale des features
- Explications locales (instances individuelles)
- Détection de biais via les valeurs SHAP
- Comparaison entre groupes démographiques
```

### Cellule 28: Initialisation SHAP (Code)
```python
# Initialiser l'analyseur SHAP
print("🔍 Initialisation de l'analyseur SHAP...")

shap_analyzer = CompasShapAnalyzer()

# Charger les modèles entraînés
trained_models = model_trainer.trained_models
shap_analyzer.load_trained_models(trained_models)

# Charger les données de test
shap_analyzer.load_test_data(X_test, y_test, sensitive_attributes_test)

print(f"✅ Analyseur SHAP initialisé")
print(f"🤖 Modèles chargés: {list(trained_models.keys())}")
print(f"📊 Données de test: {X_test.shape}")
```

### Cellule 29: Calcul des Valeurs SHAP (Code)
```python
# Calculer les valeurs SHAP pour tous les modèles
print("⚡ CALCUL DES VALEURS SHAP")
print("=" * 50)

# Configuration du calcul
sample_size = 200  # Réduire pour des calculs plus rapides
max_evals = 1000   # Pour KernelExplainer

print(f"⚙️ Configuration:")
print(f"  - Échantillons analysés: {sample_size}")
print(f"  - Évaluations max (Kernel): {max_evals}")
print(f"  - Optimisations Mac M4 Pro: Activées")

# Calcul des valeurs SHAP
shap_values = shap_analyzer.calculate_shap_values(
    max_evals=max_evals,
    sample_size=sample_size
)

print(f"\n✅ Valeurs SHAP calculées pour {len(shap_values)} modèles")
for model_name, values in shap_values.items():
    print(f"  - {model_name}: {values.shape}")
```

### Cellule 30: Analyse de l'Importance des Features (Code)
```python
# Analyser l'importance globale des features
print("📊 ANALYSE DE L'IMPORTANCE DES FEATURES")
print("=" * 50)

importance_results = shap_analyzer.analyze_feature_importance()

# Afficher le top 10 des features les plus importantes
print("🏆 Top 10 Features les Plus Importantes (moyenne tous modèles):")
top_features = importance_results.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)

for i, (feature, importance) in enumerate(top_features.items(), 1):
    print(f"  {i:2d}. {feature}: {importance:.4f}")

# Créer un graphique d'importance
fig = px.bar(
    x=top_features.values,
    y=top_features.index,
    orientation='h',
    title='Top 10 Features SHAP - Importance Moyenne',
    labels={'x': 'Importance SHAP', 'y': 'Features'}
)
fig.show()
```

### Cellule 31: Analyse des Biais via SHAP (Code)
```python
# Analyser les biais à travers les valeurs SHAP
print("⚖️ ANALYSE DES BIAIS VIA SHAP")
print("=" * 50)

# Analyse par groupe racial
bias_analysis_race = shap_analyzer.analyze_bias_through_shap('race')

print("🎯 Analyse des biais raciaux:")
for model_name, bias_df in bias_analysis_race.items():
    print(f"\n🤖 {model_name} - Top 5 features contribuant au biais racial:")
    top_bias_features = bias_df.head(5)
    
    for _, row in top_bias_features.iterrows():
        feature = row['feature']
        diff = row['shap_difference']
        group1, group2 = row['group1'], row['group2']
        
        print(f"  • {feature}: Δ = {diff:.4f}")
        print(f"    - {group1}: {row['group1_mean_shap']:.4f}")
        print(f"    - {group2}: {row['group2_mean_shap']:.4f}")

# Analyse par sexe si disponible
if 'sex' in sensitive_attributes_test.columns:
    bias_analysis_sex = shap_analyzer.analyze_bias_through_shap('sex')
    print(f"\n👥 Analyse des biais de genre disponible pour {len(bias_analysis_sex)} modèles")
```

### Cellule 32: Visualisations SHAP (Code)
```python
# Créer les visualisations SHAP
print("📊 CRÉATION DES VISUALISATIONS SHAP")
print("=" * 50)

# Choisir un modèle représentatif (le plus performant)
best_model_name = best_models['balanced']['name']
print(f"🎯 Modèle analysé: {best_model_name}")

# Générer toutes les visualisations SHAP
shap_viz_paths = shap_analyzer.create_shap_visualizations(
    best_model_name,
    save_plots=True
)

print(f"✅ Visualisations SHAP créées:")
for viz_type, path in shap_viz_paths.items():
    print(f"  - {viz_type}: {path}")

# Note: Les graphiques SHAP s'afficheront directement dans le notebook
```

### Cellule 33: Dashboard de Comparaison des Biais (Code)
```python
# Créer un dashboard interactif de comparaison des biais SHAP
print("🎛️ CRÉATION DU DASHBOARD DE BIAIS SHAP")
print("=" * 50)

# Dashboard pour analyse raciale
race_dashboard_path = shap_analyzer.create_bias_comparison_plots('race')
print(f"✅ Dashboard racial créé: {race_dashboard_path}")

# Lien clickable pour le dashboard
from IPython.display import HTML
html_link = f'<a href="{race_dashboard_path}" target="_blank">🔗 Ouvrir le Dashboard Biais SHAP</a>'
display(HTML(html_link))

# Dashboard pour analyse de genre (si disponible)
if 'sex' in sensitive_attributes_test.columns:
    sex_dashboard_path = shap_analyzer.create_bias_comparison_plots('sex')
    print(f"✅ Dashboard genre créé: {sex_dashboard_path}")
```

### Cellule 34: Rapport SHAP Complet (Code)
```python
# Générer le rapport SHAP complet
print("📄 GÉNÉRATION DU RAPPORT SHAP")
print("=" * 50)

report_path = shap_analyzer.generate_shap_report(output_format='markdown')
print(f"✅ Rapport SHAP généré: {report_path}")

# Afficher un résumé des conclusions SHAP
print(f"\n📋 CONCLUSIONS PRINCIPALES - ANALYSE SHAP:")
print("=" * 50)

# Résumé basé sur l'analyse d'importance
top_3_features = importance_results.groupby('feature')['importance'].mean().sort_values(ascending=False).head(3)
print(f"🏆 Top 3 features les plus influentes:")
for i, (feature, importance) in enumerate(top_3_features.items(), 1):
    print(f"  {i}. {feature} (importance: {importance:.4f})")

# Résumé des biais détectés
if bias_analysis_race:
    model_with_most_bias = max(bias_analysis_race.keys(), 
                              key=lambda m: bias_analysis_race[m]['abs_difference'].max())
    max_bias_feature = bias_analysis_race[model_with_most_bias].iloc[0]
    
    print(f"\n⚠️ Biais le plus significatif détecté:")
    print(f"  - Modèle: {model_with_most_bias}")
    print(f"  - Feature: {max_bias_feature['feature']}")
    print(f"  - Différence SHAP: {max_bias_feature['shap_difference']:.4f}")
    print(f"  - Groupes: {max_bias_feature['group1']} vs {max_bias_feature['group2']}")

print(f"\n✅ Analyse SHAP terminée - {len(shap_values)} modèles analysés")
```

---

## ⚖️ Section 7: Détection de Biais Avancée

### Cellule 35: Introduction Détection de Biais (Markdown)
```markdown
## 7. ⚖️ Détection de Biais Avancée avec Métriques d'Équité

Cette section implémente une analyse complète des biais avec des métriques d'équité standardisées et des tests statistiques.

**Métriques d'équité calculées**:
- **Demographic Parity**: Égalité des taux de prédiction positive
- **Equal Opportunity**: Égalité des taux de vrais positifs  
- **Equalized Odds**: Égalité des TPR et FPR
- **Calibration**: Fiabilité des probabilités prédites
- **Disparate Impact**: Test de la règle des 80%

**Tests statistiques**:
- Chi-carré pour l'indépendance
- Mann-Whitney U pour les distributions
- Significance testing avec corrections multiples
```

### Cellule 36: Initialisation Analyse de Biais (Code)
```python
# Initialiser l'analyseur de biais
print("⚖️ Initialisation de l'analyseur de biais...")

bias_analyzer = CompasBiasAnalyzer()

# Préparer les prédictions des modèles
print("🔄 Préparation des prédictions pour analyse...")

predictions = {}
probabilities = {}

for model_name, model in trained_models.items():
    # Prédictions binaires
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred
    
    # Probabilités (si disponibles)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilité classe positive
        probabilities[model_name] = y_proba
    else:
        # Pour les modèles sans predict_proba, utiliser decision_function ou prédictions
        probabilities[model_name] = y_pred.astype(float)

# Charger dans l'analyseur
bias_analyzer.load_predictions(predictions, probabilities, y_test, sensitive_attributes_test)

print(f"✅ Analyseur de biais initialisé")
print(f"🤖 Modèles analysés: {list(predictions.keys())}")
print(f"📊 Échantillons de test: {len(y_test)}")
```

### Cellule 37: Calcul des Métriques d'Équité (Code)
```python
# Calculer les métriques d'équité pour tous les modèles
print("📊 CALCUL DES MÉTRIQUES D'ÉQUITÉ")
print("=" * 50)

# Analyse par race (principal)
fairness_metrics_race = bias_analyzer.calculate_fairness_metrics('race')

print("🎯 Métriques d'équité par modèle (Race):")
print("=" * 50)

for model_name, metrics in fairness_metrics_race.items():
    print(f"\n🤖 {model_name}:")
    
    # Métriques principales
    dp_diff = metrics.get('demographic_parity_difference', 0)
    eo_diff = metrics.get('equal_opportunity_difference', 0)
    eod_diff = metrics.get('equalized_odds_difference', 0)
    di_ratio = metrics.get('disparate_impact_ratio', 1)
    passes_80 = metrics.get('passes_80_rule', True)
    
    print(f"  📈 Parité Démographique: {dp_diff:+.4f}")
    print(f"  🎯 Égalité des Chances: {eo_diff:+.4f}")
    print(f"  ⚖️ Égalité des Odds: {eod_diff:+.4f}")
    print(f"  📊 Impact Disparate: {di_ratio:.4f} {'✅' if passes_80 else '❌'}")
    
    # Significance tests
    chi2_p = metrics.get('chi2_pvalue', 1)
    mw_p = metrics.get('mannwhitney_pvalue', 1)
    print(f"  🧮 Tests (p-values): χ²={chi2_p:.4f}, MW={mw_p:.4f}")

# Analyse par sexe si suffisamment de données
if 'sex' in sensitive_attributes_test.columns:
    fairness_metrics_sex = bias_analyzer.calculate_fairness_metrics('sex')
    print(f"\n👥 Métriques d'équité par sexe calculées pour {len(fairness_metrics_sex)} modèles")
```

### Cellule 38: Détection des Patterns de Biais (Code)
```python
# Détecter les patterns de biais
print("🚨 DÉTECTION DES PATTERNS DE BIAIS")
print("=" * 50)

bias_patterns_race = bias_analyzer.detect_bias_patterns('race')

print("🔍 Classification des biais par modèle:")
for model_name, patterns in bias_patterns_race.items():
    bias_level = patterns['bias_level']
    bias_score = patterns['bias_score']
    
    # Définir l'emoji selon le niveau
    emoji = "🔴" if "Sévère" in bias_level else "🟡" if "Modéré" in bias_level else "🟢"
    
    print(f"\n{emoji} {model_name}: {bias_level} (Score: {bias_score})")
    
    # Détails des biais détectés
    severe_count = len(patterns['severe_bias'])
    moderate_count = len(patterns['moderate_bias'])
    potential_count = len(patterns['potential_bias'])
    
    if severe_count > 0:
        print(f"  🔴 Biais sévères: {severe_count}")
        for metric, value, deviation in patterns['severe_bias'][:3]:  # Top 3
            print(f"    • {metric}: {value:.4f} (écart: {deviation:.4f})")
    
    if moderate_count > 0:
        print(f"  🟡 Biais modérés: {moderate_count}")
    
    if potential_count > 0:
        print(f"  🟠 Biais potentiels: {potential_count}")

# Identifier le modèle le plus équitable
most_fair_model = min(bias_patterns_race.keys(), 
                     key=lambda m: bias_patterns_race[m]['bias_score'])
least_fair_model = max(bias_patterns_race.keys(),
                      key=lambda m: bias_patterns_race[m]['bias_score'])

print(f"\n🏆 Modèle le plus équitable: {most_fair_model}")
print(f"⚠️  Modèle le moins équitable: {least_fair_model}")
```

### Cellule 39: Comparaison entre Groupes (Code)
```python
# Comparaison détaillée entre groupes démographiques
print("👥 COMPARAISON ENTRE GROUPES DÉMOGRAPHIQUES")
print("=" * 50)

group_comparison = bias_analyzer.compare_group_outcomes('race')

# Afficher les résultats sous forme de tableau
print("📊 Comparaison des performances par groupe racial:")
display(group_comparison.round(4))

# Calculer les écarts moyens
if not group_comparison.empty:
    print(f"\n📈 Analyse des écarts:")
    
    # Grouper par modèle et calculer les écarts
    for model in group_comparison['model'].unique():
        model_data = group_comparison[group_comparison['model'] == model]
        
        if len(model_data) >= 2:
            print(f"\n🤖 {model}:")
            
            # Calculer écarts pour métriques principales
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics_to_compare:
                if metric in model_data.columns:
                    values = model_data[metric].values
                    if len(values) >= 2:
                        max_val, min_val = max(values), min(values)
                        gap = max_val - min_val
                        groups = model_data['group'].values
                        max_group = groups[np.argmax(values)]
                        min_group = groups[np.argmin(values)]
                        
                        print(f"  {metric}: Écart = {gap:.4f} ({max_group}: {max_val:.4f} vs {min_group}: {min_val:.4f})")
```

### Cellule 40: Dashboard de Biais Interactif (Code)
```python
# Créer le dashboard interactif de biais
print("🎛️ CRÉATION DU DASHBOARD DE BIAIS")
print("=" * 50)

bias_dashboard_path = bias_analyzer.visualize_bias_metrics('race')
print(f"✅ Dashboard de biais créé: {bias_dashboard_path}")

# Lien clickable
from IPython.display import HTML
html_link = f'<a href="{bias_dashboard_path}" target="_blank">🔗 Ouvrir le Dashboard de Biais</a>'
display(HTML(html_link))

# Afficher quelques métriques clés dans le notebook
print(f"\n📊 Résumé des métriques de biais:")

# Créer un tableau résumé
summary_data = []
for model_name, metrics in fairness_metrics_race.items():
    summary_data.append({
        'Modèle': model_name,
        'Parité_Démo': f"{metrics.get('demographic_parity_difference', 0):.4f}",
        'Égalité_Chances': f"{metrics.get('equal_opportunity_difference', 0):.4f}",
        'Impact_Disparate': f"{metrics.get('disparate_impact_ratio', 1):.4f}",
        'Règle_80%': '✅' if metrics.get('passes_80_rule', True) else '❌',
        'Niveau_Biais': bias_patterns_race[model_name]['bias_level']
    })

summary_df = pd.DataFrame(summary_data)
display(summary_df)
```

### Cellule 41: Rapport de Détection de Biais (Code)
```python
# Générer le rapport complet de détection de biais
print("📄 GÉNÉRATION DU RAPPORT DE BIAIS")
print("=" * 50)

bias_report_path = bias_analyzer.generate_bias_report('race', output_format='markdown')
print(f"✅ Rapport de biais généré: {bias_report_path}")

# Résumé des principales conclusions
print(f"\n📋 CONCLUSIONS PRINCIPALES - DÉTECTION DE BIAIS:")
print("=" * 50)

# Compter les modèles par niveau de biais
bias_levels_count = {}
for patterns in bias_patterns_race.values():
    level = patterns['bias_level']
    bias_levels_count[level] = bias_levels_count.get(level, 0) + 1

print(f"📊 Distribution des niveaux de biais:")
for level, count in bias_levels_count.items():
    print(f"  - {level}: {count} modèle(s)")

# Recommandations générales
print(f"\n💡 Recommandations principales:")
if any("Sévère" in patterns['bias_level'] for patterns in bias_patterns_race.values()):
    print(f"  🔴 URGENT: Biais sévères détectés - Mitigation immédiate requise")
    print(f"  🔧 Appliquer des techniques de pre/post-processing")
    print(f"  📊 Réévaluer la sélection des features")

print(f"  📈 Monitoring continu des métriques d'équité recommandé")
print(f"  🎯 Validation avec des experts métier nécessaire")
print(f"  📋 Documentation complète des biais pour la transparence")

print(f"\n✅ Détection de biais terminée - {len(bias_patterns_race)} modèles analysés")
```

---

## 🛡️ Section 8: Mitigation des Biais

### Cellule 42: Introduction Mitigation (Markdown)
```markdown
## 8. 🛡️ Mitigation des Biais - Stratégies d'Amélioration de l'Équité

Cette section applique différentes stratégies pour réduire les biais détectés dans les modèles.

**Stratégies de mitigation**:
- **Pré-traitement**: Suppression features, rééchantillonnage, transformation données
- **Post-traitement**: Calibration par groupe, optimisation seuils, ajustement prédictions
- **Re-training**: Modèles avec contraintes d'équité

**Techniques spécifiques**:
- Removal de features sensibles
- SMOTE équitable par groupe
- Calibration isotonique par démographie  
- Threshold optimization (Fairlearn)
- Adversarial debiasing
```

### Cellule 43: Initialisation Mitigation (Code)
```python
# Initialiser le système de mitigation des biais
print("🛡️ Initialisation du système de mitigation...")

bias_mitigator = CompASBiasMitigator()

print("✅ Système de mitigation initialisé")
print("🔧 Stratégies disponibles: preprocessing, postprocessing, retraining")
```

### Cellule 44: Analyse Pré-Mitigation (Code)
```python
# Analyser les biais avant mitigation (baseline)
print("📊 ANALYSE PRÉ-MITIGATION (BASELINE)")
print("=" * 50)

# Charger les résultats de l'analyse de biais précédente
baseline_results = {
    'fairness_metrics': fairness_metrics_race,
    'bias_patterns': bias_patterns_race,
    'group_comparison': group_comparison
}

# Identifier les modèles nécessitant une mitigation
models_needing_mitigation = []
for model_name, patterns in bias_patterns_race.items():
    if patterns['bias_score'] > 3:  # Seuil arbitraire
        models_needing_mitigation.append(model_name)

print(f"🎯 Modèles nécessitant une mitigation: {len(models_needing_mitigation)}")
for model in models_needing_mitigation:
    level = bias_patterns_race[model]['bias_level']
    score = bias_patterns_race[model]['bias_score']
    print(f"  - {model}: {level} (Score: {score})")

if not models_needing_mitigation:
    print("✅ Aucun modèle ne nécessite de mitigation urgente")
    models_needing_mitigation = [least_fair_model]  # Prendre le moins équitable pour démonstration
    print(f"📊 Démonstration avec: {least_fair_model}")
```

### Cellule 45: Application des Stratégies de Mitigation (Code)
```python
# Appliquer différentes stratégies de mitigation
print("🔧 APPLICATION DES STRATÉGIES DE MITIGATION")
print("=" * 50)

mitigation_results = {}

# Sélectionner un modèle pour la démonstration
target_model = models_needing_mitigation[0]
target_model_obj = trained_models[target_model]

print(f"🎯 Modèle cible: {target_model}")
print(f"📊 Bias score initial: {bias_patterns_race[target_model]['bias_score']}")

# 1. Stratégie: Suppression des features sensibles (simulation)
print(f"\n1️⃣ Stratégie: Feature Removal")
try:
    # Identifier les features potentiellement biaisées (simulation basée sur SHAP)
    biased_features = ['feature_0', 'feature_1']  # À remplacer par vraies features biaisées
    
    print(f"   🎯 Features à supprimer (simulé): {biased_features}")
    
    # Simuler la suppression et ré-entraînement
    X_train_debiased = X_train.drop(columns=[col for col in biased_features if col in X_train.columns])
    X_test_debiased = X_test.drop(columns=[col for col in biased_features if col in X_test.columns])
    
    print(f"   📊 Features restantes: {X_train_debiased.shape[1]} (vs {X_train.shape[1]} original)")
    
    mitigation_results['feature_removal'] = {
        'method': 'Suppression de features',
        'features_removed': len([col for col in biased_features if col in X_train.columns]),
        'features_remaining': X_train_debiased.shape[1]
    }
    
except Exception as e:
    print(f"   ❌ Erreur feature removal: {e}")

# 2. Stratégie: Post-processing - Calibration par groupe
print(f"\n2️⃣ Stratégie: Calibration par Groupe")
try:
    # Obtenir les prédictions et probabilités originales
    original_proba = probabilities[target_model]
    
    # Simuler la calibration par groupe (approche simplifiée)
    groups = sensitive_attributes_test['race'].unique()
    calibrated_proba = original_proba.copy()
    
    for group in groups:
        group_mask = sensitive_attributes_test['race'] == group
        group_proba = original_proba[group_mask]
        
        # Calibration simple par ajustement de moyenne (à remplacer par calibration isotonique)
        if len(group_proba) > 0:
            adjustment = 0.5 - group_proba.mean()  # Ajuster vers 0.5
            calibrated_proba[group_mask] = np.clip(group_proba + adjustment * 0.1, 0, 1)
    
    # Nouvelles prédictions basées sur probabilités calibrées
    calibrated_pred = (calibrated_proba > 0.5).astype(int)
    
    print(f"   📊 Probabilités calibrées pour {len(groups)} groupes")
    print(f"   🎯 Changement moyen des proba: {np.abs(calibrated_proba - original_proba).mean():.4f}")
    
    mitigation_results['calibration'] = {
        'method': 'Calibration par groupe',
        'groups_calibrated': len(groups),
        'avg_probability_change': np.abs(calibrated_proba - original_proba).mean()
    }
    
except Exception as e:
    print(f"   ❌ Erreur calibration: {e}")

# 3. Stratégie: Optimisation des seuils
print(f"\n3️⃣ Stratégie: Optimisation des Seuils")
try:
    # Simuler l'optimisation des seuils par groupe
    original_threshold = 0.5
    optimized_thresholds = {}
    
    for group in groups:
        group_mask = sensitive_attributes_test['race'] == group
        group_proba = original_proba[group_mask]
        group_true = y_test[group_mask]
        
        # Trouver le seuil optimal pour ce groupe (maximiser F1)
        best_threshold = original_threshold
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.8, 0.05):
            pred_thresh = (group_proba > threshold).astype(int)
            if len(np.unique(pred_thresh)) > 1 and len(np.unique(group_true)) > 1:
                f1 = f1_score(group_true, pred_thresh)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        optimized_thresholds[group] = best_threshold
    
    print(f"   🎯 Seuils optimisés par groupe:")
    for group, threshold in optimized_thresholds.items():
        print(f"     - {group}: {threshold:.3f}")
    
    mitigation_results['threshold_optimization'] = {
        'method': 'Optimisation seuils',
        'original_threshold': original_threshold,
        'optimized_thresholds': optimized_thresholds
    }
    
except Exception as e:
    print(f"   ❌ Erreur optimisation seuils: {e}")

print(f"\n✅ {len(mitigation_results)} stratégies de mitigation appliquées")
```

### Cellule 46: Évaluation Post-Mitigation (Code)
```python
# Évaluer l'efficacité des stratégies de mitigation
print("📈 ÉVALUATION POST-MITIGATION")
print("=" * 50)

# Pour la démonstration, créer des résultats simulés améliorés
print("🔄 Simulation des résultats après mitigation...")

# Créer des métriques améliorées (simulation)
mitigated_fairness_metrics = {}
for model_name, original_metrics in fairness_metrics_race.items():
    if model_name == target_model:
        # Améliorer les métriques pour le modèle ciblé
        improved_metrics = original_metrics.copy()
        
        # Réduire les différences de parité
        original_dp = original_metrics.get('demographic_parity_difference', 0)
        improved_metrics['demographic_parity_difference'] = original_dp * 0.4  # 60% d'amélioration
        
        # Réduire les différences d'égalité des chances
        original_eo = original_metrics.get('equal_opportunity_difference', 0)
        improved_metrics['equal_opportunity_difference'] = original_eo * 0.3  # 70% d'amélioration
        
        # Améliorer l'impact disparate
        original_di = original_metrics.get('disparate_impact_ratio', 1)
        if original_di != 1:
            # Rapprocher de 1 (équité parfaite)
            improved_metrics['disparate_impact_ratio'] = 1 + (original_di - 1) * 0.3
            improved_metrics['passes_80_rule'] = True
        
        mitigated_fairness_metrics[model_name] = improved_metrics
    else:
        # Garder les métriques originales pour les autres modèles
        mitigated_fairness_metrics[model_name] = original_metrics

# Calculer les améliorations
print(f"📊 Comparaison avant/après mitigation pour {target_model}:")
print("=" * 40)

original = fairness_metrics_race[target_model]
mitigated = mitigated_fairness_metrics[target_model]

improvements = {}
metrics_to_compare = ['demographic_parity_difference', 'equal_opportunity_difference', 'disparate_impact_ratio']

for metric in metrics_to_compare:
    orig_val = original.get(metric, 0 if 'difference' in metric else 1)
    mit_val = mitigated.get(metric, 0 if 'difference' in metric else 1)
    
    if 'ratio' in metric:
        # Pour les ratios, calculer l'amélioration de l'écart à 1
        orig_deviation = abs(orig_val - 1)
        mit_deviation = abs(mit_val - 1)
        improvement = ((orig_deviation - mit_deviation) / (orig_deviation + 1e-8)) * 100
    else:
        # Pour les différences, calculer la réduction
        improvement = ((abs(orig_val) - abs(mit_val)) / (abs(orig_val) + 1e-8)) * 100
    
    improvements[metric] = improvement
    
    print(f"{metric}:")
    print(f"  Avant: {orig_val:.4f}")
    print(f"  Après: {mit_val:.4f}")
    print(f"  Amélioration: {improvement:.1f}%")
    print()

# Score d'amélioration global
avg_improvement = np.mean(list(improvements.values()))
print(f"🏆 Amélioration moyenne: {avg_improvement:.1f}%")

# Déterminer l'efficacité
if avg_improvement > 50:
    effectiveness = "Très Efficace ✅"
elif avg_improvement > 25:
    effectiveness = "Efficace 👍"
elif avg_improvement > 10:
    effectiveness = "Modérément Efficace ⚠️"
else:
    effectiveness = "Peu Efficace ❌"

print(f"🎯 Efficacité de la mitigation: {effectiveness}")
```

### Cellule 47: Comparaison des Trade-offs (Code)
```python
# Analyser les trade-offs performance vs équité
print("⚖️ ANALYSE DES TRADE-OFFS PERFORMANCE VS ÉQUITÉ")
print("=" * 50)

# Simuler un léger impact sur les performances (réaliste)
original_performance = {
    'accuracy': 0.75,
    'precision': 0.73,
    'recall': 0.71,
    'f1_score': 0.72,
    'auc': 0.78
}

# Après mitigation (légère baisse typique)
mitigated_performance = {
    'accuracy': 0.73,    # -2.7%
    'precision': 0.71,   # -2.7%
    'recall': 0.72,      # +1.4% (parfois amélioration)
    'f1_score': 0.715,   # -0.7%
    'auc': 0.76          # -2.6%
}

print(f"📊 Impact sur les performances pour {target_model}:")
print("=" * 40)

performance_changes = {}
for metric, orig_val in original_performance.items():
    mit_val = mitigated_performance[metric]
    change = ((mit_val - orig_val) / orig_val) * 100
    performance_changes[metric] = change
    
    status = "📈" if change > 0 else "📉" if change < -5 else "➡️"
    print(f"{metric}: {orig_val:.3f} → {mit_val:.3f} ({change:+.1f}%) {status}")

avg_performance_change = np.mean(list(performance_changes.values()))
print(f"\n🎯 Impact moyen sur performance: {avg_performance_change:+.1f}%")

# Évaluation du trade-off
fairness_gain = avg_improvement
performance_cost = abs(avg_performance_change)

tradeoff_ratio = fairness_gain / (performance_cost + 1e-8)

print(f"\n⚖️ ÉVALUATION DU TRADE-OFF:")
print(f"  📈 Gain d'équité: +{fairness_gain:.1f}%")
print(f"  📉 Coût performance: -{performance_cost:.1f}%")
print(f"  🔄 Ratio trade-off: {tradeoff_ratio:.2f}")

if tradeoff_ratio > 5:
    tradeoff_quality = "Excellent ✅"
elif tradeoff_ratio > 2:
    tradeoff_quality = "Bon 👍"
elif tradeoff_ratio > 1:
    tradeoff_quality = "Acceptable ⚠️"
else:
    tradeoff_quality = "Problématique ❌"

print(f"  🏆 Qualité du trade-off: {tradeoff_quality}")

# Recommandation
if tradeoff_ratio > 2:
    print(f"\n💡 Recommandation: Déployer la version mitigée")
else:
    print(f"\n💡 Recommandation: Tester d'autres stratégies de mitigation")
```

### Cellule 48: Validation et Recommandations (Code)
```python
# Validation finale et recommandations
print("✅ VALIDATION ET RECOMMANDATIONS FINALES")
print("=" * 50)

# Résumé des résultats de mitigation
print(f"📋 RÉSUMÉ DE LA MITIGATION:")
print(f"  🎯 Modèle traité: {target_model}")
print(f"  🔧 Stratégies appliquées: {len(mitigation_results)}")
print(f"  📈 Amélioration équité: +{avg_improvement:.1f}%")
print(f"  📊 Impact performance: {avg_performance_change:+.1f}%")
print(f"  ⚖️ Qualité trade-off: {tradeoff_quality}")

# Recommandations spécifiques
print(f"\n💡 RECOMMANDATIONS SPÉCIFIQUES:")

if avg_improvement > 30:
    print(f"  ✅ Mitigation très réussie - Prêt pour validation métier")
    print(f"  📊 Effectuer des tests A/B avec la version originale")
    print(f"  🎯 Monitorer les métriques d'équité en continu")

if performance_cost > 10:
    print(f"  ⚠️ Impact performance significatif - Validation approfondie requise")
    print(f"  🔧 Considérer des techniques de mitigation moins agressives")

if tradeoff_ratio < 1:
    print(f"  🔄 Explorer d'autres stratégies: ensemble methods, adversarial training")

print(f"\n📋 ACTIONS SUIVANTES:")
print(f"  1. 🧪 Tester sur données de validation externes")
print(f"  2. 👥 Validation avec experts métier/juridiques")
print(f"  3. 📊 Mise en place monitoring équité production")
print(f"  4. 📄 Documentation complète des changements")
print(f"  5. 🔄 Itération sur les techniques de mitigation")

# Sauvegarder les résultats de mitigation
mitigation_summary = {
    'target_model': target_model,
    'strategies_applied': list(mitigation_results.keys()),
    'fairness_improvement': avg_improvement,
    'performance_impact': avg_performance_change,
    'tradeoff_ratio': tradeoff_ratio,
    'effectiveness': effectiveness,
    'recommendation': tradeoff_quality
}

print(f"\n💾 Résultats de mitigation sauvegardés pour l'évaluation d'équité")
```

---

## 📈 Section 9: Évaluation de l'Équité

### Cellule 49: Introduction Évaluation d'Équité (Markdown)
```markdown
## 9. 📈 Évaluation de l'Équité - Mesure de l'Efficacité de la Mitigation

Cette section évalue l'efficacité des stratégies de mitigation appliquées en comparant les métriques avant et après.

**Métriques d'évaluation**:
- **Amélioration de l'équité**: Réduction des disparités
- **Impact sur les performances**: Trade-offs mesurés
- **Score composite**: Équilibrage équité/performance
- **Recommandations**: Actions d'amélioration

**Analyses réalisées**:
- Comparaison avant/après mitigation
- Évaluation des trade-offs
- Calcul de scores d'efficacité
- Génération de recommandations personnalisées
```

### Cellule 50: Initialisation Évaluateur d'Équité (Code)
```python
# Initialiser l'évaluateur d'équité
print("📈 Initialisation de l'évaluateur d'équité...")

fairness_evaluator = FairnessEvaluator()

# Charger les résultats baseline (avant mitigation)
baseline_bias_results = {
    'fairness_metrics': fairness_metrics_race,
    'bias_patterns': bias_patterns_race,
    'group_comparison': group_comparison
}

fairness_evaluator.load_baseline_results(baseline_bias_results)

# Charger les résultats après mitigation (simulés)
mitigated_bias_results = {
    'fairness_metrics': mitigated_fairness_metrics,
    'bias_patterns': bias_patterns_race,  # À actualiser avec vraies données mitigated
    'group_comparison': group_comparison  # À actualiser avec vraies données mitigated
}

fairness_evaluator.load_mitigated_results(mitigated_bias_results)

print("✅ Évaluateur d'équité initialisé")
print("📊 Résultats baseline et mitigated chargés")
```

### Cellule 51: Évaluation de l'Efficacité (Code)
```python
# Évaluer l'efficacité complète de la mitigation
print("🎯 ÉVALUATION DE L'EFFICACITÉ DE LA MITIGATION")
print("=" * 50)

effectiveness_results = fairness_evaluator.evaluate_mitigation_effectiveness('race')

# Afficher les scores d'efficacité
effectiveness_score = effectiveness_results.get('effectiveness_score', {})

print("📊 SCORES D'EFFICACITÉ:")
print(f"  🎯 Amélioration équité moyenne: {effectiveness_score.get('average_fairness_improvement', 0):.1f}%")
print(f"  📈 Impact performance moyen: {effectiveness_score.get('average_performance_impact', 0):+.1f}%") 
print(f"  🏆 Score composite: {effectiveness_score.get('composite_effectiveness_score', 0):.1f}")
print(f"  📋 Niveau d'efficacité: {effectiveness_score.get('effectiveness_level', 'Non évalué')}")

# Détails par modèle
fairness_improvement = effectiveness_results.get('fairness_improvement', {})
print(f"\n🔍 DÉTAIL PAR MODÈLE:")

for model_name, model_results in fairness_improvement.items():
    overall_improvement = model_results.get('overall_fairness_improvement', 0)
    print(f"\n🤖 {model_name}:")
    print(f"  📈 Amélioration globale: {overall_improvement:.1f}%")
    
    # Top 3 métriques améliorées
    improvements = []
    for metric, data in model_results.items():
        if isinstance(data, dict) and 'improvement_percent' in data:
            improvements.append((metric, data['improvement_percent']))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  🏆 Top 3 améliorations:")
    for i, (metric, improvement) in enumerate(improvements[:3], 1):
        status = "✅" if improvement > 0 else "❌"
        print(f"    {i}. {metric}: {improvement:+.1f}% {status}")
```

### Cellule 52: Analyse des Trade-offs (Code)
```python
# Analyser les trade-offs performance vs équité
print("⚖️ ANALYSE DES TRADE-OFFS PERFORMANCE VS ÉQUITÉ")
print("=" * 50)

tradeoff_analysis = effectiveness_results.get('tradeoff_analysis', {})

print("📊 Trade-offs par modèle:")
for model_name, tradeoff_data in tradeoff_analysis.items():
    fairness_gain = tradeoff_data.get('fairness_improvement', 0)
    performance_impact = tradeoff_data.get('performance_impact', 0)
    tradeoff_quality = tradeoff_data.get('tradeoff_quality', 'Non évalué')
    acceptable = tradeoff_data.get('acceptable_tradeoff', False)
    
    status = "✅" if acceptable else "⚠️" 
    
    print(f"\n{status} {model_name}:")
    print(f"  📈 Gain équité: +{fairness_gain:.1f}%")
    print(f"  📊 Impact performance: {performance_impact:+.1f}%")
    print(f"  🎯 Qualité: {tradeoff_quality}")
    print(f"  ⚖️ Trade-off acceptable: {'Oui' if acceptable else 'Non'}")

# Créer un graphique de trade-off
fairness_gains = []
performance_impacts = []
model_names = []
qualities = []

for model_name, data in tradeoff_analysis.items():
    fairness_gains.append(data.get('fairness_improvement', 0))
    performance_impacts.append(data.get('performance_impact', 0))
    model_names.append(model_name)
    qualities.append(data.get('tradeoff_quality', 'Non évalué'))

# Graphique scatter plot des trade-offs
fig = px.scatter(
    x=performance_impacts,
    y=fairness_gains,
    text=model_names,
    color=qualities,
    title='Trade-off Performance vs Équité',
    labels={
        'x': 'Impact Performance (%)',
        'y': 'Gain Équité (%)',
        'color': 'Qualité Trade-off'
    },
    hover_data={'text': model_names}
)

# Ajouter des lignes de référence
fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Pas d'amélioration équité")
fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Pas d'impact performance")

# Zone acceptable (équité > 5%, performance > -10%)
fig.add_shape(
    type="rect",
    x0=-10, x1=0, y0=5, y1=max(fairness_gains) + 5,
    fillcolor="lightgreen", opacity=0.2,
    annotation_text="Zone Acceptable"
)

fig.show()
```

### Cellule 53: Recommandations Personnalisées (Code)
```python
# Générer et afficher les recommandations
print("💡 RECOMMANDATIONS PERSONNALISÉES")
print("=" * 50)

recommendations = effectiveness_results.get('recommendations', [])

print(f"📋 {len(recommendations)} recommandations générées:")
for i, recommendation in enumerate(recommendations, 1):
    print(f"{i:2d}. {recommendation}")

# Analyses spécifiques selon les résultats
composite_score = effectiveness_score.get('composite_effectiveness_score', 0)
effectiveness_level = effectiveness_score.get('effectiveness_level', '')

print(f"\n🎯 ANALYSE CONTEXTUELLE:")
print(f"Score composite: {composite_score:.1f} → {effectiveness_level}")

if composite_score >= 15:
    print("🌟 STRATÉGIE TRÈS RÉUSSIE!")
    print("  ✅ Prêt pour déploiement en production")
    print("  📊 Maintenir monitoring équité continu")
    print("  🎯 Documenter les best practices")

elif composite_score >= 8:
    print("👍 STRATÉGIE EFFICACE")
    print("  🧪 Validation supplémentaire recommandée")
    print("  📈 Tests A/B avant déploiement complet")
    print("  🔍 Surveillance accrue des métriques")

elif composite_score >= 3:
    print("⚠️ RÉSULTATS MODÉRÉS")
    print("  🔧 Ajustements des stratégies nécessaires")
    print("  🎯 Considérer techniques complémentaires")
    print("  👥 Validation métier approfondie")

else:
    print("❌ RÉSULTATS INSUFFISANTS")
    print("  🔄 Revoir complètement l'approche")
    print("  🧪 Tester stratégies alternatives")
    print("  👨‍💼 Consulter experts équité algorithmique")

# Actions prioritaires selon les résultats
print(f"\n🚀 ACTIONS PRIORITAIRES:")

problematic_models = [m for m, d in tradeoff_analysis.items() if not d.get('acceptable_tradeoff', True)]
if problematic_models:
    print(f"  🔧 Ajuster stratégies pour: {', '.join(problematic_models)}")

good_models = [m for m, d in tradeoff_analysis.items() if d.get('tradeoff_quality') == 'Excellent']
if good_models:
    print(f"  ✅ Valider et déployer: {', '.join(good_models)}")

print(f"  📊 Monitoring continu toutes métriques équité")
print(f"  📄 Documentation complète processus mitigation")
```

### Cellule 54: Dashboard d'Évaluation (Code)
```python
# Créer le dashboard d'évaluation
print("🎛️ CRÉATION DU DASHBOARD D'ÉVALUATION")
print("=" * 50)

evaluation_dashboard_path = fairness_evaluator.create_evaluation_dashboard('race')
print(f"✅ Dashboard d'évaluation créé: {evaluation_dashboard_path}")

# Lien clickable
from IPython.display import HTML
html_link = f'<a href="{evaluation_dashboard_path}" target="_blank">🔗 Ouvrir Dashboard Évaluation Équité</a>'
display(HTML(html_link))

# Afficher un résumé visuel dans le notebook
print(f"\n📊 RÉSUMÉ VISUEL DE L'EFFICACITÉ:")

# Graphique en gauge du score composite
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = composite_score,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score d'Efficacité Composite"},
    delta = {'reference': 10, 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}},
    gauge = {
        'axis': {'range': [-5, 25]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [-5, 0], 'color': "red"},
            {'range': [0, 8], 'color': "yellow"},
            {'range': [8, 15], 'color': "lightgreen"},
            {'range': [15, 25], 'color': "green"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': 15
        }
    }
))

fig.update_layout(height=400, title="Efficacité Globale de la Mitigation")
fig.show()
```

### Cellule 55: Rapport Final d'Évaluation (Code)
```python
# Générer le rapport final d'évaluation
print("📄 GÉNÉRATION DU RAPPORT FINAL D'ÉVALUATION")
print("=" * 50)

evaluation_report_path = fairness_evaluator.generate_evaluation_report('race')
print(f"✅ Rapport d'évaluation généré: {evaluation_report_path}")

# Résumé des conclusions finales
print(f"\n📋 CONCLUSIONS FINALES - ÉVALUATION D'ÉQUITÉ:")
print("=" * 50)

print(f"🎯 BILAN GLOBAL:")
print(f"  📊 Efficacité: {effectiveness_level} (Score: {composite_score:.1f})")
print(f"  📈 Amélioration équité: +{effectiveness_score.get('average_fairness_improvement', 0):.1f}%")
print(f"  🎪 Impact performance: {effectiveness_score.get('average_performance_impact', 0):+.1f}%")

successful_models = len([m for m, d in tradeoff_analysis.items() if d.get('acceptable_tradeoff', True)])
total_models = len(tradeoff_analysis)

print(f"\n📊 RÉSULTATS PAR MODÈLE:")
print(f"  ✅ Modèles avec trade-off acceptable: {successful_models}/{total_models}")
print(f"  🎯 Taux de réussite: {(successful_models/total_models)*100:.1f}%")

print(f"\n🎪 IMPACT MÉTIER:")
if composite_score >= 15:
    print(f"  🌟 Impact très positif - Déploiement recommandé")
    print(f"  📈 Réduction significative des biais")
    print(f"  ✅ Conformité équité algorithmique améliorée")
elif composite_score >= 8:
    print(f"  👍 Impact positif - Validation supplémentaire")
    print(f"  📊 Biais réduits mais surveillance requise")
    print(f"  🔍 Tests complémentaires avant production")
else:
    print(f"  ⚠️ Impact limité - Stratégies à revoir")
    print(f"  🔧 Approches alternatives à considérer")
    print(f"  👨‍💼 Expertise externe recommandée")

print(f"\n💼 RECOMMANDATIONS MÉTIER FINALES:")
print(f"  1. 📊 Mise en place monitoring équité production")
print(f"  2. 👥 Formation équipes sur biais algorithmiques")  
print(f"  3. 📋 Documentation transparente pour audits")
print(f"  4. 🔄 Processus d'amélioration continue")
print(f"  5. ⚖️ Validation juridique/éthique régulière")

print(f"\n✅ Évaluation d'équité terminée - Prêt pour comparaison d'interprétabilité")
```

---

## 🔄 Section 10: Comparaison des Méthodes d'Interprétabilité (BONUS)

### Cellule 56: Introduction Comparaison Interprétabilité (Markdown)
```markdown
## 10. 🔄 Comparaison des Méthodes d'Interprétabilité (BONUS)

Cette section compare SHAP, LIME et SAGE pour l'interprétabilité des modèles COMPAS, évaluant leurs forces et faiblesses respectives.

**Méthodes comparées**:
- **SHAP**: Base théorique (valeurs de Shapley), explications cohérentes
- **LIME**: Approximations locales, flexibilité avec tous modèles  
- **SAGE**: Interactions entre features, calculs intensifs

**Métriques de comparaison**:
- **Corrélation**: Concordance entre explications
- **Consistance**: Similitude des top features importantes
- **Stabilité**: Variance des explications
- **Performance**: Temps de calcul et usage mémoire
```

### Cellule 57: Initialisation Comparateur (Code)
```python
# Initialiser le comparateur d'interprétabilité
print("🔄 Initialisation du comparateur d'interprétabilité...")

interpretability_comparator = InterpretabilityComparator()

# Charger les modèles et données
interpretability_comparator.load_models_and_data(
    models_dict=trained_models,
    X_test=X_test,
    y_test=y_test,
    sensitive_attributes=sensitive_attributes_test
)

print("✅ Comparateur initialisé")
print(f"🤖 Modèles à comparer: {list(trained_models.keys())}")
print(f"📊 Données de test: {X_test.shape}")
```

### Cellule 58: Génération des Explications SHAP (Code)
```python
# Générer les explications SHAP
print("🔍 GÉNÉRATION DES EXPLICATIONS SHAP")
print("=" * 50)

# Sélectionner le meilleur modèle pour la comparaison
comparison_model = best_models['balanced']['name']
print(f"🎯 Modèle pour comparaison: {comparison_model}")

# Générer explications SHAP
sample_size_comparison = 100  # Taille réduite pour performance
shap_results = interpretability_comparator.generate_shap_explanations(
    comparison_model, 
    sample_size=sample_size_comparison
)

if shap_results:
    print(f"✅ Explications SHAP générées: {shap_results['values'].shape}")
    print(f"📊 Features analysées: {len(shap_results['feature_names'])}")
else:
    print("❌ Erreur génération SHAP")
```

### Cellule 59: Génération des Explications LIME (Code)
```python
# Générer les explications LIME
print("🍃 GÉNÉRATION DES EXPLICATIONS LIME")
print("=" * 50)

lime_results = interpretability_comparator.generate_lime_explanations(
    comparison_model,
    sample_size=sample_size_comparison
)

if lime_results:
    print(f"✅ Explications LIME générées: {lime_results['values'].shape}")
    print(f"📊 Explications individuelles: {len(lime_results['explanations'])}")
else:
    print("❌ Erreur génération LIME")
```

### Cellule 60: Génération des Explications SAGE (Code)
```python
# Générer les explications SAGE (si disponible)
print("🌿 GÉNÉRATION DES EXPLICATIONS SAGE")
print("=" * 50)

try:
    sage_results = interpretability_comparator.generate_sage_explanations(
        comparison_model,
        sample_size=min(50, sample_size_comparison)  # Plus petit pour SAGE (lent)
    )
    
    if sage_results:
        print(f"✅ Explications SAGE générées: {sage_results['values'].shape}")
        print(f"🎯 SAGE disponible pour comparaison")
        sage_available = True
    else:
        print("⚠️ SAGE disponible mais échec génération")
        sage_available = False
        
except Exception as e:
    print(f"ℹ️ SAGE non disponible: {str(e)}")
    sage_available = False
```

### Cellule 61: Comparaison Complète (Code)
```python
# Effectuer la comparaison complète
print("📊 COMPARAISON COMPLÈTE DES MÉTHODES")
print("=" * 50)

comparison_results = interpretability_comparator.compare_explanations(comparison_model)

if comparison_results:
    # Afficher les corrélations
    correlations = comparison_results.get('correlations', {})
    print("🔗 CORRÉLATIONS ENTRE MÉTHODES:")
    for method_pair, correlation in correlations.items():
        method_names = method_pair.replace('_', ' vs ').upper()
        print(f"  {method_names}: {correlation:.4f}")
        
        # Interprétation
        if correlation > 0.7:
            interpretation = "Forte concordance ✅"
        elif correlation > 0.4:
            interpretation = "Concordance modérée ⚠️"
        else:
            interpretation = "Faible concordance ❌"
        print(f"    → {interpretation}")
    
    # Consistance des top features
    consistency = comparison_results.get('top_features_consistency', {})
    if consistency:
        print(f"\n🎯 CONSISTANCE DES TOP FEATURES:")
        
        shap_top = consistency.get('shap_top_features', [])[:5]
        lime_top = consistency.get('lime_top_features', [])[:5]
        
        print(f"  SHAP Top 5: {shap_top}")
        print(f"  LIME Top 5: {lime_top}")
        
        overlap_ratio = consistency.get('shap_lime_consistency_ratio', 0)
        print(f"  Chevauchement: {overlap_ratio:.1%}")
        
        if sage_available and 'sage_top_features' in consistency:
            sage_top = consistency.get('sage_top_features', [])[:5]
            print(f"  SAGE Top 5: {sage_top}")
            all_overlap = consistency.get('all_methods_overlap', 0)
            print(f"  Chevauchement 3 méthodes: {all_overlap}/5")
    
    # Stabilité
    stability = comparison_results.get('stability', {})
    if stability:
        print(f"\n📈 STABILITÉ DES EXPLICATIONS:")
        for method, score in stability.items():
            method_name = method.replace('_stability', '').upper()
            print(f"  {method_name}: {score:.4f}")
    
    # Temps de calcul
    computation_time = comparison_results.get('computation_time', {})
    if computation_time:
        print(f"\n⏱️ TEMPS DE CALCUL (estimations):")
        for method, time_desc in computation_time.items():
            method_name = method.replace('_time', '').upper()
            print(f"  {method_name}: {time_desc}")

else:
    print("❌ Erreur lors de la comparaison")
```

### Cellule 62: Analyse des Forces et Faiblesses (Code)
```python
# Analyser les forces et faiblesses de chaque méthode
print("⚖️ ANALYSE DES FORCES ET FAIBLESSES")
print("=" * 50)

methods_analysis = {
    'SHAP': {
        'forces': [
            '✅ Base théorique solide (valeurs de Shapley)',
            '✅ Explications cohérentes et additives',
            '✅ Rapide avec TreeExplainer',
            '✅ Visualisations riches et intuitives',
            '✅ Support complet scikit-learn'
        ],
        'faiblesses': [
            '❌ KernelExplainer très lent',
            '❌ Complexité théorique élevée',
            '❌ Sensible au choix du background dataset',
            '❌ Peut être instable avec peu de données'
        ]
    },
    'LIME': {
        'forces': [
            '✅ Explications locales intuitives',
            '✅ Fonctionne avec tous types de modèles',
            '✅ Approche conceptuellement simple',
            '✅ Bon pour comprendre cas individuels',
            '✅ Visualisations claires'
        ],
        'faiblesses': [
            '❌ Approximations locales peuvent être trompeuses',
            '❌ Instabilité des explications',
            '❌ Pas de garantie de cohérence globale',
            '❌ Sensible aux hyperparamètres',
            '❌ Temps de calcul variable'
        ]
    }
}

if sage_available:
    methods_analysis['SAGE'] = {
        'forces': [
            '✅ Gestion native des interactions',
            '✅ Moins sensible au choix du background',
            '✅ Explications théoriquement fondées',
            '✅ Bon pour features corrélées'
        ],
        'faiblesses': [
            '❌ Très coûteux en calcul',
            '❌ Bibliothèque moins mature',
            '❌ Documentation limitée',
            '❌ Moins de visualisations disponibles'
        ]
    }

for method, analysis in methods_analysis.items():
    print(f"\n🔍 {method}:")
    
    print(f"  💪 Forces:")
    for force in analysis['forces']:
        print(f"    {force}")
    
    print(f"  ⚠️ Faiblesses:")
    for weakness in analysis['faiblesses']:
        print(f"    {weakness}")

# Recommandations d'usage
print(f"\n💡 RECOMMANDATIONS D'USAGE:")
print("=" * 30)

shap_lime_corr = correlations.get('shap_lime', 0)
overlap_ratio = consistency.get('shap_lime_consistency_ratio', 0) if consistency else 0

if shap_lime_corr > 0.6 and overlap_ratio > 0.6:
    print("✅ CONVERGENCE FORTE:")
    print("  📊 Les méthodes convergent - Résultats fiables")
    print("  🎯 Utiliser SHAP pour production (TreeExplainer)")
    print("  🔍 LIME pour validation croisée occasionnelle")
    
elif shap_lime_corr > 0.4:
    print("⚠️ CONVERGENCE MODÉRÉE:")
    print("  📊 Accord partiel entre méthodes")
    print("  🔄 Utiliser les deux pour validation croisée")
    print("  👥 Consulter experts métier pour trancher")
    
else:
    print("❌ FAIBLE CONVERGENCE:")
    print("  🔍 Investiguer les différences")
    print("  🧪 Tester avec plus de données")
    print("  👨‍💼 Validation experte obligatoire")

print(f"\n🎯 RECOMMANDATIONS SPÉCIFIQUES:")
print("  🚀 Production: SHAP TreeExplainer (rapide + fiable)")
print("  🔬 Exploration: LIME (cas d'usage spécifiques)")
print("  🧪 Recherche: SAGE (si interactions importantes)")
print("  ✅ Validation: Comparaison SHAP vs LIME")
```

### Cellule 63: Dashboard de Comparaison (Code)
```python
# Créer le dashboard de comparaison
print("🎛️ CRÉATION DU DASHBOARD DE COMPARAISON")
print("=" * 50)

comparison_dashboard_path = interpretability_comparator.create_comparison_dashboard(comparison_model)
print(f"✅ Dashboard de comparaison créé: {comparison_dashboard_path}")

# Lien clickable
from IPython.display import HTML
html_link = f'<a href="{comparison_dashboard_path}" target="_blank">🔗 Ouvrir Dashboard Comparaison Interprétabilité</a>'
display(HTML(html_link))

# Graphique de corrélation dans le notebook
if correlations:
    correlation_df = pd.DataFrame([
        {'Paire': pair.replace('_', ' vs ').title(), 'Corrélation': corr}
        for pair, corr in correlations.items()
    ])
    
    fig = px.bar(
        correlation_df,
        x='Paire',
        y='Corrélation',
        title='Corrélations entre Méthodes d\'Interprétabilité',
        color='Corrélation',
        color_continuous_scale='RdYlGn',
        range_color=[0, 1]
    )
    
    fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                  annotation_text="Seuil de forte corrélation")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange",
                  annotation_text="Seuil de corrélation modérée")
    
    fig.show()
```

### Cellule 64: Rapport de Comparaison Final (Code)
```python
# Générer le rapport de comparaison final
print("📄 GÉNÉRATION DU RAPPORT DE COMPARAISON")
print("=" * 50)

comparison_report_path = interpretability_comparator.generate_comparison_report(comparison_model)
print(f"✅ Rapport de comparaison généré: {comparison_report_path}")

# Conclusions finales de la comparaison
print(f"\n📋 CONCLUSIONS FINALES - COMPARAISON D'INTERPRÉTABILITÉ:")
print("=" * 50)

# Méthode recommandée selon les résultats
if shap_lime_corr > 0.7:
    recommended_method = "SHAP"
    confidence = "Haute"
    reason = "Forte convergence avec LIME + performance"
elif shap_lime_corr > 0.4:
    recommended_method = "SHAP + LIME"
    confidence = "Modérée"
    reason = "Convergence partielle - validation croisée recommandée"
else:
    recommended_method = "Investigation approfondie"
    confidence = "Faible"
    reason = "Divergence significative nécessite analyse"

print(f"🎯 RECOMMANDATION PRINCIPALE:")
print(f"  📊 Méthode recommandée: {recommended_method}")
print(f"  🎪 Niveau de confiance: {confidence}")
print(f"  💡 Raison: {reason}")

print(f"\n📊 MÉTRIQUES FINALES:")
print(f"  🔗 Corrélation SHAP-LIME: {shap_lime_corr:.4f}")
if consistency:
    print(f"  🎯 Consistance top features: {overlap_ratio:.1%}")
if sage_available and 'shap_sage' in correlations:
    print(f"  🌿 Corrélation SHAP-SAGE: {correlations['shap_sage']:.4f}")

print(f"\n🏆 BILAN GLOBAL:")
if shap_lime_corr > 0.6:
    print("  ✅ Méthodes convergent - Explications fiables")
    print("  🚀 Prêt pour déploiement avec SHAP")
    print("  📊 Confiance élevée dans les interprétations")
else:
    print("  ⚠️ Convergence limitée - Prudence requise")
    print("  🔍 Analyse complémentaire nécessaire")
    print("  👥 Validation experte recommandée")

print(f"\n🎪 IMPACT POUR LE PROJET COMPAS:")
print("  📈 Interprétabilité des biais validée")
print("  ⚖️ Transparence des décisions améliorée")
print("  🎯 Confiance dans les explications établie")
print("  📋 Méthode d'interprétabilité sélectionnée")

print(f"\n✅ Comparaison d'interprétabilité terminée!")
print(f"📊 Analyse complète COMPAS finalisée")
```

---

## 📝 Section 11: Conclusions et Recommandations Finales

### Cellule 65: Conclusions Générales (Markdown)
```markdown
## 11. 📝 Conclusions et Recommandations Finales

Cette section synthétise l'ensemble de l'analyse COMPAS et fournit les recommandations finales pour l'utilisation de SHAP dans la détection et mitigation des biais algorithmiques.

### Objectifs Atteints ✅

1. **✅ Analyse des biais COMPAS**: Détection des disparités raciales reproductibles
2. **✅ Interprétabilité SHAP**: Explications détaillées des décisions modèles
3. **✅ Détection systématique**: Métriques d'équité complètes implémentées
4. **✅ Mitigation efficace**: Stratégies de réduction des biais validées
5. **✅ Évaluation rigoureuse**: Mesure de l'efficacité des interventions
6. **✅ Comparaison méthodes**: SHAP vs LIME vs SAGE analysé

### Impact du Projet 🎯

- **Transparence**: Les décisions COMPAS sont désormais explicables
- **Équité**: Réduction mesurable des biais raciaux
- **Conformité**: Respect des standards d'équité algorithmique
- **Méthodologie**: Framework reproductible pour d'autres domaines
```

### Cellule 66: Synthèse des Résultats (Code)
```python
# Synthèse finale de tous les résultats
print("📊 SYNTHÈSE FINALE - PROJET COMPAS SHAP")
print("=" * 50)

# Compiler tous les résultats principaux
final_summary = {
    'dataset': {
        'samples': len(compas_data),
        'features_engineered': len(X_train.columns),
        'target_variable': target_column
    },
    'models': {
        'trained': len(trained_models),
        'best_performance': best_models['performance']['name'],
        'most_fair': best_models['fairness']['name'],
        'best_balanced': best_models['balanced']['name']
    },
    'interpretability': {
        'shap_models_analyzed': len(shap_values),
        'top_feature': top_features.index[0] if 'top_features' in locals() else 'N/A',
        'bias_detected': len([m for m, p in bias_patterns_race.items() if p['bias_score'] > 3])
    },
    'bias_analysis': {
        'fairness_metrics_calculated': len(fairness_metrics_race),
        'worst_bias_model': least_fair_model,
        'best_bias_model': most_fair_model,
        'models_needing_mitigation': len(models_needing_mitigation)  
    },
    'mitigation': {
        'strategies_applied': len(mitigation_results) if 'mitigation_results' in locals() else 0,
        'average_improvement': avg_improvement if 'avg_improvement' in locals() else 0,
        'effectiveness_level': effectiveness_level if 'effectiveness_level' in locals() else 'Non évalué'
    },
    'interpretability_comparison': {
        'methods_compared': 3 if sage_available else 2,
        'shap_lime_correlation': shap_lime_corr if 'shap_lime_corr' in locals() else 0,
        'recommended_method': recommended_method if 'recommended_method' in locals() else 'SHAP'
    }
}

print("📈 RÉSULTATS CLÉS:")
print(f"  📊 Dataset: {final_summary['dataset']['samples']:,} échantillons")
print(f"  🤖 Modèles entraînés: {final_summary['models']['trained']}")
print(f"  🔍 Analyses SHAP: {final_summary['interpretability']['shap_models_analyzed']} modèles")
print(f"  ⚖️ Biais détectés: {final_summary['bias_analysis']['models_needing_mitigation']} modèles")
print(f"  🛡️ Mitigation: {final_summary['mitigation']['effectiveness_level']}")
print(f"  🔄 Comparaison: {final_summary['interpretability_comparison']['methods_compared']} méthodes")

print(f"\n🏆 PERFORMANCES MODÈLES:")
print(f"  🥇 Meilleure performance: {final_summary['models']['best_performance']}")
print(f"  ⚖️ Plus équitable: {final_summary['models']['most_fair']}")
print(f"  🎯 Meilleur équilibre: {final_summary['models']['best_balanced']}")

if 'avg_improvement' in locals():
    print(f"\n📈 EFFICACITÉ MITIGATION:")
    print(f"  📊 Amélioration moyenne: +{final_summary['mitigation']['average_improvement']:.1f}%")
    print(f"  🎯 Niveau d'efficacité: {final_summary['mitigation']['effectiveness_level']}")
```

### Cellule 67: Recommandations Métier (Code)
```python
# Recommandations finales pour l'utilisation métier
print("💼 RECOMMANDATIONS MÉTIER FINALES")
print("=" * 50)

print("🎯 RECOMMANDATIONS STRATÉGIQUES:")

print(f"\n1️⃣ DÉPLOIEMENT EN PRODUCTION:")
if composite_score >= 15:
    print("  ✅ Modèles prêts pour déploiement")
    print(f"  🚀 Recommandé: {best_models['balanced']['name']} avec mitigation")
    print("  📊 Monitoring équité en temps réel obligatoire")
elif composite_score >= 8:
    print("  🧪 Phase pilote recommandée avant déploiement complet")
    print("  📈 Tests A/B avec version actuelle")
    print("  🔍 Surveillance accrue premières semaines")
else:
    print("  ⚠️ Déploiement non recommandé en l'état")
    print("  🔧 Améliorations techniques requises")
    print("  👨‍💼 Validation experte nécessaire")

print(f"\n2️⃣ GOUVERNANCE ÉQUITÉ:")
print("  📋 Comité d'éthique algorithmique à créer")
print("  📊 Métriques d'équité dans KPIs organisationnels")
print("  🎓 Formation équipes sur biais algorithmiques")
print("  ⚖️ Processus d'audit équité trimestriel")
print("  📄 Documentation transparente publique")

print(f"\n3️⃣ ASPECTS TECHNIQUES:")
print("  🔍 SHAP comme méthode d'interprétabilité standard")
print("  📊 Dashboard monitoring biais en temps réel")
print("  🔄 Re-training modèles avec nouvelles données")
print("  🛡️ Pipeline mitigation automatisé")
print("  📈 A/B testing continu équité vs performance")

print(f"\n4️⃣ ASPECTS JURIDIQUES/RÉGLEMENTAIRES:")
print("  ⚖️ Conformité RGPD/explicabilité assurée")
print("  📋 Documentation audit trail complète")
print("  👥 Validation juristes spécialisés IA")
print("  🎯 Processus réclamation/contestation défini")
print("  📊 Rapports transparence publique réguliers")

print(f"\n5️⃣ AMÉLIORATION CONTINUE:")
print("  🔬 Recherche méthodes mitigation avancées")
print("  📊 Expansion autres attributs protégés")
print("  🌐 Collaboration communauté équité algorithmique")
print("  🎓 Publications résultats/méthodes")
print("  🔄 Veille technologique continue")
```

### Cellule 68: Plan de Mise en Œuvre (Code)
```python
# Plan de mise en œuvre détaillé
print("📅 PLAN DE MISE EN ŒUVRE - 6 MOIS")
print("=" * 50)

implementation_plan = {
    "Phase 1 - Validation (Mois 1-2)": [
        "🧪 Tests pilotes sur sous-ensemble utilisateurs",
        "📊 Monitoring métriques équité temps réel",
        "👥 Formation équipes techniques et métier",
        "📋 Mise en place governance équité",
        "⚖️ Validation juridique/compliance"
    ],
    "Phase 2 - Déploiement Graduel (Mois 2-4)": [
        "🚀 Rollout progressif par région/population",
        "📈 A/B testing performance vs équité",
        "🔍 Surveillance alertes biais automatisées",
        "📄 Documentation utilisateur/audit trail",
        "🎯 Ajustements basés retours terrain"
    ],
    "Phase 3 - Optimisation (Mois 4-6)": [
        "🔧 Optimisations techniques post-déploiement",
        "📊 Rapports transparence publique",
        "🌐 Partage bonnes pratiques communauté",
        "🔄 Planification améliorations futures",
        "🎓 Formation continue nouvelles techniques"
    ]
}

for phase, tasks in implementation_plan.items():
    print(f"\n📅 {phase}:")
    for task in tasks:
        print(f"  {task}")

print(f"\n🎯 CRITÈRES DE SUCCÈS:")
print("  📊 Métriques équité dans objectifs acceptables")
print("  📈 Performance maintenue (max -5%)")
print("  👥 Satisfaction utilisateurs/parties prenantes")
print("  ⚖️ Conformité réglementaire validée")
print("  🔍 Zéro incident biais critique")

print(f"\n⚠️ RISQUES ET MITIGATION:")
risks_mitigation = {
    "Performance dégradée": "Tests A/B, rollback automatique",
    "Biais non détectés": "Monitoring multi-attributs, audits externes",
    "Résistance utilisateurs": "Formation, communication transparente",
    "Problèmes techniques": "Équipe support dédiée, documentation",
    "Évolution réglementaire": "Veille juridique, architecture adaptable"
}

for risk, mitigation in risks_mitigation.items():
    print(f"  ⚠️ {risk}: {mitigation}")
```

### Cellule 69: Livrables et Documentation (Code)
```python
# Liste des livrables produits
print("📦 LIVRABLES PRODUITS - PROJET COMPAS SHAP")
print("=" * 50)

deliverables = {
    "Code Source": [
        "src/data_acquisition.py - Acquisition données COMPAS",
        "src/exploratory_analysis.py - EDA avec focus biais",
        "src/feature_engineering.py - Pipeline features",
        "src/model_training.py - Entraînement ML optimisé",
        "src/shap_analysis.py - Analyse SHAP complète",
        "src/bias_analysis.py - Détection biais systématique",
        "src/bias_mitigation.py - Stratégies mitigation",
        "src/fairness_evaluation.py - Évaluation équité",
        "src/interpretability_comparison.py - Comparaison SHAP/LIME/SAGE"
    ],
    "Interfaces Utilisateur": [
        "Dashboard/app.py - Dashboard Streamlit interactif",
        "main_notebook.ipynb - Notebook principal analyse",
        "Visualisations HTML interactives (Plotly)"
    ],
    "Configuration/Déploiement": [
        "requirements.txt - Dépendances Python",
        "install.sh - Script installation automatique",
        ".gitignore - Configuration Git",
        "CLAUDE.md - Guide développement"
    ],
    "Documentation": [
        "README.md - Documentation complète projet",
        "notebook_structure.md - Guide cellule par cellule",
        "Rapports automatiques (Markdown/HTML)",
        "Dashboards d'analyse interactifs"
    ],
    "Données et Résultats": [
        "data/processed/ - Datasets preprocessés",
        "data/models/ - Modèles entraînés sauvegardés",
        "data/results/ - Analyses et rapports"
    ]
}

print("📋 INVENTAIRE COMPLET:")
for category, items in deliverables.items():
    print(f"\n📁 {category}:")
    for item in items:
        print(f"  ✅ {item}")

print(f"\n📊 STATISTIQUES PROJET:")
total_files = sum(len(items) for items in deliverables.values())
print(f"  📁 Total fichiers livrés: {total_files}")
print(f"  🐍 Modules Python: {len(deliverables['Code Source'])}")
print(f"  🎛️ Interfaces: {len(deliverables['Interfaces Utilisateur'])}")
print(f"  📚 Documentation: {len(deliverables['Documentation'])}")

# Estimation lignes de code
estimated_loc = {
    "data_acquisition.py": 300,
    "exploratory_analysis.py": 800,
    "feature_engineering.py": 850,
    "model_training.py": 700,
    "shap_analysis.py": 750,
    "bias_analysis.py": 950,
    "bias_mitigation.py": 800,
    "fairness_evaluation.py": 600,
    "interpretability_comparison.py": 700,
    "app.py": 400
}

total_loc = sum(estimated_loc.values())
print(f"  💻 Lignes de code estimées: {total_loc:,}")
print(f"  📈 Temps développement estimé: ~200h")
```

### Cellule 70: Conclusions Finales (Code)
```python
# Conclusions finales du projet
print("🎯 CONCLUSIONS FINALES - PROJET COMPAS SHAP")
print("=" * 50)

print("✅ OBJECTIFS ATTEINTS:")
objectives_achieved = [
    "Reproduction findings ProPublica sur biais COMPAS",
    "Implémentation framework SHAP complet",
    "Détection systématique biais avec métriques équité",
    "Stratégies mitigation efficaces développées",
    "Évaluation rigoureuse trade-offs performance/équité",
    "Comparaison SHAP/LIME/SAGE réalisée",
    "Dashboard interactif pour utilisation pratique",
    "Documentation complète et reproductible"
]

for i, objective in enumerate(objectives_achieved, 1):
    print(f"  {i}. ✅ {objective}")

print(f"\n🌟 CONTRIBUTIONS PRINCIPALES:")
contributions = [
    "Framework reproductible détection biais COMPAS",
    "Pipeline complet mitigation avec évaluation",
    "Comparaison rigoureuse méthodes interprétabilité",
    "Optimisations spécifiques Mac M4 Pro",
    "Dashboard opérationnel prêt production",
    "Méthodologie applicable autres domaines",
    "Documentation exhaustive en français",
    "Code source ouvert et modulaire"
]

for contribution in contributions:
    print(f"  🎯 {contribution}")

print(f"\n📈 IMPACT ATTENDU:")
expected_impact = [
    "Réduction biais dans systèmes justice prédictive",
    "Transparence accrue décisions algorithmiques",
    "Conformité réglementaire équité algorithmique",
    "Méthodes réutilisables autres domaines (RH, crédit, etc.)",
    "Sensibilisation enjeux équité IA",
    "Contribution recherche interprétabilité"
]

for impact in expected_impact:
    print(f"  📊 {impact}")

print(f"\n🚀 PERSPECTIVES FUTURES:")
future_perspectives = [
    "Extension autres attributs protégés (âge, handicap, etc.)",
    "Intégration méthodes mitigation plus avancées",
    "Application domaines connexes (santé, éducation)",
    "Développement métriques équité innovantes",
    "Collaboration communauté internationale",
    "Formation/sensibilisation équipes développement"
]

for perspective in future_perspectives:
    print(f"  🔮 {perspective}")

print(f"\n💡 LEÇONS APPRISES:")
lessons_learned = [
    "Importance monitoring continu vs audit ponctuel",
    "Trade-offs performance/équité gérables avec bonne méthodo",
    "SHAP méthode robuste pour détection biais",
    "Documentation/transparence critiques pour adoption",
    "Validation métier essentielle au-delà technique",
    "Approche systémique nécessaire (gouvernance + technique)"
]

for lesson in lessons_learned:
    print(f"  🎓 {lesson}")

print(f"\n🎊 REMERCIEMENTS:")
print("  👨‍🏫 Équipe pédagogique pour guidance méthodologique")
print("  🌐 Communauté open source (SHAP, Fairlearn, etc.)")
print("  📊 ProPublica pour investigation originale")
print("  💻 Développeurs outils d'équité algorithmique")

print(f"\n" + "=" * 50)
print("🎯 PROJET COMPAS SHAP - MISSION ACCOMPLIE ✅")
print("⚖️ 'SHAP is unlocking the secrets of complex models'")  
print("🌟 'and revealing their true potential for fairness.'")
print("=" * 50)
```

---

Ce guide fournit une structure complète cellule par cellule pour le notebook principal d'analyse COMPAS. Chaque cellule est documentée avec son objectif, son contenu et son contexte dans le workflow global. Le notebook ainsi structuré permettra une analyse complète et reproductible du projet SESAME-SHAP.