"""
Démonstration du pipeline de feature engineering COMPAS
=====================================================

Ce script démontre l'utilisation du pipeline de feature engineering
pour différents cas d'usage d'analyse de biais et de modélisation ML.

Auteur: Data Engineering Pipeline
Date: 2025-08-05
"""

import pandas as pd
import numpy as np
from feature_engineering import COMPASFeatureEngineer, create_sample_compas_data
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_pipeline_usage():
    """
    Démontre les différents usages du pipeline de feature engineering.
    """
    print("=== DÉMONSTRATION DU PIPELINE DE FEATURE ENGINEERING COMPAS ===\n")
    
    # Configuration des chemins
    base_dir = "/Users/julienrm/Workspace/M2/sesame-shap"
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Chargement des données préprocessées
    print("1. Chargement des données préprocessées...")
    
    # Trouver le fichier de métadonnées le plus récent
    processed_dir = os.path.join(data_dir, "processed")
    metadata_files = [f for f in os.listdir(processed_dir) if f.startswith('compas_metadata_')]
    
    if not metadata_files:
        print("❌ Aucun fichier de métadonnées trouvé. Veuillez d'abord exécuter le pipeline principal.")
        return
    
    latest_metadata = sorted(metadata_files)[-1]
    metadata_path = os.path.join(processed_dir, latest_metadata)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    timestamp = metadata['processing_timestamp']
    
    # Charger les différentes versions du dataset
    datasets = {}
    for dataset_type in ['full', 'no_sensitive', 'simplified']:
        train_file = f'compas_{dataset_type}_train_{timestamp}.csv'
        test_file = f'compas_{dataset_type}_test_{timestamp}.csv'
        
        train_path = os.path.join(processed_dir, train_file)
        test_path = os.path.join(processed_dir, test_file)
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Séparer features et target
            X_train = train_data.drop('two_year_recid', axis=1)
            y_train = train_data['two_year_recid']
            X_test = test_data.drop('two_year_recid', axis=1)
            y_test = test_data['two_year_recid']
            
            datasets[dataset_type] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'description': metadata['datasets_info'][dataset_type]['description']
            }
    
    print(f"✅ {len(datasets)} versions du dataset chargées")
    
    # 2. Analyse exploratoire des features
    print("\n2. Analyse exploratoire des features...")
    
    for dataset_name, data in datasets.items():
        print(f"\n📊 Dataset '{dataset_name}':")
        print(f"   - Description: {data['description']}")
        print(f"   - Features: {data['X_train'].shape[1]}")
        print(f"   - Échantillons train: {len(data['X_train'])}")
        print(f"   - Échantillons test: {len(data['X_test'])}")
        print(f"   - Taux de récidive: {data['y_train'].mean():.1%}")
        
        # Identifier les features les plus corrélées avec la target
        if len(data['X_train'].select_dtypes(include=[np.number]).columns) > 0:
            numeric_features = data['X_train'].select_dtypes(include=[np.number])
            correlations = numeric_features.corrwith(data['y_train'])
            top_corr = correlations.abs().sort_values(ascending=False).head(5)
            print(f"   - Top 5 corrélations avec récidive: {top_corr.to_dict()}")
    
    # 3. Entraînement de modèles sur chaque version
    print("\n3. Comparaison de modèles sur les différentes versions...")
    
    models_results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\n🔧 Entraînement sur dataset '{dataset_name}'...")
        
        # Modèle simple: Régression Logistique
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(data['X_train'], data['y_train'])
        lr_pred = lr_model.predict(data['X_test'])
        lr_score = lr_model.score(data['X_test'], data['y_test'])
        
        # Modèle complexe: Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(data['X_train'], data['y_train'])
        rf_pred = rf_model.predict(data['X_test'])
        rf_score = rf_model.score(data['X_test'], data['y_test'])
        
        models_results[dataset_name] = {
            'logistic_regression': {'score': lr_score, 'predictions': lr_pred},
            'random_forest': {'score': rf_score, 'predictions': rf_pred, 'model': rf_model}
        }
        
        print(f"   - Régression Logistique: {lr_score:.3f}")
        print(f"   - Random Forest: {rf_score:.3f}")
    
    # 4. Analyse de l'importance des features
    print("\n4. Analyse de l'importance des features...")
    
    for dataset_name, data in datasets.items():
        if dataset_name in models_results:
            rf_model = models_results[dataset_name]['random_forest']['model']
            feature_importance = pd.DataFrame({
                'feature': data['X_train'].columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📈 Top 10 features importantes - Dataset '{dataset_name}':")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # 5. Analyse comparative des performances
    print("\n5. Analyse comparative des performances...")
    
    comparison_df = pd.DataFrame([
        {
            'Dataset': dataset_name,
            'Features': len(datasets[dataset_name]['X_train'].columns),
            'LR_Score': results['logistic_regression']['score'],
            'RF_Score': results['random_forest']['score']
        }
        for dataset_name, results in models_results.items()
    ])
    
    print("\n📊 Comparaison des performances:")
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # 6. Analyse des biais potentiels
    print("\n6. Analyse des biais potentiels...")
    
    # Comparer les performances entre le dataset complet et sans attributs sensibles
    if 'full' in models_results and 'no_sensitive' in models_results:
        full_score = models_results['full']['random_forest']['score']
        no_sensitive_score = models_results['no_sensitive']['random_forest']['score']
        
        score_diff = full_score - no_sensitive_score
        
        print(f"📋 Impact de la suppression des attributs sensibles:")
        print(f"   - Performance avec attributs sensibles: {full_score:.3f}")
        print(f"   - Performance sans attributs sensibles: {no_sensitive_score:.3f}")
        print(f"   - Différence: {score_diff:+.3f}")
        
        if abs(score_diff) < 0.01:
            print("   ✅ Impact minimal - Bon signe pour l'équité")
        elif score_diff > 0.05:
            print("   ⚠️  Perte de performance significative - Possible sur-ajustement aux biais")
        else:
            print("   ℹ️  Impact modéré - À investiguer plus en détail")
    
    # 7. Recommandations d'usage
    print("\n7. Recommandations d'usage...")
    
    recommendations = []
    
    # Basé sur la complexité
    if 'simplified' in models_results:
        simplified_score = models_results['simplified']['random_forest']['score']
        if 'full' in models_results:
            full_score = models_results['full']['random_forest']['score']
            if (full_score - simplified_score) < 0.02:
                recommendations.append("✅ Le dataset simplifié offre des performances comparables - Recommandé pour l'interprétabilité")
    
    # Basé sur l'équité
    if 'no_sensitive' in models_results:
        recommendations.append("✅ Le dataset sans attributs sensibles est disponible pour l'analyse d'équité")
    
    # Basé sur les features importantes
    for dataset_name, data in datasets.items():
        if dataset_name in models_results:
            rf_model = models_results[dataset_name]['random_forest']['model']
            # Vérifier si des features de biais sont importantes
            feature_names = data['X_train'].columns.tolist()
            bias_features = [f for f in feature_names if any(word in f.lower() for word in ['race', 'sex', 'age_cat'])]
            
            if bias_features and dataset_name == 'full':
                feature_importance = rf_model.feature_importances_
                feature_dict = dict(zip(feature_names, feature_importance))
                important_bias_features = [f for f in bias_features if feature_dict.get(f, 0) > 0.05]
                
                if important_bias_features:
                    recommendations.append(f"⚠️  Dataset '{dataset_name}': Features de biais importantes détectées: {important_bias_features}")
    
    print("\n📝 Recommandations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    if not recommendations:
        print("   ✅ Aucune recommandation spécifique - Les datasets semblent bien équilibrés")
    
    print("\n=== DÉMONSTRATION TERMINÉE ===")
    
    return {
        'datasets_loaded': len(datasets),
        'models_trained': len(models_results),
        'comparison_results': comparison_df,
        'recommendations': recommendations
    }


def analyze_feature_distributions():
    """
    Analyse les distributions des features créées.
    """
    print("\n=== ANALYSE DES DISTRIBUTIONS DE FEATURES ===")
    
    base_dir = "/Users/julienrm/Workspace/M2/sesame-shap"
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Trouver le fichier le plus récent
    files = [f for f in os.listdir(processed_dir) if f.startswith('compas_full_') and f.endswith('.csv') and 'train' not in f and 'test' not in f]
    
    if not files:
        print("❌ Aucun dataset complet trouvé")
        return
    
    latest_file = sorted(files)[-1]
    data_path = os.path.join(processed_dir, latest_file)
    
    df = pd.read_csv(data_path)
    
    print(f"✅ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Analyse des distributions
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"\n📊 Features numériques: {len(numeric_features)}")
    print(f"📊 Features catégorielles: {len(categorical_features)}")
    
    # Statistiques descriptives pour les features numériques
    print("\n🔢 Statistiques des features numériques principales:")
    key_numeric = ['age', 'priors_count', 'decile_score', 'age_group_numeric', 'high_risk_score']
    available_numeric = [col for col in key_numeric if col in df.columns]
    
    if available_numeric:
        stats = df[available_numeric].describe()
        print(stats.round(2))
    
    # Distribution de la variable cible
    if 'two_year_recid' in df.columns:
        print(f"\n🎯 Distribution de la variable cible:")
        target_dist = df['two_year_recid'].value_counts(normalize=True)
        print(f"   - Pas de récidive (0): {target_dist[0]:.1%}")
        print(f"   - Récidive (1): {target_dist[1]:.1%}")
    
    # Analyse des corrélations
    if len(available_numeric) > 1:
        print(f"\n🔗 Matrice de corrélation (features principales):")
        corr_matrix = df[available_numeric].corr()
        print(corr_matrix.round(3))
    
    return df


if __name__ == "__main__":
    # Exécuter la démonstration complète
    results = demonstrate_pipeline_usage()
    
    # Analyser les distributions
    df_analysis = analyze_feature_distributions()
    
    print("\n" + "="*60)
    print("DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*60)