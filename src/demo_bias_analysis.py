#!/usr/bin/env python3
"""
Script de démonstration du framework d'analyse de biais COMPAS

Ce script montre comment utiliser le framework d'analyse de biais pour évaluer
l'équité algorithmique du système COMPAS et détecter les patterns de discrimination
similaires à ceux identifiés par l'enquête ProPublica.

Usage:
    python demo_bias_analysis.py

Auteur: Claude AI - Assistant Data Engineer
Date: 2025-08-05
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ajout du répertoire src au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bias_analysis import (
    BiasAnalyzer, BiasAnalysisConfig, run_comprehensive_bias_analysis,
    prepare_protected_attributes
)

def load_latest_compas_data():
    """
    Charge les dernières données COMPAS traitées
    
    Returns:
        DataFrame avec les données COMPAS
    """
    data_dir = "../data/processed"
    
    # Recherche du fichier le plus récent
    compas_files = [f for f in os.listdir(data_dir) if f.startswith('compas_full_') and f.endswith('.csv')]
    
    if not compas_files:
        raise FileNotFoundError("Aucun fichier de données COMPAS trouvé")
    
    # Tri par timestamp dans le nom de fichier
    latest_file = sorted(compas_files)[-1]
    file_path = os.path.join(data_dir, latest_file)
    
    print(f"Chargement des données: {latest_file}")
    df = pd.read_csv(file_path)
    
    return df

def demonstrate_basic_bias_analysis():
    """Démonstration de l'analyse de biais de base"""
    
    print("\n" + "="*80)
    print("DÉMONSTRATION - ANALYSE DE BIAIS COMPAS")
    print("="*80)
    
    # Chargement des données
    print("\n1. Chargement des données COMPAS...")
    df = load_latest_compas_data()
    print(f"   Données chargées: {len(df)} échantillons, {len(df.columns)} features")
    
    # Affichage des informations sur les attributs protégés
    print("\n2. Analyse des attributs protégés...")
    protected_attrs = prepare_protected_attributes(df)
    
    for attr_name, attr_values in protected_attrs.items():
        unique_values, counts = np.unique(attr_values, return_counts=True)
        print(f"   {attr_name}:")
        for val, count in zip(unique_values, counts):
            percentage = (count / len(attr_values)) * 100
            print(f"     - {val}: {count} ({percentage:.1f}%)")
    
    # Configuration de l'analyse
    print("\n3. Configuration de l'analyse de biais...")
    config = BiasAnalysisConfig(
        protected_attributes=list(protected_attrs.keys()),
        fairness_threshold=0.8,  # Règle des 80%
        statistical_significance_level=0.05,
        save_visualizations=True,
        results_dir="../data/results/bias_analysis"
    )
    print(f"   Attributs protégés: {config.protected_attributes}")
    print(f"   Seuil d'équité: {config.fairness_threshold}")
    
    # Lancement de l'analyse complète
    print("\n4. Lancement de l'analyse complète de biais...")
    bias_report = run_comprehensive_bias_analysis(
        df=df,
        y_true_col='two_year_recid'
    )
    
    return bias_report

def demonstrate_detailed_metrics_analysis(bias_report):
    """Démonstration de l'analyse détaillée des métriques"""
    
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DES MÉTRIQUES D'ÉQUITÉ")
    print("="*80)
    
    # Résumé exécutif
    executive_summary = bias_report['executive_summary']
    print(f"\n🚨 NIVEAU DE RISQUE: {executive_summary['risk_level']}")
    print(f"📊 RÉSULTATS CLÉS: {executive_summary['key_findings']}")
    print(f"⚠️  ACTIONS IMMÉDIATES: {executive_summary['recommendation_priority']}")
    
    # Analyse par attribut protégé
    fairness_metrics = bias_report['fairness_metrics']
    
    for attr_name, attr_metrics in fairness_metrics.items():
        print(f"\n" + "-"*60)
        print(f"ANALYSE POUR: {attr_name.upper()}")
        print("-"*60)
        
        # Parité démographique
        if 'demographic_parity' in attr_metrics:
            dp = attr_metrics['demographic_parity']
            print(f"\n📈 PARITÉ DÉMOGRAPHIQUE:")
            print(f"   Impact disparate: {dp['disparate_impact']:.3f}")
            print(f"   Règle des 80%: {'✅ RESPECTÉE' if dp['passes_80_rule'] else '❌ VIOLÉE'}")
            print(f"   Taux par groupe:")
            for group, rate in dp['group_rates'].items():
                print(f"     - {group}: {rate:.3f}")
        
        # Égalité des chances
        if 'equalized_odds' in attr_metrics:
            eo = attr_metrics['equalized_odds']
            print(f"\n⚖️  ÉGALITÉ DES CHANCES:")
            print(f"   Différence TPR: {eo['tpr_difference']:.3f}")
            print(f"   Différence FPR: {eo['fpr_difference']:.3f}")
            print(f"   Métrique globale: {eo['equalized_odds_difference']:.3f}")
            
            for group, metrics in eo['group_metrics'].items():
                print(f"   {group}:")
                print(f"     - TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}")
        
        # Calibration
        if 'calibration' in attr_metrics:
            cal = attr_metrics['calibration']
            print(f"\n🎯 CALIBRATION:")
            print(f"   Différence Brier Score: {cal['brier_score_difference']:.3f}")
            print(f"   Différence ECE: {cal['ece_difference']:.3f}")

def demonstrate_propublica_comparison(bias_report):
    """Démonstration de la comparaison avec ProPublica"""
    
    print("\n" + "="*80)
    print("COMPARAISON AVEC L'ENQUÊTE PROPUBLICA")
    print("="*80)
    
    propublica_benchmark = bias_report['propublica_benchmark']
    
    print(f"\n🔍 COHÉRENCE AVEC PROPUBLICA: {propublica_benchmark['consistency_with_findings'].upper()}")
    
    if 'comparison_with_propublica' in propublica_benchmark:
        comparisons = propublica_benchmark['comparison_with_propublica']
        
        for metric_name, comparison in comparisons.items():
            if isinstance(comparison, dict):
                print(f"\n📊 {metric_name.upper()}:")
                print(f"   Notre analyse: {comparison['our_finding']:.3f}")
                print(f"   Référence ProPublica: {comparison['propublica_reference']:.3f}")
                print(f"   Cohérence: {'✅' if comparison['consistent'] else '❌'} {comparison['interpretation']}")
    
    # Contexte historique ProPublica
    print(f"\n📰 CONTEXTE PROPUBLICA (2016):")
    print(f"   • Les défendeurs afro-américains étaient presque 2x plus susceptibles")
    print(f"     d'être étiquetés à haut risque que les défendeurs blancs")
    print(f"   • Taux de faux positifs: 45% (Afro-Américains) vs 23% (Blancs)")
    print(f"   • COMPAS était plus précis pour prédire la récidive chez les Blancs")

def demonstrate_mitigation_recommendations(bias_report):
    """Démonstration des recommandations de mitigation"""
    
    print("\n" + "="*80)
    print("RECOMMANDATIONS DE MITIGATION DES BIAIS")
    print("="*80)
    
    recommendations = bias_report['mitigation_recommendations']
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. 🎯 {rec['type'].upper()} - {rec['attribute']}")
        print(f"   Sévérité: {rec['severity']}")
        print(f"   Recommandation: {rec['recommendation']}")
        
        if 'technical_approach' in rec and rec['technical_approach']:
            print(f"   Approches techniques:")
            for approach in rec['technical_approach']:
                print(f"     • {approach}")

def demonstrate_visualization_generation(bias_report):
    """Démonstration de la génération de visualisations"""
    
    print("\n" + "="*80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*80)
    
    visualizations = bias_report.get('visualizations', {})
    
    if visualizations:
        print(f"\n📊 {len(visualizations)} visualisations générées:")
        
        for viz_name, viz_path in visualizations.items():
            if os.path.exists(viz_path):
                file_size = os.path.getsize(viz_path) / 1024  # KB
                print(f"   ✅ {viz_name}: {viz_path} ({file_size:.1f} KB)")
            else:
                print(f"   ❌ {viz_name}: Fichier non trouvé")
        
        print(f"\n💡 Visualisations disponibles:")
        print(f"   • Tableau de bord des métriques d'équité")
        print(f"   • Carte de chaleur des biais")
        print(f"   • Courbes ROC par groupe démographique")
        print(f"   • Graphiques de calibration")
        print(f"   • Analyse d'impact disparate")
    else:
        print("\n⚠️  Aucune visualisation générée")

def generate_summary_report(bias_report):
    """Génère un rapport de synthèse"""
    
    print("\n" + "="*80)
    print("RAPPORT DE SYNTHÈSE")
    print("="*80)
    
    metadata = bias_report['metadata']
    executive_summary = bias_report['executive_summary']
    
    print(f"\n📋 INFORMATIONS GÉNÉRALES:")
    print(f"   Modèle analysé: {metadata['model_name']}")
    print(f"   Taille de l'échantillon: {metadata['sample_size']}")
    print(f"   Attributs protégés: {', '.join(metadata['protected_attributes_analyzed'])}")
    print(f"   Date d'analyse: {metadata['analysis_timestamp']}")
    
    print(f"\n🎯 VERDICT FINAL:")
    print(f"   Niveau de risque: {executive_summary['risk_level']}")
    print(f"   Priorité d'action: {executive_summary['recommendation_priority']}")
    print(f"   Impact business: {executive_summary['business_impact']}")
    
    # Statistiques de violations
    disparate_impact = bias_report['disparate_impact_analysis']
    total_violations = sum(1 for data in disparate_impact.values() if not data['passes_80_rule'])
    
    print(f"\n📊 STATISTIQUES:")
    print(f"   Violations de la règle des 80%: {total_violations}/{len(disparate_impact)}")
    print(f"   Biais globalement détecté: {'OUI' if bias_report['bias_patterns']['overall_bias_detected'] else 'NON'}")
    
    # Sauvegarde du rapport de synthèse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"../data/results/bias_analysis/summary_report_{timestamp}.txt"
    
    try:
        os.makedirs("../data/results/bias_analysis", exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE SYNTHÈSE - ANALYSE DE BIAIS COMPAS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Niveau de risque: {executive_summary['risk_level']}\n")
            f.write(f"Violations détectées: {total_violations}\n")
            f.write(f"Recommandations: {len(bias_report['mitigation_recommendations'])}\n\n")
            
            for i, rec in enumerate(bias_report['mitigation_recommendations'], 1):
                f.write(f"{i}. {rec['type']} ({rec['severity']})\n")
                f.write(f"   {rec['recommendation']}\n\n")
        
        print(f"\n💾 Rapport de synthèse sauvegardé: {summary_path}")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde: {e}")

def main():
    """Fonction principale de démonstration"""
    
    print("🚀 DÉMARRAGE DE LA DÉMONSTRATION")
    print("Framework d'Analyse de Biais COMPAS")
    print("Détection automatique de discrimination algorithmique")
    
    try:
        # 1. Analyse de biais de base
        bias_report = demonstrate_basic_bias_analysis()
        
        # 2. Analyse détaillée des métriques
        demonstrate_detailed_metrics_analysis(bias_report)
        
        # 3. Comparaison avec ProPublica
        demonstrate_propublica_comparison(bias_report)
        
        # 4. Recommandations de mitigation
        demonstrate_mitigation_recommendations(bias_report)
        
        # 5. Génération des visualisations
        demonstrate_visualization_generation(bias_report)
        
        # 6. Rapport de synthèse
        generate_summary_report(bias_report)
        
        print(f"\n✅ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
        print(f"📁 Résultats disponibles dans: data/results/bias_analysis/")
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DE LA DÉMONSTRATION: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)