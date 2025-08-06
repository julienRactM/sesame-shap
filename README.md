# SESAME-SHAP: Analyse d'Interprétabilité et de Biais COMPAS

## 🎯 Vue d'Ensemble du Projet

**"SHAP is unlocking the secrets of complex models and revealing their true potential."**

Ce projet explore l'interprétabilité des modèles d'apprentissage automatique appliqués au dataset COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) en utilisant SHAP (SHapley Additive exPlanations) et d'autres méthodes d'explication. L'objectif principal est de détecter, analyser et mitiger les biais raciaux dans les prédictions de récidive.

### 📋 Contexte

Le système COMPAS est utilisé dans le système judiciaire américain pour évaluer le risque de récidive des accusés. Une investigation de ProPublica en 2016 a révélé des biais raciaux significatifs dans ces prédictions, montrant que les défendeurs African-American étaient disproportionnellement classés comme à haut risque par rapport aux défendeurs Caucasian.

### 🏆 Objectifs

1. **Détecter** les biais raciaux dans les prédictions COMPAS
2. **Analyser** l'interprétabilité avec SHAP, LIME et SAGE  
3. **Mitiger** les biais identifiés avec diverses stratégies
4. **Évaluer** l'efficacité des techniques de mitigation
5. **Comparer** les méthodes d'interprétabilité (BONUS)

## 🚀 Installation et Configuration

### Prérequis

- Python 3.8+
- pip ou conda
- Compte Kaggle (pour télécharger le dataset)

### Installation Rapide

```bash
# Cloner le repository
git clone <url-du-repo>
cd sesame-shap

# Lancer l'installation automatique
./install.sh
```

### Installation Manuelle

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Créer la structure des dossiers
mkdir -p data/{raw,processed,models,results}
```

### Configuration Kaggle

Pour télécharger le dataset COMPAS:

1. Créer un compte sur [Kaggle](https://www.kaggle.com)
2. Obtenir votre API key depuis votre profil Kaggle
3. Configurer kagglehub selon la documentation

## 📁 Structure du Projet

```
sesame-shap/
├── src/                              # Modules Python principaux
│   ├── data_acquisition.py          # Téléchargement dataset COMPAS
│   ├── exploratory_analysis.py      # Analyse exploratoire avec focus biais
│   ├── feature_engineering.py       # Preprocessing et feature engineering
│   ├── model_training.py            # Entraînement modèles ML
│   ├── shap_analysis.py            # Analyse SHAP principal
│   ├── bias_analysis.py            # Détection et métriques de biais
│   ├── bias_mitigation.py          # Stratégies de mitigation
│   ├── fairness_evaluation.py      # Évaluation efficacité mitigation
│   └── interpretability_comparison.py # Comparaison SHAP/LIME/SAGE
├── data/                            # Données et résultats
│   ├── raw/                        # Dataset COMPAS brut
│   ├── processed/                  # Données preprocessées
│   ├── models/                     # Modèles entraînés sauvegardés
│   └── results/                    # Résultats analyses et visualisations
├── Dashboard/                       # Dashboard interactif Streamlit
│   └── app.py                      # Application dashboard principal
├── main_notebook.ipynb             # Notebook principal d'analyse
├── requirements.txt                # Dépendances Python
├── install.sh                     # Script d'installation
├── .gitignore                     # Fichiers ignorés par Git
├── CLAUDE.md                      # Guide pour Claude Code
└── README.md                      # Ce fichier
```

## 🎲 Utilisation

### 1. Lancement du Notebook Principal

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer Jupyter Lab
jupyter lab main_notebook.ipynb
```

### 2. Dashboard Interactif

```bash
# Lancer le dashboard Streamlit
streamlit run Dashboard/app.py
```

Le dashboard sera accessible à l'adresse: http://localhost:8501

### 3. Scripts Python Individuels

```bash
# Télécharger le dataset COMPAS
python src/data_acquisition.py

# Analyse exploratoire
python src/exploratory_analysis.py

# Entraînement des modèles
python src/model_training.py

# Analyse SHAP
python src/shap_analysis.py

# Détection des biais
python src/bias_analysis.py

# Mitigation des biais
python src/bias_mitigation.py

# Évaluation de l'équité
python src/fairness_evaluation.py

# Comparaison d'interprétabilité (BONUS)
python src/interpretability_comparison.py
```

## 📊 Données et Analyse

### Dataset COMPAS

Le projet utilise le dataset COMPAS disponible sur Kaggle (`danofer/compass`) qui contient:

- **~10,000 enregistrements** de défendeurs évalués par COMPAS
- **Variables démographiques**: race, sexe, âge
- **Historique criminel**: nombre d'antécédents, gravité des charges
- **Scores COMPAS**: évaluation du risque de récidive (1-10)
- **Récidive réelle**: récidive observée dans les 2 ans

### Métriques d'Équité Analysées

- **Parité Démographique**: Égalité des taux de prédictions positives
- **Égalité des Chances**: Égalité des taux de vrais positifs
- **Égalité des Chances (FPR)**: Égalité des taux de faux positifs  
- **Impact Disparate**: Respect de la règle des 80%
- **Calibration**: Cohérence prédictions vs réalité par groupe

### Algorithmes ML Implémentés

- **Logistic Regression** (baseline)
- **Random Forest**
- **XGBoost** (optimisé Apple Silicon)
- **LightGBM** (optimisé Mac M4 Pro)
- **Support Vector Machine**
- **Neural Network** (MLP simple)

## 🔍 Méthodes d'Interprétabilité

### SHAP (Principal)

- **TreeExplainer**: Pour modèles basés sur les arbres
- **KernelExplainer**: Pour modèles complexes
- **LinearExplainer**: Pour modèles linéaires
- **Analyse des biais**: Comparaison valeurs SHAP par groupe démographique

### LIME (Comparaison)

- **Explications locales**: Approximations linéaires locales
- **Flexibilité**: Compatible avec tous types de modèles
- **Stabilité**: Analyse de la consistance des explications

### SAGE (Bonus)

- **Interactions**: Prise en compte native des interactions entre features
- **Valeurs de Shapley**: Théorie des jeux appliquée
- **Performance**: Analyse des trade-offs computationnels

## 🛡️ Stratégies de Mitigation des Biais

### Pré-traitement

- **Suppression de features sensibles**
- **Rééchantillonnage SMOTE équitable**
- **Augmentation de données consciente des biais**

### Traitement (In-processing)

- **Entraînement avec contraintes d'équité** (fairlearn)
- **Adversarial debiasing**
- **Multi-objectif optimization** (précision + équité)

### Post-traitement

- **Calibration par groupe démographique**
- **Optimisation des seuils de décision**
- **Ajustement des scores de sortie**

## 📈 Résultats et Visualisations

### Dashboard Interactif

Le dashboard Streamlit propose 8 sections:

1. **🏠 Accueil**: Vue d'ensemble et métriques globales
2. **📊 Analyse Exploratoire**: Distributions et détection de biais
3. **🤖 Modèles et Performance**: Comparaison des algorithmes ML
4. **🔍 Analyse SHAP**: Interprétabilité et importance des features
5. **⚖️ Détection des Biais**: Métriques d'équité détaillées
6. **🛡️ Mitigation des Biais**: Stratégies et trade-offs
7. **📈 Évaluation d'Équité**: Efficacité des mitigations
8. **🔄 Comparaison d'Interprétabilité**: SHAP vs LIME vs SAGE

### Rapports Générés

- **Rapport d'analyse exploratoire** (HTML/PDF)
- **Rapport SHAP complet** avec visualisations
- **Rapport de détection des biais** avec tests statistiques
- **Rapport d'évaluation d'équité** avant/après mitigation
- **Rapport de comparaison d'interprétabilité**

### Visualisations Principales

- **Summary plots SHAP** par groupe démographique
- **Waterfall plots** pour explications individuelles
- **Dependence plots** montrant interactions features
- **Matrices de confusion** par groupe racial
- **Courbes ROC** comparatives par démographie
- **Graphiques trade-off** performance vs équité

## 🧪 Tests et Validation

### Tests Statistiques

- **Test Chi² d'indépendance** entre race et prédictions
- **Test Mann-Whitney U** pour comparaisons de distributions
- **Tests de Kolmogorov-Smirnov** pour calibration
- **Intervalles de confiance** pour métriques d'équité

### Validation Croisée

- **Stratification** par cible ET attributs sensibles
- **Cross-validation temporelle** si applicable
- **Validation des performances post-mitigation**

## 🚨 Limitations et Considérations Éthiques

### Limitations Techniques

- **Données historiques**: Biais potentiels dans les données d'entraînement
- **Proxies indirects**: Variables corrélées avec race/sexe difficiles à éliminer
- **Trade-offs performance**: Mitigation peut réduire précision prédictive
- **Métriques d'équité**: Impossibilité de satisfaire toutes simultanément

### Considérations Éthiques

- **Usage responsable**: Modèles pour analyse recherche uniquement
- **Transparence**: Documentation complète des biais détectés
- **Contexte juridique**: Implications dans le système judiciaire
- **Biais historiques**: Reproduction potentielle d'injustices passées

## 🤝 Contribution et Développement

### Structure de Développement

```bash
# Tests des modules
python -m pytest tests/

# Linting du code
python -m flake8 src/

# Vérification types
python -m mypy src/

# Formatage automatique
python -m black src/
```

### Ajout de Nouvelles Features

1. Créer une branche: `git checkout -b feature/nouvelle-feature`
2. Développer dans `src/` avec tests appropriés
3. Mettre à jour documentation et notebooks
4. Créer une pull request avec description détaillée

## 📚 Références et Ressources

### Articles Académiques

- **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions" (SHAP)
- **Ribeiro et al. (2016)**: "Why Should I Trust You?" (LIME)
- **Doshi-Velez & Kim (2017)**: "Towards A Rigorous Science of Interpretable Machine Learning"

### Datasets et Outils

- **ProPublica COMPAS Analysis**: Investigation originale sur les biais COMPAS
- **Fairlearn**: Bibliothèque Microsoft pour équité algorithmique
- **AI Fairness 360**: Toolkit IBM pour détection et mitigation des biais

### Documentation Technique

- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Documentation**: https://github.com/marcotcr/lime
- **Fairlearn Guide**: https://fairlearn.org/

## 📞 Support et Contact

### Issues et Bugs

Signaler les problèmes via les GitHub Issues avec:
- Description détaillée du problème
- Étapes pour reproduire
- Logs d'erreur si disponibles
- Configuration système

### Questions et Discussions

Utiliser les GitHub Discussions pour:
- Questions sur l'interprétation des résultats
- Suggestions d'améliorations
- Discussions sur les aspects éthiques
- Partage d'analyses complémentaires

## 📄 Licence et Citation

### Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour les détails complets.

### Citation

Si vous utilisez ce projet dans vos recherches, veuillez citer:

```bibtex
@software{sesame_shap_2025,
  title={SESAME-SHAP: COMPAS Bias Analysis with SHAP Interpretability},
  author={Projet SESAME-SHAP},
  year={2025},
  url={https://github.com/votre-repo/sesame-shap}
}
```

---

**⚖️ "Sésame, ouvre-toi" - Déverrouillant les secrets des modèles complexes pour révéler leur véritable potentiel équitable.**

*Développé avec ❤️ pour la recherche en IA éthique et interprétable*