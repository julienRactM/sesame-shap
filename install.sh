#!/bin/bash

echo "🔧 Installation du projet COMPAS SHAP Analysis..."

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Créer un environnement virtuel si il n'existe pas
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🚀 Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier que l'environnement virtuel est bien activé
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Environnement virtuel activé: $VIRTUAL_ENV"
else
    echo "❌ Erreur: Impossible d'activer l'environnement virtuel"
    exit 1
fi

# Mettre à jour pip
echo "⬆️ Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "📚 Installation des dépendances Python..."
pip install -r requirements.txt

# Créer les dossiers de données si ils n'existent pas
echo "📁 Création de la structure des dossiers..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p data/results

# Vérifier l'installation de Jupyter
echo "🔍 Vérification de l'installation Jupyter..."
if command -v jupyter &> /dev/null; then
    echo "✅ Jupyter est installé et prêt à utiliser."
else
    echo "❌ Problème avec l'installation de Jupyter."
fi

echo "✅ Installation terminée!"
echo ""
echo "Pour commencer:"
echo "1. Activez l'environnement virtuel: source venv/bin/activate"
echo "2. Lancez Jupyter Lab: jupyter lab main_notebook.ipynb"
echo "3. Ou lancez le dashboard: cd Dashboard && python app.py"
echo ""
echo "📖 Consultez le README.md pour plus d'informations."