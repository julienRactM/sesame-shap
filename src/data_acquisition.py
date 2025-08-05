"""
Module d'acquisition de données pour le dataset COMPAS

Ce module gère le téléchargement et le chargement du dataset COMPAS 
depuis Kaggle en utilisant kagglehub. Il fournit des fonctions pour:
- Télécharger le dataset COMPAS
- Charger les données dans des DataFrames pandas
- Valider la présence des fichiers attendus
- Obtenir des informations sur les datasets

Auteur: Projet SESAME-SHAP
Date: 2025
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import kagglehub

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompasDataAcquisition:
    """
    Classe pour gérer l'acquisition et le chargement des données COMPAS.
    
    Cette classe encapsule toutes les fonctionnalités nécessaires pour télécharger,
    valider et charger le dataset COMPAS depuis Kaggle.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialise l'acquisition des données COMPAS.
        
        Args:
            data_dir (str): Répertoire de destination pour les données brutes
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = "danofer/compass"
        
        # Fichiers attendus dans le dataset COMPAS
        self.expected_files = [
            "compas-scores-raw.csv",
            "cox-violent-parsed.csv",
            "cox-violent-parsed_filt.csv",
            "propublica_data_for_fairml.csv"
        ]
        
        # Créer le répertoire de données s'il n'existe pas
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_compas_data(self, force_reload: bool = False) -> str:
        """
        Télécharge le dataset COMPAS depuis Kaggle.
        
        Args:
            force_reload (bool): Force le re-téléchargement même si les fichiers existent
            
        Returns:
            str: Chemin vers le répertoire contenant les données téléchargées
            
        Raises:
            Exception: Si le téléchargement échoue ou si les fichiers attendus sont manquants
        """
        try:
            logger.info(f"Début du téléchargement du dataset COMPAS: {self.dataset_name}")
            
            # Vérifier si les données existent déjà
            if not force_reload and self._check_existing_data():
                logger.info("Les données COMPAS existent déjà localement")
                return str(self.data_dir)
            
            # Télécharger le dataset via kagglehub
            logger.info("Téléchargement en cours depuis Kaggle...")
            download_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset téléchargé vers: {download_path}")
            
            # Copier les fichiers vers notre répertoire de données
            self._copy_files_to_data_dir(download_path)
            
            # Valider que tous les fichiers attendus sont présents
            self._validate_downloaded_files()
            
            logger.info("Téléchargement et validation terminés avec succès")
            return str(self.data_dir)
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données COMPAS: {str(e)}")
            raise Exception(f"Échec du téléchargement des données COMPAS: {str(e)}")
    
    def _check_existing_data(self) -> bool:
        """
        Vérifie si les données COMPAS existent déjà localement.
        
        Returns:
            bool: True si tous les fichiers attendus sont présents
        """
        for filename in self.expected_files:
            file_path = self.data_dir / filename
            if not file_path.exists():
                return False
        return True
    
    def _copy_files_to_data_dir(self, source_path: str) -> None:
        """
        Copie les fichiers depuis le répertoire de téléchargement kagglehub
        vers notre répertoire de données.
        
        Args:
            source_path (str): Chemin source du téléchargement kagglehub
        """
        source_dir = Path(source_path)
        
        # Copier tous les fichiers CSV trouvés (récursivement)
        csv_files = list(source_dir.rglob("*.csv"))
        
        if not csv_files:
            raise Exception(f"Aucun fichier CSV trouvé dans {source_path}")
        
        for csv_file in csv_files:
            # Éviter les fichiers cachés ou temporaires
            if csv_file.name.startswith('._'):
                continue
            
            # Vérifier que c'est bien un fichier (pas un répertoire)
            if not csv_file.is_file():
                continue
                
            destination = self.data_dir / csv_file.name
            shutil.copy2(csv_file, destination)
            logger.info(f"Fichier copié: {csv_file.name}")
    
    def _validate_downloaded_files(self) -> None:
        """
        Valide que tous les fichiers attendus sont présents et non vides.
        
        Raises:
            Exception: Si des fichiers sont manquants ou vides
        """
        missing_files = []
        empty_files = []
        
        for filename in self.expected_files:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                missing_files.append(filename)
            elif file_path.stat().st_size == 0:
                empty_files.append(filename)
        
        if missing_files:
            raise Exception(f"Fichiers manquants: {missing_files}")
        
        if empty_files:
            raise Exception(f"Fichiers vides: {empty_files}")
        
        logger.info("Validation des fichiers téléchargés: OK")
    
    def load_compas_data(self) -> Dict[str, pd.DataFrame]:
        """
        Charge les datasets COMPAS dans des DataFrames pandas.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionnaire contenant les DataFrames chargés
            
        Raises:
            Exception: Si le chargement des données échoue
        """
        try:
            logger.info("Chargement des données COMPAS...")
            
            # Vérifier que les données existent
            if not self._check_existing_data():
                logger.warning("Données manquantes, tentative de téléchargement...")
                self.download_compas_data()
            
            dataframes = {}
            
            # Charger chaque fichier CSV
            for filename in self.expected_files:
                file_path = self.data_dir / filename
                
                try:
                    # Déterminer la clé du dictionnaire basée sur le nom du fichier
                    key = filename.replace('.csv', '').replace('-', '_')
                    
                    # Charger le CSV avec gestion d'erreur
                    df = pd.read_csv(file_path, encoding='utf-8')
                    dataframes[key] = df
                    
                    logger.info(f"Chargé {filename}: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                    
                except UnicodeDecodeError:
                    # Essayer avec un encodage différent si UTF-8 échoue
                    df = pd.read_csv(file_path, encoding='latin-1')
                    dataframes[key] = df
                    logger.info(f"Chargé {filename} (latin-1): {df.shape[0]} lignes, {df.shape[1]} colonnes")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de {filename}: {str(e)}")
                    raise
            
            logger.info("Chargement des données terminé avec succès")
            return dataframes
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données COMPAS: {str(e)}")
            raise Exception(f"Échec du chargement des données COMPAS: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Retourne des informations détaillées sur les datasets COMPAS.
        
        Returns:
            Dict[str, Dict]: Informations sur chaque dataset
            
        Raises:
            Exception: Si l'analyse des données échoue
        """
        try:
            logger.info("Analyse des informations des datasets...")
            
            dataframes = self.load_compas_data()
            dataset_info = {}
            
            for name, df in dataframes.items():
                info = {
                    'nom_fichier': f"{name.replace('_', '-')}.csv",
                    'nombre_lignes': len(df),
                    'nombre_colonnes': len(df.columns),
                    'colonnes': list(df.columns),
                    'types_donnees': df.dtypes.to_dict(),
                    'valeurs_manquantes': df.isnull().sum().to_dict(),
                    'taille_memoire_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                    'apercu_donnees': df.head().to_dict('records')[:3]  # Premières 3 lignes
                }
                
                # Statistiques descriptives pour les colonnes numériques
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    info['statistiques_numeriques'] = df[numeric_cols].describe().to_dict()
                
                dataset_info[name] = info
                logger.info(f"Analysé {name}: {info['nombre_lignes']} lignes, {info['nombre_colonnes']} colonnes")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des datasets: {str(e)}")
            raise Exception(f"Échec de l'analyse des datasets: {str(e)}")
    
    def get_available_files(self) -> List[str]:
        """
        Retourne la liste des fichiers CSV disponibles dans le répertoire de données.
        
        Returns:
            List[str]: Liste des noms de fichiers disponibles
        """
        if not self.data_dir.exists():
            return []
        
        return [f.name for f in self.data_dir.glob("*.csv")]


# Fonctions utilitaires pour faciliter l'utilisation du module
def download_compas_data(data_dir: str = "data/raw", force_reload: bool = False) -> str:
    """
    Fonction utilitaire pour télécharger le dataset COMPAS.
    
    Args:
        data_dir (str): Répertoire de destination
        force_reload (bool): Force le re-téléchargement
        
    Returns:
        str: Chemin vers les données téléchargées
    """
    acquisition = CompasDataAcquisition(data_dir)
    return acquisition.download_compas_data(force_reload)


def load_compas_data(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Fonction utilitaire pour charger les données COMPAS.
    
    Args:
        data_dir (str): Répertoire des données
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionnaire des DataFrames
    """
    acquisition = CompasDataAcquisition(data_dir)
    return acquisition.load_compas_data()


def get_dataset_info(data_dir: str = "data/raw") -> Dict[str, Dict]:
    """
    Fonction utilitaire pour obtenir les informations des datasets.
    
    Args:
        data_dir (str): Répertoire des données
        
    Returns:
        Dict[str, Dict]: Informations détaillées sur les datasets
    """
    acquisition = CompasDataAcquisition(data_dir)
    return acquisition.get_dataset_info()


if __name__ == "__main__":
    """
    Script de test pour vérifier le fonctionnement du module.
    """
    try:
        print("=== Test du module d'acquisition COMPAS ===")
        
        # Initialiser l'acquisition
        acquisition = CompasDataAcquisition()
        
        # Télécharger les données
        print("\n1. Téléchargement des données...")
        data_path = acquisition.download_compas_data()
        print(f"Données téléchargées vers: {data_path}")
        
        # Charger les données
        print("\n2. Chargement des données...")
        dataframes = acquisition.load_compas_data()
        print(f"Datasets chargés: {list(dataframes.keys())}")
        
        # Obtenir les informations
        print("\n3. Analyse des informations...")
        info = acquisition.get_dataset_info()
        
        for name, dataset_info in info.items():
            print(f"\n--- {name} ---")
            print(f"Lignes: {dataset_info['nombre_lignes']}")
            print(f"Colonnes: {dataset_info['nombre_colonnes']}")
            print(f"Taille: {dataset_info['taille_memoire_mb']} MB")
        
        print("\n=== Test terminé avec succès ===")
        
    except Exception as e:
        print(f"Erreur lors du test: {str(e)}")