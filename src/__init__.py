"""
Package SESAME-SHAP

Module principal pour l'analyse d'équité et d'explicabilité des algorithmes
de justice prédictive, avec focus sur le dataset COMPAS.
"""

from .data_acquisition import (
    CompasDataAcquisition,
    download_compas_data,
    load_compas_data,
    get_dataset_info
)

from .bias_mitigation import (
    BiasMitigationFramework,
    FairnessAwareCalibrator
)

__version__ = "1.0.0"
__author__ = "Projet SESAME-SHAP"

__all__ = [
    "CompasDataAcquisition",
    "download_compas_data", 
    "load_compas_data",
    "get_dataset_info",
    "BiasMitigationFramework",
    "FairnessAwareCalibrator"
]