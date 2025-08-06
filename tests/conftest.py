"""
Configuration and fixtures for pytest test suite.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_compas_data():
    """Create sample COMPAS-like dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic COMPAS data
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'priors_count': np.random.poisson(2, n_samples),
        'days_b_screening_arrest': np.random.normal(0, 50, n_samples),
        'c_jail_in': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'c_jail_out': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'c_charge_degree': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
        'score_text': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'decile_score': np.random.randint(1, 11, n_samples),
        'two_year_recid': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'compas_screening_date': pd.date_range('2013-01-01', periods=n_samples, freq='D')[:n_samples]
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Higher priors -> higher recidivism
    high_priors_mask = df['priors_count'] > 3
    df.loc[high_priors_mask, 'two_year_recid'] = np.random.choice([0, 1], high_priors_mask.sum(), p=[0.3, 0.7])
    
    # Higher decile score -> higher recidivism
    high_score_mask = df['decile_score'] > 7
    df.loc[high_score_mask, 'two_year_recid'] = np.random.choice([0, 1], high_score_mask.sum(), p=[0.2, 0.8])
    
    return df

@pytest.fixture
def processed_data(sample_compas_data):
    """Create processed training and test data."""
    df = sample_compas_data.copy()
    
    # Basic preprocessing
    df['sex_encoded'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['race_encoded'] = pd.get_dummies(df['race'], prefix='race').iloc[:, 0]  # Just take first column
    df['charge_degree_encoded'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
    
    # Features and target
    feature_cols = ['age', 'sex_encoded', 'race_encoded', 'priors_count', 
                   'days_b_screening_arrest', 'charge_degree_encoded', 'decile_score']
    
    X = df[feature_cols].fillna(0)
    y = df['two_year_recid']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Sensitive attributes
    sensitive_attrs = df[['race', 'sex', 'age']].loc[X_test.index].reset_index(drop=True)
    
    return {
        'X_train': X_train.reset_index(drop=True),
        'X_test': X_test.reset_index(drop=True),
        'y_train': y_train.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
        'sensitive_attrs': sensitive_attrs,
        'full_data': df
    }

@pytest.fixture
def trained_models(processed_data):
    """Create pre-trained models for testing."""
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path

@pytest.fixture
def mock_results_dir(temp_dir):
    """Create temporary results directory structure."""
    results_dir = temp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (results_dir / "bias_analysis").mkdir(exist_ok=True)
    (results_dir / "shap_analysis").mkdir(exist_ok=True)
    (results_dir / "model_training").mkdir(exist_ok=True)
    (results_dir / "mitigation").mkdir(exist_ok=True)
    
    return str(results_dir)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Suppress warnings during tests
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seeds
    np.random.seed(42)
    
    yield
    
    # Cleanup if needed
    pass