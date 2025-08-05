# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a COMPAS bias analysis project using SHAP (SHapley Additive exPlanations) for model interpretability. The project analyzes the COMPAS Recidivism Risk Score dataset to detect and mitigate bias in predictive models used in criminal justice.

## Development Commands

### Project Setup
```bash
# Install dependencies and setup environment
./install.sh

# Install Python dependencies only
pip install -r requirements.txt
```

### Data Management
```bash
# Download COMPAS dataset (handled in notebooks/scripts)
python src/data_acquisition.py

# Run exploratory data analysis
python src/exploratory_analysis.py
```

### Model Training and Analysis
```bash
# Train models with bias analysis
python src/model_training.py

# Run SHAP interpretability analysis
python src/shap_analysis.py

# Run bias detection and mitigation
python src/bias_analysis.py
```

### Dashboard
```bash
# Launch interactive dashboard
cd Dashboard && python app.py
```

### Notebook Development
```bash
# Launch Jupyter for main analysis notebook
jupyter lab main_notebook.ipynb
```

## Architecture

### Core Structure
- `src/`: Python modules for data processing, modeling, and analysis
- `data/`: Dataset storage and processed data
- `Dashboard/`: Streamlit/Flask dashboard for interactive analysis
- `main_notebook.ipynb`: Primary analysis notebook

### Key Modules
- `data_acquisition.py`: COMPAS dataset download via kagglehub
- `exploratory_analysis.py`: Comprehensive EDA with bias focus
- `feature_engineering.py`: Data preprocessing and feature creation
- `model_training.py`: ML model training and comparison framework
- `shap_analysis.py`: SHAP interpretability analysis
- `bias_analysis.py`: Bias detection and fairness metrics
- `bias_mitigation.py`: Strategies for bias reduction
- `interpretability_comparison.py`: LIME/SAGE comparison (bonus)

### Data Pipeline
1. Raw COMPAS data → `data/raw/`
2. Processed features → `data/processed/`
3. Model artifacts → `data/models/`
4. Analysis results → `data/results/`

## Key Dependencies
- **Core ML**: scikit-learn, pandas, numpy
- **Interpretability**: shap, lime, sage-ml
- **Visualization**: matplotlib, seaborn, plotly
- **Bias Analysis**: fairlearn, aif360
- **Dashboard**: streamlit or flask, dash
- **Data**: kagglehub for dataset access

## Development Notes

### COMPAS Dataset Context
- Contains ~10,000 criminal records with recidivism scores
- Focus on racial bias detection (African American vs Caucasian)
- Key target: `two_year_recid` (binary recidivism within 2 years)
- Sensitive attributes: `race`, `sex`, `age`

### Model Interpretability Focus
- Primary: SHAP (TreeExplainer, KernelExplainer)
- Comparison: LIME and SAGE for bonus analysis
- Visualizations: summary plots, dependence plots, waterfall charts

### Bias Analysis Approach
- Fairness metrics: demographic parity, equalized odds, calibration
- Before/after mitigation comparison
- Feature impact analysis on protected groups

### Notebook Structure
The main notebook should follow this cell-by-cell structure:
1. Introduction and context
2. Data acquisition and loading
3. Exploratory data analysis with bias focus
4. Feature engineering and preprocessing
5. Model training and evaluation
6. SHAP interpretability analysis
7. Bias detection and visualization
8. Bias mitigation strategies
9. Fairness evaluation and comparison
10. Conclusions and recommendations

### French Documentation
All documentation (README.md, notebook markdown cells) should be in French as per project requirements.