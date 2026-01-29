# Mortality Risk Modeling

This project implements predictive models to estimate **mortality risk and probability** based on historical hospital patient data. The models are designed to support clinical decision-making, quality management, and exploratory analysis of patient outcomes.

## Project Overview

- The dataset used for development consists of **dummy patient data** that mimics the structure of over 30,000 real cases collected over the past 10 years.
- The repository is structured to allow seamless replacement of the dummy dataset with **real, sensitive hospital data (PHI)** when deployed in a secure environment.
- The workflow includes:
  - **Data preprocessing:** handling missing values, encoding categorical variables, transforming binary columns, and splitting data into training and testing sets.
  - **Modeling:** logistic regression models with baseline (L2), L1 (lasso), and optional elastic-net regularization.
  - **Evaluation:** standard classification metrics (precision, recall, F1-score, accuracy) and ROC-AUC.
  - **Model persistence:** trained models are saved in a separate folder (`models/`), with the option to load them for later inference.

## Folder Structure

```bash
project/
├─ app/ # Streamlit dashboard application
│ └─ app.py
├─ data/ # Contains dummy dataset for testing
├─ models/ # Saved model files (\*.joblib), not tracked in Git
├─ scripts/ # Preprocessing and training scripts
│ ├─ preprocess.py
│ └─ train_model.py
├─ config.py # Configuration and hyperparameters
├─ run_app.sh # Script to run the Streamlit app
└─ README.md
```

## Getting Started

1. Clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Run the training script on dummy data:

```bash
python scripts/train_model.py
```

4. Run the interactive Streamlit dashboard:

```bash
# Using the provided script
./run_app.sh

# Or directly with streamlit
streamlit run app/app.py
```

The dashboard allows you to interactively adjust parameters (season, time of day, unit, etc.) to see predicted mortality rates in real-time.

**Note**: This project uses dummy data for development purposes. All real patient data (PHI) must be handled in secure hospital environments and is not included in this repository.

## Future Work

- Hyperparameter tuning for optimal L1/L2 regularization.
- Extension to other predictive models or machine learning pipelines.
- Deployment on secure hospital systems for model evaluation with real patient data.
