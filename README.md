# Employee Performance Predictor using Data Analytics

A beginner-friendly people-analytics project that creates a synthetic HR dataset, explores workforce signals, trains a classification model, evaluates results, and provides an interactive Streamlit demonstration.

> **Responsible-use note:** All data is fictional and the target labels are generated from a transparent simulation formula. This project is for learning and portfolio demonstration only. It is not validated for real HR decisions. Predictions must not be used as the sole basis for promotion, compensation, performance management, or termination.

## Problem statement
Explore whether historical work and development signals can be used to classify a *simulated* next-cycle performance band (`Low`, `Medium`, `High`). The example demonstrates data preparation, model training, evaluation, feature importance, and communication of limitations.

## Features
- Reproducible synthetic dataset with 1,200 fictional records
- Missing-value and data-quality report
- Exploratory charts and numeric correlation heatmap
- Scikit-learn preprocessing pipeline with imputation and one-hot encoding
- Random Forest classifier with stratified train/test split
- Accuracy, macro-F1, classification report, confusion matrix
- Permutation feature importance
- Saved model and sample predictions
- Streamlit dashboard for hypothetical employee inputs
- Generic coaching and learning suggestions

## Architecture
```text
Synthetic employee records
          |
          v
Data validation + EDA -----> charts and quality report
          |
          v
Train/test split (stratified)
          |
          v
Preprocessing pipeline (imputation + encoding + scaling)
          |
          v
Random Forest classifier
          |
          +----> evaluation metrics + confusion matrix
          +----> permutation feature importance
          +----> saved model
                       |
                       v
              Streamlit demo dashboard
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, scikit-learn, Joblib, Streamlit.

## Setup (Windows PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
streamlit run app.py
```

If Python 3.13 is unavailable, use `py -m venv .venv` and activate the environment. On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
streamlit run app.py
```

## Outputs
- `data/employee_performance.csv` — generated fictional dataset
- `outputs/data_quality_report.json` — row counts, missing values and class balance
- `outputs/evaluation_metrics.json` and `outputs/classification_report.txt` — evaluation results
- `outputs/feature_importance.csv` — permutation importance
- `outputs/sample_predictions.csv` — example hold-out predictions
- `models/employee_performance_model.joblib` — trained model pipeline
- `images/` — generated EDA and evaluation charts

Metric values depend on the code and environment; run the pipeline to obtain the current output. High scores on synthetic data do not establish real-world validity.

## Run the dashboard
After `python main.py` completes, run `streamlit run app.py`. The local browser page lets you explore the generated dataset and submit hypothetical feature values to see a model prediction and generic coaching prompts.

## Repository structure
```text
Employee-Performance-Predictor/
├── data/
├── images/
├── models/
├── outputs/
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

Folders and generated outputs are created by `main.py` when it runs.

## Screenshots to add
Save screenshots in `screenshots/`:
1. `01_dataset_preview.png` — dataset opened in VS Code or notebook
2. `02_performance_distribution.png` — target distribution chart
3. `03_correlation_heatmap.png` — correlation chart
4. `04_confusion_matrix.png` — evaluation chart
5. `05_feature_importance.png` — feature importance chart
6. `06_terminal_results.png` — terminal evaluation output
7. `07_dashboard.png` — running Streamlit dashboard
8. `08_prediction_demo.png` — prediction and suggested coaching topics

## Limitations and ethical considerations
- The dataset and labels are synthetic and do not reflect any real employer.
- Historical scores, manager feedback, attendance and productivity signals can be noisy or context-dependent.
- Performance data can encode structural inequities. Do not include sensitive attributes in the model for employment decisions without a lawful, reviewed purpose.
- Any real deployment would require consent and privacy review, job-relevant validation, bias and accessibility audits, explainability, monitoring, employee recourse, and human oversight.
- Use outputs as prompts for supportive conversations, not definitive judgments.

## Future improvements
- Compare Logistic Regression and Random Forest with cross-validation
- Add a time-based evaluation design with genuine longitudinal data
- Add group-level fairness and error analysis where legally and ethically appropriate
- Add model versioning, drift monitoring, and a governed API
- Replace synthetic data only with appropriately authorized, privacy-protected data

## License
MIT (see `LICENSE`).

## Author
Student portfolio project — Data Analytics and Machine Learning.
