"""End-to-end synthetic employee performance prediction pipeline."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
IMAGES = ROOT / "images"
for folder in (DATA, OUTPUTS, MODELS, IMAGES):
    folder.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "perf_band_next"

def create_dataset(n=1200):
    """Create fictional data; target is derived from simulated work signals."""
    rng = np.random.default_rng(RANDOM_STATE)
    departments = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "HR"]
    levels = ["Junior", "Mid", "Senior", "Lead"]
    salary_bands = ["Band 1", "Band 2", "Band 3", "Band 4"]
    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
        "department": rng.choice(departments, n),
        "job_level": rng.choice(levels, n, p=[.30, .38, .23, .09]),
        "salary_band": rng.choice(salary_bands, n),
        "experience_years": rng.integers(0, 26, n),
        "projects_count": rng.integers(1, 10, n),
        "avg_task_delay_days": np.clip(rng.normal(3.5, 3.2, n), 0, 18).round(1),
        "on_time_delivery_rate": rng.beta(5, 2, n).round(3),
        "bug_count": rng.poisson(3.0, n),
        "code_review_score": np.clip(rng.normal(3.4, .8, n), 1, 5).round(2),
        "training_hours": np.clip(rng.normal(18, 10, n), 0, 60).round(1),
        "certifications_count": rng.poisson(1.2, n),
        "sick_days": rng.poisson(3, n),
        "unplanned_absences": rng.poisson(2, n),
        "peer_feedback_score": np.clip(rng.normal(3.5, .7, n), 1, 5).round(2),
        "manager_score": np.clip(rng.normal(3.4, .75, n), 1, 5).round(2),
        "kudos_count": rng.poisson(2, n),
        "billable_hours_ratio": rng.beta(5, 3, n).round(3),
    })
    # A transparent synthetic label formula, not a real-world HR rule.
    score = (
        28 * df["on_time_delivery_rate"]
        + 5.5 * df["code_review_score"]
        + 4.0 * df["peer_feedback_score"]
        + 4.0 * df["manager_score"]
        + 0.22 * df["training_hours"].clip(upper=40)
        + 1.5 * df["certifications_count"].clip(upper=4)
        + 0.8 * df["kudos_count"].clip(upper=6)
        + 1.2 * df["projects_count"]
        - 1.15 * df["avg_task_delay_days"]
        - 1.1 * df["bug_count"]
        - 1.0 * df["unplanned_absences"]
        - 0.35 * df["sick_days"]
        + rng.normal(0, 7, n)
    )
    low_cut, high_cut = score.quantile([.33, .68])
    df[TARGET] = np.where(score < low_cut, "Low",
                   np.where(score < high_cut, "Medium", "High"))
    # Introduce a few realistic missing values for cleaning pipeline demonstration.
    for col in ["training_hours", "peer_feedback_score", "avg_task_delay_days"]:
        idx = rng.choice(df.index, size=max(1, int(n * .025)), replace=False)
        df.loc[idx, col] = np.nan
    return df

def run_pipeline():
    df = create_dataset()
    df.to_csv(DATA / "employee_performance.csv", index=False)
    df.head(20).to_csv(OUTPUTS / "dataset_preview.csv", index=False)

    # Quality report
    quality = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "target_distribution": {k: int(v) for k, v in df[TARGET].value_counts().items()},
    }
    (OUTPUTS / "data_quality_report.json").write_text(json.dumps(quality, indent=2))

    # EDA charts
    counts = df[TARGET].value_counts().reindex(["Low", "Medium", "High"])
    ax = counts.plot(kind="bar", rot=0, title="Synthetic Performance Band Distribution")
    ax.set_xlabel("Performance band"); ax.set_ylabel("Employee count")
    plt.tight_layout(); plt.savefig(IMAGES / "01_performance_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots()
    df.boxplot(column="training_hours", by=TARGET, ax=ax)
    fig.suptitle(""); ax.set_title("Training Hours by Synthetic Performance Band")
    ax.set_xlabel("Performance band"); ax.set_ylabel("Training hours")
    plt.tight_layout(); plt.savefig(IMAGES / "02_training_by_band.png", dpi=150); plt.close()

    numeric = df.select_dtypes(include="number").drop(columns=[], errors="ignore")
    corr_cols = ["experience_years", "projects_count", "avg_task_delay_days",
                 "on_time_delivery_rate", "bug_count", "code_review_score",
                 "training_hours", "certifications_count", "sick_days",
                 "unplanned_absences", "peer_feedback_score", "manager_score",
                 "kudos_count", "billable_hours_ratio"]
    fig, ax = plt.subplots(figsize=(11, 8))
    numeric[corr_cols].corr().round(2).pipe(lambda c: __import__("seaborn").heatmap(c, annot=True, cmap="coolwarm", center=0, ax=ax))
    ax.set_title("Numeric Feature Correlation")
    plt.tight_layout(); plt.savefig(IMAGES / "03_correlation_heatmap.png", dpi=150); plt.close()

    # Keep identifiers and target out of the feature matrix.
    X = df.drop(columns=["employee_id", TARGET])
    y = df[TARGET]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, num_cols),
        ("categorical", categorical_pipe, cat_cols),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=RANDOM_STATE, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": "RandomForestClassifier",
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "note": "Synthetic data and synthetic labels; metrics are educational and do not represent real-world HR performance."
    }
    (OUTPUTS / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(OUTPUTS / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, pred, zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=["Low", "Medium", "High"], cmap="Blues", values_format="d"
    )
    disp.ax_.set_title("Confusion Matrix — Hold-out Test Set")
    plt.tight_layout(); plt.savefig(IMAGES / "04_confusion_matrix.png", dpi=150); plt.close()

    # Permutation importance measured on held-out rows; original columns stay readable.
    perm = permutation_importance(model, X_test, y_test, n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1)
    imp = pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(OUTPUTS / "feature_importance.csv", index=False)
    top = imp.head(12).sort_values("importance")
    ax = top.plot.barh(x="feature", y="importance", legend=False, title="Top Features — Permutation Importance")
    ax.set_xlabel("Mean decrease in macro-F1 when shuffled"); ax.set_ylabel("Feature")
    plt.tight_layout(); plt.savefig(IMAGES / "05_feature_importance.png", dpi=150); plt.close()

    joblib.dump(model, MODELS / "employee_performance_model.joblib")
    # Example predictions for reproducible proof.
    sample = X_test.head(10).copy()
    sample["actual_band"] = y_test.loc[sample.index]
    sample["predicted_band"] = model.predict(X_test.head(10))
    sample.to_csv(OUTPUTS / "sample_predictions.csv", index=False)

    print("\\n=== Employee Performance Predictor: run complete ===")
    print(f"Dataset: {DATA / 'employee_performance.csv'}")
    print(f"Rows: {len(df)} | Features used: {X.shape[1]}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("\\nSaved model: models/employee_performance_model.joblib")
    print("Saved charts: images/")
    print("Saved reports: outputs/")

if __name__ == "__main__":
    run_pipeline()
