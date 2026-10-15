"""Interactive demo dashboard for the synthetic employee performance model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "employee_performance_model.joblib"
DATA_PATH = ROOT / "data" / "employee_performance.csv"

st.set_page_config(page_title="Employee Performance Predictor", page_icon="📊", layout="wide")
st.title("📊 Employee Performance Predictor")
st.caption("Educational people-analytics simulation using synthetic data — not for employment decisions.")
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or dataset not found. Run `python main.py` in the project folder first.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
st.info("All records are fictional. A predicted band is a model output, not a judgment of an employee's value.")
left, right = st.columns(2)
left.metric("Synthetic employee records", f"{len(df):,}")
right.metric("Departments represented", df["department"].nunique())

st.subheader("Explore the dataset")
st.dataframe(df.drop(columns=["employee_id"]).head(15), use_container_width=True)

st.subheader("Predict a hypothetical employee")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        department = st.selectbox("Department", sorted(df["department"].unique()))
        job_level = st.selectbox("Job level", ["Junior", "Mid", "Senior", "Lead"])
        salary_band = st.selectbox("Salary band", ["Band 1", "Band 2", "Band 3", "Band 4"])
        experience_years = st.slider("Experience (years)", 0, 30, 4)
        projects_count = st.slider("Projects count", 0, 15, 4)
    with c2:
        avg_task_delay_days = st.slider("Average task delay (days)", 0.0, 20.0, 3.0, .5)
        on_time_delivery_rate = st.slider("On-time delivery rate", 0.0, 1.0, .75, .01)
        bug_count = st.slider("Bug count", 0, 20, 3)
        code_review_score = st.slider("Code review score (1–5)", 1.0, 5.0, 3.5, .1)
        training_hours = st.slider("Training hours", 0.0, 80.0, 20.0, 1.0)
    with c3:
        certifications_count = st.slider("Certifications", 0, 10, 1)
        sick_days = st.slider("Sick days", 0, 30, 3)
        unplanned_absences = st.slider("Unplanned absences", 0, 20, 1)
        peer_feedback_score = st.slider("Peer feedback (1–5)", 1.0, 5.0, 3.5, .1)
        manager_score = st.slider("Previous manager feedback (1–5)", 1.0, 5.0, 3.5, .1)
        kudos_count = st.slider("Recognition / kudos count", 0, 15, 2)
        billable_hours_ratio = st.slider("Billable-hours ratio", 0.0, 1.0, .65, .01)
    submitted = st.form_submit_button("Predict hypothetical performance band")

if submitted:
    row = pd.DataFrame([{
        "department": department, "job_level": job_level, "salary_band": salary_band,
        "experience_years": experience_years, "projects_count": projects_count,
        "avg_task_delay_days": avg_task_delay_days, "on_time_delivery_rate": on_time_delivery_rate,
        "bug_count": bug_count, "code_review_score": code_review_score,
        "training_hours": training_hours, "certifications_count": certifications_count,
        "sick_days": sick_days, "unplanned_absences": unplanned_absences,
        "peer_feedback_score": peer_feedback_score, "manager_score": manager_score,
        "kudos_count": kudos_count, "billable_hours_ratio": billable_hours_ratio
    }])
    prediction = model.predict(row)[0]
    st.subheader(f"Predicted band: {prediction}")
    if hasattr(model, "predict_proba"):
        probs = pd.DataFrame({"Band": model.classes_, "Model probability": model.predict_proba(row)[0]})
        st.bar_chart(probs.set_index("Band"))
    actions = []
    if on_time_delivery_rate < .65 or avg_task_delay_days > 5:
        actions.append("Review workload, task dependencies, and sprint planning with the employee.")
    if code_review_score < 3 or bug_count >= 6:
        actions.append("Offer pair programming, code review coaching, or a quality-focused learning plan.")
    if training_hours < 10 or certifications_count == 0:
        actions.append("Discuss relevant training or certification options aligned with the employee's goals.")
    if peer_feedback_score < 3 or manager_score < 3:
        actions.append("Arrange a supportive one-to-one to clarify feedback and agree on measurable goals.")
    if not actions:
        actions.append("Discuss career goals and maintain a balanced development plan; do not infer that no support is needed.")
    st.markdown("**Suggested coaching discussion topics**")
    for action in actions[:3]:
        st.write(f"- {action}")
    st.caption("Recommendations are generic rules for demonstration. Validate context with the employee; do not use this output alone for appraisal, promotion, pay, or termination.")
