"""Interactive Streamlit Clinical Decision Support Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.data.loader import get_feature_and_target_matrices, load_raw_heart_disease_data
from src.inference.predict import ClinicalRiskPredictor
from src.models.explain import explain_individual_patient_prediction

st.set_page_config(
    page_title="CardioRisk Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
)

ARTIFACT_PATH = Path("artifacts/heart_disease_pipeline.joblib")
RAW_DATA_PATH = Path("heartdisease_raw.data")


@st.cache_resource
def get_predictor():
    if not ARTIFACT_PATH.exists():
        return None
    return ClinicalRiskPredictor(ARTIFACT_PATH)


@st.cache_data
def get_baseline_data():
    df = load_raw_heart_disease_data(RAW_DATA_PATH)
    x, _ = get_feature_and_target_matrices(df)
    return x


st.title("🩺 CardioRisk - Clinical Heart Disease Risk Prediction")
st.markdown(
    "Real-time clinical decision support tool providing calibrated cardiovascular "
    "risk assessment and patient-specific explainability."
)

predictor = get_predictor()

if predictor is None:
    st.error(
        "Model artifact missing! Please run `python main.py` first to train and save."
    )
    st.stop()

baseline_df = get_baseline_data()

st.sidebar.header("📋 Patient Clinical Parameters")

with st.sidebar:
    st.subheader("Demographics & Vitals")
    age = st.slider("Age (years)", min_value=20, max_value=90, value=58)
    sex_opts = [("Male", 1.0), ("Female", 0.0)]
    sex = st.selectbox("Sex", options=sex_opts, format_func=lambda x: x[0])[1]
    trestbps = st.slider("Resting BP (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.slider("Serum Chol (mg/dl)", min_value=100, max_value=600, value=240)

    st.subheader("Cardiovascular Examination")
    cp_opts = [
        ("Typical Angina (1)", 1.0),
        ("Atypical Angina (2)", 2.0),
        ("Non-anginal Pain (3)", 3.0),
        ("Asymptomatic (4)", 4.0),
    ]
    cp = st.selectbox("Chest Pain Type", options=cp_opts, format_func=lambda x: x[0])[1]

    thalach = st.slider("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150)
    exang_opts = [("No", 0.0), ("Yes", 1.0)]
    exang = st.selectbox(
        "Exercise Angina", options=exang_opts, format_func=lambda x: x[0]
    )[1]
    oldpeak = st.slider(
        "ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1
    )
    slope_opts = [
        ("Upsloping (1)", 1.0),
        ("Flat (2)", 2.0),
        ("Downsloping (3)", 3.0),
    ]
    slope = st.selectbox(
        "ST Slope", options=slope_opts, format_func=lambda x: x[0]
    )[1]

    st.subheader("Diagnostic Indicators")
    fbs_opts = [("No (False)", 0.0), ("Yes (True)", 1.0)]
    fbs = st.selectbox(
        "FBS > 120 mg/dl", options=fbs_opts, format_func=lambda x: x[0]
    )[1]
    ecg_opts = [
        ("Normal (0)", 0.0),
        ("ST-T Abnormality (1)", 1.0),
        ("LV Hypertrophy (2)", 2.0),
    ]
    restecg = st.selectbox(
        "Resting ECG", options=ecg_opts, format_func=lambda x: x[0]
    )[1]
    ca = st.selectbox(
        "Major Fluoroscopy Vessels", options=[0.0, 1.0, 2.0, 3.0]
    )
    thal_opts = [
        ("Normal (3)", 3.0),
        ("Fixed Defect (6)", 6.0),
        ("Reversible Defect (7)", 7.0),
    ]
    thal = st.selectbox(
        "Thallium Scintigraphy", options=thal_opts, format_func=lambda x: x[0]
    )[1]

patient_payload = {
    "age": float(age),
    "sex": float(sex),
    "cp": float(cp),
    "trestbps": float(trestbps),
    "chol": float(chol),
    "fbs": float(fbs),
    "restecg": float(restecg),
    "thalach": float(thalach),
    "exang": float(exang),
    "oldpeak": float(oldpeak),
    "slope": float(slope),
    "ca": float(ca),
    "thal": float(thal),
}

patient_df = pd.DataFrame([patient_payload])

# Run inference
res = predictor.predict_patient_risk(patient_df)
explanation = explain_individual_patient_prediction(
    predictor.pipeline, patient_df, baseline_df
)

# Dashboard Layout
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader("Risk Stratification Outcome")
    prob_pct = res["risk_probability"] * 100
    is_high = res["risk_tier"] == "High Risk"

    if is_high:
        st.error(f"### ⚠️ {res['risk_tier']}")
    else:
        st.success(f"### ✅ {res['risk_tier']}")

    st.progress(res["risk_probability"])
    st.metric("Predicted Disease Probability", f"{prob_pct:.1f}%")
    st.caption(f"Decision Threshold Applied: {res['threshold_applied']:.2f}")

with col2:
    st.subheader("Clinical Recommendation")
    if is_high:
        st.warning(res["actionable_recommendation"])
        st.markdown(
            "- **Next Steps:** Order diagnostic angiography or stress test.\n"
            "- Initiate aggressive risk-factor reduction therapy."
        )
    else:
        st.info(res["actionable_recommendation"])
        st.markdown(
            "- **Next Steps:** Maintain annual cardiovascular follow-ups.\n"
            "- Adhere to preventive lifestyle and dietary guidelines."
        )

with col3:
    st.subheader("Model Validation Metrics")
    meta = predictor.metadata.get("metrics", {})
    if meta:
        st.metric("Sensitivity (Recall)", f"{meta.get('sensitivity', 0):.1%}")
        st.metric("Specificity", f"{meta.get('specificity', 0):.1%}")
        st.metric("ROC-AUC", f"{meta.get('roc_auc', 0):.3f}")

st.divider()

st.subheader("🔍 Patient-Level Explainability (Top Clinical Drivers)")
st.caption("Attribution quantifying feature impact relative to the baseline cohort.")

factors = explanation["top_contributing_factors"]
factors_df = pd.DataFrame(factors)
factors_df["Direction"] = factors_df["impact"].apply(
    lambda x: "Increases Risk ⬆️" if x > 0 else "Decreases Risk ⬇️"
)
factors_df["Impact Magnitude"] = factors_df["impact"].abs()
factors_df = factors_df.sort_values(by="Impact Magnitude", ascending=False)

exp_col1, exp_col2 = st.columns([1.5, 1])
with exp_col1:
    st.dataframe(
        factors_df[["feature", "impact", "Direction"]].rename(
            columns={"feature": "Biomarker", "impact": "Attribution Impact"}
        ),
        use_container_width=True,
    )
with exp_col2:
    st.bar_chart(factors_df.set_index("feature")["impact"])
