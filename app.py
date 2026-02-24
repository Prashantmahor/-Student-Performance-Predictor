

import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load model & scaler
# ----------------------------
model = pickle.load(open("knn_model.pkl", "rb"))

try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    scaler_features = scaler.n_features_in_
except:
    scaler = None
    scaler_features = None

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🎓 Student Performance Predictor")
st.write("Predict student GPA using demographics and activities")

# ----------------------------
# Inputs
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 15, 25, 18)
    StudyTimeWeekly = st.slider("Study Time Weekly", 0.0, 40.0, 10.0)
    Absences = st.slider("Absences", 0, 50, 5)

with col2:
    Tutoring = st.selectbox("Tutoring", ["No", "Yes"])
    ParentalSupport = st.selectbox("Parental Support", ["Low", "Medium", "High"])
    Extracurricular = st.selectbox("Extracurricular", ["No", "Yes"])
    Sports = st.selectbox("Sports", ["No", "Yes"])
    Music = st.selectbox("Music", ["No", "Yes"])

GradeClass = st.slider("Grade Class", 0.0, 4.0, 2.0)

# ----------------------------
# Encoding
# ----------------------------
Tutoring = 1 if Tutoring == "Yes" else 0
Extracurricular = 1 if Extracurricular == "Yes" else 0
Sports = 1 if Sports == "Yes" else 0
Music = 1 if Music == "Yes" else 0

support_map = {"Low": 0, "Medium": 1, "High": 2}
ParentalSupport = support_map[ParentalSupport]

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict GPA"):

    # UI feature set (9 features from your notebook)
    features = np.array([[
        Age,
        StudyTimeWeekly,
        Absences,
        Tutoring,
        ParentalSupport,
        Extracurricular,
        Sports,
        Music,
        GradeClass
    ]])

    # ---------- ALIGN WITH SCALER ----------
    if scaler is not None:
        expected = scaler_features
        current = features.shape[1]

        if current < expected:
            padding = np.zeros((1, expected - current))
            features = np.hstack([features, padding])

        elif current > expected:
            features = features[:, :expected]

        features = scaler.transform(features)

    prediction = model.predict(features)[0]

    # GPA interpretation
    if prediction >= 3.5:
        label = "Excellent"
        color = "green"
    elif prediction >= 3.0:
        label = "Good"
        color = "blue"
    elif prediction >= 2.0:
        label = "Average"
        color = "orange"
    else:
        label = "At Risk"
        color = "red"

    st.markdown(
        f"<h2 style='color:{color}'>Predicted GPA: {prediction:.2f} ({label})</h2>",
        unsafe_allow_html=True
    )