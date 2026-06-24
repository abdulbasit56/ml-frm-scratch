import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "Linear Regression"))
sys.path.append(str(Path(__file__).resolve().parent / "Logistic Regression"))

from regressor import load_model as load_reg_model, predict_new as predict_admission
from classifier import load_model as load_clf_model, predict_new as predict_pass

import traceback



st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")
st.title(" Student Success Predictor")
st.caption("Linear & Logistic Regression — built from scratch with NumPy")

@st.cache_resource
def get_regression_model():
    return load_reg_model(prefix='admission_model')

@st.cache_resource
def get_classifier_model():
    return load_clf_model(prefix='student_model')

    
tab1, tab2 = st.tabs(["Admission Chance", "Pass / Fail"])


# ── TAB 1: LINEAR REGRESSION ─────────────────────────────────────────────────
with tab1:
    st.subheader("Graduate Admission Probability")
    st.write("Enter a student's academic profile to predict their chance of admission (0–1).")

    col1, col2 = st.columns(2)
    with col1:
        gre   = st.number_input("GRE Score",           min_value=260, max_value=340, value=310)
        toefl = st.number_input("TOEFL Score",         min_value=0,   max_value=120, value=100)
        rating = st.number_input("University Rating",  min_value=1,   max_value=5,   value=3)
    with col2:
        sop   = st.number_input("SOP Strength",        min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        lor   = st.number_input("LOR Strength",        min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        cgpa  = st.number_input("CGPA (out of 10)",    min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        research = st.selectbox("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("Predict Admission Chance", use_container_width=True):
        try:
            w, b, mu, sigma = get_regression_model()
            result = predict_admission([gre, toefl, rating, sop, lor, cgpa, research], w, b, mu, sigma)
            result = float(np.clip(result, 0, 1))
            st.metric("Predicted Chance of Admission", f"{result:.2%}")
            if result >= 0.75:
                st.success("Strong chance of admission.")
            elif result >= 0.5:
                st.warning("Moderate chance — consider strengthening your profile.")
            else:
                st.error("Low chance — significant improvement needed.")
        except FileNotFoundError:
            st.error("Model weights not found. Run `Linear Regression/train_save.py` first.")


# ── TAB 2: LOGISTIC REGRESSION 
with tab2:
    st.subheader("Student Pass / Fail Classifier")
    st.write("Enter a student's study profile to predict whether they will pass or fail.")

    col3, col4 = st.columns(2)
    with col3:
        attendance = st.slider("Attendance (%)",          min_value=0,  max_value=100, value=75)
        homework   = st.slider("Homework Completion (%)", min_value=0,  max_value=100, value=75)
    with col4:
        midterm    = st.slider("Midterm Score (%)",       min_value=0,  max_value=100, value=65)
        study_hrs  = st.slider("Study Hours / Week",      min_value=0,  max_value=20,  value=7)

    if st.button("Predict Pass / Fail", use_container_width=True):
        try:
            w, b, mu, sigma = get_classifier_model()
            result = predict_pass([attendance, homework, midterm, study_hrs], w, b, mu, sigma)
            if result == 1:
                st.success("Prediction: PASS!")
            else:
                st.error("Prediction: FAIL!!")
        except FileNotFoundError:
            st.error("Model weights not found. Run `Logistic Regression/train_save.py` first.")