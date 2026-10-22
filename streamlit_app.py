import streamlit as st
import joblib
import numpy as np

# Load models and encoders
titanic_model   = joblib.load('titanic_model.pkl')
le_sex          = joblib.load('titanic_le_sex.pkl')
le_embarked     = joblib.load('titanic_le_embarked.pkl')
heart_model     = joblib.load('heart_model.pkl')
le_cp           = joblib.load('heart_le_cp.pkl')

st.set_page_config(page_title="ML Prediction App", page_icon="🤖")

task = st.sidebar.radio("Select Task", ["Task 1: Titanic Survival", "Task 2: Heart Disease"])

# ── TASK 1 ───────────────────────────────────────────────────────────────────
if task == "Task 1: Titanic Survival":
    st.title("🚢 Titanic Survival Predictor")
    st.write("Enter passenger details to predict survival.")

    pclass   = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex      = st.selectbox("Sex", ["male", "female"])
    age      = st.slider("Age", 1, 80, 30)
    sibsp    = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch    = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare     = st.number_input("Fare ($)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    if st.button("Predict Survival"):
        sex_enc = le_sex.transform([sex])[0]
        emb_enc = le_embarked.transform([embarked])[0]
        features = np.array([[pclass, sex_enc, age, sibsp, parch, fare, emb_enc]])
        result = titanic_model.predict(features)[0]
        if result == 1:
            st.success("✅ This passenger **Survived**!")
        else:
            st.error("❌ This passenger **Did Not Survive**.")

# ── TASK 2 ───────────────────────────────────────────────────────────────────
else:
    st.title("❤️ Heart Disease Predictor")
    st.write("Enter patient details to predict heart disease.")

    age     = st.slider("Age (years)", 20, 80, 55)
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    cp      = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                           format_func=lambda x: {
                               1: "1 - Typical Angina",
                               2: "2 - Atypical Angina",
                               3: "3 - Non-Anginal Pain",
                               4: "4 - Asymptomatic"}[x])

    if st.button("Predict Heart Disease"):
        cp_enc   = le_cp.transform([cp])[0]
        features = np.array([[age, thalach, oldpeak, cp_enc]])
        result   = heart_model.predict(features)[0]
        if result == 1:
            st.error("🔴 **Heart Disease Detected**")
        else:
            st.success("🟢 **No Heart Disease Detected**")
