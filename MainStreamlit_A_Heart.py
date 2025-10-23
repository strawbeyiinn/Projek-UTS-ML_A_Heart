import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
model = load_model("BestModel_CLF_GradientBoosting_Heart.pkl")

st.title("🫀Heart Disease Prediction🫀")

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age (tahun)", min_value=1, max_value=120, value=50)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=250, value=120)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
with col2:
    RestingECG = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

submitted = st.button("Prediksi")

MAPS = {
    "Sex": {"Female": 0, "Male": 1},
    "ExerciseAngina": {"No": 0, "Yes": 1},
    "FastingBS": {"No": 0, "Yes": 1},
    "ChestPainType": {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3},
    "RestingECG": {"LVH": 0, "Normal": 1, "ST": 2},
    "ST_Slope": {"Down": 0, "Flat": 1, "Up": 2},
}

NUMERIC_COLS = [
    "Age","RestingBP","Cholesterol","MaxHR","Oldpeak",
    "Sex","ExerciseAngina","FastingBS","ChestPainType","RestingECG","ST_Slope"
]

def encode_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, m in MAPS.items():
        if col in df.columns and isinstance(df.at[0, col], str):
            if df.at[0, col] not in m:
                raise ValueError(f"Nilai '{df.at[0, col]}' pada kolom '{col}' tidak dikenal. Pilihan: {list(m.keys())}")
            df.at[0, col] = m[df.at[0, col]]
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise")
    return df

if submitted:
    raw_input = pd.DataFrame({
        "Age": [Age],
        "Sex": [Sex],
        "ChestPainType": [ChestPainType],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "FastingBS": [FastingBS],
        "RestingECG": [RestingECG],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [ExerciseAngina],
        "Oldpeak": [Oldpeak],
        "ST_Slope": [ST_Slope],
    })

    used_encoded = False
    try:
        yhat = model.predict(raw_input)[0]
        proba = model.predict_proba(raw_input)[0][1] if hasattr(model, "predict_proba") else None
    except Exception:
        input_num = encode_input(raw_input)
        yhat = model.predict(input_num)[0]
        proba = model.predict_proba(input_num)[0][1] if hasattr(model, "predict_proba") else None
        used_encoded = True

    st.subheader(" Hasil Prediksi 📊 ")
    if yhat == 1:
        st.error("🚨 Pasien berisiko memiliki penyakit jantung 🚨")
    else:
        st.success("✅ Pasien kemungkinan tidak memiliki penyakit jantung ✅")

    if proba is not None:
        st.write(f"Probabilitas positif: **{proba:.2%}**")

st.markdown("------------")
