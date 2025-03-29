import streamlit as st
import joblib
import numpy as np

# Load the trained model
rf = joblib.load('tuned_random_forest_model.pkl')

# Custom Styles
st.set_page_config(page_title="Health Cost Predictor", page_icon="💰", layout="centered")

st.title("💰 Health Insurance Cost Predictor")
st.markdown("### Predict your expected medical costs based on key health indicators.")

# Sidebar for instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.write("1. Enter your details on the left.")
    st.write("2. Click 'Predict Costs' to estimate your medical expenses.")
    st.write("3. Results will be displayed below.")

# Layout for inputs
col1, col2 = st.columns(2)

# User Inputs
with col1:
    age = st.slider("👤 Enter your Age:", min_value=0, max_value=120, value=30, step=1)
    st.markdown(f" **You selected:** `{age} years`")

    height = st.slider("📏 Enter your Height (cm):", min_value=50, max_value=250, value=170, step=1)
    st.markdown(f" **You selected:** `{height} cm`")

    weight = st.slider("⚖️ Enter your Weight (kg):", min_value=10, max_value=500, value=70, step=1)
    st.markdown(f" **You selected:** `{weight} kg`")

    children = st.slider("👶 How many children live in your household?", min_value=0, max_value=10, value=0, step=1)
    st.markdown(f" **You selected:** `{children} children`")

with col2:
    sex = st.radio("Gender", ["Male", "Female"])
    smoker = st.radio("Do you Smoke?", ["No", "Yes"])
    region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"])

# Preprocessing function
def preprocess(age, sex, height, weight, smoker, region, children):
    age = float(age)
    height = float(height)
    weight = float(weight)
    children = int(children)

    # BMI Calculation
    bmi = round(weight / ((height / 100) ** 2), 2)
    bmi_category_encoded = 0 if bmi <= 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3

    # Encoding Categorical Features
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    age_group_encoded = 0 if age <= 25 else 1 if age <= 40 else 2 if age <= 64 else 3
    region_mapping = {"northwest": [0, 0, 1], "southeast": [0, 1, 0], "southwest": [1, 0, 0], "northeast": [0, 0, 0]}
    region_encoded = region_mapping[region]

    features = np.array([age_group_encoded, sex, bmi_category_encoded, smoker, *region_encoded, children]).reshape(1, -1)
    return features

# Prediction Button
if st.button("Predict Costs"):
    features = preprocess(age, sex, height, weight, smoker, region, children)
    prediction = rf.predict(features)[0]
    
    st.success(f"Your Estimated Medical Costs: **${prediction:,.2f}**")
