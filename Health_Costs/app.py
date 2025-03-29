import streamlit as st
import joblib
import numpy as np
# Load model
rf = joblib.load('tuned_random_forest_model.pkl')

# feature engineering function
def preprocess(age,sex, height, weight, smoker, region, children):
    # Define bmi
    bmi = weight / ((height/100) ** 2)
    if  bmi <= 18.5:
        bmi_category_encoded = 0
    elif bmi < 25:
        bmi_category_encoded = 1
    elif bmi < 30:
        bmi_category_encoded = 2
    else:
        bmi_category_encoded = 3
    # Sex
    sex =  1 if sex == "male" else 0
    # Smoker
    smoker = 1 if smoker == "Yes" else 0
    # Age
    if age <= 25:
        age_group_encoded = 0
    elif age <= 40:
        age_group_encoded = 1
    elif age <= 64:
        age_group_encoded = 2
    else:
        age_group_encoded = 3
    # Region
    region_mapping = {"northwest":[0, 0, 1], "southeast": [0, 1, 0], "southwest": [1, 0, 0], "northeast":[0, 0, 0]}
    region_encoded = region_mapping[region]
    
    features = np.array([age_group_encoded, bmi_category_encoded, smoker,  *region_encoded, children]).reshape(1, -1)
    return features

# user inputs
age = st.number_input("Age", min_value=0, max_value=120, step=1)
height = st.number_input("Height (cm)", min_value = 50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value = 10, max_value=500, step=1)
sex = st.selectbox("Gender", ["male", "female"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest", "northeast"])
children = st.number_input("Number of children", min_value=0, max_value=10, step=1)

# Make predictions
if st.button("Predict Costs"):
    features = preprocess(age, height, weight, sex, smoker, region, children)
    prediction = rf.predict(features)[0]
    st.write(f"Estimated Costs will be: ${prediction:.2f}")

    