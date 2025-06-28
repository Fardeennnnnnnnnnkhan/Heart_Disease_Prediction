import streamlit as st
import pandas as pd
import joblib 

model= joblib.load('../LR_heart.pkl')
scaler = joblib.load('../scaler.pkl')
expected_columns = joblib.load('../columns.pkl')

st.title("Heart Disease Prediction App")
st.write("This app predicts the presence of heart disease based on user input.")
st.markdown("Kindly fill in the details below:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
Fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
resting_bp = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0)
max_heart_rate = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
Oldpeak = st.slider("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of ST Segment", ["Up","Flat", "Down"])
if st.button("Predict"):
    # Step 1: Create base input dictionary (clean names)
    raw_input = {
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [Fasting_blood_sugar],
        'RestingECG': [resting_bp],  # NOTE: If you have separate variable for ECG, use that instead
        'MaxHR': [max_heart_rate],
        'ExerciseAngina': ['Y' if exercise_angina == 1 else 'N'],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [slope]
    }

    # Step 2: Create DataFrame
    input_df = pd.DataFrame(raw_input)

    # Step 3: One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Step 4: Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Step 5: Reorder columns
    input_encoded = input_encoded[expected_columns]

    # Step 6: Scale
    scaled_input = scaler.transform(input_encoded)

    # Step 7: Predict
    prediction = model.predict(scaled_input)

    # Step 8: Show result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
