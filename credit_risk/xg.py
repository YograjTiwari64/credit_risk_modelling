import streamlit as st
import pickle
import numpy as np

# Load the trained models
with open('rf_classifier.pkl', 'rb') as rf_file:
    rf_classifier = pickle.load(rf_file)

with open('xgb_classifier.pkl', 'rb') as xgb_file:
    xgb_classifier = pickle.load(xgb_file)

with open('dt_model_xgb.pkl', 'rb') as dt_file:
    dt_classifier = pickle.load(dt_file)

# Define the input fields for the web app
st.title("Credit Risk Modeling")

st.header("Input the following values to get a prediction")

# Input fields
ROI = st.number_input("ROI (Rate of Interest)", min_value=0.0, max_value=100.0, step=0.01)
ROC = st.number_input("ROC (Rate on Credit)", min_value=0.0, max_value=100.0, step=0.01)
CASA = st.number_input("CASA (Credit-Adjusted Spread)", min_value=0.0, max_value=100.0, step=0.01)
RATE_INTEREST = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, step=0.01)
EMI = st.number_input("EMI (Equated Monthly Installment)", min_value=0.0, max_value=100000.0, step=100.0)
AGE = st.number_input("AGE", min_value=18, max_value=100, step=1)
Total_TL = st.number_input("Total Trade Lines", min_value=0)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
Tot_Active_TL = st.number_input("Total Active Trade Lines", min_value=0)
Tot_Closed_TL = st.number_input("Total Closed Trade Lines", min_value=0)

# Model selection for prediction
model_option = st.selectbox("Choose the model for prediction", ["Random Forest", "XGBoost", "Decision Tree"])

# Button to trigger prediction
if st.button("Predict Credit Risk"):
    # Prepare the feature vector for prediction
    input_data = [ROI, ROC, CASA, RATE_INTEREST, EMI, AGE, Total_TL, Credit_Score, Tot_Active_TL, Tot_Closed_TL]

    # Assuming the model was trained on 54 features, you need to append default values to reach 54 features
    default_values = [0] * (54 - len(input_data))  # Placeholder for the remaining features
    input_data += default_values

    input_data = np.array(input_data).reshape(1, -1)  # Reshape for prediction

    # Model prediction based on selection
    if model_option == "Random Forest":
        prediction = rf_classifier.predict(input_data)
    elif model_option == "XGBoost":
        prediction = xgb_classifier.predict(input_data)
    else:
        prediction = dt_classifier.predict(input_data)

    # Display the result
    if prediction[0] == 1:
        st.success("The credit is likely to be approved!")
    else:
        st.error("The credit is likely to be rejected!")

# Optional: Display the selected inputs
st.write("Selected Input values:")
st.write(f"ROI: {ROI}, ROC: {ROC}, CASA: {CASA}, Interest Rate: {RATE_INTEREST}, EMI: {EMI}, AGE: {AGE}, Total Trade Lines: {Total_TL}, Credit Score: {Credit_Score}, Total Active Trade Lines: {Tot_Active_TL}, Total Closed Trade Lines: {Tot_Closed_TL}")
#streamlit run xg.py