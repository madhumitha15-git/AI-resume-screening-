import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

tenure = st.number_input("Tenure (months)")
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict"):

    input_data = np.array([[tenure, monthly_charges, total_charges]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer will Stay")