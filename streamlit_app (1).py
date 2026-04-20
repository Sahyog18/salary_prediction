
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("Salary Prediction for Data Science Lovers")
st.write("This app predicts salary based on various input features using a Linear Regression model.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('best_model.pkl', 'rb'))
        return model
    except FileNotFoundError:
        st.error("Error: best_model.pkl not found. Please ensure it's in the same directory as this app.")
        st.stop()

# Load dataset (for dropdown values )
try:
    df = pd.read_csv("Salary_Dataset_DataScienceLovers.csv")
except:
    st.error("Dataset not found")
    st.stop()

    

model = load_model()

st.header("Enter Employee Details")

# Input features (assuming numerical input for encoded categorical features)
rating = st.number_input('Rating (e.g., 3.8)', min_value=1.0, max_value=5.0, value=3.8, step=0.1)
company_name = st.number_input('Company Name (Encoded Numerical Value)', min_value=0, value=5554)
job_title = st.number_input('Job Title (Encoded Numerical Value)', min_value=0, value=23)
salaries_reported = st.number_input('Salaries Reported', min_value=1.0, value=3.0, step=1.0)
location = st.number_input('Location (Encoded Numerical Value)', min_value=0, value=0)
employment_status = st.number_input('Employment Status (Encoded Numerical Value)', min_value=0, value=0)
job_roles = st.number_input('Job Roles (Encoded Numerical Value)', min_value=0, value=0)

# Make prediction
if st.button('Predict Salary'):
    # Create a DataFrame for prediction
    features = pd.DataFrame([[
        rating,
        company_name,
        job_title,
        salaries_reported,
        location,
        employment_status,
        job_roles
    ]], columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

    prediction = model.predict(features)[0]

    st.success(f"Predicted Salary: ₹{prediction:,.2f}")
    st.write("**Note:** For 'Company Name', 'Job Title', 'Location', 'Employment Status', and 'Job Roles', please enter the numerical values that correspond to your data's label encoding. The app assumes you have the correct encoded values.")
