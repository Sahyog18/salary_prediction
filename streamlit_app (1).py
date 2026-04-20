import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💰 Salary Prediction App")
st.write("Predict salary using categorical inputs (dropdowns).")

# Load model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Load dataset
try:
    df = pd.read_csv("Salary_Dataset_DataScienceLovers.csv")
except:
    st.error("Dataset not found")
    st.stop()

st.header("📌 Enter Employee Details")

# Create encoders for each categorical column
encoders = {}

categorical_cols = [
    'Company Name', 'Job Title', 'Location',
    'Employment Status', 'Job Roles'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Dropdown inputs (original values)
rating = st.number_input('Rating', min_value=1.0, max_value=5.0, value=3.8)

company_name = st.selectbox(
    'Company Name', encoders['Company Name'].classes_
)

job_title = st.selectbox(
    'Job Title', encoders['Job Title'].classes_
)

location = st.selectbox(
    'Location', encoders['Location'].classes_
)

employment_status = st.selectbox(
    'Employment Status', encoders['Employment Status'].classes_
)

job_roles = st.selectbox(
    'Job Roles', encoders['Job Roles'].classes_
)

salaries_reported = st.number_input(
    'Salaries Reported', min_value=1, value=3
)

# Prediction
if st.button('🚀 Predict Salary'):
    try:
        # Encode selected values
        features = pd.DataFrame([{
            'Rating': rating,
            'Company Name': encoders['Company Name'].transform([company_name])[0],
            'Job Title': encoders['Job Title'].transform([job_title])[0],
            'Salaries Reported': salaries_reported,
            'Location': encoders['Location'].transform([location])[0],
            'Employment Status': encoders['Employment Status'].transform([employment_status])[0],
            'Job Roles': encoders['Job Roles'].transform([job_roles])[0]
        }])

        prediction = model.predict(features)[0]

        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
