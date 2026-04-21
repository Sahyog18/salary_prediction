import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Salary Predictor", layout="centered")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    color: #00f5c4;
}
.stButton>button {
    background-color: #00f5c4;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stSelectbox, .stNumberInput {
    background-color: #262730;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1>💰 Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Predict salaries using Machine Learning")

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
    st.error("❌ Dataset not found")
    st.stop()

# Encoders
encoders = {}
categorical_cols = [
    'Company Name', 'Job Title', 'Location',
    'Employment Status', 'Job Roles'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Layout using columns
st.markdown("## 📌 Enter Details")

col1, col2 = st.columns(2)

with col1:
    rating = st.number_input("⭐ Rating", 1.0, 5.0, 3.8)
    company_name = st.selectbox("🏢 Company", encoders['Company Name'].classes_)
    job_title = st.selectbox("💼 Job Title", encoders['Job Title'].classes_)

with col2:
    location = st.selectbox("📍 Location", encoders['Location'].classes_)
    employment_status = st.selectbox("📄 Employment Status", encoders['Employment Status'].classes_)
    job_roles = st.selectbox("🧑‍💻 Job Role", encoders['Job Roles'].classes_)

salaries_reported = st.number_input("📊 Salaries Reported", 1, 100, 3)

st.markdown("---")

# Prediction Button
if st.button("🚀 Predict Salary"):
    try:
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

        st.markdown("## 🎯 Result")
        st.success(f"💰 Estimated Salary: ₹ {prediction:,.0f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.markdown("### ⚡ Built with Streamlit | AI Project")
