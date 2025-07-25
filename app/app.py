import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\salary_model.pkl")
encoder = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\encoder.pkl")
mlb = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\mlb.pkl")

# Page config
st.set_page_config(
    page_title="AI Job Salary Predictor 💼",
    page_icon="💰",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f9fc;
            font-family: 'Segoe UI', sans-serif;
        }
        .salary-box {
            background-color: #d4edda;
            padding: 20px;
            border-radius: 12px;
            color: #155724;
            font-weight: bold;
            font-size: 24px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>💼 AI Job Salary Predictor 💰</h1>
    <p style='text-align: center; color: gray;'>Estimate your salary in Indian Rupees based on experience, title, skills & location</p>
    <hr style='border:1px solid #f2f2f2;'>
""", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)
with col1:
    experience = st.selectbox("👤 Experience Level", ["Entry", "Mid", "Senior", "Executive"])
    location = st.selectbox("📍 Company Location", ["India", "USA", "UK", "Remote", "Other"])

with col2:
    title = st.text_input("💼 Job Title", "Data Scientist")
    skills = st.text_input("🧠 Required Skills (comma separated)", "Python, TensorFlow")

# Predict
if st.button("🔮 Predict Salary"):
    input_cat = encoder.transform([[experience, title, location]])
    input_skills = mlb.transform([[s.strip() for s in skills.split(",")]])
    input_all = np.concatenate((input_cat, input_skills), axis=1)
    salary_usd = model.predict(input_all)[0]
    salary_inr = salary_usd * 83  # USD to INR approx.

    st.markdown(f"<div class='salary-box'>💰 Estimated Salary: ₹ {salary_inr:,.0f}</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📘 About")
st.sidebar.info("""
This app uses a machine learning model trained on AI job data to predict salaries based on multiple factors.

💻 Created with ❤️ by Bhagyashri Sonar
""")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
