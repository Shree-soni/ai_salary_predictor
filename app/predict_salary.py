import joblib
import numpy as np

# Load model and encoders
model = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\salary_model.pkl")
encoder = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\encoder.pkl")
mlb = joblib.load(r"C:\Users\ADMIN\Desktop\ai_salary_predictor\models\mlb.pkl")

def predict_salary(experience_level, job_title, company_location, required_skills):
    # Encode categorical features
    cat_input = encoder.transform([[experience_level, job_title, company_location]])
    
    # Format skills (assumes comma-separated string input)
    skill_list = [skill.strip() for skill in required_skills.split(",")]
    skill_input = mlb.transform([skill_list])
    
    # Combine both inputs
    final_input = np.hstack((cat_input, skill_input))
    
    # Predict salary (USD to INR)
    salary_usd = model.predict(final_input)[0]
    salary_inr = salary_usd * 83.0  # USD to INR
    return round(salary_inr, 2)
