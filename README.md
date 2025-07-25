# 💼 AI Job Salary Predictor

This is a web-based Machine Learning application that predicts the salary of an AI-related job based on various features such as experience level, job title, required skills, and company location.

The predicted salary is displayed in Indian Rupees (INR) using a trained Linear Regression model.

## 🚀 Features

- Predicts AI job salary in **INR ₹**
- Handles multiple skills input
- Clean and interactive **Streamlit** UI
- Easy to use for recruiters, job seekers, and analysts

---

## 🧠 Machine Learning Workflow

- Data Preprocessing with Pandas and Scikit-learn
- Multi-label binarization for skills
- One-hot encoding for categorical features
- Trained using **Linear Regression**
- Model Evaluation with **R² Score**

---

## 🧰 Tech Stack

| Category      | Tools Used                        |
|---------------|------------------------------------|
| Language       | Python 3.x                         |
| Libraries      | pandas, numpy, matplotlib, seaborn |
| ML Toolkit     | scikit-learn, joblib               |
| Web Framework  | Streamlit                          |
| IDE            | VS Code / Jupyter Notebook         |

---

## 📁 Project Structure

ai_salary_predictor/
│
├── app/
│ ├── app.py # Streamlit frontend
│ └── predict_salary.py # Salary prediction logic
│
├── data/
│ └── ai_jobs.csv # Cleaned dataset
│
├── models/
│ ├── salary_model.pkl # Trained ML model
│ ├── encoder.pkl # OneHotEncoder for categorical features
│ └── mlb.pkl # MultiLabelBinarizer for skills
│
├── notebook/
│ └── salary_model.ipynb # EDA + Model training
│
└── README.md # You're reading it!

## ⚙️ How to Run the App

1. **Clone this repository**  
   ```bash
   git clone https://github.com/Shree-soni/ai_salary_predictor.git
   cd ai_salary_predictor

Install dependencies

pip install -r requirements.txt
Train the model (if not already)
Open and run:

notebook/salary_model.ipynb
Start Streamlit app

streamlit run app/app.py
Open http://localhost:8501 in your browser.

📌 Sample Input

Experience Level: Senior
Job Title: Machine Learning Engineer
Company Location: India
Required Skills: Python, TensorFlow, Deep Learning
👉 Output: Estimated Salary: ₹ 18,52,000 INR

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and share with credit.

🙌 Credits
Developed by Bhagyashri Sonar
Project idea inspired by AI Job Market Trends 📊

⭐️ Support
If you found this useful, consider giving the repo a ⭐️ on GitHub.

### Optional: `requirements.txt` (if you need it)

pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
