# Chronic-Disease-Prediction-
# 🩺 Chronic Disease Prediction Website

This project is a web-based application that predicts the likelihood of chronic diseases using a trained Machine Learning (ML) model.  
It combines a **Flask backend** (for model predictions) with a **simple HTML/JavaScript frontend** (for user input).

---

## 📌 Features
- User-friendly web interface for entering patient details (age, symptoms, etc.)
- Flask API backend that loads the ML model and performs predictions
- Real-time disease prediction results
- Easily extendable for more diseases, features, or advanced models

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **Flask** (backend web framework)
- **scikit-learn / joblib** (for ML model training and saving)
- **HTML, CSS, JavaScript** (frontend)

---
```
## 📂 Project Structure
disease_prediction_project/
│── app.py # Flask backend
│── disease_model.pkl # Saved ML model
│── requirements.txt # Dependencies
│── templates/
│ └── index.html # Frontend (form for input + results)
│── static/
└── style.css # (Optional) Styling
```
yaml
Copy code

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/chronic-disease-prediction.git
   cd chronic-disease-prediction
Create a virtual environment (optional but recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the Flask server

bash
Copy code
python app.py
Open the app in browser

cpp
📊 Model Details
Trained on chronic disease dataset (custom or open-source)

Example features:

Age

Symptoms (Fever, Cough, Fatigue, Headache, etc.)

Output:

Predicted disease class (e.g., Diabetes, Heart Disease, etc.)

🚀 Future Improvements
Add authentication (login/signup for patients/doctors)

Improve UI with React or Streamlit

Deploy on Heroku / AWS / Render

Expand to multiple diseases with probability scores

🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to change.

👩‍💻 Developed by: M. Nivetha
