# ⬡ KORE — Clinical Outcome Risk Evaluator

> AI-Powered Hospital Readmission Prediction System

[![Live Demo](https://img.shields.io/badge/Live-Demo-02C39A?style=for-the-badge)](https://hospital-readmission-prediction-kpnl.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![Groq AI](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=for-the-badge)](https://groq.com)

## 🌐 Live Demo
**[https://hospital-readmission-prediction-kpnl.onrender.com](https://hospital-readmission-prediction-kpnl.onrender.com)**

---

## 🏥 What is KORE?

KORE predicts whether a diabetic patient will be readmitted to hospital within 30 days of discharge — with **91% accuracy** and **0.957 ROC-AUC score**.

Hospital readmissions cost the US healthcare system **$26 billion annually**. KORE helps hospitals identify high-risk patients BEFORE discharge so they can take preventive action.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 ML Prediction | Random Forest model, 91% accuracy, 0.957 ROC-AUC |
| 🩺 AI Doctor Report | Auto-generated medical summary using Llama 3.3 70B |
| ✅ AI Recommendations | 4 prioritized action items for hospital staff |
| 💬 Multilingual Chatbot | Supports English, Hindi, Urdu, Arabic, French, Spanish, Chinese |
| 📈 AI Insights | Auto-analyzes trends from prediction history |
| 📋 Prediction History | SQLite database stores all predictions |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.13 |
| ML Model | Scikit-learn Random Forest |
| Class Balancing | SMOTE (imbalanced-learn) |
| Web Framework | Flask + Flask-SQLAlchemy |
| AI Layer | Groq API — Llama 3.3 70B |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Dashboard | Power BI |
| Deployment | Render.com + Gunicorn |
| Version Control | GitHub |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 91% |
| Precision | 94% |
| Recall | 87% |
| F1-Score | 91% |
| ROC-AUC | 0.957 |

---

## 🗂️ Project Structure
```
hospital_readmission/
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration
│   ├── 02_cleaning.ipynb       # Data preprocessing
│   └── 03_modeling.py          # ML model training
├── outputs/
│   ├── rf_model.pkl            # Trained model
│   ├── cleaned_data.csv        # Processed dataset
│   └── feature_importance.png  # Feature analysis
├── webapp/
│   ├── app.py                  # Flask application
│   ├── templates/
│   │   ├── index.html          # Main prediction page
│   │   ├── history.html        # Prediction history
│   │   └── insights.html       # AI insights dashboard
│   ├── requirements.txt
│   └── Procfile
└── README.md
```

---

## 🚀 Run Locally
```bash
# Clone the repository
git clone https://github.com/Faizan1310/hospital-readmission-prediction.git
cd hospital-readmission-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r webapp/requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > webapp/.env

# Run the app
cd webapp
python app.py
```

---

## 📈 Dataset

- **Source:** Diabetes 130-US Hospitals Dataset (UCI / Kaggle)
- **Records:** 101,766 patient encounters
- **Features:** 50 clinical attributes
- **Period:** 10 years (1999-2008)
- **Hospitals:** 130 US hospitals

---

## 👨‍💻 Developer

**Faizan Khan**
- GitHub: [@Faizan1310](https://github.com/Faizan1310)
- Project: KORE — Clinical Outcome Risk Evaluator

---

## 📄 License

This project is open source and available under the MIT License.