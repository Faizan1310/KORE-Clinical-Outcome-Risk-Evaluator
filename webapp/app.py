from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kore-clinical-outcome-risk-evaluator'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
import os
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///predictions.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Prediction(db.Model):
    id =db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=True)
    patient_id = db.Column(db.String(50), nullable=True)
    hospital_name = db.Column(db.String(200), nullable=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    time_in_hospital = db.Column(db.Integer)
    num_medications = db.Column(db.Integer)
    num_lab_procedures = db.Column(db.Integer)
    number_diagnoses = db.Column(db.Integer)
    insulin = db.Column(db.String(5))
    change = db.Column(db.String(5))
    risk = db.Column(db.String(20))
    probability = db.Column(db.Float)
    ai_report = db.Column(db.Text)
    ai_recommendations = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    subject = db.Column(db.String(200))
    message = db.Column(db.Text)
    rating = db.Column(db.String(10))
    date = db.Column(db.DateTime, default=datetime.utcnow)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

model = None
feature_names = None

def load_model():
    global model, feature_names
    if model is None:
        with open('../outputs/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = pd.read_csv('../outputs/cleaned_data.csv')
        feature_names = sample.drop(columns=['readmitted_30']).columns.tolist()

def generate_ai_report(patient_data, probability, risk):
    prompt = f"""You are a medical AI assistant. Based on the following patient data, generate a concise professional medical summary report.

Patient Data:
- Age Group: {patient_data['age']} (1=youngest, 9=oldest)
- Gender: {patient_data['gender']}
- Time in Hospital: {patient_data['time_in_hospital']} days
- Number of Medications: {patient_data['num_medications']}
- Number of Lab Procedures: {patient_data['num_lab_procedures']}
- Number of Diagnoses: {patient_data['number_diagnoses']}
- Insulin Given: {patient_data['insulin']}
- Medication Changed: {patient_data['change']}
- Readmission Risk: {risk} ({probability}% probability)

Write a 3-4 sentence professional medical summary explaining the patient's risk profile, contributing factors, and urgency level."""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

def generate_recommendations(patient_data, probability, risk):
    prompt = f"""You are a medical AI assistant. Based on this patient's readmission risk profile, provide 4 specific actionable recommendations.

Patient Profile:
- Age Group: {patient_data['age']} (1=youngest, 9=oldest)
- Time in Hospital: {patient_data['time_in_hospital']} days
- Medications: {patient_data['num_medications']}
- Lab Procedures: {patient_data['num_lab_procedures']}
- Diagnoses: {patient_data['number_diagnoses']}
- Insulin: {patient_data['insulin']}
- Medication Changed: {patient_data['change']}
- Risk Level: {risk} ({probability}%)

Provide exactly 4 recommendations. Format each as:
[PRIORITY] Action: Description"""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
@login_required
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact', methods=['GET', 'POST'])
@login_required
def contact():
    success = False
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        rating = request.form.get('rating')
        
        # Save to database
        feedback = Feedback(
            name=name,
            email=email,
            subject=subject,
            message=message,
            rating=rating
        )
        db.session.add(feedback)
        db.session.commit()
        success = True
    
    return render_template('contact.html', success=success)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
  try:
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_id = request.form.get('patient_id', 'N/A')
    hospital_name = request.form.get('hospital_name', 'Unknown Hospital')  
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    time_in_hospital = int(request.form['time_in_hospital'])
    num_medications = int(request.form['num_medications'])
    num_lab_procedures = int(request.form['num_lab_procedures'])
    number_diagnoses = int(request.form['number_diagnoses'])
    insulin = int(request.form['insulin'])
    change = int(request.form['change'])
    admission_type_id = int(request.form.get('admission_type_id', 1))
    discharge_disposition_id = int(request.form.get('discharge_disposition_id', 1))
    num_procedures = int(request.form.get('num_procedures', 0))
    number_inpatient = int(request.form.get('number_inpatient', 0))
    
    load_model()
    full_features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    full_features['age'] = age
    full_features['gender'] = gender
    full_features['time_in_hospital'] = time_in_hospital
    full_features['num_medications'] = num_medications
    full_features['num_lab_procedures'] = num_lab_procedures
    full_features['number_diagnoses'] = number_diagnoses
    full_features['insulin'] = insulin
    full_features['change'] = change
    full_features['admission_type_id'] = admission_type_id
    full_features['discharge_disposition_id'] = discharge_disposition_id
    full_features['num_procedures'] = num_procedures
    full_features['number_inpatient'] = number_inpatient

    probability = model.predict_proba(full_features)[0][1]
    risk = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
    color = "#e53e3e" if probability >= 0.5 else "#38a169"
    prob_percent = float(round(probability * 100, 2))

    patient_data = {
        'age': age, 'gender': 'Male' if gender == 1 else 'Female',
        'time_in_hospital': time_in_hospital,
        'num_medications': num_medications,
        'num_lab_procedures': num_lab_procedures,
        'number_diagnoses': number_diagnoses,
        'insulin': 'Yes' if insulin == 1 else 'No',
        'change': 'Yes' if change == 1 else 'No'
    }

    ai_report = generate_ai_report(patient_data, prob_percent, risk)
    ai_recommendations = generate_recommendations(patient_data, prob_percent, risk)

    record = Prediction(
        user_id=current_user.id,
        patient_name=patient_name,
        patient_id=patient_id,
        hospital_name=hospital_name,
        age=age,
        gender='Male' if gender == 1 else 'Female',
        time_in_hospital=time_in_hospital,
        num_medications=num_medications,
        num_lab_procedures=num_lab_procedures,
        number_diagnoses=number_diagnoses,
        insulin='Yes' if insulin == 1 else 'No',
        change='Yes' if change == 1 else 'No',
        risk=risk, probability=prob_percent,
        ai_report=ai_report,
        ai_recommendations=ai_recommendations
    )
    db.session.add(record)
    db.session.commit()

    importance = model.feature_importances_
    top_features = pd.Series(importance, index=feature_names).nlargest(5)
    chart_labels = top_features.index.tolist()
    chart_values = [round(v * 100, 2) for v in top_features.values.tolist()]

    return render_template('index.html',
                           prediction=risk,
                           probability=prob_percent,
                           color=color,
                           chart_labels=chart_labels,
                           chart_values=chart_values,
                           ai_report=ai_report,
                           ai_recommendations=ai_recommendations,
                           age_val=age,
                           gender_val='Male' if gender == 1 else 'Female',
                           time_val=time_in_hospital,
                           med_val=num_medications,
                           lab_val=num_lab_procedures,
                           diag_val=number_diagnoses,
                           insulin_val='Yes' if insulin == 1 else 'No',
                           change_val='Yes' if change == 1 else 'No')
  except Exception as e:
    return f"Prediction error: {str(e)}", 500
@app.route('/history')
@login_required
def history():
    records = Prediction.query.order_by(Prediction.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/insights')
@login_required
def insights():
    records = Prediction.query.all()
    if not records:
        return render_template('insights.html', insights=None)

    total = len(records)
    high_risk = sum(1 for r in records if r.risk == "HIGH RISK")
    avg_prob = round(sum(r.probability for r in records) / total, 2)

    prompt = f"""You are a medical data analyst. Analyze these hospital readmission prediction statistics and provide insights.

Statistics:
- Total Predictions: {total}
- High Risk Patients: {high_risk} ({round(high_risk/total*100, 1)}%)
- Low Risk Patients: {total - high_risk} ({round((total-high_risk)/total*100, 1)}%)
- Average Risk Probability: {avg_prob}%
- Average Medications: {round(sum(r.num_medications for r in records)/total, 1)}
- Average Hospital Stay: {round(sum(r.time_in_hospital for r in records)/total, 1)} days
- Average Diagnoses: {round(sum(r.number_diagnoses for r in records)/total, 1)}

Provide 4 key insights and trends from this data. Be specific and actionable. Format as numbered list."""

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    insights_text = message.choices[0].message.content

    return render_template('insights.html',
                           insights=insights_text,
                           total=total,
                           high_risk=high_risk,
                           avg_prob=avg_prob)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    language = data.get('language', 'English')
    patient_context = data.get('patient_context', '')

    prompt = f"""You are KORE AI Assistant — a multilingual medical AI assistant for KORE (Clinical Outcome Risk Evaluator), a hospital readmission prediction system.

{f'Current Patient Context: {patient_context}' if patient_context else ''}

The user is communicating in {language}. Always respond in the same language as the user's message.

User message: {user_message}

Provide a helpful, accurate, and empathetic response. Keep it concise (2-3 sentences max)."""

    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"response": message.choices[0].message.content})
    except Exception as e:
        return jsonify({"response": "I'm sorry, something went wrong. Please try again."})

@app.route('/clear_history')
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    return redirect(url_for('history'))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Username already exists!")
        hashed_password = generate_password_hash(password)
        is_admin = User.query.count() == 0  # First user becomes admin
        user = User(username=username, email=email, password=hashed_password, is_admin=is_admin)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password!")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))
@app.route('/sitemap.xml')
def sitemap():
    return '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://kore-ai-fnf8.onrender.com/</loc></url>
    <url><loc>https://kore-ai-fnf8.onrender.com/about</loc></url>
    <url><loc>https://kore-ai-fnf8.onrender.com/contact</loc></url>
</urlset>''', 200, {'Content-Type': 'application/xml'}
@app.route('/migrate-columns-kore2026')
def migrate_columns():
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text('ALTER TABLE prediction ADD COLUMN IF NOT EXISTS patient_name VARCHAR(100)'))
            conn.execute(db.text('ALTER TABLE prediction ADD COLUMN IF NOT EXISTS patient_id VARCHAR(50)'))
            conn.execute(db.text('ALTER TABLE prediction ADD COLUMN IF NOT EXISTS hospital_name VARCHAR(200)'))
            conn.commit()
        return "Migration successful! New columns added!"
    except Exception as e:
        return f"Migration error: {str(e)}"
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user_predictions = Prediction.query.filter_by(
            user_id=current_user.id
        ).order_by(Prediction.date.desc()).all()
    except:
        user_predictions = []
    total = len(user_predictions)
    high_risk = sum(1 for p in user_predictions if p.risk == 'HIGH RISK')
    return render_template('dashboard.html',
                           user=current_user,
                           predictions=user_predictions,
                           total=total,
                           high_risk=high_risk)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return redirect(url_for('dashboard'))
    try:
        all_users = User.query.all()
        try:
            all_predictions = Prediction.query.order_by(Prediction.date.desc()).all()
        except:
            all_predictions = []
        try:
            all_feedback = Feedback.query.order_by(Feedback.date.desc()).all()
        except:
            all_feedback = []
        total_users = len(all_users)
        total_predictions = len(all_predictions)
        high_risk = sum(1 for p in all_predictions if p.risk == 'HIGH RISK')
        return render_template('admin.html',
                               users=all_users,
                               predictions=all_predictions,
                               feedback=all_feedback,
                               total_users=total_users,
                               total_predictions=total_predictions,
                               high_risk=high_risk)
    except Exception as e:
        return f"Admin error: {str(e)}"
        
@app.before_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
    