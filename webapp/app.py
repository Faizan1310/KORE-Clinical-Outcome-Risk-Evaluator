from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
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
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Load model
with open('../outputs/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
sample = pd.read_csv('../outputs/cleaned_data.csv')
feature_names = sample.drop(columns=['readmitted_30']).columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    time_in_hospital = int(request.form['time_in_hospital'])
    num_medications = int(request.form['num_medications'])
    num_lab_procedures = int(request.form['num_lab_procedures'])
    number_diagnoses = int(request.form['number_diagnoses'])
    insulin = int(request.form['insulin'])
    change = int(request.form['change'])

    # Build feature array
    full_features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    full_features['age'] = age
    full_features['gender'] = gender
    full_features['time_in_hospital'] = time_in_hospital
    full_features['num_medications'] = num_medications
    full_features['num_lab_procedures'] = num_lab_procedures
    full_features['number_diagnoses'] = number_diagnoses
    full_features['insulin'] = insulin
    full_features['change'] = change

    probability = model.predict_proba(full_features)[0][1]
    risk = "HIGH RISK" if probability >= 0.5 else "LOW RISK"
    color = "#e53e3e" if probability >= 0.5 else "#38a169"

    # Save to database
    record = Prediction(
        age=age, gender='Male' if gender==1 else 'Female',
        time_in_hospital=time_in_hospital,
        num_medications=num_medications,
        num_lab_procedures=num_lab_procedures,
        number_diagnoses=number_diagnoses,
        insulin='Yes' if insulin==1 else 'No',
        change='Yes' if change==1 else 'No',
        risk=risk,
        probability=round(probability * 100, 2)
    )
    db.session.add(record)
    db.session.commit()

    # Feature importance for chart
    importance = model.feature_importances_
    top_features = pd.Series(importance, index=feature_names).nlargest(5)
    chart_labels = top_features.index.tolist()
    chart_values = [round(v * 100, 2) for v in top_features.values.tolist()]

    return render_template('index.html',
                           prediction=risk,
                           probability=round(probability * 100, 2),
                           color=color,
                           chart_labels=chart_labels,
                           chart_values=chart_values)

@app.route('/history')
def history():
    records = Prediction.query.order_by(Prediction.date.desc()).all()
    return render_template('history.html', records=records)

@app.route('/clear_history')
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)