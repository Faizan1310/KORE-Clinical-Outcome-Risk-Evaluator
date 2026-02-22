# 🏥 Hospital Readmission Prediction

## Project Overview
This project predicts whether a diabetic patient will be readmitted to the hospital within 30 days of discharge using machine learning. Early identification of high-risk patients can help hospitals take preventive measures and reduce unnecessary readmissions.

## Problem Statement
Hospital readmissions cost the US healthcare system over $26 billion annually. Many readmissions are preventable with better care planning. This project builds a data-driven solution to identify high-risk patients before discharge.

## Dataset
- **Source:** Diabetes 130-US Hospitals Dataset (UCI/Kaggle)
- **Size:** 101,766 patient records, 50 features
- **Period:** 10 years of clinical care data from 130 US hospitals

## Tools & Technologies
- **Python** — Data cleaning, analysis, and machine learning
- **SQL** — Data exploration
- **Power BI** — Interactive dashboard
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Project Structure
```
hospital_readmission/
├── data/                  ← Original dataset
├── notebooks/
│   ├── 01_exploration.ipynb   ← Data exploration
│   ├── 02_cleaning.ipynb      ← Data cleaning
│   ├── 03_modeling.py         ← ML model
│   └── 04_export.py           ← Power BI export
├── outputs/
│   ├── cleaned_data.csv
│   ├── powerbi_data.csv
│   ├── rf_model.pkl
│   └── feature_importance.png
```

## Key Findings
- Only 11% of patients were readmitted within 30 days (class imbalance handled with SMOTE)
- Top predictors: medication change, primary diagnosis, number of lab procedures
- Older patients and those with more medications are at higher risk

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 91% |
| Precision | 94% |
| Recall | 87% |
| ROC-AUC | 0.957 |

## Dashboard
Interactive Power BI dashboard showing:
- Total patients and readmission KPIs
- Readmissions by age group
- Predicted risk by number of medications
- Readmission breakdown (predicted vs actual)

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order (01 → 02 → 03 → 04)
4. Open Power BI dashboard from outputs folder