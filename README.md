# 🔴 Customer Churn Predictor — SaaS ML Project
### Joyline | Machine Learning System

---

## 📁 Project Structure

```
customer_churn_predictor/
│
├── data/                          ← Put your dataset CSV here
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Customer_Churn_Predictor.ipynb   ← MAIN NOTEBOOK (run this)
│
├── src/
│   └── predict.py                 ← Standalone prediction script
│
├── models/                        ← Auto-created after training
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── outputs/                       ← Auto-created — all plots saved here
│   ├── churn_distribution.png
│   ├── tenure_charges_analysis.png
│   ├── correlation_heatmap.png
│   ├── categorical_analysis.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── all_confusion_matrices.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run (Step-by-Step)

### Step 1 — Install dependencies
Open terminal in VS Code and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Get the Dataset
**Option A (Manual):**
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it inside the `data/` folder

**Option B (Auto):**
The notebook will auto-generate sample data if the file is not found!

### Step 3 — Open the Notebook
```bash
cd notebooks
jupyter notebook Customer_Churn_Predictor.ipynb
```
OR open it directly in VS Code with the Jupyter extension.

### Step 4 — Run all cells top to bottom
Click **"Run All"** or press `Shift+Enter` on each cell.

### Step 5 — Use the Prediction System
After training, run the standalone predictor:
```bash
cd src
python predict.py
```

---

## 📊 Project Phases

| Day | Phase | What Happens |
|-----|-------|--------------|
| Day 1 | Data Loading & Preprocessing | Load CSV, fix types, encode categories |
| Day 2 | EDA + Baseline Model | Visualize data, train Logistic Regression |
| Day 3 | Model Improvement | Train Decision Tree & Random Forest |
| Day 4 | Evaluation & Selection | Compare all models, pick best |
| Day 5 | Prediction System | Interactive CLI predictor |

---

## 🤖 Models Used

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Uses StandardScaler |
| Decision Tree | Improved | Max depth = 5 |
| Random Forest | Improved | 100 estimators |

**Primary Metric: Recall** (to catch as many churning customers as possible)

---

## 🎯 Features Used

| Feature | Description |
|---------|-------------|
| tenure | Months with company |
| MonthlyCharges | Monthly bill |
| TotalCharges | Total billed |
| Contract | Month-to-month / 1yr / 2yr |
| InternetService | DSL / Fiber / None |
| PaymentMethod | Electronic check, etc. |
| SeniorCitizen | 0 or 1 |
| + 12 more | Service subscriptions |

---

## 🔮 How to Predict a Single Customer

```python
import joblib, pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

customer = {
    'tenure': 3,
    'MonthlyCharges': 95.5,
    'Contract': 0,           # 0 = Month-to-month
    # ... all other features
}
# See notebook Day 5 for full example
```

---

## ✅ Success Criteria

- [x] Accuracy ≥ 80%
- [x] High Recall for churn prediction
- [x] Working prediction system
- [x] EDA with visualizations
- [x] Model comparison and selection

---

**Tech Stack:** Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Jupyter
