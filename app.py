"""
================================================
  Customer Churn Predictor — Flask Web App
  Joyline SaaS | ML Project
================================================
Run:
    pip install flask
    python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ─── Load saved model, scaler, feature columns ───
BASE = os.path.dirname(__file__)

model        = joblib.load(os.path.join(BASE, 'models', 'best_model.pkl'))
scaler       = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
feature_cols = joblib.load(os.path.join(BASE, 'models', 'feature_columns.pkl'))


def predict_churn(customer_data: dict) -> dict:
    """
    Exact same function from the Jupyter Notebook — Day 5.
    Takes encoded customer dict, returns prediction + probabilities.
    """
    input_df = pd.DataFrame([customer_data])

    # Ensure all columns exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    # Apply scaling only for Logistic Regression
    model_name = type(model).__name__
    if model_name == 'LogisticRegression':
        input_array = scaler.transform(input_df)
    else:
        input_array = input_df  # keep DataFrame for feature names

    prediction  = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    churn_prob  = float(probability[1])
    retain_prob = float(probability[0])

    risk = 'HIGH' if churn_prob > 0.65 else ('MEDIUM' if churn_prob > 0.40 else 'LOW')

    actions = {
        'HIGH': [
            'Offer an immediate discount or loyalty bonus',
            'Assign a dedicated customer success manager',
            'Propose a contract upgrade incentive',
            'Schedule a personal check-in call urgently'
        ],
        'MEDIUM': [
            'Send a re-engagement email campaign',
            'Offer a feature walkthrough or demo session',
            'Provide a limited-time upgrade offer'
        ],
        'LOW': [
            'Continue normal engagement cycle',
            'Offer referral or loyalty incentives',
            'Monitor usage patterns monthly'
        ]
    }

    return {
        'prediction'        : int(prediction),
        'prediction_label'  : 'Will Churn' if prediction == 1 else 'Will Not Churn',
        'churn_probability' : round(churn_prob * 100, 2),
        'retain_probability': round(retain_prob * 100, 2),
        'risk_level'        : risk,
        'model_used'        : model_name,
        'actions'           : actions[risk]
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        customer = {
            'gender'          : int(data.get('gender', 1)),
            'SeniorCitizen'   : int(data.get('SeniorCitizen', 0)),
            'Partner'         : int(data.get('Partner', 0)),
            'Dependents'      : int(data.get('Dependents', 0)),
            'tenure'          : int(data.get('tenure', 12)),
            'PhoneService'    : int(data.get('PhoneService', 1)),
            'MultipleLines'   : int(data.get('MultipleLines', 0)),
            'InternetService' : int(data.get('InternetService', 1)),
            'OnlineSecurity'  : int(data.get('OnlineSecurity', 0)),
            'OnlineBackup'    : int(data.get('OnlineBackup', 0)),
            'DeviceProtection': int(data.get('DeviceProtection', 0)),
            'TechSupport'     : int(data.get('TechSupport', 0)),
            'StreamingTV'     : int(data.get('StreamingTV', 0)),
            'StreamingMovies' : int(data.get('StreamingMovies', 0)),
            'Contract'        : int(data.get('Contract', 0)),
            'PaperlessBilling': int(data.get('PaperlessBilling', 1)),
            'PaymentMethod'   : int(data.get('PaymentMethod', 0)),
            'MonthlyCharges'  : float(data.get('MonthlyCharges', 70.0)),
            'TotalCharges'    : float(data.get('TotalCharges', 840.0)),
        }

        result = predict_churn(customer)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print('=' * 50)
    print('  Customer Churn Predictor — Web App')
    print('   ML Project')
    print('=' * 50)
    print(f'  Model loaded : {type(model).__name__}')
    print(f'  Features     : {len(feature_cols)}')
    print('  Open         : http://127.0.0.1:5000')
    print('=' * 50)
    app.run(debug=True)