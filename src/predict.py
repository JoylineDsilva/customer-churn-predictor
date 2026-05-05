"""
=============================================================
  Customer Churn Prediction System — Standalone Script
  Joyline | SaaS ML Project
  Run this after training the model in the Jupyter Notebook
=============================================================
Usage:
    python src/predict.py
"""

import joblib
import pandas as pd
import os


def load_models():
    """Load saved model, scaler, and feature columns."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    model  = joblib.load(os.path.join(model_path, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
    cols   = joblib.load(os.path.join(model_path, 'feature_columns.pkl'))
    return model, scaler, cols


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for a customer.

    Parameters
    ----------
    customer_data : dict
        Dictionary with encoded feature values.

    Returns
    -------
    dict
        Prediction label, probabilities, and risk level.
    """
    model, scaler, feature_cols = load_models()

    input_df = pd.DataFrame([customer_data])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # Try scaling (works for LR; tree models ignore it gracefully)
    try:
        input_array = scaler.transform(input_df)
    except Exception:
        input_array = input_df.values

    prediction  = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]

    churn_prob = probability[1]
    risk = 'HIGH 🔴' if churn_prob > 0.7 else ('MEDIUM 🟡' if churn_prob > 0.4 else 'LOW 🟢')

    return {
        'prediction'        : 'YES — Will Churn ⚠️'  if prediction == 1 else 'NO — Will Not Churn ✅',
        'churn_probability' : f'{churn_prob * 100:.2f}%',
        'retain_probability': f'{probability[0] * 100:.2f}%',
        'risk_level'        : risk,
    }


def interactive_predictor():
    """Run an interactive CLI predictor."""
    print('\n' + '=' * 55)
    print('   🔮 CUSTOMER CHURN PREDICTION SYSTEM')
    print('   Joyline SaaS — ML Prediction Tool')
    print('=' * 55)

    def get_val(prompt, default, cast=int):
        raw = input(f'  {prompt} [default={default}]: ').strip()
        return cast(raw) if raw else default

    customer = {
        'gender'          : get_val('Gender         (0=Female, 1=Male)', 1),
        'SeniorCitizen'   : get_val('Senior Citizen (0=No, 1=Yes)', 0),
        'Partner'         : get_val('Partner        (0=No, 1=Yes)', 0),
        'Dependents'      : get_val('Dependents     (0=No, 1=Yes)', 0),
        'tenure'          : get_val('Tenure (months)', 12),
        'PhoneService'    : get_val('Phone Service  (0=No, 1=Yes)', 1),
        'MultipleLines'   : get_val('Multiple Lines (0=No,1=Yes,2=No svc)', 0),
        'InternetService' : get_val('Internet       (0=DSL,1=Fiber,2=None)', 1),
        'OnlineSecurity'  : get_val('Online Security(0=No,1=Yes,2=No svc)', 0),
        'OnlineBackup'    : get_val('Online Backup  (0=No,1=Yes,2=No svc)', 0),
        'DeviceProtection': get_val('Device Protect (0=No,1=Yes,2=No svc)', 0),
        'TechSupport'     : get_val('Tech Support   (0=No,1=Yes,2=No svc)', 0),
        'StreamingTV'     : get_val('Streaming TV   (0=No,1=Yes,2=No svc)', 0),
        'StreamingMovies' : get_val('Streaming Mov  (0=No,1=Yes,2=No svc)', 0),
        'Contract'        : get_val('Contract       (0=Monthly,1=1yr,2=2yr)', 0),
        'PaperlessBilling': get_val('Paperless Bill (0=No, 1=Yes)', 1),
        'PaymentMethod'   : get_val('Payment        (0=Elec,1=Mail,2=Bank,3=Card)', 0),
        'MonthlyCharges'  : get_val('Monthly Charges ($)', 70.0, float),
        'TotalCharges'    : get_val('Total Charges  ($)', 840.0, float),
    }

    result = predict_churn(customer)

    print('\n' + '=' * 55)
    print('   📊 PREDICTION RESULT')
    print('=' * 55)
    for key, val in result.items():
        label = key.replace('_', ' ').title()
        print(f'  {label:<22}: {val}')
    print('=' * 55)

    # Business recommendation
    risk = result['risk_level']
    print('\n💡 Recommended Action:')
    if 'HIGH' in risk:
        print('  → Offer a discount or loyalty bonus IMMEDIATELY')
        print('  → Assign a customer success manager')
        print('  → Offer contract upgrade incentive')
    elif 'MEDIUM' in risk:
        print('  → Send engagement email / check-in call')
        print('  → Offer feature demo or training session')
    else:
        print('  → Customer is healthy — continue normal engagement')
    print()


if __name__ == '__main__':
    interactive_predictor()
