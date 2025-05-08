# This file should be in /utils/churn_utils.py
import os
import joblib
import pandas as pd
from pathlib import Path


def get_project_root():
    """Return the project root directory."""
    # Assuming this file is in utils/churn_utils.py
    current_file = Path(__file__)
    return current_file.parent.parent


def get_model_path():
    """Return standardized path to model directory"""
    return os.path.join(get_project_root(), 'models')


def get_data_path():
    """Return standardized path to data directory"""
    return os.path.join(get_project_root(), 'data')


def get_artifacts_path():
    """Return standardized path to artifacts directory"""
    artifacts_dir = os.path.join(get_project_root(), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


def load_model_components():
    """Load all model components"""
    model_dir = get_model_path()
    model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    expected_columns = joblib.load(os.path.join(model_dir, 'model_columns.pkl'))

    # Try to load thresholds
    try:
        thresholds = joblib.load(os.path.join(model_dir, 'optimal_thresholds.pkl'))
    except:
        thresholds = {'default': 0.5, 'f1_threshold': 0.5, 'business_threshold': 0.5}

    return model, scaler, expected_columns, thresholds


def validate_customer_data(data, raise_exception=True):
    """
    Validate customer data for churn prediction.

    Args:
        data: Dictionary or DataFrame with customer data
        raise_exception: Whether to raise exception on validation failure

    Returns:
        (is_valid, error_message) tuple
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Check for minimum required fields
    required_fields = [
        'tenure',
        'MonthlyCharges',
        'Contract'
    ]

    missing_fields = [field for field in required_fields if field not in data.columns]
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        if raise_exception:
            raise ValueError(error_msg)
        return False, error_msg

    # Validate field types and values
    validations = [
        (data['tenure'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(),
         "Tenure must be a number"),
        (data['MonthlyCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(),
         "MonthlyCharges must be a number")
    ]

    for is_valid, error_msg in validations:
        if not is_valid:
            if raise_exception:
                raise ValueError(error_msg)
            return False, error_msg

    return True, "Data is valid"


def predict_single_customer(data, model, scaler, expected_columns, threshold=0.5):
    """
    Make churn prediction for a single customer.

    Args:
        data: Dictionary with customer data
        model: Loaded ML model
        scaler: Fitted scaler
        expected_columns: List of model input columns
        threshold: Prediction threshold (default 0.5)

    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Process categorical variables
    df_encoded = pd.get_dummies(df)

    # Handle missing columns
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure columns are in the right order
    df_encoded = df_encoded[expected_columns]

    # Scale features
    X_scaled = scaler.transform(df_encoded)

    # Make prediction
    churn_probability = model.predict_proba(X_scaled)[0][1]
    churn_prediction = churn_probability >= threshold

    return {
        'customer_id': data.get('customerID', 'Unknown'),
        'churn_probability': float(churn_probability),
        'will_churn': bool(churn_prediction),
        'risk_level': 'High' if churn_probability > 0.7 else 'Medium' if churn_probability > 0.3 else 'Low'
    }