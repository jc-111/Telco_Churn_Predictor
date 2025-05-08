# data_validation.py

import pandas as pd

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

    required_fields = [
        'tenure',
        'MonthlyCharges',
        'InternetService',
        'Contract',
        'PaymentMethod'
    ]

    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in data.columns]
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        if raise_exception:
            raise ValueError(error_msg)
        return False, error_msg

    # Validate field types and values
    validations = [
        (data['tenure'].apply(lambda x: isinstance(x, (int, float)) and x >= 0).all(),
         "Tenure must be a non-negative number"),
        (data['MonthlyCharges'].apply(lambda x: isinstance(x, (int, float)) and x > 0).all(),
         "MonthlyCharges must be a positive number"),
        (data['Contract'].isin(['Month-to-month', 'One year', 'Two year']).all(),
         "Contract must be 'Month-to-month', 'One year', or 'Two year'"),
        (data['InternetService'].isin(['DSL', 'Fiber optic', 'No']).all(),
         "InternetService must be 'DSL', 'Fiber optic', or 'No'")
    ]

    for is_valid, error_msg in validations:
        if not is_valid:
            if raise_exception:
                raise ValueError(error_msg)
            return False, error_msg

    return True, "Data is valid"