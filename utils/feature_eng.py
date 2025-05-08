"""
Feature engineering functions.
"""
import pandas as pd
import numpy as np


def create_features(df):
    """Create additional features for churn prediction."""
    df = df.copy()

    # Add tenure-related features
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero

    # Calculate revenue-to-tenure ratio (higher values may indicate higher churn risk)
    df['Revenue_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero

    # Create tenure buckets (non-linear relationship with churn)
    df['Tenure_Bucket'] = pd.cut(df['tenure'],
                                bins=[0, 6, 12, 24, 36, 60, float('inf')],
                                labels=['0-6mo', '6-12mo', '1-2yr', '2-3yr', '3-5yr', '5yr+'])

    # Add service count features if data hasn't been encoded yet
    if 'InternetService_No' not in df.columns:
        # Count the services each customer has
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']

        # Service count feature - Count services where value is 'Yes'
        df['ServiceCount'] = df[service_columns].apply(
            lambda x: sum(1 for val in x if val == 'Yes'), axis=1)

        # Calculate Monthly Charges per Service
        df['Cost_per_Service'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)

        # Create service-related features
        df['HasInternetService'] = df['InternetService'] != 'No'
        df['HasPhoneService'] = df['PhoneService'] == 'Yes'

        # Create contract type feature
        df['IsMonthToMonth'] = df['Contract'] == 'Month-to-month'

        # Interaction features
        df['Contract_Tenure'] = df['tenure'] * (df['Contract'] == 'Month-to-month').astype(int)

        # Security features combination
        df['NoSecurity'] = ((df['OnlineSecurity'] == 'No') &
                           (df['OnlineBackup'] == 'No') &
                           (df['DeviceProtection'] == 'No')).astype(int)

        # Payment risk - electronic check with month-to-month
        df['HighRiskPayment'] = ((df['PaymentMethod'] == 'Electronic check') &
                                (df['Contract'] == 'Month-to-month')).astype(int)

        # Senior citizen with high monthly charges
        if 'SeniorCitizen' in df.columns:
            df['Senior_HighCharge'] = ((df['SeniorCitizen'] == 1) &
                                      (df['MonthlyCharges'] > df['MonthlyCharges'].median())).astype(int)

    return df


def get_important_features():
    """Return list of most important features based on analysis."""
    return [
        'Contract_Month-to-month',
        'tenure',
        'TotalCharges',
        'MonthlyCharges',
        'InternetService_Fiber optic',
        'PaymentMethod_Electronic check',
        'OnlineSecurity_No',
        'TechSupport_No',
        'PaperlessBilling_Yes',
        'Contract_Two year',
        'Revenue_per_Tenure',
        'ServiceCount',
        'NoSecurity',
        'HighRiskPayment',
        'Contract_Tenure'
    ]