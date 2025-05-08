"""
Data loading and preprocessing functions.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(file_path):
    """Load and clean the dataset."""
    # Load data
    df = pd.read_csv(file_path)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Remove rows with missing values
    df = df.dropna()

    # Convert SeniorCitizen from 0/1 to No/Yes for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

    # Convert target to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for modeling."""
    # Remove customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler