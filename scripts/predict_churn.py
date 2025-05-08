# predict_churn.py
import pandas as pd
import joblib
import argparse
import os
from pathlib import Path


def get_model_path():
    """Return standardized path to model directory"""
    base_dir = Path(__file__).parent.parent
    return os.path.join(base_dir, 'models')


def predict_customer_churn(customer_data_path, threshold_type='default'):
    """
    Predict churn for customers in the provided CSV file.

    Args:
        customer_data_path: Path to CSV file with customer data
        threshold_type: Type of threshold to use ('default', 'f1', 'business')

    Returns:
        DataFrame with original data plus churn predictions
    """
    # Load model and preprocessing components
    model_dir = get_model_path()
    try:
        model = joblib.load(os.path.join(model_dir, 'churn_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        expected_columns = joblib.load(os.path.join(model_dir, 'model_columns.pkl'))

        # Load thresholds if available
        try:
            thresholds = joblib.load(os.path.join(model_dir, 'optimal_thresholds.pkl'))
            if threshold_type == 'f1':
                optimal_threshold = thresholds.get('f1_threshold', 0.5)
                print(f"Using F1-optimal threshold: {optimal_threshold:.3f}")
            elif threshold_type == 'business':
                optimal_threshold = thresholds.get('business_threshold', 0.5)
                print(f"Using business-optimal threshold: {optimal_threshold:.3f}")
            else:
                optimal_threshold = 0.5
                print(f"Using default threshold: {optimal_threshold:.3f}")
        except Exception as e:
            print(f"Error loading thresholds: {str(e)}")
            optimal_threshold = 0.5
            print(f"Thresholds file not found. Using default: {optimal_threshold:.3f}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find model files in {model_dir}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading model components: {str(e)}")

    # Load customer data
    try:
        customer_df = pd.read_csv(customer_data_path)
    except Exception as e:
        raise Exception(f"Error reading customer data from {customer_data_path}: {str(e)}")

    # Store customer IDs
    customer_ids = customer_df['customerID'] if 'customerID' in customer_df.columns else None

    # Prepare data
    try:
        customer_encoded = pd.get_dummies(customer_df.drop('customerID', axis=1, errors='ignore'))

        # Add missing columns
        for col in expected_columns:
            if col not in customer_encoded.columns:
                customer_encoded[col] = 0

        # Ensure columns are in the right order
        available_columns = [col for col in expected_columns if col in customer_encoded.columns]
        if len(available_columns) == 0:
            raise ValueError("No matching columns found between the data and the model's expected columns.")

        customer_encoded = customer_encoded[available_columns]

        # Scale features
        X_scaled = scaler.transform(customer_encoded)
    except Exception as e:
        raise Exception(f"Error preparing customer data for prediction: {str(e)}")

    # Make predictions
    try:
        churn_probabilities = model.predict_proba(X_scaled)[:, 1]
        churn_predictions = (churn_probabilities >= optimal_threshold).astype(int)
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

    # Create results DataFrame
    try:
        if customer_ids is not None:
            results = pd.DataFrame({
                'customerID': customer_ids,
                'churn_probability': churn_probabilities,
                'predicted_churn': churn_predictions
            })
        else:
            results = pd.DataFrame({
                'churn_probability': churn_probabilities,
                'predicted_churn': churn_predictions
            })

        # Add risk level
        results['risk_level'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.7, 1],
            labels=['Low', 'Medium', 'High']
        )

        return results
    except Exception as e:
        raise Exception(f"Error formatting prediction results: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('input_file', help='Path to CSV file with customer data')
    parser.add_argument('output_file', help='Path to save predictions')
    parser.add_argument('--threshold', choices=['default', 'f1', 'business'], default='default',
                        help='Threshold type to use for predictions')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    try:
        # Make predictions
        predictions = predict_customer_churn(args.input_file, threshold_type=args.threshold)

        # Save predictions
        predictions.to_csv(args.output_file, index=False)

        print(f"Predictions saved to {args.output_file}")
        print(f"Found {predictions['predicted_churn'].sum()} customers at risk of churning out of {len(predictions)}")

        if args.debug:
            print("\nPrediction Details:")
            print(predictions.head())

    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()