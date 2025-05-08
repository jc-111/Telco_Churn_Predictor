"""
Flask API for churn prediction.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
from utils.churn_utils import get_model_path, validate_customer_data, predict_single_customer, load_model_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model and preprocessing components
try:
    logger.info("Loading model and preprocessing components...")
    model, scaler, expected_columns, thresholds = load_model_components()
    logger.info("Model and components loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    scaler = None
    expected_columns = None
    thresholds = {'default': 0.5, 'f1_threshold': 0.5, 'business_threshold': 0.5}


@app.route('/predict', methods=['POST'])
def predict_churn():
    """Predict churn for a customer."""
    try:
        # Check if model is loaded
        if model is None or scaler is None or expected_columns is None:
            return jsonify({'error': 'Model not loaded. Please check logs.'}), 500

        # Get data from request
        data = request.get_json()
        logger.info(f"Received prediction request for customer: {data.get('customerID', 'Unknown')}")

        # Validate data
        is_valid, error_msg = validate_customer_data(data, raise_exception=False)
        if not is_valid:
            logger.warning(f"Validation error: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Make prediction
        result = predict_single_customer(data, model, scaler, expected_columns)

        logger.info(f"Prediction for customer {result['customer_id']}: {result['churn_probability']:.2f} probability of churn")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict churn for multiple customers."""
    try:
        # Check if model is loaded
        if model is None or scaler is None or expected_columns is None:
            return jsonify({'error': 'Model not loaded. Please check logs.'}), 500

        # Get data from request
        data = request.get_json()

        if not isinstance(data, list):
            logger.warning("Invalid request format: expected a list of customer data")
            return jsonify({'error': 'Expected a list of customer data'}), 400

        logger.info(f"Received batch prediction request for {len(data)} customers")

        # Process each customer
        results = []
        for customer in data:
            # Validate data
            is_valid, error_msg = validate_customer_data(customer, raise_exception=False)
            if not is_valid:
                results.append({
                    'customer_id': customer.get('customerID', 'Unknown'),
                    'error': error_msg
                })
                continue

            # Make prediction
            result = predict_single_customer(customer, model, scaler, expected_columns)
            results.append(result)

        logger.info(f"Completed batch prediction for {len(data)} customers")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return "Churn Prediction API is running! Send POST requests to /predict or /predict_batch"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if model is None or scaler is None or expected_columns is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model components not loaded. Check logs.'
        }), 500

    return jsonify({
        'status': 'healthy',
        'message': 'API is operational'
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get feature importance if possible
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            top_features = [{expected_columns[i]: float(importance[i])}
                           for i in indices[:10]]
            feature_importance = top_features

        return jsonify({
            'model_type': type(model).__name__,
            'feature_count': len(expected_columns),
            'top_features': feature_importance,
            'prediction_threshold': 0.5
        })
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Churn Prediction API...")
    app.run(debug=True, port=5000)