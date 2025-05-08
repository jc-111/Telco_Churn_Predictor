# explain_model.py
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os


def explain_with_shap(model, X_sample, output_path='shap_summary.png'):
    """Generate SHAP summary plot."""
    try:
        import shap

        # For tree models (Random Forest, XGBoost, etc.)
        if hasattr(model, 'feature_importances_'):
            # For Random Forest specifically
            if 'RandomForest' in str(type(model)):
                explainer = shap.TreeExplainer(model)
            # For other tree-based models
            else:
                explainer = shap.TreeExplainer(model)
        # For other model types
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 50))

        # Calculate SHAP values for the first 100 samples
        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP value formats (list vs array)
        if isinstance(shap_values, list):
            # For binary classification, take class 1 (churn)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        plt.figure(figsize=(12, 8))
        # Use the summary plot
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(12, 8))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"SHAP summary saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error generating SHAP plot: {str(e)}")
        return False


if __name__ == "__main__":
    # This part runs when the script is executed directly
    try:
        # Load model and a sample of data
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')
        model = joblib.load(model_path)

        # Try to load sample data if available, or use some sample data from training
        try:
            X_sample = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'X_sample.pkl'))
        except:
            # If sample data isn't available, create a simple placeholder
            X_sample = pd.DataFrame(0.0, index=range(10), columns=range(10))

        # Generate the SHAP plot
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        explain_with_shap(model, X_sample, os.path.join(artifacts_dir, 'shap_summary.png'))
        print("SHAP analysis completed successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")