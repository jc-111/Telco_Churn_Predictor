import os
import sys
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so you can import from utils and other app modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_prep import load_and_clean_data, prepare_data
from utils.feature_eng import create_features
from model_training import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_catboost,
    train_gradient_boosting,
    train_ensemble,
    evaluate_model,
    find_optimal_threshold,
    find_optimal_threshold_by_business_cost,
    handle_class_imbalance,
    save_model_evaluation_metrics,
    save_roc_curve_data,
    save_precision_recall_curve_data
)


def main():
    print("======= Churn Prediction Pipeline =======")

    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("\nStep 1: Loading and preparing data...")
    df = load_and_clean_data(file_path)

    print("\nStep 2: Engineering features...")
    df = create_features(df)

    print("\nStep 3: Splitting data and scaling features...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data(df)

    print("\nStep 3.5: Handling class imbalance...")
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_scaled, y_train, method='smote')

    print("\nStep 4: Training models...")

    print("\nTraining Logistic Regression...")
    log_model = train_logistic_regression(X_train_balanced, y_train_balanced)
    log_results = evaluate_model(log_model, X_test_scaled, y_test)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train_balanced, y_train_balanced)
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train_balanced, y_train_balanced)
    xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test)

    print("\nTraining LightGBM...")
    lgb_model = train_lightgbm(X_train_balanced, y_train_balanced)
    lgb_results = evaluate_model(lgb_model, X_test_scaled, y_test)

    print("\nTraining CatBoost...")
    cat_model = train_catboost(X_train_balanced, y_train_balanced)
    cat_results = evaluate_model(cat_model, X_test_scaled, y_test)

    print("\nTraining Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train_balanced, y_train_balanced)
    gb_results = evaluate_model(gb_model, X_test_scaled, y_test)

    print("\nTraining Ensemble Model...")
    ensemble_models = {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'cat': cat_model,
        'gb': gb_model
    }
    ensemble_model = train_ensemble(ensemble_models, X_train_balanced, y_train_balanced)
    ensemble_results = evaluate_model(ensemble_model, X_test_scaled, y_test)

    print("\nStep 5: Selecting best model based on F1 score...")
    model_results = [
        (log_model, log_results, "Logistic Regression"),
        (rf_model, rf_results, "Random Forest"),
        (xgb_model, xgb_results, "XGBoost"),
        (lgb_model, lgb_results, "LightGBM"),
        (cat_model, cat_results, "CatBoost"),
        (gb_model, gb_results, "Gradient Boosting"),
        (ensemble_model, ensemble_results, "Ensemble")
    ]

    model, results, model_name = max(model_results, key=lambda x: x[1]['f1_score'])
    print(f"Selected model: {model_name} with F1 score: {results['f1_score']:.4f}")

    print("\nStep 5.5: Finding optimal threshold using F1 score...")
    best_f1_threshold, best_f1 = find_optimal_threshold(y_test, results['y_prob'])
    print(f"Optimal threshold (F1): {best_f1_threshold:.3f} (F1 = {best_f1:.3f})")

    print("\nStep 5.6: Finding threshold based on business cost-benefit...")
    business_thresh, net_benefit = find_optimal_threshold_by_business_cost(
        y_test, results['y_prob'], v_saved=1000, c_retention=100
    )
    print(f"Optimal threshold (Business): {business_thresh:.3f}, Net Benefit = ${net_benefit:,.0f}")

    print("\nStep 6: Saving model and components...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'churn_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(models_dir, 'model_columns.pkl'))

    # Save thresholds for later use
    joblib.dump({
        'f1_threshold': best_f1_threshold,
        'business_threshold': business_thresh,
        'default_threshold': 0.5
    }, os.path.join(models_dir, 'optimal_thresholds.pkl'))

    # Save evaluation metrics
    save_model_evaluation_metrics(results, output_dir=models_dir)

    # Save ROC and PR curve data for visualization
    save_roc_curve_data(y_test, results['y_prob'], output_dir=models_dir)
    save_precision_recall_curve_data(y_test, results['y_prob'], output_dir=models_dir)

    print("Model and components saved successfully!")

    print("\nStep 7: Plotting feature importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Feature Importances")
        plt.bar(range(10), importances[indices], align='center')
        plt.xticks(range(10), [X_train.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'feature_importance.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Feature importance plot saved to {output_path}")

        # Also save features and their importance
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        feature_importance_df.to_csv(os.path.join(models_dir, 'feature_importance.csv'), index=False)
        print("Feature importance CSV saved to models/feature_importance.csv")

    print("\nStep 8: Performing customer segmentation...")
    customer_segments = perform_customer_segmentation(X_train_scaled, X_train, y_train)
    print("Customer segments identified:")
    print(customer_segments)

    print("\nStep 9: Generating SHAP feature impact visualization...")
    try:
        from explain_model import explain_with_shap
        # Use a small sample of the test data
        sample_size = min(100, len(X_test_scaled))
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        explain_with_shap(model, X_test_scaled[:sample_size],
                          os.path.join(artifacts_dir, 'shap_summary.png'))
        print("SHAP visualization generated successfully.")
    except Exception as e:
        print(f"Failed to generate SHAP visualization: {str(e)}")

    print("\n======= Pipeline Completed Successfully! =======")
    print(f"Model accuracy: {results['accuracy']:.4f}")
    print(f"Model precision: {results['precision']:.4f}")
    print(f"Model recall: {results['recall']:.4f}")
    print(f"Model F1 score: {results['f1_score']:.4f}")
    print(f"Model ROC AUC: {results['roc_auc']:.4f}")


def perform_customer_segmentation(X_train_scaled, X_train, y_train):
    from sklearn.cluster import KMeans

    cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_train_scaled[:, :3])

    data_with_clusters = X_train.copy()
    data_with_clusters['Cluster'] = clusters
    data_with_clusters['Churn'] = y_train.values

    cluster_churn = pd.DataFrame()
    cluster_churn['ChurnRate'] = data_with_clusters.groupby('Cluster')['Churn'].mean().round(3)
    cluster_churn['Count'] = data_with_clusters.groupby('Cluster').size()
    cluster_churn['Percentage'] = (cluster_churn['Count'] / cluster_churn['Count'].sum() * 100).round(1)
    cluster_profiles = data_with_clusters.groupby('Cluster')[cluster_features].mean().round(1)
    cluster_results = pd.merge(cluster_churn, cluster_profiles, left_index=True, right_index=True)

    # Save customer segments to artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    cluster_results.to_csv(os.path.join(artifacts_dir, 'customer_segments.csv'))

    plt.figure(figsize=(12, 6))
    plt.scatter(data_with_clusters['tenure'], data_with_clusters['MonthlyCharges'],
                c=clusters, cmap='viridis', s=50, alpha=0.7)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Monthly Charges ($)')
    plt.title('Customer Segments')
    plt.colorbar(label='Cluster')
    plt.savefig(os.path.join(artifacts_dir, 'customer_segments.png'))

    print("Customer segmentation completed!")
    return cluster_results


if __name__ == "__main__":
    main()