"""
Model training and evaluation functions.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import joblib
import os


def handle_class_imbalance(X_train, y_train, method='smote'):
    """Apply class balancing technique to training data."""
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
            print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts()}")
            return X_balanced, y_balanced
        except ImportError:
            print("SMOTE not available. Install imblearn package for SMOTE. Using original data.")
            return X_train, y_train

    elif method == 'class_weight':
        # Return original data with a note that class_weight='balanced' should be used
        print("Using class_weight='balanced' in model training")
        return X_train, y_train

    elif method == 'undersampling':
        try:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
            print(f"Original class distribution: {pd.Series(y_train).value_counts()}")
            print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts()}")
            return X_balanced, y_balanced
        except ImportError:
            print("RandomUnderSampler not available. Install imblearn package. Using original data.")
            return X_train, y_train

    return X_train, y_train


def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000, class_weight=class_weight, solver='liblinear', C=0.1)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, class_weight='balanced'):
    """Train a random forest model with optimized parameters."""
    model = RandomForestClassifier(
        n_estimators=200,      # Increased from 100
        max_depth=12,          # Increased from 10
        min_samples_split=5,   # Added parameter
        min_samples_leaf=2,    # Reduced from 4
        max_features='sqrt',   # Added parameter
        bootstrap=True,        # Added parameter
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1              # Use all available cores
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, scale_pos_weight=None):
    """Train an XGBoost model."""
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        # Recommended value for imbalanced data: sum(negative_cases) / sum(positive_cases)
        scale_pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=200
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, is_unbalanced=True):
    """Train a LightGBM model."""
    model = lgb.LGBMClassifier(
        is_unbalanced=is_unbalanced,
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=10
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, class_weights=None):
    """Train a CatBoost model."""
    # Set class weights if not provided
    if class_weights is None:
        pos_weight = len(y_train) / max(1, sum(y_train))
        class_weights = {0: 1.0, 1: pos_weight}

    model = CatBoostClassifier(
        verbose=0,
        class_weights=class_weights,
        iterations=200,
        depth=6,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting model."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_ensemble(models, X_train, y_train):
    """Train an ensemble model using multiple base models."""
    # Create named estimators from models dictionary
    estimators = [(name, model) for name, model in models.items()]

    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft'  # Use probability outputs for voting
    )

    # Train the ensemble
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Calculate specific metrics
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold that maximizes F1 score."""
    thresholds = np.linspace(0.1, 0.9, 101)  # 0.1 to 0.9 with step 0.01
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Also find optimal thresholds for precision and recall
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))

    print(f"Best threshold for F1: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.plot(thresholds, precision_scores, 'g-', label='Precision')
    plt.plot(thresholds, recall_scores, 'r-', label='Recall')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Optimal Threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    threshold_plot_path = os.path.join(models_dir, 'threshold_optimization.png')
    plt.savefig(threshold_plot_path)
    print(f"Threshold optimization plot saved to {threshold_plot_path}")

    return best_threshold, best_f1


def find_optimal_threshold_by_business_cost(y_true, y_prob, v_saved=1000, c_retention=100):
    """Find threshold that maximizes business benefit."""
    thresholds = np.linspace(0.1, 0.9, 101)  # 0.1 to 0.9 with step 0.01
    net_benefits = []

    for threshold in thresholds:
        # Make predictions with this threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Count true positives (correctly identified churners)
        tp = ((y_pred == 1) & (y_true == 1)).sum()

        # Count false positives (falsely identified as churners)
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        # Calculate cost and benefit
        retention_cost = (tp + fp) * c_retention  # Cost of retention efforts for all predicted churners
        saved_revenue = tp * v_saved  # Revenue saved from retained customers
        net_benefit = saved_revenue - retention_cost

        net_benefits.append(net_benefit)

    # Find threshold with maximum net benefit
    best_idx = np.argmax(net_benefits)
    best_threshold = thresholds[best_idx]
    best_benefit = net_benefits[best_idx]

    print(f"Best threshold for business value: {best_threshold:.2f} (Net Benefit: ${best_benefit:,.2f})")

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, 'g-')
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Optimal Threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit ($)')
    plt.title('Business Value Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    business_plot_path = os.path.join(models_dir, 'business_optimization.png')
    plt.savefig(business_plot_path)
    print(f"Business optimization plot saved to {business_plot_path}")

    return best_threshold, best_benefit


def save_model_evaluation_metrics(model_results, output_dir='models'):
    """Save model evaluation metrics to JSON file."""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Create a version of results that's JSON serializable
    serializable_results = {
        'accuracy': float(model_results['accuracy']),
        'precision': float(model_results['precision']),
        'recall': float(model_results['recall']),
        'f1_score': float(model_results['f1_score']),
        'roc_auc': float(model_results['roc_auc']),
        'avg_precision': float(model_results['avg_precision'])
    }

    output_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Model evaluation metrics saved to {output_path}")


def save_roc_curve_data(y_true, y_prob, output_dir='models'):
    """Save ROC curve data for later use in visualization."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Save as numpy array
    output_path = os.path.join(output_dir, 'roc_data.npz')
    np.savez(output_path, fpr=fpr, tpr=tpr, thresholds=thresholds)

    print(f"ROC curve data saved to {output_path}")


def save_precision_recall_curve_data(y_true, y_prob, output_dir='models'):
    """Save precision-recall curve data for visualization."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Save as numpy array
    output_path = os.path.join(output_dir, 'pr_curve_data.npz')
    np.savez(output_path, precision=precision, recall=recall, thresholds=thresholds)

    print(f"Precision-recall curve data saved to {output_path}")