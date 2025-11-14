"""
XGBoost Model for Predicting Appendicitis Severity
This script trains an XGBoost model to predict the'Severity' column
using pre-split and balanced training/test data.

Features:
- Uses pre-split training data (already balanced with RandomOverSampler)
- Removes duplicate samples from training data
- Uses pre-split test data
- Performance metrics: AUROC and AUPR
- No cross-validation - direct training on training set
- No additional oversampling (data is already balanced)
"""

import pandas as pd
import numpy as np
import subprocess
subprocess.check_call(["pip", "install", "xgboost"])

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    precision_recall_fscore_support,
    make_scorer
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """Load the pre-split training and test datasets"""
    print("=" * 70)
    print("Loading pre-split training and test data...")
    print("=" * 70)

    # Load training data
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Training data loaded. Shape: {train_df.shape}")

    # Load test data
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test data loaded. Shape: {test_df.shape}")

    print(f"\nColumns in dataset: {train_df.columns.tolist()}")

    return train_df, test_df


def remove_duplicates(train_df, test_df):
    """Remove duplicate samples from training and test datasets"""
    print("\n" + "=" * 70)
    print("Removing duplicate samples...")
    print("=" * 70)

    # Check for duplicates in training set
    train_duplicates_before = train_df.duplicated().sum()
    print(f"\nTraining set:")
    print(f"  Shape before removing duplicates: {train_df.shape}")
    print(f"  Number of duplicate rows: {train_duplicates_before}")

    # Remove duplicates from training set
    train_df_clean = train_df.drop_duplicates()
    train_duplicates_removed = train_df.shape[0] - train_df_clean.shape[0]
    print(f"Duplicates removed: {train_duplicates_removed}")
    print(f"  Shape after removing duplicates: {train_df_clean.shape}")

    # Check for duplicates in test set
    test_duplicates_before = test_df.duplicated().sum()
    print(f"\nTest set:")
    print(f"  Shape before removing duplicates: {test_df.shape}")
    print(f"  Number of duplicate rows: {test_duplicates_before}")

    # Remove duplicates from test set
    test_df_clean = test_df.drop_duplicates()
    test_duplicates_removed = test_df.shape[0] - test_df_clean.shape[0]
    print(f"  Duplicates removed: {test_duplicates_removed}")
    print(f"  Shape after removing duplicates: {test_df_clean.shape}")

    return train_df_clean, test_df_clean


def prepare_features(train_df, test_df):
    """
    Prepare features for training by excluding specified columns
    Excluded columns:
    - US_Performed, US_Number, Severity, Management, Diagnosis
    """
    print("\n" + "=" * 70)
    print("Preparing features...")
    print("=" * 70)

    # Columns to exclude
    excluded_columns = ['US_Performed', 'US_Number', 'Severity', 'Management', 'Diagnosis']

    # Check which excluded columns exist in the dataset
    existing_excluded = [col for col in excluded_columns if col in train_df.columns]
    print(f"\nExcluded columns found in dataset: {existing_excluded}")

    # Get feature columns (all columns except excluded ones)
    feature_columns = [col for col in train_df.columns if col not in excluded_columns]
    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")

    # Separate features and target for training set
    X_train = train_df[feature_columns].copy()
    y_train = train_df['Severity'].copy()

    # Separate features and target for test set
    X_test = test_df[feature_columns].copy()
    y_test = test_df['Severity'].copy()

    print(f"\nTraining set:")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Target shape: {y_train.shape}")
    print(f"\nTraining set Severity distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTraining set Severity distribution (%):")
    print(y_train.value_counts(normalize=True).sort_index() * 100)

    print(f"\nTest set:")
    print(f"  Features shape: {X_test.shape}")
    print(f"  Target shape: {y_test.shape}")
    print(f"\nTest set Severity distribution:")
    print(y_test.value_counts().sort_index())
    print(f"\nTest set Severity distribution (%):")
    print(y_test.value_counts(normalize=True).sort_index() * 100)

    return X_train, X_test, y_train, y_test, feature_columns


def encode_categorical_features(X_train, X_test):
    """Encode categorical features if any"""
    print("\n" + "=" * 70)
    print("Checking for categorical features...")
    print("=" * 70)

    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) > 0:
        print(f"\nCategorical columns found: {categorical_cols}")

        # One-hot encode categorical variables
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

        # Ensure train and test have the same columns
        # Get all columns from training set
        train_cols = set(X_train_encoded.columns)
        test_cols = set(X_test_encoded.columns)

        # Add missing columns to test set with zeros
        for col in train_cols - test_cols:
            X_test_encoded[col] = 0

        # Remove extra columns from test set
        X_test_encoded = X_test_encoded[X_train_encoded.columns]

        print(f"After encoding:")
        print(f"  Training features shape: {X_train_encoded.shape}")
        print(f"  Test features shape: {X_test_encoded.shape}")

        return X_train_encoded, X_test_encoded
    else:
        print("\nNo categorical features found.")
        return X_train, X_test


def train_xgboost(X_train, y_train):
    """
    Train XGBoost model without cross-validation

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels

    Returns:
    --------
    model : trained XGBClassifier
    """
    print("\n" + "=" * 70)
    print("Training XGBoost model (No Cross-Validation)...")
    print("=" * 70)

    # Train with specified parameters
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # Train model on full training set
    print("\nTraining on full training set...")
    model.fit(X_train, y_train, verbose=False)

    print("\nModel Configuration:")
    print("-" * 70)
    print(f"  n_estimators: {model.n_estimators}")
    print(f"  max_depth: {model.max_depth}")
    print(f"  learning_rate: {model.learning_rate}")
    print(f"  subsample: {model.subsample}")
    print(f"  colsample_bytree: {model.colsample_bytree}")
    print(f"  min_child_weight: {model.min_child_weight}")
    print(f"  gamma: {model.gamma}")
    print(f"  reg_alpha: {model.reg_alpha}")
    print(f"  reg_lambda: {model.reg_lambda}")
    print(f"  random_state: {model.random_state}")

    print("\nModel training completed successfully.")

    return model


def calculate_auroc_aupr(y_true, y_pred_proba):
    """
    Calculate AUROC and AUPR metrics

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class

    Returns:
    --------
    auroc : float
        Area Under ROC Curve
    aupr : float
        Area Under Precision-Recall Curve
    """
    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_pred_proba)

    # Calculate AUPR (Average Precision)
    aupr = average_precision_score(y_true, y_pred_proba)

    return auroc, aupr


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the trained model with AUROC and AUPR metrics"""
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probability predictions for AUROC and AUPR
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auroc, train_aupr = calculate_auroc_aupr(y_train, y_train_proba)

    print("\n" + "-" * 70)
    print("Training Set Performance:")
    print("-" * 70)
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  AUROC (Area Under ROC Curve): {train_auroc:.4f}")
    print(f"  AUPR (Area Under Precision-Recall Curve): {train_aupr:.4f}")

    # Testing metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auroc, test_aupr = calculate_auroc_aupr(y_test, y_test_proba)

    print("\n" + "-" * 70)
    print("Test Set Performance:")
    print("-" * 70)
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  AUROC (Area Under ROC Curve): {test_auroc:.4f}")
    print(f"  AUPR (Area Under Precision-Recall Curve): {test_aupr:.4f}")

    # Classification report
    print("\n" + "-" * 70)
    print("Classification Report (Test Set):")
    print("-" * 70)
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    print("\n" + "-" * 70)
    print("Confusion Matrix (Test Set):")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - XGBoost (No CV)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'results/confusion_matrix_xgboost.png'")
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'XGBoost (AUROC = {test_auroc:.4f})', linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - XGBoost (No CV)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve_xgboost.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved as 'results/roc_curve_xgboost.png'")
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'XGBoost (AUPR = {test_aupr:.4f})', linewidth=2, color='green')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - XGBoost (No CV)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve_xgboost.png', dpi=300, bbox_inches='tight')
    print("Precision-Recall curve saved as 'results/precision_recall_curve_xgboost.png'")
    plt.close()

    # Combined ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    axes[0].plot(fpr, tpr, label=f'AUROC = {test_auroc:.4f}', linewidth=2, color='blue')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, label=f'AUPR = {test_aupr:.4f}', linewidth=2, color='green')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('XGBoost Performance (No Cross-Validation)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/combined_curves_xgboost.png', dpi=300, bbox_inches='tight')
    print("Combined curves saved as 'results/combined_curves_xgboost.png'")
    plt.close()

    return y_train_pred, y_test_pred, test_auroc, test_aupr


def plot_class_distribution(y_train, y_test):
    """Plot class distribution for training and test sets"""
    print("\n" + "=" * 70)
    print("Plotting Class Distribution")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set
    train_counts = y_train.value_counts().sort_index()
    axes[0].bar(train_counts.index, train_counts.values, color='lightgreen', edgecolor='black', alpha=0.8)
    axes[0].set_title('Training Set Class Distribution\n(After Removing Duplicates)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Severity Class', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts.values):
        axes[0].text(train_counts.index[i], v + 10, str(v), ha='center', va='bottom', fontsize=11)

    # Test set
    test_counts = y_test.value_counts().sort_index()
    axes[1].bar(test_counts.index, test_counts.values, color='lightcoral', edgecolor='black', alpha=0.8)
    axes[1].set_title('Test Set Class Distribution\n(After Removing Duplicates)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Severity Class', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_counts.values):
        axes[1].text(test_counts.index[i], v + 5, str(v), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('output/class_distribution_xgboost.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'output/class_distribution_xgboost.png'")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from XGBoost"""
    print("\n" + "=" * 70)
    print("Feature Importance Analysis")
    print("=" * 70)

    # Get feature importances
    importances = model.feature_importances_

    # Create dataframe for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    # Display top features
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 70)
    print(feature_importance.head(top_n).to_string(index=False))

    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - XGBoost (No CV)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved as 'output/feature_importance_xgboost.png'")
    plt.close()

    return feature_importance


def plot_learning_curve(model, X_train, X_test, y_train, y_test):
    """Plot learning curve showing training progress"""
    print("\n" + "=" * 70)
    print("Analyzing Learning Curve")
    print("=" * 70)

    # Train model with evaluation set to track performance
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model_eval = XGBClassifier(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        learning_rate=model.learning_rate,
        subsample=model.subsample,
        colsample_bytree=model.colsample_bytree,
        min_child_weight=model.min_child_weight,
        gamma=model.gamma,
        reg_alpha=model.reg_alpha,
        reg_lambda=model.reg_lambda,
        random_state=model.random_state,
        n_jobs=model.n_jobs,
        eval_metric='logloss'
    )

    model_eval.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Get evaluation results
    results = model_eval.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # Plot log loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train', linewidth=2)
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test', linewidth=2)
    ax.legend(fontsize=11)
    plt.ylabel('Log Loss', fontsize=12)
    plt.xlabel('Number of Boosting Rounds', fontsize=12)
    plt.title('XGBoost Learning Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/learning_curve_xgboost.png', dpi=300, bbox_inches='tight')
    print("Learning curve plot saved as 'output/learning_curve_xgboost.png'")
    plt.close()


def plot_estimator_performance(model, X_train, X_test, y_train, y_test):
    """Plot how performance changes with number of estimators"""
    print("\n" + "=" * 70)
    print("Analyzing Performance vs Number of Estimators")
    print("=" * 70)

    n_estimators_range = range(1, model.n_estimators + 1, max(1, model.n_estimators // 20))
    train_scores = []
    test_scores = []

    print("Testing different numbers of estimators...")
    for n in n_estimators_range:
        # Create model with n estimators
        temp_model = XGBClassifier(
            n_estimators=n,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            min_child_weight=model.min_child_weight,
            gamma=model.gamma,
            reg_alpha=model.reg_alpha,
            reg_lambda=model.reg_lambda,
            random_state=model.random_state,
            n_jobs=model.n_jobs,
            eval_metric='logloss'
        )
        temp_model.fit(X_train, y_train, verbose=False)

        train_proba = temp_model.predict_proba(X_train)[:, 1]
        test_proba = temp_model.predict_proba(X_test)[:, 1]

        train_score = roc_auc_score(y_train, train_proba)
        test_score = roc_auc_score(y_test, test_proba)

        train_scores.append(train_score)
        test_scores.append(test_score)# Plot
    plt.figure(figsize=(12, 6))
    plt.plot(n_estimators_range, train_scores, label='Training Score', marker='o', linewidth=2)
    plt.plot(n_estimators_range, test_scores, label='Test Score', marker='s', linewidth=2)
    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.title('XGBoost Performance vs Number of Estimators', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/estimators_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"Estimators vs Performance plot saved as 'output/estimators_vs_performance.png'")
    plt.close()


def save_results_summary(model, test_auroc, test_aupr, feature_importance, y_test, y_test_pred):
    """Save a summary of results to a text file"""
    print("\n" + "=" * 70)
    print("Saving Results Summary")
    print("=" * 70)

    with open('results/xgboost_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBoost Model Results (No Cross-Validation)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Training Method:\n")
        f.write("-" * 70 + "\n")
        f.write("Direct training on full training set (No Cross-Validation)\n")
        f.write("Duplicate samples removed before training\n")
        f.write("  Single train-test split evaluation\n\n")

        f.write("Model Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  n_estimators: {model.n_estimators}\n")
        f.write(f"  max_depth: {model.max_depth}\n")
        f.write(f"  learning_rate: {model.learning_rate}\n")
        f.write(f"  subsample: {model.subsample}\n")
        f.write(f"  colsample_bytree: {model.colsample_bytree}\n")
        f.write(f"  min_child_weight: {model.min_child_weight}\n")
        f.write(f"  gamma: {model.gamma}\n")
        f.write(f"  reg_alpha: {model.reg_alpha}\n")
        f.write(f"  reg_lambda: {model.reg_lambda}\n")
        f.write(f"  random_state: {model.random_state}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Test Set Performance:\n")
        f.write("=" * 70 + "\n")
        f.write(f"AUROC: {test_auroc:.4f}\n")
        f.write(f"AUPR: {test_aupr:.4f}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_test_pred))

        f.write("\n" + "=" * 70 + "\n")
        f.write("Top 20 Feature Importances:\n")
        f.write("=" * 70 + "\n")
        f.write(feature_importance.head(20).to_string(index=False))
        f.write("\n")
        print("Results summary saved as 'results/xgboost_summary.txt'")


def main():
    """Main execution function"""
    os.chdir("../../")

    print("\n" + "=" * 70)
    print("XGBoost for Appendicitis Severity Prediction")
    print("Training Method: Direct Training (No Cross-Validation)")
    print("Duplicate Removal: Enabled")
    print("=" * 70)

    # Define file paths
    train_path = 'data/train_severity1_resampled.csv'
    test_path = 'data/test_severity1.csv'

    # Load pre-split data
    train_df, test_df = load_data(train_path, test_path)

    # Remove duplicate samples
    train_df, test_df = remove_duplicates(train_df, test_df)

    # Prepare features
    X_train, X_test, y_train, y_test, feature_columns = prepare_features(train_df, test_df)

    # Encode categorical features
    X_train, X_test = encode_categorical_features(X_train, X_test)

    # Update feature names after encoding
    final_feature_names = X_train.columns.tolist()
    print(f"\nFinal number of features: {len(final_feature_names)}")

    # Note: XGBoost doesn't require feature scaling
    print("\n" + "=" * 70)
    print("Note: XGBoost does not require feature scaling")
    print("=" * 70)

    # Plot class distribution
    plot_class_distribution(y_train, y_test)

    # Train model (No Cross-Validation)
    model = train_xgboost(X_train, y_train)

    # Evaluate model with AUROC and AUPR
    y_train_pred, y_test_pred, test_auroc, test_aupr = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )

    # Plot feature importance
    feature_importance = plot_feature_importance(model, final_feature_names, top_n=20)

    # Plot learning curve
    plot_learning_curve(model, X_train, X_test, y_train, y_test)

    # Plot estimator performance analysis
    plot_estimator_performance(model, X_train, X_test, y_train, y_test)

    # Save results summary
    save_results_summary(model, test_auroc, test_aupr, feature_importance, y_test, y_test_pred)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Training Method: Direct Training (No Cross-Validation)")
    print(f"Duplicate Removal: Enabled")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPR: {test_aupr:.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nAll results saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()