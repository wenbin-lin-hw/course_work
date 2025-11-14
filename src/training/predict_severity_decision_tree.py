"""
Decision Tree Model for Predicting Appendicitis Severity
This script trains a decision tree model to predict the'Severity' column
using pre-split and balanced training/test data.

Features:
- Uses pre-split training data (already balanced with RandomOverSampler)
- Uses pre-split test data
- Performance metrics: AUROC and AUPR
- No additional oversampling (data is already balanced)


"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    make_scorer,
    precision_score,
    recall_score,
    f1_score
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


def train_decision_tree(X_train, y_train):
    """
    Train decision tree model

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels

    Returns:
    --------
    model : trained DecisionTreeClassifier
    """
    print("\n" + "=" * 70)
    print("Training Decision Tree model...")
    print("=" * 70)

    # Train with default parameters
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        criterion='gini',
        random_state=42
    )

    # Train model on full training set
    model.fit(X_train, y_train)

    print("\nModel Configuration:")
    print("-" * 70)
    print(f"  max_depth: {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  min_samples_leaf: {model.min_samples_leaf}")
    print(f"  criterion: {model.criterion}")

    print("\nModel training completed.")

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
    plt.title('Confusion Matrix - Decision Tree')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_decision_tree.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'results/confusion_matrix_decision_tree.png'")
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'Decision Tree (AUROC = {test_auroc:.4f})', linewidth=2, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Decision Tree', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curve_decision_tree.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved as 'results/roc_curve_decision_tree.png'")
    plt.close()

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'Decision Tree (AUPR = {test_aupr:.4f})', linewidth=2, color='green')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Decision Tree', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/precision_recall_curve_decision_tree.png', dpi=300, bbox_inches='tight')
    print("Precision-Recall curve saved as 'results/precision_recall_curve_decision_tree.png'")
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

    plt.suptitle('Decision Tree Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/combined_curves_decision_tree.png', dpi=300, bbox_inches='tight')
    print("Combined curves saved as 'results/combined_curves_decision_tree.png'")
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
    axes[0].set_title('Training Set Class Distribution\n(Already Balanced)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Severity Class', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts.values):
        axes[0].text(train_counts.index[i], v + 10, str(v), ha='center', va='bottom', fontsize=11)

    # Test set
    test_counts = y_test.value_counts().sort_index()
    axes[1].bar(test_counts.index, test_counts.values, color='lightcoral', edgecolor='black', alpha=0.8)
    axes[1].set_title('Test Set Class Distribution\n(Original Distribution)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Severity Class', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_counts.values):
        axes[1].text(test_counts.index[i], v + 5, str(v), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('output/class_distribution_decision_tree.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'output/class_distribution_decision_tree.png'")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from decision tree"""
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
    plt.title(f'Top {top_n} Feature Importances - Decision Tree', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/feature_importance_decision_tree.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved as 'output/feature_importance_decision_tree.png'")
    plt.close()

    return feature_importance


def plot_decision_tree_structure(model, feature_names, max_depth=3):
    """Plot the decision tree structure (limited depth for visualization)"""
    print("\n" + "=" * 70)
    print("Plotting Decision Tree Structure")
    print("=" * 70)

    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=max_depth
    )
    plt.title(f'Decision Tree Structure (Max Depth={max_depth} for visualization)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/decision_tree_structure.png', dpi=300, bbox_inches='tight')
    print(f"Decision tree structure saved as 'output/decision_tree_structure.png'")
    print(f"Note: Only showing first {max_depth} levels for clarity")
    plt.close()


def save_results_summary(model, test_auroc, test_aupr, y_test, y_test_pred):
    """Save a summary of results to a text file"""
    print("\n" + "=" * 70)
    print("Saving Results Summary")
    print("=" * 70)

    with open('results/decision_tree_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Decision Tree Model Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  max_depth: {model.max_depth}\n")
        f.write(f"  min_samples_split: {model.min_samples_split}\n")
        f.write(f"  min_samples_leaf: {model.min_samples_leaf}\n")
        f.write(f"  criterion: {model.criterion}\n")

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


def main():
    os.chdir("../../")
    """Main execution function"""
    print("\n" + "=" * 70)
    print("Decision Tree for Appendicitis Severity Prediction")
    print("=" * 70)

    # Define file paths
    train_path = 'data/appendicitis/train_data_balanced.csv'
    test_path = 'data/appendicitis/test_data.csv'

    # Load pre-split data
    train_df, test_df = load_data(train_path, test_path)

    # Prepare features
    X_train, X_test, y_train, y_test, feature_columns = prepare_features(train_df, test_df)

    # Encode categorical features
    X_train, X_test = encode_categorical_features(X_train, X_test)

    # Update feature names after encoding
    final_feature_names = X_train.columns.tolist()
    print(f"\nFinal number of features: {len(final_feature_names)}")

    # Plot class distribution
    plot_class_distribution(y_train, y_test)

    # Train model
    model = train_decision_tree(X_train, y_train)

    # Evaluate model with AUROC and AUPR
    y_train_pred, y_test_pred, test_auroc, test_aupr = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )


    # Plot decision tree structure
    plot_decision_tree_structure(model, final_feature_names, max_depth=3)

    # Save results summary
    save_results_summary(model, test_auroc, test_aupr, y_test, y_test_pred)

    # Calculate additional metrics for final summary
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPR: {test_aupr:.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("\nAll results saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()