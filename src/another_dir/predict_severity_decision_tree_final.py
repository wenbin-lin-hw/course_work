"""
Decision Tree Model for Predicting Appendicitis Severity with Pre-resampled Data
This script trains a decision tree model to predict the 'Severity' column
using the processed appendicitis dataset with pre-resampled training data.

Training data: data/train_severity1_resampled.csv (already oversampled)
Test data: data/test_severity1.csv
Cross-validation: 5-fold CV

Note: Oversampling has been done before running this script, so SMOTE, 
RandomOverSampler, and ADASYN methods are not needed.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(train_filepath, test_filepath):
    """Load the appendicitis training and test datasets from CSV files"""
    print("Loading data...")
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)
    print(f"Training data loaded successfully. Shape: {train_df.shape}")
    print(f"Test data loaded successfully. Shape: {test_df.shape}")
    print(f"\nColumns in training dataset: {train_df.columns.tolist()}")
    print(f"\nColumns in test dataset: {test_df.columns.tolist()}")
    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Prepare features for training by excluding target column
    
    Training target: Severity
    Test target: Severity
    """
    print("\n" + "="*60)
    print("Preparing features...")
    
    # Get feature columns (all columns except target)
    train_feature_columns = [col for col in train_df.columns if col != 'Severity']
    test_feature_columns = [col for col in test_df.columns if col != 'Severity']
    
    print(f"\nTraining feature columns ({len(train_feature_columns)}): {train_feature_columns}")
    print(f"Test feature columns ({len(test_feature_columns)}): {test_feature_columns}")
    
    # Separate features and target
    X_train = train_df[train_feature_columns].copy()
    y_train = train_df['Severity'].copy()
    
    X_test = test_df[test_feature_columns].copy()
    y_test = test_df['Severity'].copy()
    
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    print(f"\nTraining target variable (Severity) distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTraining target variable distribution percentage:")
    print(y_train.value_counts(normalize=True).sort_index() * 100)
    
    print(f"\nTest target variable (Severity) distribution:")
    print(y_test.value_counts().sort_index())
    print(f"\nTest target variable distribution percentage:")
    print(y_test.value_counts(normalize=True).sort_index() * 100)
    
    return X_train, X_test, y_train, y_test, train_feature_columns


def encode_categorical_features(X):
    """Encode categorical features if any"""
    print("\n" + "="*60)
    print("Checking for categorical features...")
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"Categorical columns found: {categorical_cols}")
        
        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"After encoding, feature shape: {X_encoded.shape}")
        return X_encoded
    else:
        print("No categorical features found.")
        return X


def train_decision_tree(X_train, y_train, tune_hyperparameters=True, cv_folds=5):
    """
    Train decision tree model with optional hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    tune_hyperparameters : bool, default=True
        Whether to perform hyperparameter tuning using GridSearchCV
    cv_folds : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    model : trained DecisionTreeClassifier
    best_params : dict of best parameters (if tuning enabled)
    cv_scores : cross-validation scores
    """
    print("\n" + "="*60)
    print("Training Decision Tree model...")
    
    if tune_hyperparameters:
        print(f"Performing hyperparameter tuning with {cv_folds}-fold Cross-Validation...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
        
        # Create base model
        base_model = DecisionTreeClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_scores = grid_search.best_score_
        
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest {cv_folds}-fold cross-validation score: {cv_scores:.4f}")
        
    else:
        print(f"Training with default parameters and {cv_folds}-fold cross-validation...")
        # Train with reasonable default parameters
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            criterion='gini',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        print(f"\n{cv_folds}-fold Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        best_params = None
    
    print("Model training completed.")
    
    return model, best_params, cv_scores


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    # Testing accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\n" + "-"*60)
    print("Classification Report (Test Set):")
    print("-"*60)
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\n" + "-"*60)
    print("Confusion Matrix (Test Set):")
    print("-"*60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - Decision Tree (with Over-sampling)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../../results/confusion_matrix_decision_tree_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix_decision_tree_oversampling.png'")
    plt.close()
    
    # ROC curve and AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        y_test_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_proba)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Decision Tree (with Over-sampling)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../../results/roc_curve_decision_tree_oversampling.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved as 'roc_curve_decision_tree_oversampling.png'")
        plt.close()
    
    return y_train_pred, y_test_pred


def plot_class_distribution(y_train, y_test):
    """Plot class distribution for training and test sets"""
    print("\n" + "="*60)
    print("Plotting Class Distribution")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set (pre-resampled)
    train_counts = y_train.value_counts().sort_index()
    axes[0].bar(train_counts.index, train_counts.values, color='lightgreen', edgecolor='black')
    axes[0].set_title('Training Set\n(Pre-resampled Data)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts.values):
        axes[0].text(train_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    
    # Test set
    test_counts = y_test.value_counts().sort_index()
    axes[1].bar(test_counts.index, test_counts.values, color='lightcoral', edgecolor='black')
    axes[1].set_title('Test Set')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Samples')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_counts.values):
        axes[1].text(test_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../../output/class_distribution_decision_tree_oversampling.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'class_distribution_decision_tree_oversampling.png'")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from decision tree"""
    print("\n" + "="*60)
    print("Feature Importance")
    print("="*60)
    
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
    print(feature_importance.head(top_n).to_string(index=False))
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - Decision Tree (with Over-sampling)')
    plt.tight_layout()
    plt.savefig('../../output/feature_importance_decision_tree_oversampling.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature importance plot saved as 'feature_importance_decision_tree_oversampling.png'")
    plt.close()
    
    return feature_importance


def plot_decision_tree_structure(model, feature_names, max_depth=3):
    """Plot the decision tree structure (limited depth for visualization)"""
    print("\n" + "="*60)
    print("Plotting Decision Tree Structure")
    print("="*60)
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=max_depth
    )
    plt.title(f'Decision Tree Structure (Max Depth={max_depth} for visualization)')
    plt.tight_layout()
    plt.savefig('../../output/decision_tree_structure_oversampling.png', dpi=300, bbox_inches='tight')
    print(f"Decision tree structure saved as 'decision_tree_structure_oversampling.png'")
    print(f"Note: Only showing first {max_depth} levels for clarity")
    plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("Decision Tree for Appendicitis Severity Prediction")
    print("WITH PRE-RESAMPLED TRAINING DATA")
    print("Using 5-fold Cross-Validation")
    print("="*60)
    
    # Load data
    train_filepath = '../../data/train_severity1_resampled.csv'
    test_filepath = '../../data/test_severity1.csv'
    train_df, test_df = load_data(train_filepath, test_filepath)
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_columns = prepare_features(train_df, test_df)
    
    # Encode categorical features
    X_train = encode_categorical_features(X_train)
    X_test = encode_categorical_features(X_test)
    
    # Update feature names after encoding
    final_feature_names = X_train.columns.tolist()
    print(f"\nFinal number of features: {len(final_feature_names)}")
    
    # Ensure test set has the same features
    # Add missing columns with zeros
    for col in final_feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    # Remove extra columns and reorder
    X_test = X_test[final_feature_names]
    print(f"Test set aligned with training features: {len(X_test.columns)} features")
    
    # Plot class distribution
    plot_class_distribution(y_train, y_test)
    
    # Train model with hyperparameter tuning using 5-fold CV
    cv_folds = 5
    print(f"\nUsing {cv_folds}-fold cross-validation for model training and evaluation")
    model, best_params, cv_scores = train_decision_tree(
        X_train, y_train, tune_hyperparameters=True, cv_folds=cv_folds
    )
    
    # Evaluate model
    y_train_pred, y_test_pred = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )
    
    # Plot feature importance
    feature_importance = plot_feature_importance(model, final_feature_names, top_n=20)
    
    # Plot decision tree structure
    plot_decision_tree_structure(model, final_feature_names, max_depth=3)
    
    print("\n" + "="*60)
    print("Script completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
