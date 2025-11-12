"""
Logistic Regression Model for Predicting Appendicitis Severity with Over-sampling
This script trains a logistic regression model to predict the 'Severity' column
using the processed appendicitis dataset with SMOTE over-sampling to balance classes.

Excluded columns for training:
- US_Performed
- US_Number
- Severity (target variable)
- Management
- Diagnosis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the appendicitis dataset from Excel file"""
    print("Loading data...")
    df = pd.read_excel(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"\nColumns in dataset: {df.columns.tolist()}")
    return df


def prepare_features(df):
    """
    Prepare features for training by excluding specified columns
    
    Excluded columns:
    - US_Performed, US_Number, Severity, Management, Diagnosis
    """
    print("\n" + "="*60)
    print("Preparing features...")
    
    # Columns to exclude
    excluded_columns = ['US_Performed', 'US_Number', 'Severity', 'Management', 'Diagnosis']
    
    # Check which excluded columns exist in the dataset
    existing_excluded = [col for col in excluded_columns if col in df.columns]
    print(f"Excluded columns found in dataset: {existing_excluded}")
    
    # Get feature columns (all columns except excluded ones)
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
    
    # Separate features and target
    X = df[feature_columns].copy()
    y = df['Severity'].copy()
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget variable distribution (BEFORE over-sampling):")
    print(y.value_counts().sort_index())
    print(f"\nTarget variable distribution percentage (BEFORE over-sampling):")
    print(y.value_counts(normalize=True).sort_index() * 100)
    
    return X, y, feature_columns



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


def apply_oversampling(X_train, y_train, method='SMOTE'):
    """
    Apply over-sampling to balance the training dataset
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    method : str, default='SMOTE'
        Over-sampling method: 'SMOTE', 'RandomOverSampler', or 'ADASYN'
    
    Returns:
    --------
    X_resampled, y_resampled : resampled training data
    """
    print("\n" + "="*60)
    print(f"Applying Over-sampling Method: {method}")
    print("="*60)
    
    # Display original class distribution
    print("\nOriginal training set class distribution:")
    original_distribution = Counter(y_train)
    for label, count in sorted(original_distribution.items()):
        print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.2f}%)")
    
    # Apply over-sampling
    if method == 'SMOTE':
        # SMOTE: Synthetic Minority Over-sampling Technique
        oversampler = SMOTE(random_state=42, k_neighbors=5)
        print("\nUsing SMOTE (Synthetic Minority Over-sampling Technique)")
        print("SMOTE generates synthetic samples by interpolating between existing minority samples")
        
    elif method == 'RandomOverSampler':
        # Random Over-sampling: Randomly duplicate minority class samples
        oversampler = RandomOverSampler(random_state=42)
        print("\nUsing Random Over-sampling")
        print("Randomly duplicates minority class samples")
        
    elif method == 'ADASYN':
        # ADASYN: Adaptive Synthetic Sampling
        oversampler = ADASYN(random_state=42, n_neighbors=5)
        print("\nUsing ADASYN (Adaptive Synthetic Sampling)")
        print("ADASYN generates more synthetic samples for harder-to-learn minority samples")
    
    else:
        raise ValueError(f"Unknown over-sampling method: {method}")
    
    # Resample the training data
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    
    # Display new class distribution
    print("\nResampled training set class distribution:")
    resampled_distribution = Counter(y_resampled)
    for label, count in sorted(resampled_distribution.items()):
        print(f"  Class {label}: {count} samples ({count/len(y_resampled)*100:.2f}%)")
    
    print(f"\nTraining set size:")
    print(f"  Before over-sampling: {len(y_train)} samples")
    print(f"  After over-sampling: {len(y_resampled)} samples")
    print(f"  Increase: {len(y_resampled) - len(y_train)} samples ({(len(y_resampled)/len(y_train) - 1)*100:.2f}%)")
    
    return X_resampled, y_resampled


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    Important for Logistic Regression as it's sensitive to feature scales
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    
    Returns:
    --------
    X_train_scaled, X_test_scaled, scaler
    """
    print("\n" + "="*60)
    print("Scaling features...")
    print("Note: Feature scaling is crucial for Logistic Regression")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled using StandardScaler")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, tune_hyperparameters=True):
    """
    Train logistic regression model with optional hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features (should be scaled)
    y_train : array-like
        Training labels
    tune_hyperparameters : bool, default=True
        Whether to perform hyperparameter tuning using GridSearchCV
    
    Returns:
    --------
    model : trained LogisticRegression
    best_params : dict of best parameters (if tuning enabled)
    """
    print("\n" + "="*60)
    print("Training Logistic Regression model...")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with GridSearchCV...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l2', 'none'],  # Regularization type
            'solver': ['lbfgs', 'saga'],  # Optimization algorithm
            'max_iter': [500, 1000, 2000],  # Maximum iterations
            'class_weight': [None, 'balanced']  # Handle class imbalance
        }
        
        # Create base model
        base_model = LogisticRegression(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
        
    else:
        print("Training with default parameters...")
        # Train with reasonable default parameters
        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        best_params = None
    
    print("Model training completed.")
    
    return model, best_params


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
    plt.title('Confusion Matrix - Logistic Regression (with Over-sampling)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../../results/confusion_matrix_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix_logistic_regression_oversampling.png'")
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
        plt.title('ROC Curve - Logistic Regression (with Over-sampling)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../../results/roc_curve_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved as 'roc_curve_logistic_regression_oversampling.png'")
        plt.close()
    
    return y_train_pred, y_test_pred


def plot_class_distribution(y_original, y_resampled, y_test):
    """Plot class distribution before and after over-sampling"""
    print("\n" + "="*60)
    print("Plotting Class Distribution")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original training set
    original_counts = y_original.value_counts().sort_index()
    axes[0].bar(original_counts.index, original_counts.values, color='skyblue', edgecolor='black')
    axes[0].set_title('Original Training Set\n(Before Over-sampling)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(original_counts.values):
        axes[0].text(original_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    
    # Resampled training set
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    axes[1].bar(resampled_counts.index, resampled_counts.values, color='lightgreen', edgecolor='black')
    axes[1].set_title('Resampled Training Set\n(After Over-sampling)')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Number of Samples')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(resampled_counts.values):
        axes[1].text(resampled_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    
    # Test set (unchanged)
    test_counts = y_test.value_counts().sort_index()
    axes[2].bar(test_counts.index, test_counts.values, color='lightcoral', edgecolor='black')
    axes[2].set_title('Test Set\n(Unchanged)')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Number of Samples')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_counts.values):
        axes[2].text(test_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../../output/class_distribution_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'class_distribution_logistic_regression_oversampling.png'")
    plt.close()


def plot_feature_coefficients(model, feature_names, top_n=20):
    """
    Plot feature coefficients from logistic regression
    Coefficients indicate the importance and direction of influence
    """
    print("\n" + "="*60)
    print("Feature Coefficients")
    print("="*60)
    
    # Get coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        
        # Create dataframe for visualization
        feature_coef = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        feature_coef = feature_coef.sort_values('Abs_Coefficient', ascending=False)
        
        # Display top features
        print(f"\nTop {top_n} Most Important Features (by absolute coefficient):")
        print(feature_coef.head(top_n)[['Feature', 'Coefficient']].to_string(index=False))
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_coef.head(top_n)
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Coefficients - Logistic Regression (with Over-sampling)\n(Green=Positive, Red=Negative)')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig('../../output/feature_coefficients_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature coefficients plot saved as 'feature_coefficients_logistic_regression_oversampling.png'")
        plt.close()
        
        return feature_coef
    else:
        print("Model does not have coefficients (might be using a different penalty)")
        return None


def plot_probability_distribution(model, X_test, y_test):
    """Plot predicted probability distribution"""
    print("\n" + "="*60)
    print("Plotting Probability Distribution")
    print("="*60)
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)
    
    # For binary classification
    if y_proba.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram for each class
        for class_label in np.unique(y_test):
            mask = y_test == class_label
            plt.hist(y_proba[mask, 1], bins=30, alpha=0.6, label=f'Actual Class {class_label}')
        
        plt.xlabel('Predicted Probability (Class 1)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities - Logistic Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../../output/probability_distribution_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
        print("Probability distribution plot saved as 'probability_distribution_logistic_regression_oversampling.png'")
        plt.close()


def compare_methods(X_train, X_test, y_train, y_test, final_feature_names):
    """Compare different over-sampling methods"""
    print("\n" + "="*60)
    print("Comparing Different Over-sampling Methods")
    print("="*60)
    
    methods = ['SMOTE', 'RandomOverSampler', 'ADASYN']
    results = []
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing Method: {method}")
        print(f"{'='*60}")
        
        # Apply over-sampling
        X_train_resampled, y_train_resampled = apply_oversampling(X_train, y_train, method=method)
        
        # Scale features
        X_train_scaled, X_test_scaled, _ = scale_features(X_train_resampled, X_test)
        
        # Train model (without hyperparameter tuning for speed)
        model, _ = train_logistic_regression(X_train_scaled, y_train_resampled, tune_hyperparameters=False)
        
        # Evaluate
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Get precision, recall, f1 for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
        
        results.append({
            'Method': method,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"\n{method} Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # Plot comparison
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.bar(results_df['Method'], results_df[metric], color=colors)
        ax.set_xlabel('Over-sampling Method')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim([0, 1])
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/oversampling_methods_comparison_logistic_regression.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'oversampling_methods_comparison_logistic_regression.png'")
    plt.close()
    
    return results_df


def perform_cross_validation(model, X, y, cv=5):
    """Perform cross-validation to assess model stability"""
    print("\n" + "="*60)
    print("Performing Cross-Validation")
    print("="*60)
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\nCross-Validation Results ({cv}-fold):")
    print(f"  Scores: {cv_scores}")
    print(f"  Mean Accuracy: {cv_scores.mean():.4f}")
    print(f"  Standard Deviation: {cv_scores.std():.4f}")
    print(f"  Min Accuracy: {cv_scores.min():.4f}")
    print(f"  Max Accuracy: {cv_scores.max():.4f}")
    
    # Plot CV scores
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, cv+1), cv_scores, color='steelblue', alpha=0.7)
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'{cv}-Fold Cross-Validation Scores - Logistic Regression')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../../output/cross_validation_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nCross-validation plot saved as 'cross_validation_logistic_regression_oversampling.png'")
    plt.close()
    
    return cv_scores


def main():
    """Main execution function"""
    print("="*60)
    print("Logistic Regression for Appendicitis Severity Prediction")
    print("WITH OVER-SAMPLING FOR CLASS BALANCE")
    print("="*60)
    
    # Load data
    filepath = '../../data/appendicitis/processed_appendicitis_data_final.xlsx'
    df = load_data(filepath)
    
    # Prepare features
    X, y, feature_columns = prepare_features(df)

    
    # Encode categorical features
    X = encode_categorical_features(X)
    
    # Update feature names after encoding
    final_feature_names = X.columns.tolist()
    print(f"\nFinal number of features: {len(final_feature_names)}")
    
    # Split data into train and test sets
    print("\n" + "="*60)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Store original y_train for comparison
    y_train_original = y_train.copy()
    
    # Apply over-sampling (default: SMOTE)
    X_train_resampled, y_train_resampled = apply_oversampling(
        X_train, y_train, method='SMOTE'
    )
    
    # Plot class distribution
    plot_class_distribution(y_train_original, y_train_resampled, y_test)
    
    # Scale features (IMPORTANT for Logistic Regression)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)
    
    # Train model with resampled data and hyperparameter tuning
    model, best_params = train_logistic_regression(
        X_train_scaled, y_train_resampled, tune_hyperparameters=True
    )
    
    # Evaluate model
    y_train_pred, y_test_pred = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train_resampled, y_test
    )
    
    # Plot feature coefficients
    feature_coef = plot_feature_coefficients(model, final_feature_names, top_n=20)
    
    # Plot probability distribution
    plot_probability_distribution(model, X_test_scaled, y_test)
    
    # Perform cross-validation
    cv_scores = perform_cross_validation(model, X_train_scaled, y_train_resampled, cv=5)
    
    # Compare different over-sampling methods
    print("\n" + "="*60)
    print("BONUS: Comparing All Over-sampling Methods")
    print("="*60)
    comparison_results = compare_methods(X_train, X_test, y_train, y_test, final_feature_names)
    print("\nComparison Results:")
    print(comparison_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - ../../results/confusion_matrix_logistic_regression_oversampling.png")
    print("  - ../../results/roc_curve_logistic_regression_oversampling.png")
    print("  - ../../output/class_distribution_logistic_regression_oversampling.png")
    print("  - ../../output/feature_coefficients_logistic_regression_oversampling.png")
    print("  - ../../output/probability_distribution_logistic_regression_oversampling.png")
    print("  - ../../output/cross_validation_logistic_regression_oversampling.png")
    print("  - ../../results/oversampling_methods_comparison_logistic_regression.png")


if __name__ == "__main__":
    main()
