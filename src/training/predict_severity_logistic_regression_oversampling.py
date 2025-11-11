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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
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


def handle_missing_values(X):
    """Handle missing values in the dataset"""
    print("\n" + "="*60)
    print("Checking for missing values...")
    
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values found:")
        print(missing_counts[missing_counts > 0])
        
        # Fill missing values with median for numerical columns
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['float64', 'int64']:
                    X[col].fillna(X[col].median(), inplace=True)
                    print(f"  - Filled {col} with median value")
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
                    print(f"  - Filled {col} with mode value")
    else:
        print("No missing values found.")
    
    return X


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


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train logistic regression model"""
    print("\n" + "="*60)
    print("Training Logistic Regression model...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    # Using max_iter=1000 to ensure convergence
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        multi_class='auto',
        class_weight=None  # No class weight since we're using over-sampling
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")
    
    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
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
    plt.savefig('confusion_matrix_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix_logistic_regression_oversampling.png'")
    plt.close()
    
    # ROC curve and AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
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
        plt.savefig('roc_curve_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('class_distribution_oversampling.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'class_distribution_oversampling.png'")
    plt.close()


def plot_feature_importance(model, feature_names):
    """Plot feature importance (coefficients) from logistic regression"""
    print("\n" + "="*60)
    print("Feature Importance (Coefficients)")
    print("="*60)
    
    # Get coefficients
    if len(model.classes_) == 2:
        # Binary classification
        coefficients = model.coef_[0]
    else:
        # Multi-class classification - use average absolute coefficients
        coefficients = np.mean(np.abs(model.coef_), axis=0)
    
    # Create dataframe for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute value
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    # Display top 20 features
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Feature Coefficients - Logistic Regression (with Over-sampling)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('feature_importance_logistic_regression_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance_logistic_regression_oversampling.png'")
    plt.close()
    
    return feature_importance


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
        
        # Train model
        model, scaler, X_train_scaled, X_test_scaled = train_logistic_regression(
            X_train_resampled, X_test, y_train_resampled, y_test
        )
        
        # Evaluate
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        results.append({
            'Method': method,
            'Test Accuracy': test_accuracy
        })
        
        print(f"\n{method} - Test Accuracy: {test_accuracy:.4f}")
    
    # Plot comparison
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Method'], results_df['Test Accuracy'], color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('Over-sampling Method')
    plt.ylabel('Test Accuracy')
    plt.title('Comparison of Over-sampling Methods')
    plt.ylim([0, 1])
    for i, v in enumerate(results_df['Test Accuracy']):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('oversampling_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'oversampling_methods_comparison.png'")
    plt.close()
    
    return results_df


def main():
    """Main execution function"""
    print("="*60)
    print("Logistic Regression for Appendicitis Severity Prediction")
    print("WITH OVER-SAMPLING FOR CLASS BALANCE")
    print("="*60)
    
    # Load data
    filepath = 'data/appendicitis/processed_appendicitis_data_final.xlsx'
    df = load_data(filepath)
    
    # Prepare features
    X, y, feature_columns = prepare_features(df)
    
    # Handle missing values
    X = handle_missing_values(X)
    
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
    
    # Train model with resampled data
    model, scaler, X_train_scaled, X_test_scaled = train_logistic_regression(
        X_train_resampled, X_test, y_train_resampled, y_test
    )
    
    # Evaluate model
    y_train_pred, y_test_pred = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train_resampled, y_test
    )
    
    # Plot feature importance
    feature_importance = plot_feature_importance(model, final_feature_names)
    
    # Compare different over-sampling methods
    print("\n" + "="*60)
    print("BONUS: Comparing All Over-sampling Methods")
    print("="*60)
    comparison_results = compare_methods(X_train, X_test, y_train, y_test, final_feature_names)
    print("\nComparison Results:")
    print(comparison_results)
    
    # Save model and scaler
    print("\n" + "="*60)
    print("Saving model and scaler...")
    import joblib
    joblib.dump(model, 'logistic_regression_model_oversampling.pkl')
    joblib.dump(scaler, 'scaler_oversampling.pkl')
    joblib.dump(final_feature_names, 'feature_names_oversampling.pkl')
    print("Model saved as 'logistic_regression_model_oversampling.pkl'")
    print("Scaler saved as 'scaler_oversampling.pkl'")
    print("Feature names saved as 'feature_names_oversampling.pkl'")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    print("\nGenerated Files:")
    print("  1. confusion_matrix_logistic_regression_oversampling.png")
    print("  2. roc_curve_logistic_regression_oversampling.png (if binary classification)")
    print("  3. class_distribution_oversampling.png")
    print("  4. feature_importance_logistic_regression_oversampling.png")
    print("  5. oversampling_methods_comparison.png")
    print("  6. logistic_regression_model_oversampling.pkl")
    print("  7. scaler_oversampling.pkl")
    print("  8. feature_names_oversampling.pkl")


if __name__ == "__main__":
    main()
