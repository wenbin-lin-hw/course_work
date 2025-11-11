"""
K-Nearest Neighbors (KNN) Model for Predicting Appendicitis Severity with Over-sampling
This script trains a KNN model to predict the 'Severity' column
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
from sklearn.neighbors import KNeighborsClassifier
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


def train_knn(X_train, y_train, tune_hyperparameters=True):
    """
    Train KNN model with optional hyperparameter tuning
    
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
    model : trained KNeighborsClassifier
    best_params : dict of best parameters (if tuning enabled)
    """
    print("\n" + "="*60)
    print("Training K-Nearest Neighbors (KNN) model...")
    print("Note: KNN is a distance-based algorithm, so feature scaling is CRITICAL!")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with GridSearchCV...")
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2],  # p=1 for manhattan, p=2 for euclidean
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        # Create base model
        base_model = KNeighborsClassifier()
        
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
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            p=2,
            algorithm='auto'
        )
        model.fit(X_train, y_train)
        best_params = None
    
    print("Model training completed.")
    
    return model, best_params


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
    plt.title('Confusion Matrix - KNN (with Over-sampling)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix_knn_oversampling.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix_knn_oversampling.png'")
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
        plt.title('ROC Curve - KNN (with Over-sampling)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve_knn_oversampling.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved as 'roc_curve_knn_oversampling.png'")
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
    plt.savefig('class_distribution_knn_oversampling.png', dpi=300, bbox_inches='tight')
    print("Class distribution plot saved as 'class_distribution_knn_oversampling.png'")
    plt.close()


def plot_k_vs_accuracy(X_train, X_test, y_train, y_test, k_range=range(1, 31)):
    """Plot accuracy vs number of neighbors (k)"""
    print("\n" + "="*60)
    print("Analyzing K vs Accuracy")
    print("="*60)
    
    train_accuracies = []
    test_accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        
        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_accuracies, label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(k_range, test_accuracies, label='Testing Accuracy', marker='s', linewidth=2)
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('KNN: Accuracy vs Number of Neighbors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('knn_k_vs_accuracy_oversampling.png', dpi=300, bbox_inches='tight')
    print("K vs Accuracy plot saved as 'knn_k_vs_accuracy_oversampling.png'")
    plt.close()
    
    # Find optimal k
    optimal_k = k_range[np.argmax(test_accuracies)]
    print(f"\nOptimal K based on test accuracy: {optimal_k}")
    print(f"Test accuracy at optimal K: {max(test_accuracies):.4f}")


def compare_methods(X_train, X_test, y_train, y_test, scaler):
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
        
        # Scale the resampled data
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (without hyperparameter tuning for speed)
        model, _ = train_knn(X_train_scaled, y_train_resampled, tune_hyperparameters=False)
        
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
    plt.savefig('oversampling_methods_comparison_knn.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'oversampling_methods_comparison_knn.png'")
    plt.close()
    
    return results_df


def main():
    """Main execution function"""
    print("="*60)
    print("K-Nearest Neighbors (KNN) for Appendicitis Severity Prediction")
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
    
    # IMPORTANT: Scale features for KNN (distance-based algorithm)
    print("\n" + "="*60)
    print("Scaling features for KNN...")
    print("KNN is sensitive to feature scales, so standardization is essential!")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed.")
    
    # Train model with resampled and scaled data
    model, best_params = train_knn(
        X_train_scaled, y_train_resampled, tune_hyperparameters=True
    )
    
    # Evaluate model
    y_train_pred, y_test_pred = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train_resampled, y_test
    )
    
    # Plot K vs Accuracy
    plot_k_vs_accuracy(X_train_scaled, X_test_scaled, y_train_resampled, y_test, k_range=range(1, 31))
    
    # Compare different over-sampling methods
    print("\n" + "="*60)
    print("BONUS: Comparing All Over-sampling Methods")
    print("="*60)
    comparison_results = compare_methods(X_train, X_test, y_train, y_test, StandardScaler())
    print("\nComparison Results:")
    print(comparison_results.to_string(index=False))
    
    # Save model and scaler
    print("\n" + "="*60)
    print("Saving model and scaler...")
    import joblib
    joblib.dump(model, 'knn_model_oversampling.pkl')
    joblib.dump(scaler, 'scaler_knn_oversampling.pkl')
    joblib.dump(final_feature_names, 'feature_names_knn_oversampling.pkl')
    if best_params:
        joblib.dump(best_params, 'best_params_knn_oversampling.pkl')
    print("Model saved as 'knn_model_oversampling.pkl'")
    print("Scaler saved as 'scaler_knn_oversampling.pkl'")
    print("Feature names saved as 'feature_names_knn_oversampling.pkl'")
    if best_params:
        print("Best parameters saved as 'best_params_knn_oversampling.pkl'")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    print("\nGenerated Files:")
    print("  1. confusion_matrix_knn_oversampling.png")
    print("  2. roc_curve_knn_oversampling.png (if binary classification)")
    print("  3. class_distribution_knn_oversampling.png")
    print("  4. knn_k_vs_accuracy_oversampling.png")
    print("  5. oversampling_methods_comparison_knn.png")
    print("  6. knn_model_oversampling.pkl")
    print("  7. scaler_knn_oversampling.pkl")
    print("  8. feature_names_knn_oversampling.pkl")
    print("  9. best_params_knn_oversampling.pkl")
    
    print("\nModel Summary:")
    print(f"  Number of Neighbors (K): {model.n_neighbors}")
    print(f"  Weights: {model.weights}")
    print(f"  Distance Metric: {model.metric}")
    print(f"  Algorithm: {model.algorithm}")


if __name__ == "__main__":
    main()
