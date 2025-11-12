"""
Decision Tree Classifier for Severity Prediction
This script trains a Decision Tree classifier to predict the Severity target
(uncomplicated vs complicated appendicitis) using tabular data.
Excludes 'US_Performed' and 'US_Number' columns from training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("Decision Tree Classifier for Severity Prediction")
print("="*80)

# ============================================================================
# 1. Load the preprocessed data
# ============================================================================
print("\n[1] Loading preprocessed data...")
data_path = "../../data/appendicitis/processed_appendicitis_data_final.xlsx"
df = pd.read_excel(data_path)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns in dataset:")
print(df.columns.tolist())

# Check if target variable exists
if 'Severity' not in df.columns:
    raise ValueError("Target variable 'Severity' not found in the dataset!")

print(f"\nTarget variable distribution:")
print(df['Severity'].value_counts())
print(f"\nTarget variable distribution (%):")
print(df['Severity'].value_counts(normalize=True) * 100)

# ============================================================================
# 2. Prepare features and target
# ============================================================================
print("\n[2] Preparing features and target...")

# Define columns to exclude from training
exclude_columns = ['US_Performed', 'US_Number', 'Severity','Management','Diagnosis']

# Get feature columns (all columns except target and excluded columns)
feature_columns = [col for col in df.columns if col not in exclude_columns]

print(f"\nExcluded columns: {exclude_columns}")
print(f"Number of features: {len(feature_columns)}")
print(f"\nFeature columns:")
print(feature_columns)

# Separate features and target
X = df[feature_columns]
y = df['Severity']

# Handle any remaining missing values
if X.isnull().sum().sum() > 0:
    print(f"\nWarning: Found {X.isnull().sum().sum()} missing values in features.")
    print("Filling missing values with column median/mode...")
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ============================================================================
# 3. Split the data
# ============================================================================
print("\n[3] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining set target distribution:")
print(y_train.value_counts())
print(f"\nTest set target distribution:")
print(y_test.value_counts())

# ============================================================================
# 4. Train baseline Decision Tree model
# ============================================================================
print("\n[4] Training baseline Decision Tree model...")
dt_baseline = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    criterion='gini'
)
dt_baseline.fit(X_train, y_train)

# Make predictions
y_train_pred_baseline = dt_baseline.predict(X_train)
y_test_pred_baseline = dt_baseline.predict(X_test)

# Evaluate baseline model
print("\n--- Baseline Model Performance ---")
print("\nTraining Set:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_baseline):.4f}")

print("\nTest Set:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_baseline, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred_baseline, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred_baseline, average='weighted'):.4f}")

print(f"\nTree depth: {dt_baseline.get_depth()}")
print(f"Number of leaves: {dt_baseline.get_n_leaves()}")

# ============================================================================
# 5. Hyperparameter tuning with GridSearchCV
# ============================================================================
print("\n[5] Performing hyperparameter tuning with GridSearchCV...")

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

dt_grid = DecisionTreeClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=dt_grid,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("Fitting GridSearchCV... (this may take a while)")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation F1-score: {grid_search.best_score_:.4f}")

# Get the best model
dt_best = grid_search.best_estimator_

# ============================================================================
# 6. Evaluate the best model
# ============================================================================
print("\n[6] Evaluating the best model...")

# Make predictions with best model
y_train_pred = dt_best.predict(X_train)
y_test_pred = dt_best.predict(X_test)
y_test_pred_proba = dt_best.predict_proba(X_test)

# Calculate metrics
print("\n--- Best Model Performance ---")
print("\nTraining Set:")
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy: {train_accuracy:.4f}")

print("\nTest Set:")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_test_pred,
                          target_names=['Uncomplicated', 'Complicated']))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# Calculate ROC-AUC if binary classification
if len(np.unique(y)) == 2:
    roc_auc = roc_auc_score(y_test, y_test_pred_proba[:, 1])
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Tree statistics
print(f"\nTree depth: {dt_best.get_depth()}")
print(f"Number of leaves: {dt_best.get_n_leaves()}")

# ============================================================================
# 7. Cross-validation
# ============================================================================
print("\n[7] Performing cross-validation...")
cv_scores = cross_val_score(dt_best, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"Cross-validation F1-scores: {cv_scores}")
print(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# 8. Feature importance analysis
# ============================================================================
print("\n[8] Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': dt_best.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 20 Most Important Features ---")
print(feature_importance.head(20).to_string(index=False))

# ============================================================================
# 9. Visualizations
# ============================================================================
print("\n[9] Creating visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Confusion Matrix Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Uncomplicated', 'Complicated'],
            yticklabels=['Uncomplicated', 'Complicated'])
plt.title('Confusion Matrix - Decision Tree', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. Feature Importance (Top 20)
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 3. ROC Curve (if binary classification)
if len(np.unique(y)) == 2:
    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba[:, 1])
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

# 4. Model Performance Comparison
ax4 = plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [test_accuracy, test_precision, test_recall, test_f1]
bars = plt.bar(metrics, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylim([0, 1.0])
plt.ylabel('Score')
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='0.8 threshold')
for i, (bar, score) in enumerate(zip(bars, scores)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.legend()

# 5. Training vs Test Accuracy
ax5 = plt.subplot(2, 3, 5)
comparison = pd.DataFrame({
    'Dataset': ['Training', 'Test'],
    'Accuracy': [train_accuracy, test_accuracy]
})
bars = plt.bar(comparison['Dataset'], comparison['Accuracy'],
               color=['#2ca02c', '#1f77b4'])
plt.ylim([0, 1.0])
plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', linewidth=2, markersize=8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--',
            label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold')
plt.ylabel('F1-Score')
plt.title('Cross-Validation F1-Scores', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(cv_scores) + 1))
plt.ylim([max(0, cv_scores.min() - 0.05), min(1.0, cv_scores.max() + 0.05)])
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../results/severity_decision_tree_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'results/severity_decision_tree_results.png'")

# ============================================================================
# 10. Visualize the Decision Tree (simplified version)
# ============================================================================
print("\n[10] Creating decision tree visualization...")

# Create a simplified tree visualization (max_depth=3 for readability)
fig2, ax = plt.subplots(figsize=(25, 15))
plot_tree(
    dt_best,
    max_depth=3,  # Limit depth for visualization clarity
    feature_names=feature_columns,
    class_names=['Uncomplicated', 'Complicated'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title('Decision Tree (depth limited to 3 for visualization)',
          fontsize=16, fontweight='bold', pad=20)
plt.savefig('../../output/severity_decision_tree_structure.png',
            dpi=300, bbox_inches='tight')
print("Tree structure saved as 'output/severity_decision_tree_structure.png'")

# ============================================================================
# 11. Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nDataset: {data_path}")
print(f"Total samples: {len(df)}")
print(f"Number of features: {len(feature_columns)}")
print(f"Features excluded: US_Performed, US_Number")
print(f"\nTarget variable: Severity")
print(f"Classes: Uncomplicated (0) vs Complicated (1)")
print(f"\nBest Model: Decision Tree Classifier")
print(f"Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"\nTest Set Performance:")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall: {test_recall:.4f}")
print(f"  - F1-Score: {test_f1:.4f}")
if len(np.unique(y)) == 2:
    print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"\nCross-Validation F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"\nTree Statistics:")
print(f"  - Depth: {dt_best.get_depth()}")
print(f"  - Number of leaves: {dt_best.get_n_leaves()}")
print(f"\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

plt.show()
