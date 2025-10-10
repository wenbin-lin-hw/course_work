import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

warnings.filterwarnings('ignore')


class TabNetDiagnosisPredictor:
    def __init__(self, data_path):
        """Initialize TabNet Diagnosis Predictor"""
        self.data_path = data_path
        self.data = None
        self.target_column = 'Diagnosis'
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR DIAGNOSIS PREDICTION")
        print("=" * 80)

        try:
            # Try to load from different sheet names
            try:
                self.data = pd.read_excel(self.data_path, sheet_name='Processed_Data')
                print("✓ Data loaded from 'Processed_Data' sheet")
            except:
                self.data = pd.read_excel(self.data_path)
                print("✓ Data loaded from default sheet")

            print(f"Original data shape: {self.data.shape}")

            # Check if target column exists
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")

            # Display target variable distribution
            print(f"\n{self.target_column} Distribution:")
            print("-" * 40)
            target_counts = self.data[self.target_column].value_counts().sort_index()
            for value, count in target_counts.items():
                pct = (count / len(self.data)) * 100
                print(f"  {value}: {count:4d} samples ({pct:5.1f}%)")

            # Remove rows with missing target values
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=[self.target_column])
            final_rows = len(self.data)

            if final_rows < initial_rows:
                print(f"Removed {initial_rows - final_rows} rows with missing {self.target_column}")

            # Check class distribution for stratification
            min_class_count = target_counts.min()
            if min_class_count < 10:
                print(
                    f"⚠ Warning: Minimum class count is {min_class_count}, which may cause issues with stratification")

            # Prepare feature columns
            self.feature_columns = [col for col in self.data.columns
                                    if col not in self.exclude_columns]

            print(f"\nFeature Selection:")
            print(f"Total columns: {len(self.data.columns)}")
            print(f"Excluded columns ({len(self.exclude_columns)}): {self.exclude_columns}")
            print(f"Feature columns: {len(self.feature_columns)}")

            # Display sample of feature columns
            print(f"\nSample feature columns (first 15):")
            for i, col in enumerate(self.feature_columns[:15]):
                print(f"  {i + 1:2d}. {col}")

            if len(self.feature_columns) > 15:
                print(f"  ... and {len(self.feature_columns) - 15} more columns")

            return True

        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False

    def preprocess_features(self):
        """Preprocess features for TabNet"""
        print(f"\nFeature Preprocessing:")
        print("-" * 30)

        # Get features and target
        X = self.data[self.feature_columns].copy()
        y = self.data[self.target_column].copy()

        print(f"Initial shapes - Features: {X.shape}, Target: {y.shape}")

        # Handle missing values in features
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]

        if len(features_with_missing) > 0:
            print(f"\nHandling missing values in {len(features_with_missing)} features:")
            for feature, count in features_with_missing.head(10).items():
                pct = (count / len(X)) * 100
                print(f"  {feature[:30]:<30}: {count:4d} ({pct:5.1f}%)")

            if len(features_with_missing) > 10:
                print(f"  ... and {len(features_with_missing) - 10} more features")

            # Fill missing values
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    # Numerical: use median
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    # Categorical: use mode or 'Unknown'
                    mode_value = X[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    X[col].fillna(fill_value, inplace=True)

        # Convert categorical features to numeric
        categorical_features = []
        numeric_features = []

        for col in X.columns:
            if X[col].dtype == 'object':
                categorical_features.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            else:
                numeric_features.append(col)

        print(f"Feature types - Categorical: {len(categorical_features)}, Numeric: {len(numeric_features)}")

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        target_mapping = dict(zip(self.label_encoder.classes_,
                                  range(len(self.label_encoder.classes_))))

        print(f"\nTarget encoding mapping:")
        for original, encoded in target_mapping.items():
            count = np.sum(y_encoded == encoded)
            print(f"  {original} -> {encoded} ({count} samples)")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        print(f"\n✓ Preprocessing completed")
        print(f"Final shapes - Features: {X_scaled.shape}, Target: {y_encoded.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")

        return X_scaled, y_encoded

    def create_tabnet_model(self, input_dim, output_dim):
        """Create TabNet model with optimized hyperparameters"""
        print(f"\nCreating TabNet Model:")
        print("-" * 25)

        # Optimized TabNet hyperparameters for diagnosis prediction
        tabnet_params = {
            'n_d': 64,  # Width of the decision prediction layer
            'n_a': 64,  # Width of the attention embedding for each mask
            'n_steps': 6,  # Number of steps in the architecture
            'gamma': 1.3,  # Coefficient for feature reusage in the masks
            'n_independent': 2,  # Number of independent GLU layers at each step
            'n_shared': 2,  # Number of shared GLU layers at each step
            'lambda_sparse': 1e-3,  # Sparsity regularization coefficient
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': 'entmax',  # Either "sparsemax" or "entmax"
            'scheduler_params': {"step_size": 50, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"Model configuration:")
        for key, value in tabnet_params.items():
            if key not in ['optimizer_fn', 'scheduler_fn']:
                print(f"  {key}: {value}")

        # Create TabNet model
        self.model = TabNetClassifier(**tabnet_params)

        print(f"✓ TabNet model created for {input_dim} features, {output_dim} classes")
        print(f"Device: {self.device}")

        return self.model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train TabNet model with class balancing"""
        print(f"\nTraining TabNet Model:")
        print("-" * 25)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")

        # Display class distribution in training set
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print(f"\nTraining set class distribution:")
        for class_idx, count in zip(unique_train, counts_train):
            class_name = self.label_encoder.classes_[class_idx]
            pct = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({pct:.1f}%)")

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=unique_train, y=y_train)
        class_weight_dict = dict(zip(unique_train, class_weights))

        print(f"\nClass weights for balancing:")
        for class_idx, weight in class_weight_dict.items():
            class_name = self.label_encoder.classes_[class_idx]
            print(f"  {class_name}: {weight:.3f}")

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=150,
            patience=20,
            batch_size=512,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        print("✓ Model training completed")

        return self.model, history

    def evaluate_model(self, X_test, y_test, fold_num=None):
        """Evaluate the trained model"""
        fold_info = f" (Fold {fold_num})" if fold_num else ""
        print(f"\nModel Evaluation{fold_info}:")
        print("-" * 30)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Get classes present in test set
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [str(self.label_encoder.classes_[i]) for i in unique_test_classes]

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    labels=unique_test_classes,
                                    target_names=target_names,
                                    digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_test_classes)
        print(f"\nConfusion Matrix:")
        print(cm)

        # ROC AUC calculation
        try:
            if len(unique_test_classes) == 2:
                # Binary classification
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multiclass classification
                auc_score = roc_auc_score(y_test, y_pred_proba,
                                          multi_class='ovr', average='macro')
            print(f"ROC AUC Score: {auc_score:.4f}")
        except Exception as e:
            auc_score = None
            print(f"ROC AUC Score: Could not calculate ({str(e)})")

        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'target_names': target_names,
            'unique_classes': unique_test_classes
        }

    def create_visualizations(self, results, fold=None):
        """Create comprehensive visualization plots"""
        fold_suffix = f"_fold_{fold}" if fold is not None else ""

        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'TabNet Diagnosis Prediction Results{fold_suffix}',
                     fontsize=16, fontweight='bold')

        # 1. Confusion Matrix Heatmap
        ax1 = axes[0, 0]
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=results['target_names'],
                    yticklabels=results['target_names'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. Feature Importance
        ax2 = axes[0, 1]
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Get top 20 features
            top_indices = np.argsort(importances)[-20:]
            top_features = [self.feature_columns[i] for i in top_indices]
            top_importances = importances[top_indices]

            bars = ax2.barh(range(len(top_features)), top_importances, color='skyblue')
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([feat[:25] + '...' if len(feat) > 25 else feat
                                 for feat in top_features], fontsize=8)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Top 20 Feature Importances')
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Feature Importance\nNot Available',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Feature Importance')

        # 3. Prediction Distribution
        ax3 = axes[0, 2]
        y_pred = results['y_pred']
        unique_preds, pred_counts = np.unique(y_pred, return_counts=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_preds)))
        bars = ax3.bar(range(len(unique_preds)), pred_counts, color=colors)
        ax3.set_xticks(range(len(unique_preds)))
        ax3.set_xticklabels([results['target_names'][np.where(results['unique_classes'] == pred)[0][0]]
                             for pred in unique_preds], rotation=45)
        ax3.set_ylabel('Prediction Count')
        ax3.set_title('Prediction Distribution')

        # Add value labels on bars
        for bar, count in zip(bars, pred_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(count)}', ha='center', va='bottom')

        # 4. Model Performance Metrics
        ax4 = axes[1, 0]
        metrics = ['Accuracy']
        values = [results['accuracy']]
        colors_metrics = ['lightblue']

        if results['auc_score'] is not None:
            metrics.append('ROC AUC')
            values.append(results['auc_score'])
            colors_metrics.append('lightcoral')

        bars = ax4.bar(metrics, values, color=colors_metrics)
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics')
        ax4.set_ylim(0, 1.1)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # 5. Class-wise Performance (Precision, Recall, F1)
        ax5 = axes[1, 1]
        if len(results['target_names']) <= 5:  # Only if reasonable number of classes
            try:
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    results['y_true'] if 'y_true' in results else np.arange(len(results['y_pred'])),
                    results['y_pred'],
                    labels=results['unique_classes'],
                    average=None,
                    zero_division=0
                )

                x = np.arange(len(results['target_names']))
                width = 0.25

                ax5.bar(x - width, precision, width, label='Precision', alpha=0.8)
                ax5.bar(x, recall, width, label='Recall', alpha=0.8)
                ax5.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

                ax5.set_xlabel('Classes')
                ax5.set_ylabel('Score')
                ax5.set_title('Class-wise Performance')
                ax5.set_xticks(x)
                ax5.set_xticklabels(results['target_names'], rotation=45)
                ax5.legend()
                ax5.set_ylim(0, 1.1)
            except:
                ax5.text(0.5, 0.5, 'Class-wise Performance\nNot Available',
                         ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, 'Too many classes\nto display',
                     ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Class-wise Performance')

        # 6. Prediction Confidence Distribution
        ax6 = axes[1, 2]
        if results['y_pred_proba'] is not None:
            max_proba = np.max(results['y_pred_proba'], axis=1)
            ax6.hist(max_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax6.axvline(np.mean(max_proba), color='red', linestyle='--',
                        label=f'Mean: {np.mean(max_proba):.3f}')
            ax6.set_xlabel('Maximum Prediction Probability')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Prediction Confidence Distribution')
            ax6.legend()
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Prediction Probabilities\nNot Available',
                     ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Prediction Confidence')

        plt.tight_layout()
        filename = f'../../pictures/tabnet_diagnosis_results{fold_suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved: {filename}")

    def stratified_cross_validation(self, n_folds=5, test_ratio=1 / 2):
        """Run stratified cross-validation with 2:1 train:test ratio"""
        print(f"\n" + "=" * 80)
        print(f"STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
        print(f"Train:Test Ratio = {(1 - test_ratio):.1f}:{test_ratio:.1f}")
        print("=" * 80)

        # Prepare data
        X, y = self.preprocess_features()

        # Check minimum class size for stratification
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)

        if min_class_count < n_folds:
            print(f"⚠ Warning: Minimum class count ({min_class_count}) < n_folds ({n_folds})")
            print("Reducing n_folds to minimum class count")
            n_folds = min_class_count

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'=' * 60}")
            print(f"FOLD {fold}/{n_folds}")
            print('=' * 60)

            # Split into train+val and test (2:1 ratio)
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            print(f"Train+Val set: {len(X_train_val)} samples")
            print(f"Test set: {len(X_test)} samples")
            print(f"Actual ratio: {len(X_train_val) / len(X_test):.2f}:1")

            # Further split train_val into train and validation (80:20)
            val_size = 0.2

            # Check if stratified split is possible for train/val split
            unique_train_val, counts_train_val = np.unique(y_train_val, return_counts=True)
            min_train_val_count = np.min(counts_train_val)

            if min_train_val_count >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_size,
                    stratify=y_train_val, random_state=42
                )
                print(f"Used stratified train/validation split")
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_size,
                    random_state=42
                )
                print(f"Used random train/validation split (stratification not possible)")

            print(f"Final split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Create and train model
            self.create_tabnet_model(X_train.shape[1], len(unique_classes))
            model, history = self.train_model(X_train, y_train, X_val, y_val)

            # Evaluate model
            fold_results = self.evaluate_model(X_test, y_test, fold_num=fold)
            fold_results['fold'] = fold
            fold_results['y_true'] = y_test  # Store true labels for visualization
            cv_results.append(fold_results)

            # Create visualization for this fold
            self.create_visualizations(fold_results, fold=fold)

        return cv_results

    def summarize_cv_results(self, cv_results):
        """Summarize cross-validation results with comprehensive statistics"""
        print(f"\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

        # Extract metrics
        accuracies = [result['accuracy'] for result in cv_results]
        auc_scores = [result['auc_score'] for result in cv_results
                      if result['auc_score'] is not None]

        # Accuracy results
        print(f"Accuracy Results by Fold:")
        print("-" * 30)
        for i, acc in enumerate(accuracies, 1):
            print(f"Fold {i}: {acc:.4f}")

        print(f"\nAccuracy Statistics:")
        print(f"Mean:    {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Median:  {np.median(accuracies):.4f}")
        print(f"Min:     {np.min(accuracies):.4f}")
        print(f"Max:     {np.max(accuracies):.4f}")
        print(f"Range:   {np.max(accuracies) - np.min(accuracies):.4f}")

        # ROC AUC results
        if auc_scores:
            print(f"\nROC AUC Results by Fold:")
            print("-" * 30)
            for i, auc in enumerate(auc_scores, 1):
                print(f"Fold {i}: {auc:.4f}")

            print(f"\nROC AUC Statistics:")
            print(f"Mean:    {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            print(f"Median:  {np.median(auc_scores):.4f}")
            print(f"Min:     {np.min(auc_scores):.4f}")
            print(f"Max:     {np.max(auc_scores):.4f}")
            print(f"Range:   {np.max(auc_scores) - np.min(auc_scores):.4f}")

        # Overall performance assessment
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nOverall Performance Assessment:")
        print("-" * 35)
        if mean_accuracy >= 0.9:
            performance_level = "Excellent"
        elif mean_accuracy >= 0.8:
            performance_level = "Good"
        elif mean_accuracy >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Needs Improvement"

        print(f"Performance Level: {performance_level}")
        print(f"Model Consistency: {'High' if std_accuracy < 0.05 else 'Moderate' if std_accuracy < 0.1 else 'Low'}")
        print(f"Stability (Std Dev): {std_accuracy:.4f}")

        # Save comprehensive results
        summary_data = []
        for i, result in enumerate(cv_results, 1):
            summary_data.append({
                'Fold': i,
                'Accuracy': result['accuracy'],
                'ROC_AUC': result['auc_score'],
                'Test_Samples': len(result['y_pred'])
            })

        summary_df = pd.DataFrame(summary_data)

        # Add summary statistics
        summary_stats = {
            'Fold': 'Mean',
            'Accuracy': np.mean(accuracies),
            'ROC_AUC': np.mean(auc_scores) if auc_scores else None,
            'Test_Samples': np.mean([len(result['y_pred']) for result in cv_results])
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_stats])], ignore_index=True)

        summary_stats_std = {
            'Fold': 'Std Dev',
            'Accuracy': np.std(accuracies),
            'ROC_AUC': np.std(auc_scores) if auc_scores else None,
            'Test_Samples': np.std([len(result['y_pred']) for result in cv_results])
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_stats_std])], ignore_index=True)

        # Save results
        summary_df.to_csv('../../results/tabnet_diagnosis_cv_results.csv', index=False)
        print(f"\nResults saved to: tabnet_diagnosis_cv_results.csv")

        return summary_df

    def run_complete_experiment(self):
        """Run the complete diagnosis prediction experiment"""
        print("Starting TabNet Diagnosis Prediction Experiment...")
        print("Target: Diagnosis (appendicitis classification)")

        # Load and prepare data
        if not self.load_and_prepare_data():
            print("Failed to load data. Experiment terminated.")
            return False

        # Run stratified cross-validation with 2:1 train:test ratio
        cv_results = self.stratified_cross_validation(n_folds=5, test_ratio=1 / 3)

        # Summarize results
        summary = self.summarize_cv_results(cv_results)

        print(f"\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated files:")
        print("- tabnet_diagnosis_cv_results.csv (Cross-validation results)")
        for i in range(1, len(cv_results) + 1):
            print(f"- tabnet_diagnosis_results_fold_{i}.png (Fold {i} visualization)")

        print(f"\nExperiment Summary:")
        print(f"- Model: TabNet Classifier")
        print(f"- Target: {self.target_column}")
        print(f"- Features: {len(self.feature_columns)}")
        print(f"- Cross-validation: Stratified {len(cv_results)}-fold")
        print(f"- Train:Test ratio: ~2:1")
        print(
            f"- Final accuracy: {np.mean([r['accuracy'] for r in cv_results]):.4f} ± {np.std([r['accuracy'] for r in cv_results]):.4f}")

        return True


def main():
    """Main function"""
    print("TabNet Diagnosis Prediction for Appendicitis Data")
    print("=" * 60)
    print("Configuration:")
    print("- Target: Diagnosis")
    print("- Excluded features: US_Performed, US_Number, Management, Severity, Diagnosis")
    print("- Cross-validation: Stratified 5-fold")
    print("- Train:Test ratio: 2:1")
    print()

    # Create predictor
    predictor = TabNetDiagnosisPredictor('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run experiment
    success = predictor.run_complete_experiment()

    if success:
        print("\n✓ TabNet Diagnosis Prediction completed successfully!")
    else:
        print("\n✗ TabNet Diagnosis Prediction failed!")


if __name__ == "__main__":
    main()