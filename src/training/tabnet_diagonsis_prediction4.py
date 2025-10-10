import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

import matplotlib
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')


class TabNetDiagnosisTrainer:
    def __init__(self, data_path):
        """Initialize TabNet Diagnosis Trainer"""
        self.data_path = data_path
        self.data = None
        self.target_column = 'Diagnosis'
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR DIAGNOSIS PREDICTION")
        print("=" * 80)

        try:
            # Try different sheet names
            try:
                self.data = pd.read_excel(self.data_path, sheet_name='Processed_Data')
                print("✓ Data loaded from 'Processed_Data' sheet")
            except:
                self.data = pd.read_excel(self.data_path)
                print("✓ Data loaded from default sheet")

            print(f"Original data shape: {self.data.shape}")

            # Check target column
            if self.target_column not in self.data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")

            # Display target distribution
            print(f"\n{self.target_column} Distribution:")
            print("-" * 40)
            target_counts = self.data[self.target_column].value_counts().sort_index()
            total_samples = len(self.data)

            for value, count in target_counts.items():
                pct = (count / total_samples) * 100
                print(f"  {value}: {count:4d} samples ({pct:5.1f}%)")

            print(f"\nTotal samples: {total_samples}")
            print(f"Number of classes: {len(target_counts)}")

            # Remove missing target values
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=[self.target_column])
            final_rows = len(self.data)

            if final_rows < initial_rows:
                removed = initial_rows - final_rows
                print(f"Removed {removed} rows with missing {self.target_column}")

            # Check for stratification feasibility
            min_class_count = target_counts.min()
            if min_class_count < 3:
                print(f"⚠ Warning: Minimum class has only {min_class_count} samples")
                print("This may cause issues with stratified sampling")

            # Prepare feature columns
            available_columns = set(self.data.columns)
            exclude_set = set(self.exclude_columns)

            # Find which excluded columns actually exist
            existing_excluded = exclude_set.intersection(available_columns)
            missing_excluded = exclude_set - available_columns

            if existing_excluded:
                print(f"\nExcluding {len(existing_excluded)} columns: {list(existing_excluded)}")
            if missing_excluded:
                print(f"Note: {len(missing_excluded)} excluded columns not found: {list(missing_excluded)}")

            self.feature_columns = [col for col in self.data.columns
                                    if col not in existing_excluded]

            print(f"\nFeature Selection Summary:")
            print(f"Total columns: {len(self.data.columns)}")
            print(f"Feature columns: {len(self.feature_columns)}")

            # Show sample features
            print(f"\nSample feature columns (first 20):")
            for i, col in enumerate(self.feature_columns[:20], 1):
                print(f"  {i:2d}. {col}")

            if len(self.feature_columns) > 20:
                print(f"  ... and {len(self.feature_columns) - 20} more columns")

            return True

        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False

    def preprocess_data(self):
        """Preprocess features and target"""
        print(f"\nData Preprocessing:")
        print("-" * 30)

        # Extract features and target
        X = self.data[self.feature_columns].copy()
        y = self.data[self.target_column].copy()

        print(f"Initial shapes - Features: {X.shape}, Target: {y.shape}")
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)

        # Show target encoding mapping
        target_mapping = dict(zip(self.label_encoder.classes_,
                                  range(len(self.label_encoder.classes_))))
        print(f"\nTarget encoding mapping:")
        for original, encoded in target_mapping.items():
            count = np.sum(y_encoded == encoded)
            print(f"  '{original}' -> {encoded} ({count} samples)")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        print(f"\n✓ Preprocessing completed")
        print(f"Final shapes - Features: {X_scaled.shape}, Target: {y_encoded.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")

        return X_scaled, y_encoded

    def split_data_stratified(self, X, y, test_size=1 / 3):
        """Split data with stratified sampling for 2:1 train:test ratio"""
        print(f"\nStratified Data Splitting:")
        print("-" * 30)

        print(f"Target test size: {test_size:.3f} (Train:Test = {1 - test_size:.1f}:{test_size:.1f})")

        # Check if stratified split is possible
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)

        if min_class_count < 2:
            print(f"⚠ Warning: Cannot use stratified split (min class count: {min_class_count})")
            print("Using random split instead")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            stratify_used = False
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            stratify_used = True
            print(f"✓ Used stratified sampling")

        # Display split results
        actual_ratio = len(X_train) / len(X_test)
        print(f"\nSplit Results:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Actual ratio: {actual_ratio:.2f}:1")

        # Show class distribution in splits
        print(f"\nClass distribution in splits:")
        print(f"{'Class':<15} {'Original':<10} {'Train':<10} {'Test':<10} {'Train %':<10} {'Test %':<10}")
        print("-" * 70)

        for class_idx in unique_classes:
            class_name = self.label_encoder.classes_[class_idx]
            original_count = np.sum(y == class_idx)
            train_count = np.sum(y_train == class_idx)
            test_count = np.sum(y_test == class_idx)
            train_pct = (train_count / len(y_train)) * 100
            test_pct = (test_count / len(y_test)) * 100

            print(f"{class_name:<15} {original_count:<10} {train_count:<10} {test_count:<10} "
                  f"{train_pct:<10.1f} {test_pct:<10.1f}")

        return X_train, X_test, y_train, y_test, stratify_used

    def create_tabnet_model(self, input_dim, output_dim):
        """Create and configure TabNet model"""
        print(f"\nCreating TabNet Model:")
        print("-" * 25)

        # TabNet hyperparameters optimized for diagnosis prediction
        tabnet_params = {
            'n_d': 64,  # Decision layer width
            'n_a': 64,  # Attention layer width
            'n_steps': 6,  # Number of steps
            'gamma': 1.3,  # Feature reusage coefficient
            'n_independent': 2,  # Independent GLU layers
            'n_shared': 2,  # Shared GLU layers
            'lambda_sparse': 1e-3,  # Sparsity regularization
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': 'entmax',  # Masking type
            'scheduler_params': {"step_size": 50, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"Model configuration:")
        key_params = ['n_d', 'n_a', 'n_steps', 'gamma', 'lambda_sparse']
        for key in key_params:
            print(f"  {key}: {tabnet_params[key]}")

        print(f"  Device: {self.device}")
        print(f"  Input features: {input_dim}")
        print(f"  Output classes: {output_dim}")

        # Create model
        self.model = TabNetClassifier(**tabnet_params)

        print(f"✓ TabNet model created successfully")
        return self.model

    def train_model(self, X_train, y_train, validation_split=0.2):
        """Train the TabNet model"""
        print(f"\nTraining TabNet Model:")
        print("-" * 25)

        # Create validation split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=validation_split,
            stratify=y_train if len(np.unique(y_train)) > 1 and np.min(np.bincount(y_train)) >= 2 else None,
            random_state=42
        )

        print(f"Training samples: {len(X_train_fit)}")
        print(f"Validation samples: {len(X_val)}")

        # Show training class distribution
        unique_train_classes, train_counts = np.unique(y_train_fit, return_counts=True)
        print(f"\nTraining set class distribution:")
        for class_idx, count in zip(unique_train_classes, train_counts):
            class_name = self.label_encoder.classes_[class_idx]
            pct = (count / len(y_train_fit)) * 100
            print(f"  {class_name}: {count} ({pct:.1f}%)")

        # Calculate class weights for imbalanced data
        if len(unique_train_classes) > 1:
            class_weights = compute_class_weight('balanced',
                                                 classes=unique_train_classes,
                                                 y=y_train_fit)
            class_weight_dict = dict(zip(unique_train_classes, class_weights))

            print(f"\nClass weights for balancing:")
            for class_idx, weight in class_weight_dict.items():
                class_name = self.label_encoder.classes_[class_idx]
                print(f"  {class_name}: {weight:.3f}")

        # Train the model
        print(f"\nStarting training...")

        history = self.model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=150,
            patience=25,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
            drop_last=False
        )

        print("✓ Model training completed")
        return history

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print(f"\nModel Evaluation:")
        print("-" * 20)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Basic accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Get class information for reports
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [str(self.label_encoder.classes_[i]) for i in unique_test_classes]

        # Classification report
        print(f"\nDetailed Classification Report:")
        print("=" * 50)
        report = classification_report(y_test, y_pred,
                                       labels=unique_test_classes,
                                       target_names=target_names,
                                       digits=4)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_test_classes)
        print(f"\nConfusion Matrix:")
        print(cm)

        # ROC AUC calculation
        try:
            n_classes = len(unique_test_classes)
            if n_classes == 2:
                # Binary classification
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                print(f"ROC AUC Score: {auc_score:.4f}")
            elif n_classes > 2:
                # Multi-class classification
                auc_score = roc_auc_score(y_test, y_pred_proba,
                                          multi_class='ovr', average='macro')
                print(f"ROC AUC Score (macro-avg): {auc_score:.4f}")
            else:
                auc_score = None
        except Exception as e:
            auc_score = None
            print(f"ROC AUC Score: Could not calculate - {str(e)}")

        # Store results
        self.results = {
            'accuracy': accuracy,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'target_names': target_names,
            'unique_classes': unique_test_classes,
            'classification_report': report
        }

        return self.results

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            print("No results to visualize. Run evaluation first.")
            return

        print(f"\nCreating visualizations...")

        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TabNet Diagnosis Prediction Results', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=self.results['target_names'],
                    yticklabels=self.results['target_names'])
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

            y_pos = np.arange(len(top_features))
            bars = ax2.barh(y_pos, top_importances, color='lightblue')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f[:25] + '...' if len(f) > 25 else f
                                 for f in top_features], fontsize=8)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Top 20 Feature Importances')
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Feature Importance\nNot Available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Importance')

        # 3. Class Distribution
        ax3 = axes[0, 2]
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']

        true_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()

        x = np.arange(len(self.results['target_names']))
        width = 0.35

        ax3.bar(x - width / 2, [true_counts.get(i, 0) for i in self.results['unique_classes']],
                width, label='True', alpha=0.8)
        ax3.bar(x + width / 2, [pred_counts.get(i, 0) for i in self.results['unique_classes']],
                width, label='Predicted', alpha=0.8)

        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Count')
        ax3.set_title('True vs Predicted Class Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.results['target_names'], rotation=45)
        ax3.legend()

        # 4. Performance Metrics
        ax4 = axes[1, 0]
        metrics = ['Accuracy']
        values = [self.results['accuracy']]
        colors = ['lightblue']

        if self.results['auc_score'] is not None:
            metrics.append('ROC AUC')
            values.append(self.results['auc_score'])
            colors.append('lightcoral')

        bars = ax4.bar(metrics, values, color=colors)
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics')
        ax4.set_ylim(0, 1.1)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # 5. Prediction Confidence
        ax5 = axes[1, 1]
        if self.results['y_pred_proba'] is not None:
            max_proba = np.max(self.results['y_pred_proba'], axis=1)
            ax5.hist(max_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax5.axvline(np.mean(max_proba), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(max_proba):.3f}')
            ax5.set_xlabel('Maximum Prediction Probability')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Prediction Confidence Distribution')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Prediction Probabilities\nNot Available',
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Prediction Confidence')

        # 6. Model Summary
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = f"""
Model Summary:
━━━━━━━━━━━━━━━━━━━━━
Target: {self.target_column}
Features: {len(self.feature_columns)}
Classes: {len(self.results['unique_classes'])}
Device: {self.device}

Results:
━━━━━━━━━━━━━━━━━━━━━
Accuracy: {self.results['accuracy']:.4f}
"""
        if self.results['auc_score']:
            summary_text += f"ROC AUC: {self.results['auc_score']:.4f}\n"

        summary_text += f"""
Test Samples: {len(self.results['y_true'])}

Class Distribution:
━━━━━━━━━━━━━━━━━━━━━
"""
        for i, class_name in enumerate(self.results['target_names']):
            count = np.sum(self.results['y_true'] == self.results['unique_classes'][i])
            pct = (count / len(self.results['y_true'])) * 100
            summary_text += f"{class_name}: {count} ({pct:.1f}%)\n"

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        # Save visualization
        filename = '../../results/tabnet_diagnosis_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ Visualization saved: {filename}")

    def save_results(self):
        """Save detailed results to files"""
        if not self.results:
            print("No results to save. Run evaluation first.")
            return

        print(f"\nSaving results...")

        # Create results summary
        results_summary = {
            'Metric': ['Accuracy', 'ROC_AUC', 'Test_Samples', 'Number_of_Classes', 'Number_of_Features'],
            'Value': [
                self.results['accuracy'],
                self.results['auc_score'] if self.results['auc_score'] else 'N/A',
                len(self.results['y_true']),
                len(self.results['unique_classes']),
                len(self.feature_columns)
            ]
        }

        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv('../../output/tabnet_diagnosis_summary.csv', index=False)

        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'True_Label': [self.label_encoder.classes_[i] for i in self.results['y_true']],
            'Predicted_Label': [self.label_encoder.classes_[i] for i in self.results['y_pred']],
            'Correct': self.results['y_true'] == self.results['y_pred']
        })

        # Add prediction probabilities
        if self.results['y_pred_proba'] is not None:
            for i, class_name in enumerate(self.label_encoder.classes_):
                if i < self.results['y_pred_proba'].shape[1]:
                    predictions_df[f'Prob_{class_name}'] = self.results['y_pred_proba'][:, i]

        predictions_df.to_csv('../../output/tabnet_diagnosis_predictions.csv', index=False)

        # Save confusion matrix
        cm_df = pd.DataFrame(self.results['confusion_matrix'],
                             index=self.results['target_names'],
                             columns=self.results['target_names'])
        cm_df.to_csv('../../output/tabnet_diagnosis_confusion_matrix.csv')

        print(f"✓ Results saved:")
        print(f"  - tabnet_diagnosis_summary.csv")
        print(f"  - tabnet_diagnosis_predictions.csv")
        print(f"  - tabnet_diagnosis_confusion_matrix.csv")
        print(f"  - tabnet_diagnosis_results.png")

    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("Starting Complete TabNet Diagnosis Training Pipeline")
        print("=" * 70)

        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print("Failed to load data. Training terminated.")
            return False

        # Step 2: Preprocess data
        X, y = self.preprocess_data()

        # Step 3: Split data with stratification (2:1 ratio)
        X_train, X_test, y_train, y_test, stratify_used = self.split_data_stratified(X, y)

        # Step 4: Create model
        self.create_tabnet_model(X_train.shape[1], len(np.unique(y)))

        # Step 5: Train model
        training_history = self.train_model(X_train, y_train)

        # Step 6: Evaluate model
        results = self.evaluate_model(X_test, y_test)

        # Step 7: Create visualizations
        self.create_visualizations()

        # Step 8: Save results
        self.save_results()

        # Final summary
        print(f"\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print(f"Final Results Summary:")
        print(f"- Target Variable: {self.target_column}")
        print(f"- Training Samples: {len(X_train)}")
        print(f"- Test Samples: {len(X_test)}")
        print(f"- Features Used: {len(self.feature_columns)}")
        print(f"- Stratified Sampling: {'Yes' if stratify_used else 'No'}")
        print(f"- Final Accuracy: {results['accuracy']:.4f}")

        if results['auc_score']:
            print(f"- ROC AUC Score: {results['auc_score']:.4f}")

        return True


def main():
    """Main execution function"""
    print("TabNet Diagnosis Prediction Training")
    print("=" * 50)
    print("Configuration:")
    print("- Target: Diagnosis")
    print("- Excluded: US_Performed, US_Number, Management, Severity, Diagnosis")
    print("- Sampling: Stratified")
    print("- Train:Test Ratio: 2:1")
    print()

    # Create trainer
    trainer = TabNetDiagnosisTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run training
    success = trainer.run_complete_training()

    if success:
        print("\n✓ TabNet Diagnosis Training completed successfully!")
    else:
        print("\n✗ TabNet Diagnosis Training failed!")


if __name__ == "__main__":
    main()