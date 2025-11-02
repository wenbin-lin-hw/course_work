import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

warnings.filterwarnings('ignore')


class TabNetManagementPredictor:
    def __init__(self, data_path):
        """Initialize TabNet Management Predictor"""
        self.data_path = data_path
        self.data = None
        self.target_column = 'Management'
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING AND PREPARING APPENDICITIS DATA FOR MANAGEMENT PREDICTION")
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
            print("-" * 30)
            target_counts = self.data[self.target_column].value_counts()
            for value, count in target_counts.items():
                pct = (count / len(self.data)) * 100
                print(f"  {value}: {count} ({pct:.1f}%)")

            # Remove rows with missing target values
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=[self.target_column])
            final_rows = len(self.data)

            if final_rows < initial_rows:
                print(f"Removed {initial_rows - final_rows} rows with missing {self.target_column}")

            # Prepare feature columns
            self.feature_columns = [col for col in self.data.columns
                                    if col not in self.exclude_columns]

            print(f"\nFeature Selection:")
            print(f"Total columns: {len(self.data.columns)}")
            print(f"Excluded columns: {self.exclude_columns}")
            print(f"Feature columns: {len(self.feature_columns)}")

            # Display first 10 feature columns
            print(f"\nFirst 10 feature columns:")
            for i, col in enumerate(self.feature_columns[:10]):
                print(f"  {i + 1:2d}. {col}")

            if len(self.feature_columns) > 10:
                print(f"  ... and {len(self.feature_columns) - 10} more columns")

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

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Handle missing values in features
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]

        if len(features_with_missing) > 0:
            print(f"\nFeatures with missing values:")
            for feature, count in features_with_missing.items():
                pct = (count / len(X)) * 100
                print(f"  {feature}: {count} ({pct:.1f}%)")

            # Fill missing values (simple strategy for TabNet)
            print("Filling missing values...")
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)

        # Convert categorical features to numeric
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object':
                categorical_features.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if categorical_features:
            print(f"Encoded {len(categorical_features)} categorical features")

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        target_mapping = dict(zip(self.label_encoder.classes_,
                                  range(len(self.label_encoder.classes_))))

        print(f"\nTarget encoding mapping:")
        for original, encoded in target_mapping.items():
            print(f"  {original} -> {encoded}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        print(f"✓ Feature preprocessing completed")
        print(f"Final feature shape: {X_scaled.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")

        return X_scaled, y_encoded

    def create_tabnet_model(self, input_dim, output_dim):
        """Create TabNet model"""
        print(f"\nCreating TabNet Model:")
        print("-" * 25)

        # TabNet hyperparameters
        tabnet_params = {
            'n_d': 32,  # Width of the decision prediction layer
            'n_a': 32,  # Width of the attention embedding for each mask
            'n_steps': 5,  # Number of steps in the architecture
            'gamma': 1.5,  # Coefficient for feature reusage in the masks
            'n_independent': 2,  # Number of independent GLU layers at each step
            'n_shared': 2,  # Number of shared GLU layers at each step
            'lambda_sparse': 1e-4,  # Sparsity regularization coefficient
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'mask_type': 'entmax',  # Either "sparsemax" or "entmax"
            'scheduler_params': {"step_size": 50, "gamma": 0.9},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"TabNet Parameters:")
        for key, value in tabnet_params.items():
            if key not in ['optimizer_fn', 'scheduler_fn']:
                print(f"  {key}: {value}")

        # Create TabNet model
        self.model = TabNetClassifier(**tabnet_params)

        print(f"✓ TabNet model created")
        print(f"Device: {self.device}")

        return self.model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train TabNet model"""
        print(f"\nTraining TabNet Model:")
        print("-" * 25)

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")

        # Calculate class weights for imbalanced data
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        print(f"Class weights: {class_weight_dict}")

        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy'],
            max_epochs=150,
            patience=25,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
            drop_last=False,
            weights=1  # You can use class_weights here if needed
        )

        print("✓ Model training completed")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print(f"\nModel Evaluation:")
        print("-" * 20)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report with proper target names
        # Get unique classes present in test set
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [str(self.label_encoder.classes_[i]) for i in unique_test_classes]

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    labels=unique_test_classes,
                                    target_names=target_names))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # ROC AUC for multiclass
        try:
            if len(np.unique(y_test)) > 2:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"ROC AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
            print("ROC AUC Score: Could not calculate")

        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'target_names': target_names
        }

    def create_visualizations(self, results, fold=None):
        """Create visualization plots"""
        fold_suffix = f"_fold_{fold}" if fold is not None else ""

        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'TabNet Management Prediction Results{fold_suffix}', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix Heatmap
        ax1 = axes[0, 0]
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=results['target_names'],
                    yticklabels=results['target_names'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. Feature Importance (if available)
        ax2 = axes[0, 1]
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Get top 15 features
            top_indices = np.argsort(importances)[-15:]
            top_features = [self.feature_columns[i] for i in top_indices]
            top_importances = importances[top_indices]

            bars = ax2.barh(range(len(top_features)), top_importances)
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([feat[:20] + '...' if len(feat) > 20 else feat for feat in top_features])
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Top 15 Feature Importances')

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'Feature Importance\nNot Available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Importance')

        # 3. Prediction Distribution
        ax3 = axes[1, 0]
        y_pred = results['y_pred']
        pred_counts = pd.Series(y_pred).value_counts().sort_index()

        bars = ax3.bar(range(len(pred_counts)), pred_counts.values)
        ax3.set_xticks(range(len(pred_counts)))
        ax3.set_xticklabels([results['target_names'][i] for i in pred_counts.index], rotation=45)
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Distribution')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        # 4. Model Performance Metrics
        ax4 = axes[1, 1]
        metrics = ['Accuracy']
        values = [results['accuracy']]

        if results['auc_score'] is not None:
            metrics.append('ROC AUC')
            values.append(results['auc_score'])

        bars = ax4.bar(metrics, values, color=['skyblue', 'lightcoral'][:len(metrics)])
        ax4.set_ylabel('Score')
        ax4.set_title('Model Performance Metrics')
        ax4.set_ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        filename = f'../../results/tabnet_management_results{fold_suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved: {filename}")

    def cross_validation_experiment(self, n_folds=5):
        """Run cross-validation experiment"""
        print(f"\n" + "=" * 80)
        print(f"CROSS-VALIDATION EXPERIMENT ({n_folds}-FOLD)")
        print("=" * 80)

        # Prepare data
        X, y = self.preprocess_features()

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'=' * 50}")
            print(f"FOLD {fold}/{n_folds}")
            print('=' * 50)

            # Split data
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Further split training into train/validation
            # Check if stratified split is possible
            unique_classes, class_counts = np.unique(y_train_fold, return_counts=True)
            min_class_count = np.min(class_counts)

            if min_class_count >= 2:
                # Use stratified split if all classes have at least 2 samples
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_fold, y_train_fold, test_size=0.2,
                    stratify=y_train_fold, random_state=42
                )
            else:
                # Use regular split if stratified split is not possible
                print(f"Warning: Some classes have only 1 sample. Using regular split instead of stratified.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_fold, y_train_fold, test_size=0.2,
                    random_state=42
                )

            # Create and train model
            self.create_tabnet_model(X_train.shape[1], len(np.unique(y)))
            self.train_model(X_train, y_train, X_val, y_val)

            # Evaluate model
            fold_results = self.evaluate_model(X_test_fold, y_test_fold)
            fold_results['fold'] = fold
            cv_results.append(fold_results)

            # Create visualization for this fold
            self.create_visualizations(fold_results, fold=fold)

        return cv_results

    def summarize_cv_results(self, cv_results):
        """Summarize cross-validation results"""
        print(f"\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

        # Extract metrics
        accuracies = [result['accuracy'] for result in cv_results]
        auc_scores = [result['auc_score'] for result in cv_results if result['auc_score'] is not None]

        print(f"Accuracy Results:")
        print("-" * 20)
        for i, acc in enumerate(accuracies, 1):
            print(f"Fold {i}: {acc:.4f}")

        print(f"\nAccuracy Statistics:")
        print(f"Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Min:  {np.min(accuracies):.4f}")
        print(f"Max:  {np.max(accuracies):.4f}")

        if auc_scores:
            print(f"\nROC AUC Results:")
            print("-" * 20)
            for i, auc in enumerate(auc_scores, 1):
                print(f"Fold {i}: {auc:.4f}")

            print(f"\nROC AUC Statistics:")
            print(f"Mean: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            print(f"Min:  {np.min(auc_scores):.4f}")
            print(f"Max:  {np.max(auc_scores):.4f}")

        # Save results summary
        summary_df = pd.DataFrame({
            'Fold': range(1, len(cv_results) + 1),
            'Accuracy': accuracies,
            'ROC_AUC': [result['auc_score'] for result in cv_results]
        })

        summary_df.to_csv('../../results/tabnet_management_cv_results.csv', index=False)
        print(f"\nResults saved to: tabnet_management_cv_results.csv")

        return summary_df

    def run_complete_experiment(self):
        """Run the complete experiment"""
        print("Starting TabNet Management Prediction Experiment...")

        # Load and prepare data
        if not self.load_and_prepare_data():
            print("Failed to load data. Experiment terminated.")
            return False

        # Run cross-validation experiment
        cv_results = self.cross_validation_experiment(n_folds=5)

        # Summarize results
        summary = self.summarize_cv_results(cv_results)

        print(f"\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Generated files:")
        print("- tabnet_management_cv_results.csv")
        for i in range(1, 6):
            print(f"- tabnet_management_results_fold_{i}.png")

        return True


def main():
    """Main function"""
    print("TabNet Management Prediction for Appendicitis Data")
    print("=" * 60)

    # Create predictor
    predictor = TabNetManagementPredictor('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run experiment
    success = predictor.run_complete_experiment()

    if success:
        print("\n✓ TabNet Management Prediction completed successfully!")
    else:
        print("\n✗ TabNet Management Prediction failed!")


if __name__ == "__main__":
    main()