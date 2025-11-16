import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings
import time
from datetime import datetime
import matplotlib

# matplotlib.use("TkAgg")

warnings.filterwarnings('ignore')
milliseconds = int(round(time.time() * 1000))
import os


class TabNetMultiTargetTrainer:
    def __init__(self, data_path=None, n_folds=5, balancing=0, remove_single_class=False):
        """Initialize TabNet Multi-Target Trainer
        data_path: Path to the data file
        n_folds: Number of folds for cross-validation
        balancing: Type of balancing to apply (0 for no balancing, 1 for balancing)
        remove_single_class: Whether to remove classes with only one sample
        """
        self.data_path = data_path
        self.data = None
        self.target_column = None
        self.target_columns = ['Diagnosis', 'Severity', 'Management']
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.result = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_folds = n_folds
        # you can you rebalance to adjust for class imbalance if needed, set value to 1 to enable balancing
        self.balancing = balancing
        self.label_encoder = LabelEncoder()
        self.model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.remove_single_class = remove_single_class
        self.model_dir = "../../saved_models"

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR MULTI-TARGET PREDICTION")
        print("=" * 80)

        try:

            self.data = pd.read_excel(self.data_path, sheet_name='Processed_Data')
            print(f"Original data shape: {self.data.shape}")
            if self.remove_single_class:
                print("\nremove_single_class tag is True")
                print("\n Try to remove one single class of Management...")
                management_counts = self.data['Management'].value_counts()
                single_value_categories = management_counts[management_counts == 1].index.tolist()
                print(f"\nthe single  categories are: {single_value_categories}")
                print(f"\nthe number of rows needed to be deleted is : {len(single_value_categories)}")
                self.data = self.data[~self.data['Management'].isin(single_value_categories)].copy()

            # Display target distributions
            print(f"\nTarget Variable Distributions:")
            print("-" * 50)

            for target in self.target_columns:
                print(f"\n{target}:")
                target_counts = self.data[target].value_counts().sort_index()
                total = len(self.data[target].dropna())

                for value, count in target_counts.items():
                    pct = (count / total) * 100
                    print(f"  {value}: {count:4d} samples ({pct:5.1f}%)")

                # Special handling for Management (check for rare classes)
                if target == 'Management':
                    min_count = target_counts.min()
                    print(f"  Minimum class count: {min_count}")
                    if min_count <= 2:
                        print(f"  ⚠ Warning: Very rare classes detected (≤2 samples)")
                        print(f"  Will use custom stratification strategy")

            # Remove rows with any missing target values
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=self.target_columns)
            final_rows = len(self.data)

            if final_rows < initial_rows:
                print(f"\nRemoved {initial_rows - final_rows} rows with missing target values")
            print(f"Final data shape: {self.data.shape}")
            # Prepare feature columns
            self.feature_columns = [col for col in self.data.columns
                                    if col not in self.exclude_columns]
            print(f"\nFeature Selection:")
            print(f"Total columns: {len(self.data.columns)}")
            print(f"Excluded columns: {list(self.exclude_columns)}")
            print(f"Feature columns: {len(self.feature_columns)}")
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

    def create_stratified_split_with_rare_classes(self, X, target_encoded, test_size=1 / 3):
        """Create stratified split handling rare classes in Management"""
        print(f"\nCustom Stratified Splitting:")
        print("-" * 35)
        if self.target_column == 'Management' and not self.remove_single_class:
            # Get Management target for stratification strategy
            unique_mgmt, counts_mgmt = np.unique(target_encoded, return_counts=True)
            # Identify rare classes (≤2 samples)
            rare_classes = unique_mgmt[counts_mgmt <= 2]
            common_classes = unique_mgmt[counts_mgmt > 2]
            # if len(rare_classes) > 0:
            print(f"\nRare classes detected: {rare_classes}")
            print(f"Common classes: {common_classes}")
            # Strategy: Put all rare class samples in training set
            rare_indices = np.where(np.isin(target_encoded, rare_classes))[0]
            common_indices = np.where(np.isin(target_encoded, common_classes))[0]
            print(f"Rare class samples: {len(rare_indices)} (will go to training)")
            print(f"Common class samples: {len(common_indices)} (will be split)")
            y_common = target_encoded[common_indices]
            # Can stratify common classes
            train_common_idx, test_common_idx = train_test_split(
                common_indices, test_size=test_size,
                stratify=y_common, random_state=42
            )
            # Combine rare (all to train) + common split
            train_idx = np.concatenate([rare_indices, train_common_idx])
            test_idx = test_common_idx
            # Create splits
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = target_encoded[train_idx]
            y_test = target_encoded[test_idx]
        else:
            # Can stratify common classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, target_encoded, test_size=test_size,
                stratify=target_encoded, random_state=42
            )

        # Display split results
        actual_ratio = len(X_train) / len(X_test)
        print(f"\nSplit Results:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Actual ratio: {actual_ratio:.2f}:1")
        if self.n_folds < 3:
            X_val = X_train[:100]
            y_val = y_train[:100]
        else:

            if self.target_column != 'Management':
                # Further split training into train/validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2,
                    stratify=y_train, random_state=42
                )
            else:
                # Further split training into train/validation without stratification
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2,
                    random_state=42
                )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_tabnet_model(self, input_dim, output_dim, target_name):
        """Create TabNet model for specific target"""
        print(f"\nCreating TabNet model for {target_name}:")
        print("-" * 30)

        # Adjust parameters based on target complexity
        if target_name == 'Diagnosis':
            # Binary classification - simpler model
            config = {
                'n_d': 64, 'n_a': 64, 'n_steps': 6, 'gamma': 1.3,
                'lambda_sparse': 1e-3
            }
        elif target_name == 'Severity':
            # Multi-class with moderate complexity
            config = {
                'n_d': 64, 'n_a': 64, 'n_steps': 6, 'gamma': 1.3,
                'lambda_sparse': 1e-3
            }
        else:  # Management
            # Most complex with rare classes
            config = {
                'n_d': 64, 'n_a': 64, 'n_steps': 6, 'gamma': 1.3,
                'lambda_sparse': 1e-3
            }

        tabnet_params = {
            'n_d': config['n_d'],
            'n_a': config['n_a'],
            'n_steps': config['n_steps'],
            'gamma': config['gamma'],
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': config['lambda_sparse'],
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=1.5e-2, weight_decay=1e-5),
            'mask_type': 'entmax',
            'scheduler_params': {"step_size": 40, "gamma": 0.85},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"Model config: n_d={config['n_d']}, n_a={config['n_a']}, "
              f"n_steps={config['n_steps']}, classes={output_dim}")

        self.model = TabNetClassifier(**tabnet_params)
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
        # Adjust training parameters based on target
        if self.target_column == 'Management':
            # More epochs for complex/imbalanced target
            max_epochs = 250
            patience = 25
            batch_size = 32  # Smaller for rare classes
            virtual_batch_size = 16
            drop_last = False
        elif self.target_column == 'Severity':
            max_epochs = 150
            patience = 25
            batch_size = 32
            virtual_batch_size = 16
            drop_last = False
        elif self.target_column == 'Diagnosis':  # Diagnosis
            max_epochs = 150
            patience = 25
            batch_size = 32
            virtual_batch_size = 16
            drop_last = False
        if len(set(y_train)) == len(set(y_val)):
            # Train the model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_name=['validation'],
                eval_metric=['accuracy', 'logloss'],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=drop_last,
                weights=self.balancing

                # max_epochs=150,
                # patience=25,
                # batch_size=64,
                # virtual_batch_size=32,

            )
        else:
            self.model.fit(
                X_train, y_train,
                # eval_set=[(X_val, y_val)],
                # eval_name=['validation'],
                eval_metric=['accuracy', 'logloss'],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=drop_last,
                weights=self.balancing

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
        filename = f'../../results/tabnet_{self.target_column.lower()}_results{fold_suffix}_{self.timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # plt.show()

        print(f"Visualization saved: {filename}")

    def cross_validation_experiment(self):
        """Run cross-validation experiment"""
        print(f"\n" + "=" * 80)
        print(f"CROSS-VALIDATION EXPERIMENT ({self.n_folds}-FOLD)")
        print("=" * 80)

        # Prepare data
        X, y = self.preprocess_features()
        cv_results = []
        if self.n_folds < 3:
            self.n_folds = 1
        for fold in range(self.n_folds):
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_stratified_split_with_rare_classes(X, y)
            self.create_tabnet_model(X_train.shape[1], len(np.unique(y)), self.target_column)
            self.train_model(X_train, y_train, X_val, y_val)
            model_path = os.path.join(self.model_dir,
                                      f"tabnet_{self.target_column.lower()}_{self.timestamp}_{fold}.zip")
            self.model.save_model(model_path)
            print(f"✓ Saved {self.target_column} {fold} fold model: {model_path}")
            preprocessor_data = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'exclude_columns': self.exclude_columns,
                'timestamp': self.timestamp,
                'remove_single_class': self.remove_single_class,
                'n_folds': self.n_folds,
                'balancing': self.balancing,
                'target_column': self.target_column
            }
            preprocessor_path = os.path.join(self.model_dir,
                                             f"preprocessor_{self.target_column.lower()}_{self.timestamp}_{fold}.pkl")
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor_data, f)
            print(f"✓ Saved {self.target_column} {fold} fold preprocessor: {preprocessor_path}")

            # Evaluate model
            fold_results = self.evaluate_model(X_test, y_test)
            fold_results['fold'] = fold
            cv_results.append(fold_results)
            # Save model info
            model_info = {
                'target_column': self.target_column,
                'feature_columns': self.feature_columns,
                'timestamp': self.timestamp,
                'device': str(self.device),
                'results_summary': {
                    'accuracy': fold_results['accuracy'],
                    'classes': fold_results['target_names']
                }
            }

            info_path = os.path.join(self.model_dir,
                                     f"model_info_{self.target_column.lower()}_{self.timestamp}_fold_{fold}.pkl")
            with open(info_path, 'wb') as f:
                pickle.dump(model_info, f)

            print(f"✓ Saved model info: {info_path}")
            # Create visualization for this fold
            self.create_visualizations(fold_results, fold)

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

        summary_df.to_csv(f'../../results/tabnet_{self.target_column.lower()}_cv_results_{self.timestamp}.csv',
                          index=False)
        print(f"\nResults saved to: tabnet_{self.target_column.lower()}_cv_results_{self.timestamp}.csv")

        return summary_df

    def load_model_and_preprocessor(self, target_column, model_path, preprocessor_path,model_info_path):
        """Load previously saved models and preprocessors"""
        print(f"\nLoading models and preprocessors  from {model_path} and {preprocessor_path}...")

        try:
            # Load preprocessors
            with open(preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)

            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.feature_columns = preprocessor_data['feature_columns']
            self.target_columns = preprocessor_data['target_columns']
            self.exclude_columns = preprocessor_data['exclude_columns']
            self.target_column = target_column

            # Load feature encoders
            for key, value in preprocessor_data.items():
                if key.startswith('feature_encoder_'):
                    setattr(self, key, value)

            print(f"✓ Loaded preprocessors")

            # Load models

            if os.path.exists(model_path):
                model = TabNetClassifier()
                model.load_model(model_path)
                self.model = model
                print(f"✓ Loaded {self.target_column} model")
            else:
                print(f"⚠ Model file not found for {self.target_column}: {model_path}")

            # Load model info
            with open(model_info_path, 'rb') as f:
                model_info = pickle.load(f)

            print(f"✓ Loaded model info from {model_info_path}")
            print(f"Models loaded: {list(self.models.keys())}")

            return True

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False

    def predict_new_data(self, new_data_path, target_column, model_path, preprocessor_path,model_info_path):
        """Predict on new data using saved models"""
        print(f"\n{'='*80}")
        print("PREDICTING ON NEW DATA")
        print('='*80)


        if not self.load_model_and_preprocessor(target_column,model_path, preprocessor_path,model_info_path):
            print("Failed to load models")
            return None

        # Load new data
        try:
            self.data_path = new_data_path
            self.load_and_prepare_data()



            # Preprocess new data (without fitting transformers)
            X_scaled, y_encoded = self.preprocess_features()
            print(f"Preprocessed data shape: {X_scaled.shape}")


            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)
            print("-"*100)
            print(y_pred)
            print("-" * 100)
            print(y_pred_proba)





            return y_pred, y_pred_proba

        except Exception as e:
            print(f"✗ Error in prediction: {e}")
            return None

    def run_complete_experiment(self):
        """Run the complete experiment"""

        # Load and prepare data
        if not self.load_and_prepare_data():
            print("Failed to load data. Experiment terminated.")
            return False

        # Run cross-validation experiment
        for target_column in self.target_columns:
            self.target_column = target_column
            print(f"Starting TabNet {self.target_column.lower()} Prediction Experiment...")
            cv_results = self.cross_validation_experiment()

            # Summarize results
            summary = self.summarize_cv_results(cv_results)

            print(f"\n" + "=" * 80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")

        return True


def main():
    """Main function"""
    print(f"TabNet Prediction Diagnosis, Severity and Management for Appendicitis Data")
    print("=" * 60)

    # Create predictor
    predictor = TabNetMultiTargetTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx', n_folds=5,
                                         balancing=1, remove_single_class=True)

    # Run experiment
    success = predictor.run_complete_experiment()

    if success:
        print("\n✓ TabNet Prediction Diagnosis, Severity and Management Prediction completed successfully!")
    else:
        print("\n✗ TabNet Prediction Diagnosis, Severity and Management Prediction failed!")


if __name__ == "__main__":
    main()
    # predictor = TabNetMultiTargetTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx', n_folds=5,
    #                                      balancing=1, remove_single_class=True)
    # predictor.predict_new_data('/Users/user/Documents/IR/course_work/F21DL-group13-appendicitis/data/appendicitis/processed_appendicitis_data_final.xlsx','Diagnosis','/Users/user/Documents/IR/course_work/F21DL-group13-appendicitis/saved_models/tabnet_diagnosis_20251019_161301_0.zip.zip','/Users/user/Documents/IR/course_work/F21DL-group13-appendicitis/saved_models/preprocessor_diagnosis_20251019_161301_0.pkl','/Users/user/Documents/IR/course_work/F21DL-group13-appendicitis/saved_models/model_info_diagnosis_20251019_161301_fold_0.pkl')
