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

warnings.filterwarnings('ignore')
milliseconds = int(round(time.time() * 1000))




class TabNetMultiTargetTrainer:
    def __init__(self, data_path, n_folds=5, rebalance=False):
        """Initialize TabNet Multi-Target Trainer"""
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_folds = n_folds
        self.rebalance = rebalance
        self.label_encoder = LabelEncoder()
        self.model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR MULTI-TARGET PREDICTION")
        print("=" * 80)

        try:

            self.data = pd.read_excel(self.data_path, sheet_name='Processed_Data')
            print(f"Original data shape: {self.data.shape}")

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

    def preprocess_data(self):
        """Preprocess features for all targets"""
        print(f"\nData Preprocessing:")
        print("-" * 30)

        # Extract features
        X = self.data[self.feature_columns].copy()
        print(f"Features shape: {X.shape}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Features scaled and preprocessed")

        # Prepare targets with label encoding
        targets_encoded = {}
        for target in self.target_columns:
            y = self.data[target].copy()
            le = LabelEncoder()
            targets_encoded[target] = le.fit_transform(y)
            self.label_encoders[target] = le

            print(f"\n{target} encoding:")
            for original, encoded in zip(le.classes_, range(len(le.classes_))):
                count = np.sum(targets_encoded[target] == encoded)
                print(f"  '{original}' -> {encoded} ({count} samples)")

        return X_scaled, targets_encoded

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

    def create_stratified_split_with_rare_classes(self, X, targets_encoded, test_size=1 / 3):
        """Create stratified split handling rare classes in Management"""
        print(f"\nCustom Stratified Splitting:")
        print("-" * 35)

        # Get Management target for stratification strategy
        y_management = targets_encoded['Management']
        unique_mgmt, counts_mgmt = np.unique(y_management, return_counts=True)

        print(f"Management class distribution:")
        for cls, count in zip(unique_mgmt, counts_mgmt):
            class_name = self.label_encoders['Management'].classes_[cls]
            print(f"  Class {cls} ('{class_name}'): {count} samples")

        # Identify rare classes (≤2 samples)
        rare_classes = unique_mgmt[counts_mgmt <= 2]
        common_classes = unique_mgmt[counts_mgmt > 2]

        # if len(rare_classes) > 0:
        print(f"\nRare classes detected: {rare_classes}")
        print(f"Common classes: {common_classes}")

        # Strategy: Put all rare class samples in training set
        rare_indices = np.where(np.isin(y_management, rare_classes))[0]
        common_indices = np.where(np.isin(y_management, common_classes))[0]

        print(f"Rare class samples: {len(rare_indices)} (will go to training)")
        print(f"Common class samples: {len(common_indices)} (will be split)")


        y_common = y_management[common_indices]




        # Can stratify common classes
        train_common_idx, test_common_idx = train_test_split(
            common_indices, test_size=test_size,
            stratify=y_common, random_state=42
        )
        print(f"Stratified split on common classes successful")

        # Combine rare (all to train) + common split
        train_idx = np.concatenate([rare_indices, train_common_idx])
        test_idx = test_common_idx




        # Create splits
        X_train, X_test = X[train_idx], X[test_idx]

        targets_train = {}
        targets_test = {}
        for target in self.target_columns:
            targets_train[target] = targets_encoded[target][train_idx]
            targets_test[target] = targets_encoded[target][test_idx]

        # Display split results
        actual_ratio = len(X_train) / len(X_test)
        print(f"\nSplit Results:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Actual ratio: {actual_ratio:.2f}:1")

        # Show distribution in splits for each target
        for target in self.target_columns:
            print(f"\n{target} distribution in splits:")
            y_train_target = targets_train[target]
            y_test_target = targets_test[target]

            unique_classes = np.unique(np.concatenate([y_train_target, y_test_target]))
            for cls in unique_classes:
                class_name = self.label_encoders[target].classes_[cls]
                train_count = np.sum(y_train_target == cls)
                test_count = np.sum(y_test_target == cls)
                print(f"  '{class_name}': Train={train_count}, Test={test_count}")
        X_val = X_train[:95]
        y_val = targets_train[:95]
        y_train = targets_train
        y_test = targets_test
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



#     def train_target_model(self, X_train, y_train, X_test, y_test, target_name):
#         """Train model for specific target"""
#         print(f"\n{'=' * 60}")
#         print(f"TRAINING MODEL FOR {target_name}")
#         print('=' * 60)
#
#         # Get number of classes
#         n_classes = len(np.unique(np.concatenate([y_train, y_test])))
#
#         # Create model
#         model = self.create_tabnet_model(X_train.shape[1], n_classes, target_name)
#
#         # Create validation split from training data
#         val_size = 0.2
#         unique_train, counts_train = np.unique(y_train, return_counts=True)
#         min_train_count = np.min(counts_train)
#
#         # Check if we can create a proper validation split
#         if len(y_train) < 10 or min_train_count == 1:
#             # Too few samples or singleton classes - skip validation
#             print(f"Insufficient data for validation split - using training data as validation")
#             X_train_fit = X_train
#             y_train_fit = y_train
#             X_val = X_train[:min(50, len(X_train))]  # Use small subset as pseudo-validation
#             y_val = y_train[:min(50, len(y_train))]
#         else:
#             # Try stratified split, fall back to ensuring all training classes in validation
#             try:
#                 if min_train_count >= 2:
#                     X_train_fit, X_val, y_train_fit, y_val = train_test_split(
#                         X_train, y_train, test_size=val_size,
#                         stratify=y_train, random_state=42
#                     )
#
#                     # Check if validation set has classes not in training set
#                     unique_train_fit = set(np.unique(y_train_fit))
#                     unique_val = set(np.unique(y_val))
#
#                     if not unique_val.issubset(unique_train_fit):
#                         # Re-split to ensure all validation classes are in training
#                         print(f"Validation contains classes not in training - adjusting split")
#                         raise ValueError("Need to adjust split")
#
#                     print(f"Used stratified validation split")
#                 else:
#                     raise ValueError("Cannot stratify")
#
#             except (ValueError, Exception):
#                 # Use random split with manual class balancing
#                 print(f"Using custom validation split to ensure class compatibility")
#
#                 # Ensure each class has at least one sample in training
#                 indices_by_class = {}
#                 for cls in unique_train:
#                     indices_by_class[cls] = np.where(y_train == cls)[0]
#
#                 train_indices = []
#                 val_indices = []
#
#                 for cls, cls_indices in indices_by_class.items():
#                     n_cls_samples = len(cls_indices)
#                     if n_cls_samples == 1:
#                         # Single sample - put in training
#                         train_indices.extend(cls_indices)
#                     else:
#                         # Multiple samples - split but ensure at least 1 in training
#                         n_val = max(1, int(n_cls_samples * val_size))
#                         n_train = n_cls_samples - n_val
#
#                         np.random.seed(42)
#                         shuffled_indices = np.random.permutation(cls_indices)
#                         train_indices.extend(shuffled_indices[:n_train])
#                         val_indices.extend(shuffled_indices[n_train:])
#
#                 X_train_fit = X_train[train_indices]
#                 y_train_fit = y_train[train_indices]
#                 X_val = X_train[val_indices] if val_indices else X_train[:5]  # Fallback
#                 y_val = y_train[val_indices] if val_indices else y_train[:5]  # Fallback
#
#         print(f"Training: {len(X_train_fit)}, Validation: {len(X_val)}, Test: {len(X_test)}")
#
#         # Show class distribution
#         print(f"\nTraining set class distribution:")
#         for cls in unique_train:
#             class_name = self.label_encoders[target_name].classes_[cls]
#             count = np.sum(y_train_fit == cls)
#             pct = (count / len(y_train_fit)) * 100
#             print(f"  '{class_name}': {count} ({pct:.1f}%)")
#
#         # Calculate class weights for classes present in training data
#         unique_train_fit, counts_train_fit = np.unique(y_train_fit, return_counts=True)
#
#         if len(unique_train_fit) > 1:
#             class_weights = compute_class_weight('balanced',
#                                                  classes=unique_train_fit,
#                                                  y=y_train_fit)
#             print(f"\nClass weights:")
#             for cls, weight in zip(unique_train_fit, class_weights):
#                 class_name = self.label_encoders[target_name].classes_[cls]
#                 print(f"  '{class_name}': {weight:.3f}")
#         else:
#             print(f"\nOnly one class in training set - no class weighting applied")
#             class_weights = None
#
#         # Adjust training parameters based on target
#         if target_name == 'Management':
#             # More epochs for complex/imbalanced target
#             max_epochs = 250
#             patience = 25
#             batch_size = 8  # Smaller for rare classes
#             virtual_batch_size =4
#         elif target_name == 'Severity':
#             max_epochs = 150
#             patience = 25
#             batch_size = 8
#             virtual_batch_size = 4
#         else:  # Diagnosis
#             max_epochs = 150
#             patience = 25
#             batch_size = 8
#             virtual_batch_size=4
#         print(f"\nTraining parameters:")
#         print(f"  Max epochs: {max_epochs}")
#         print(f"  Patience: {patience}")
#         print(f"  Batch size: {batch_size}")
#
#         # Final check on data size before training
#         if len(X_train_fit) < batch_size:
#             print(f"Warning: Training data ({len(X_train_fit)}) smaller than batch size ({batch_size})")
#             batch_size = max(2, len(X_train_fit))
#             virtual_batch_size = max(2, batch_size // 2)
#             print(f"Adjusted batch_size to {batch_size}, virtual_batch_size to {virtual_batch_size}")
#
#         # Ensure we don't have singleton batches with drop_last=True
#         drop_last = True if len(X_train_fit) > batch_size else False
#
#         print(f"\nFinal training parameters:")
#         print(f"  Training samples: {len(X_train_fit)}")
#         print(f"  Batch size: {batch_size}")
#         print(f"  Virtual batch size: {virtual_batch_size}")
#         print(f"  Drop last: {drop_last}")
#
#         # Train model
#         print(f"\nStarting training...")
#         history = model.fit(
#             X_train_fit, y_train_fit,
#             eval_set=[(X_val, y_val)],
#             eval_name=['validation'],
#             eval_metric=['accuracy', 'logloss'],
#             max_epochs=max_epochs,
#             patience=patience,
#             batch_size=batch_size,
#             virtual_batch_size=4,
#             num_workers=0,
#             drop_last=drop_last  # This ensures no singleton batches
#         )
#
#         print(f"✓ Training completed for {target_name}")
#
#         # Store model
#         self.models[target_name] = model
#
#         return model, history
#
#     def evaluate_target_model(self, X_test, y_test, target_name):
#         """Evaluate model for specific target"""
#         print(f"\nEvaluating {target_name} model:")
#         print("-" * 30)
#
#         model = self.models[target_name]
#
#         # Make predictions
#         y_pred = model.predict(X_test)
#         y_pred_proba = model.predict_proba(X_test)
#
#         # Calculate accuracy
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f"Accuracy: {accuracy:.4f}")
#
#         # Get class information
#         unique_classes = np.unique(np.concatenate([y_test, y_pred]))
#         target_names = [str(self.label_encoders[target_name].classes_[i])
#                         for i in unique_classes]
#
#         # Classification report
#         print(f"\nClassification Report:")
#         report = classification_report(y_test, y_pred,
#                                        labels=unique_classes,
#                                        target_names=target_names,
#                                        digits=4)
#         print(report)
#
#         # Confusion matrix
#         cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
#         print(f"\nConfusion Matrix:")
#         print(cm)
#
#         # ROC AUC
#         try:
#             if len(unique_classes) == 2:
#                 auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
#             else:
#                 auc_score = roc_auc_score(y_test, y_pred_proba,
#                                           multi_class='ovr', average='macro')
#             print(f"ROC AUC: {auc_score:.4f}")
#         except:
#             auc_score = None
#             print("ROC AUC: Could not calculate")
#
#         # Store results
#         results = {
#             'accuracy': accuracy,
#             'y_true': y_test,
#             'y_pred': y_pred,
#             'y_pred_proba': y_pred_proba,
#             'confusion_matrix': cm,
#             'auc_score': auc_score,
#             'target_names': target_names,
#             'unique_classes': unique_classes,
#             'classification_report': report
#         }
#
#         self.results[target_name] = results
#         return results
#
#     def create_comprehensive_visualization(self):
#         """Create comprehensive visualization for all targets"""
#         print(f"\nCreating comprehensive visualizations...")
#
#         fig, axes = plt.subplots(3, 3, figsize=(20, 18))
#         fig.suptitle('TabNet Multi-Target Prediction Results', fontsize=16, fontweight='bold')
#
#         for idx, target in enumerate(self.target_columns):
#             if target not in self.results:
#                 continue
#
#             results = self.results[target]
#
#             # Confusion Matrix
#             ax_cm = axes[idx, 0]
#             cm = results['confusion_matrix']
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
#                         xticklabels=results['target_names'],
#                         yticklabels=results['target_names'])
#             ax_cm.set_title(f'{target} - Confusion Matrix')
#             ax_cm.set_xlabel('Predicted')
#             ax_cm.set_ylabel('Actual')
#
#             # Class Distribution
#             ax_dist = axes[idx, 1]
#             y_true = results['y_true']
#             y_pred = results['y_pred']
#
#             true_counts = pd.Series(y_true).value_counts().sort_index()
#             pred_counts = pd.Series(y_pred).value_counts().sort_index()
#
#             x = np.arange(len(results['target_names']))
#             width = 0.35
#
#             ax_dist.bar(x - width / 2, [true_counts.get(i, 0) for i in results['unique_classes']],
#                         width, label='True', alpha=0.8)
#             ax_dist.bar(x + width / 2, [pred_counts.get(i, 0) for i in results['unique_classes']],
#                         width, label='Predicted', alpha=0.8)
#
#             ax_dist.set_xlabel('Classes')
#             ax_dist.set_ylabel('Count')
#             ax_dist.set_title(f'{target} - Class Distribution')
#             ax_dist.set_xticks(x)
#             ax_dist.set_xticklabels(results['target_names'], rotation=45)
#             ax_dist.legend()
#
#             # Performance Metrics
#             ax_perf = axes[idx, 2]
#             metrics = ['Accuracy']
#             values = [results['accuracy']]
#             colors = ['lightblue']
#
#             if results['auc_score'] is not None:
#                 metrics.append('ROC AUC')
#                 values.append(results['auc_score'])
#                 colors.append('lightcoral')
#
#             bars = ax_perf.bar(metrics, values, color=colors)
#             ax_perf.set_ylabel('Score')
#             ax_perf.set_title(f'{target} - Performance')
#             ax_perf.set_ylim(0, 1.1)
#
#             # Add value labels
#             for bar, value in zip(bars, values):
#                 height = bar.get_height()
#                 ax_perf.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
#                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
#
#         plt.tight_layout()
#         filename = f'../../results/tabnet_multi_target_results_{milliseconds}.png'
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         plt.show()
#
#         print(f"✓ Visualization saved: {filename}")
#
#     def save_all_results(self):
#         """Save results for all targets"""
#         print(f"\nSaving results...")
#
#         # Create summary DataFrame
#         summary_data = []
#         for target in self.target_columns:
#             if target in self.results:
#                 results = self.results[target]
#                 summary_data.append({
#                     'Target': target,
#                     'Accuracy': results['accuracy'],
#                     'ROC_AUC': results['auc_score'] if results['auc_score'] else 'N/A',
#                     'Test_Samples': len(results['y_true']),
#                     'Classes': len(results['unique_classes'])
#                 })
#
#         summary_df = pd.DataFrame(summary_data)
#         summary_df.to_csv(f'../../output/tabnet_multi_target_summary_{milliseconds}.csv', index=False)
#
#         # Save detailed results for each target
#         for target in self.target_columns:
#             if target in self.results:
#                 results = self.results[target]
#
#                 # Predictions
#                 pred_df = pd.DataFrame({
#                     'True_Label': [self.label_encoders[target].classes_[i] for i in results['y_true']],
#                     'Predicted_Label': [self.label_encoders[target].classes_[i] for i in results['y_pred']],
#                     'Correct': results['y_true'] == results['y_pred']
#                 })
#
#                 # Add probabilities
#                 if results['y_pred_proba'] is not None:
#                     for i, class_name in enumerate(self.label_encoders[target].classes_):
#                         if i < results['y_pred_proba'].shape[1]:
#                             pred_df[f'Prob_{class_name}'] = results['y_pred_proba'][:, i]
#
#                 pred_df.to_csv(f'../../output/tabnet_{target.lower()}_predictions_{milliseconds}.csv', index=False)
#
#                 # Confusion matrix
#                 cm_df = pd.DataFrame(results['confusion_matrix'],
#                                      index=results['target_names'],
#                                      columns=results['target_names'])
#                 cm_df.to_csv(f'../../output/tabnet_{target.lower()}_confusion_matrix_{milliseconds}.csv')
#
#         print(f"✓ Results saved:")
#         print(f"  - tabnet_multi_target_summary_{milliseconds}.csv")
#         for target in self.target_columns:
#             if target in self.results:
#                 print(f"  - tabnet_{target.lower()}_predictions_{milliseconds}.csv")
#                 print(f"  - tabnet_{target.lower()}_confusion_matrix_{milliseconds}.csv")
#
#     def run_complete_training(self):
#         """Run complete multi-target training pipeline"""
#         print("Starting TabNet Multi-Target Training Pipeline")
#         print("=" * 70)
#
#         # Load and prepare data
#         if not self.load_and_prepare_data():
#             return False
#
#         # Preprocess data
#         X, targets_encoded = self.preprocess_data()
#
#         # Create custom stratified split
#         X_train, X_test, targets_train, targets_test = self.create_stratified_split_with_rare_classes(
#             X, targets_encoded
#         )
#
#         # Train models for each target
#         for target in self.target_columns:
#             y_train = targets_train[target]
#             y_test = targets_test[target]
#
#             # Train model
#             model, history = self.train_target_model(X_train, y_train, X_test, y_test, target)
#
#             # Evaluate model
#             self.evaluate_target_model(X_test, y_test, target)
#
#         # Create visualizations
#         self.create_comprehensive_visualization()
#
#         # Save results
#         self.save_all_results()
#
#         # Final summary
#         print(f"\n{'=' * 70}")
#         print("MULTI-TARGET TRAINING COMPLETED!")
#         print("=" * 70)
#
#         print(f"\nFinal Results Summary:")
#         for target in self.target_columns:
#             if target in self.results:
#                 acc = self.results[target]['accuracy']
#                 auc = self.results[target]['auc_score']
#                 n_classes = len(self.results[target]['unique_classes'])
#                 print(f"- {target}: Accuracy={acc:.4f}, Classes={n_classes}",
#                       end="")
#                 if auc:
#                     print(f", AUC={auc:.4f}")
#                 else:
#                     print()
#
#         print(f"\nFeatures used: {len(self.feature_columns)}")
#         print(f"Training samples: {len(X_train)}")
#         print(f"Test samples: {len(X_test)}")
#
#         return True
#
#
# def main():
#     """Main execution function"""
#     print("TabNet Multi-Target Prediction Training")
#     print("=" * 50)
#     print("Targets: Diagnosis, Severity, Management")
#     print("Strategy: Separate models for each target")
#     print("Special handling: Management rare classes")
#     print("Split ratio: 2:1 (train:test)")
#     print()
#
#     # Create trainer
#     trainer = TabNetMultiTargetTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx')
#
#     # Run training
#     success = trainer.run_complete_training()
#
#     if success:
#         print("\n✓ Multi-target training completed successfully!")
#     else:
#         print("\n✗ Multi-target training failed!")
#
#
# if __name__ == "__main__":
#     main()


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
        filename = f'../../results/tabnet_{self.target_column.lower()}_results{fold_suffix}_{self.timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved: {filename}")

    def cross_validation_experiment(self):
        """Run cross-validation experiment"""
        print(f"\n" + "=" * 80)
        print(f"CROSS-VALIDATION EXPERIMENT ({self.n_folds}-FOLD)")
        print("=" * 80)

        # Prepare data
        X, y = self.preprocess_features()
        cv_results = []

        if self.n_folds >2:
            # Cross-validation
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)


            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                print(f"\n{'=' * 50}")
                print(f"FOLD {fold}/{self.n_folds}")
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
                self.create_tabnet_model(X_train.shape[1], len(np.unique(y)),self.target_column)
                self.train_model(X_train, y_train, X_val, y_val)

                # Evaluate model
                fold_results = self.evaluate_model(X_test_fold, y_test_fold)
                fold_results['fold'] = fold
                cv_results.append(fold_results)

                # Create visualization for this fold
                self.create_visualizations(fold_results, fold=fold)
        else:
            X, y = self.preprocess_data()
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_stratified_split_with_rare_classes(X,y)
            self.create_tabnet_model(X_train.shape[1], len(np.unique(y)), self.target_column)
            self.train_model(X_train, y_train, X_val, y_val)
            # Evaluate model
            fold_results = self.evaluate_model(X_test, y_test)
            fold_results['fold'] = 0
            cv_results.append(fold_results)

            # Create visualization for this fold
            self.create_visualizations(fold_results, fold=0)

        return cv_results

    def summarize_cv_results(self,  cv_results):
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

        summary_df.to_csv(f'../../results/tabnet_{self.target_column.lower()}_cv_results_{self.timestamp}.csv', index=False)
        print(f"\nResults saved to: tabnet_{self.target_column.lower()}_cv_results_{self.timestamp}.csv")

        return summary_df

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
            print("=" * 80)
            print("Generated files:")
            print("- tabnet_management_cv_results.csv")
            for i in range(1, 6):
                print(f"- tabnet_{self.target_column.lower()}_results_fold_{i}.png")

        return True


def main():
    """Main function"""
    print(f"TabNet Prediction Diagnosis, Severity and Management for Appendicitis Data")
    print("=" * 60)

    # Create predictor
    predictor = TabNetMultiTargetTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx',n_folds=1)

    # Run experiment
    success = predictor.run_complete_experiment()

    if success:
        print("\n✓ TabNet Prediction Diagnosis, Severity and Management   Prediction completed successfully!")
    else:
        print("\n✗ TabNet Prediction Diagnosis, Severity and Management  Prediction failed!")


if __name__ == "__main__":
    main()