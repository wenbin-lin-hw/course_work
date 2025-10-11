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
import time

warnings.filterwarnings('ignore')
milliseconds = int(round(time.time() * 1000))


class TabNetSingleSplitTrainer:
    def __init__(self, data_path):
        """Initialize TabNet Single Split Trainer - Direct Train/Test Split"""
        self.data_path = data_path
        self.data = None
        self.target_columns = ['Diagnosis', 'Severity', 'Management']
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR SINGLE SPLIT TRAINING")
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

            # Check target columns
            missing_targets = [col for col in self.target_columns if col not in self.data.columns]
            if missing_targets:
                print(f"✗ Missing target columns: {missing_targets}")
                return False

            print(f"✓ All target columns found")

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
            available_columns = set(self.data.columns)
            exclude_set = set(self.exclude_columns)
            existing_excluded = exclude_set.intersection(available_columns)

            self.feature_columns = [col for col in self.data.columns
                                    if col not in existing_excluded]

            print(f"\nFeature Selection:")
            print(f"Total columns: {len(self.data.columns)}")
            print(f"Excluded columns: {list(existing_excluded)}")
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

        # Handle missing values in features
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]

        if len(features_with_missing) > 0:
            print(f"Handling {len(features_with_missing)} features with missing values...")

            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    mode_values = X[col].mode()
                    fill_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                    X[col].fillna(fill_value, inplace=True)

        # Encode categorical features
        categorical_count = 0
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                categorical_count += 1

        print(f"Encoded {categorical_count} categorical features")

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

    def create_stratified_split_with_rare_classes(self, X, targets_encoded, test_size=1 / 3):
        """Create stratified split handling rare classes in Management"""
        print(f"\nCustom Stratified Splitting (Single Split):")
        print("-" * 45)

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

        if len(rare_classes) > 0:
            print(f"\nRare classes detected: {rare_classes}")
            print(f"Common classes: {common_classes}")

            # Strategy: Put all rare class samples in training set
            rare_indices = np.where(np.isin(y_management, rare_classes))[0]
            common_indices = np.where(np.isin(y_management, common_classes))[0]

            print(f"Rare class samples: {len(rare_indices)} (will go to training)")
            print(f"Common class samples: {len(common_indices)} (will be split)")

            if len(common_indices) == 0:
                # All classes are rare - use simple random split
                print("All classes are rare - using random split")
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)), test_size=test_size, random_state=42
                )
            else:
                # Split only common classes with stratification
                y_common = y_management[common_indices]

                # Check if stratification is still possible with common classes
                unique_common, counts_common = np.unique(y_common, return_counts=True)
                min_common_count = np.min(counts_common)

                if min_common_count >= 2:
                    # Can stratify common classes
                    train_common_idx, test_common_idx = train_test_split(
                        common_indices, test_size=test_size,
                        stratify=y_common, random_state=42
                    )
                    print(f"Stratified split on common classes successful")
                else:
                    # Cannot stratify even common classes
                    train_common_idx, test_common_idx = train_test_split(
                        common_indices, test_size=test_size, random_state=42
                    )
                    print(f"Random split on common classes (stratification not possible)")

                # Combine rare (all to train) + common split
                train_idx = np.concatenate([rare_indices, train_common_idx])
                test_idx = test_common_idx

        else:
            # No rare classes - use standard stratified split
            print("No rare classes - using standard stratified split")
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=test_size,
                stratify=y_management, random_state=42
            )

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

        return X_train, X_test, targets_train, targets_test

    def create_tabnet_model(self, input_dim, output_dim, target_name):
        """Create TabNet model for specific target"""
        print(f"\nCreating TabNet model for {target_name}:")
        print("-" * 30)

        # Unified configuration for all targets (from original script)
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
            'seed': 48,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"Model config: n_d={config['n_d']}, n_a={config['n_a']}, "
              f"n_steps={config['n_steps']}, classes={output_dim}")

        model = TabNetClassifier(**tabnet_params)
        return model

    def train_target_model(self, X_train, y_train, X_test, y_test, target_name):
        """Train model for specific target"""
        print(f"\n{'=' * 60}")
        print(f"TRAINING MODEL FOR {target_name}")
        print('=' * 60)

        # Get number of classes
        n_classes = len(np.unique(np.concatenate([y_train, y_test])))

        # Create model
        model = self.create_tabnet_model(X_train.shape[1], n_classes, target_name)

        # Create validation split from training data
        val_size = 0.2
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        min_train_count = np.min(counts_train)

        # Check if we can create a proper validation split
        if len(y_train) < 10 or min_train_count == 1:
            # Too few samples or singleton classes - skip validation
            print(f"Insufficient data for validation split - using training data as validation")
            X_train_fit = X_train
            y_train_fit = y_train
            X_val = X_train[:min(50, len(X_train))]  # Use small subset as pseudo-validation
            y_val = y_train[:min(50, len(y_train))]
        else:
            # Try stratified split, fall back to ensuring all training classes in validation
            try:
                if min_train_count >= 2:
                    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                        X_train, y_train, test_size=val_size,
                        stratify=y_train, random_state=42
                    )

                    # Check if validation set has classes not in training set
                    unique_train_fit = set(np.unique(y_train_fit))
                    unique_val = set(np.unique(y_val))

                    if not unique_val.issubset(unique_train_fit):
                        # Re-split to ensure all validation classes are in training
                        print(f"Validation contains classes not in training - adjusting split")
                        raise ValueError("Need to adjust split")

                    print(f"Used stratified validation split")
                else:
                    raise ValueError("Cannot stratify")

            except (ValueError, Exception):
                # Use random split with manual class balancing
                print(f"Using custom validation split to ensure class compatibility")

                # Ensure each class has at least one sample in training
                indices_by_class = {}
                for cls in unique_train:
                    indices_by_class[cls] = np.where(y_train == cls)[0]

                train_indices = []
                val_indices = []

                for cls, cls_indices in indices_by_class.items():
                    n_cls_samples = len(cls_indices)
                    if n_cls_samples == 1:
                        # Single sample - put in training
                        train_indices.extend(cls_indices)
                    else:
                        # Multiple samples - split but ensure at least 1 in training
                        n_val = max(1, int(n_cls_samples * val_size))
                        n_train = n_cls_samples - n_val

                        np.random.seed(48)
                        shuffled_indices = np.random.permutation(cls_indices)
                        train_indices.extend(shuffled_indices[:n_train])
                        val_indices.extend(shuffled_indices[n_train:])

                X_train_fit = X_train[train_indices]
                y_train_fit = y_train[train_indices]
                X_val = X_train[val_indices] if val_indices else X_train[:5]  # Fallback
                y_val = y_train[val_indices] if val_indices else y_train[:5]  # Fallback

        print(f"Training: {len(X_train_fit)}, Validation: {len(X_val)}, Test: {len(X_test)}")

        # Show class distribution
        print(f"\nTraining set class distribution:")
        for cls in unique_train:
            class_name = self.label_encoders[target_name].classes_[cls]
            count = np.sum(y_train_fit == cls)
            pct = (count / len(y_train_fit)) * 100
            print(f"  '{class_name}': {count} ({pct:.1f}%)")

        # Calculate class weights for classes present in training data
        unique_train_fit, counts_train_fit = np.unique(y_train_fit, return_counts=True)

        if len(unique_train_fit) > 1:
            class_weights = compute_class_weight('balanced',
                                                 classes=unique_train_fit,
                                                 y=y_train_fit)
            print(f"\nClass weights:")
            for cls, weight in zip(unique_train_fit, class_weights):
                class_name = self.label_encoders[target_name].classes_[cls]
                print(f"  '{class_name}': {weight:.3f}")
        else:
            print(f"\nOnly one class in training set - no class weighting applied")
            class_weights = None

        # Adjust training parameters based on target (from original script)
        if target_name == 'Management':
            # More epochs for complex/imbalanced target
            max_epochs = 250
            patience = 25
            batch_size = 8  # Smaller for rare classes
            virtual_batch_size = 4
        elif target_name == 'Severity':
            max_epochs = 150
            patience = 25
            batch_size = 8
            virtual_batch_size = 4
        else:  # Diagnosis
            max_epochs = 150
            patience = 25
            batch_size = 8
            virtual_batch_size = 4

        print(f"\nTraining parameters:")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Patience: {patience}")
        print(f"  Batch size: {batch_size}")

        # Final check on data size before training
        if len(X_train_fit) < batch_size:
            print(f"Warning: Training data ({len(X_train_fit)}) smaller than batch size ({batch_size})")
            batch_size = max(2, len(X_train_fit))
            virtual_batch_size = max(2, batch_size // 2)
            print(f"Adjusted batch_size to {batch_size}, virtual_batch_size to {virtual_batch_size}")

        # Ensure we don't have singleton batches with drop_last=True
        drop_last = True if len(X_train_fit) > batch_size else False

        print(f"\nFinal training parameters:")
        print(f"  Training samples: {len(X_train_fit)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Virtual batch size: {virtual_batch_size}")
        print(f"  Drop last: {drop_last}")

        # Train model
        print(f"\nStarting training...")
        history = model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy', 'logloss'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=drop_last  # This ensures no singleton batches
        )

        print(f"✓ Training completed for {target_name}")

        # Store model
        self.models[target_name] = model

        return model, history

    def evaluate_target_model(self, X_test, y_test, target_name):
        """Evaluate model for specific target"""
        print(f"\nEvaluating {target_name} model:")
        print("-" * 30)

        model = self.models[target_name]

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Get class information
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [str(self.label_encoders[target_name].classes_[i])
                        for i in unique_classes]

        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y_test, y_pred,
                                       labels=unique_classes,
                                       target_names=target_names,
                                       digits=4)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        print(f"\nConfusion Matrix:")
        print(cm)

        # ROC AUC
        try:
            if len(unique_classes) == 2:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba,
                                          multi_class='ovr', average='macro')
            print(f"ROC AUC: {auc_score:.4f}")
        except:
            auc_score = None
            print("ROC AUC: Could not calculate")

        # Store results
        results = {
            'accuracy': accuracy,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'auc_score': auc_score,
            'target_names': target_names,
            'unique_classes': unique_classes,
            'classification_report': report
        }

        self.results[target_name] = results
        return results

    def create_comprehensive_visualization(self):
        """Create comprehensive visualization for all targets"""
        print(f"\nCreating comprehensive visualizations...")

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('TabNet Single Split Training Results', fontsize=16, fontweight='bold')

        for idx, target in enumerate(self.target_columns):
            if target not in self.results:
                continue

            results = self.results[target]

            # Confusion Matrix
            ax_cm = axes[idx, 0]
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=results['target_names'],
                        yticklabels=results['target_names'])
            ax_cm.set_title(f'{target} - Confusion Matrix')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')

            # Class Distribution
            ax_dist = axes[idx, 1]
            y_true = results['y_true']
            y_pred = results['y_pred']

            true_counts = pd.Series(y_true).value_counts().sort_index()
            pred_counts = pd.Series(y_pred).value_counts().sort_index()

            x = np.arange(len(results['target_names']))
            width = 0.35

            ax_dist.bar(x - width / 2, [true_counts.get(i, 0) for i in results['unique_classes']],
                        width, label='True', alpha=0.8)
            ax_dist.bar(x + width / 2, [pred_counts.get(i, 0) for i in results['unique_classes']],
                        width, label='Predicted', alpha=0.8)

            ax_dist.set_xlabel('Classes')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title(f'{target} - Class Distribution')
            ax_dist.set_xticks(x)
            ax_dist.set_xticklabels(results['target_names'], rotation=45)
            ax_dist.legend()

            # Performance Metrics
            ax_perf = axes[idx, 2]
            metrics = ['Accuracy']
            values = [results['accuracy']]
            colors = ['lightblue']

            if results['auc_score'] is not None:
                metrics.append('ROC AUC')
                values.append(results['auc_score'])
                colors.append('lightcoral')

            bars = ax_perf.bar(metrics, values, color=colors)
            ax_perf.set_ylabel('Score')
            ax_perf.set_title(f'{target} - Performance')
            ax_perf.set_ylim(0, 1.1)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_perf.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        filename = f'../../results/tabnet_single_split_results_{milliseconds}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✓ Visualization saved: {filename}")

    def save_all_results(self):
        """Save results for all targets"""
        print(f"\nSaving results...")

        # Create summary DataFrame
        summary_data = []
        for target in self.target_columns:
            if target in self.results:
                results = self.results[target]
                summary_data.append({
                    'Target': target,
                    'Accuracy': results['accuracy'],
                    'ROC_AUC': results['auc_score'] if results['auc_score'] else 'N/A',
                    'Test_Samples': len(results['y_true']),
                    'Classes': len(results['unique_classes']),
                    'Timestamp': milliseconds
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'../../output/tabnet_single_split_summary_{milliseconds}.csv', index=False)

        # Save detailed results for each target
        for target in self.target_columns:
            if target in self.results:
                results = self.results[target]

                # Predictions
                pred_df = pd.DataFrame({
                    'True_Label': [self.label_encoders[target].classes_[i] for i in results['y_true']],
                    'Predicted_Label': [self.label_encoders[target].classes_[i] for i in results['y_pred']],
                    'Correct': results['y_true'] == results['y_pred']
                })

                # Add probabilities
                if results['y_pred_proba'] is not None:
                    for i, class_name in enumerate(self.label_encoders[target].classes_):
                        if i < results['y_pred_proba'].shape[1]:
                            pred_df[f'Prob_{class_name}'] = results['y_pred_proba'][:, i]

                pred_df.to_csv(f'../../output/tabnet_single_split_{target.lower()}_predictions_{milliseconds}.csv', index=False)

                # Confusion matrix
                cm_df = pd.DataFrame(results['confusion_matrix'],
                                     index=results['target_names'],
                                     columns=results['target_names'])
                cm_df.to_csv(f'../../output/tabnet_single_split_{target.lower()}_confusion_matrix_{milliseconds}.csv')

        print(f"✓ Results saved:")
        print(f"  - tabnet_single_split_summary_{milliseconds}.csv")
        for target in self.target_columns:
            if target in self.results:
                print(f"  - tabnet_single_split_{target.lower()}_predictions_{milliseconds}.csv")
                print(f"  - tabnet_single_split_{target.lower()}_confusion_matrix_{milliseconds}.csv")

    def run_complete_training(self):
        """Run complete single split training pipeline"""
        print("Starting TabNet Single Split Training Pipeline")
        print("=" * 70)
        print("Configuration:")
        print("- Single Train/Test Split (No Cross-Validation)")
        print("- Ratio: 2:1 (train:test)")
        print("- Separate models for each target")
        print("- Custom stratification for rare classes")
        print()

        # Load and prepare data
        if not self.load_and_prepare_data():
            print("✗ Data preparation failed")
            return False

        # Preprocess data
        X, targets_encoded = self.preprocess_data()

        # Create single stratified split
        X_train, X_test, targets_train, targets_test = self.create_stratified_split_with_rare_classes(
            X, targets_encoded
        )

        # Train models for each target
        print(f"\n{'=' * 70}")
        print("TRAINING PHASE - Single Split")
        print('=' * 70)

        for target in self.target_columns:
            y_train = targets_train[target]
            y_test = targets_test[target]

            try:
                # Train model
                model, history = self.train_target_model(X_train, y_train, X_test, y_test, target)

                # Evaluate model
                self.evaluate_target_model(X_test, y_test, target)

            except Exception as e:
                print(f"✗ Error training {target}: {e}")
                continue

        # Create visualizations
        if self.results:
            self.create_comprehensive_visualization()

            # Save results
            self.save_all_results()

            # Final summary
            print(f"\n{'=' * 70}")
            print("SINGLE SPLIT TRAINING COMPLETED!")
            print("=" * 70)

            print(f"\nFinal Results Summary:")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features used: {len(self.feature_columns)}")
            print(f"Timestamp: {milliseconds}")

            for target in self.target_columns:
                if target in self.results:
                    acc = self.results[target]['accuracy']
                    auc = self.results[target]['auc_score']
                    n_classes = len(self.results[target]['unique_classes'])
                    print(f"- {target}: Accuracy={acc:.4f}, Classes={n_classes}", end="")
                    if auc:
                        print(f", AUC={auc:.4f}")
                    else:
                        print()

            return True
        else:
            print("✗ No successful training completed")
            return False


def main():
    """Main execution function"""
    print("TabNet Single Split Multi-Target Training")
    print("=" * 50)
    print("Targets: Diagnosis, Severity, Management")
    print("Strategy: Single train/test split (no k-fold)")
    print("Special handling: Management rare classes")
    print("Split ratio: 2:1 (train:test)")
    print()

    # Create trainer
    trainer = TabNetSingleSplitTrainer('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run training
    success = trainer.run_complete_training()

    if success:
        print("\n✓ Single split training completed successfully!")
        print("Check generated files for detailed results and visualizations.")
    else:
        print("\n✗ Single split training failed!")


if __name__ == "__main__":
    main()