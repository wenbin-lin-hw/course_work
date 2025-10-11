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
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class TabNetTrainerWithPrediction:
    def __init__(self, data_path):
        """Initialize TabNet Trainer with Model Saving and Prediction Capabilities"""
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

        # Create timestamp for unique naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model directory
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_prepare_data(self):
        """Load and prepare the data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA FOR TRAINING")
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

    def preprocess_data(self, data=None, fit_transformers=True):
        """Preprocess features for training or prediction"""
        print(f"\nData Preprocessing:")
        print("-" * 30)

        # Use provided data or self.data
        if data is None:
            data = self.data
            print("Using training data for preprocessing")
        else:
            print("Using provided data for preprocessing")

        # Extract features
        X = data[self.feature_columns].copy()
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
                if fit_transformers:
                    # Training mode - fit new encoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    # Store encoder for this column (not target encoders)
                    setattr(self, f'feature_encoder_{col}', le)
                else:
                    # Prediction mode - use existing encoder
                    if hasattr(self, f'feature_encoder_{col}'):
                        le = getattr(self, f'feature_encoder_{col}')
                        # Handle unseen categories
                        unique_values = X[col].astype(str).unique()
                        known_classes = set(le.classes_)
                        unknown_values = [v for v in unique_values if v not in known_classes]

                        if unknown_values:
                            print(f"Warning: Unknown categories in {col}: {unknown_values}")
                            # Replace unknown values with most frequent known class
                            most_frequent = le.classes_[0]  # Use first class as default
                            X[col] = X[col].astype(str).replace(unknown_values, most_frequent)

                        X[col] = le.transform(X[col].astype(str))
                    else:
                        print(f"Warning: No encoder found for {col}, skipping encoding")
                categorical_count += 1

        print(f"Processed {categorical_count} categorical features")

        # Scale features
        if fit_transformers:
            # Training mode - fit scaler
            X_scaled = self.scaler.fit_transform(X)
            print(f"✓ Features scaled and fitted")
        else:
            # Prediction mode - use existing scaler
            X_scaled = self.scaler.transform(X)
            print(f"✓ Features scaled using existing scaler")

        # Process targets if this is training data
        if fit_transformers and data is self.data:
            targets_encoded = {}
            for target in self.target_columns:
                y = data[target].copy()
                le = LabelEncoder()
                targets_encoded[target] = le.fit_transform(y)
                self.label_encoders[target] = le

                print(f"\n{target} encoding:")
                for original, encoded in zip(le.classes_, range(len(le.classes_))):
                    count = np.sum(targets_encoded[target] == encoded)
                    print(f"  '{original}' -> {encoded} ({count} samples)")

            return X_scaled, targets_encoded
        else:
            return X_scaled

    def create_stratified_split(self, X, targets_encoded, test_size=1/3):
        """Create stratified split handling rare classes"""
        print(f"\nStratified Data Splitting:")
        print("-" * 35)

        # Use Management for stratification (most complex)
        y_management = targets_encoded['Management']
        unique_mgmt, counts_mgmt = np.unique(y_management, return_counts=True)

        print(f"Management class distribution:")
        for cls, count in zip(unique_mgmt, counts_mgmt):
            class_name = self.label_encoders['Management'].classes_[cls]
            print(f"  Class {cls} ('{class_name}'): {count} samples")

        # Handle rare classes
        rare_classes = unique_mgmt[counts_mgmt <= 2]

        if len(rare_classes) > 0:
            print(f"\nRare classes detected: {rare_classes}")
            # Put rare classes in training set
            rare_indices = np.where(np.isin(y_management, rare_classes))[0]
            common_indices = np.where(~np.isin(y_management, rare_classes))[0]

            if len(common_indices) > 0:
                y_common = y_management[common_indices]
                unique_common, counts_common = np.unique(y_common, return_counts=True)

                if np.min(counts_common) >= 2:
                    train_common_idx, test_idx = train_test_split(
                        common_indices, test_size=test_size,
                        stratify=y_common, random_state=42
                    )
                    print("Used stratified split for common classes")
                else:
                    train_common_idx, test_idx = train_test_split(
                        common_indices, test_size=test_size, random_state=42
                    )
                    print("Used random split for common classes")

                train_idx = np.concatenate([rare_indices, train_common_idx])
            else:
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)), test_size=test_size, random_state=42
                )
        else:
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

        print(f"\nSplit Results:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Ratio: {len(X_train)/len(X_test):.2f}:1")

        return X_train, X_test, targets_train, targets_test

    def create_tabnet_model(self, input_dim, output_dim, target_name):
        """Create TabNet model for specific target"""
        tabnet_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 6,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=1.5e-2, weight_decay=1e-5),
            'mask_type': 'entmax',
            'scheduler_params': {"step_size": 40, "gamma": 0.85},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'seed': 42,
            'verbose': 1,
            'device_name': str(self.device)
        }

        print(f"\nCreating TabNet model for {target_name}")
        print(f"Input features: {input_dim}, Output classes: {output_dim}")

        model = TabNetClassifier(**tabnet_params)
        return model

    def train_target_model(self, X_train, y_train, target_name):
        """Train model for specific target with validation split"""
        print(f"\n{'=' * 60}")
        print(f"TRAINING MODEL FOR {target_name}")
        print('=' * 60)

        n_classes = len(np.unique(y_train))
        model = self.create_tabnet_model(X_train.shape[1], n_classes, target_name)

        # Create validation split
        val_size = 0.2
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_count = np.min(class_counts)

        if len(y_train) >= 10 and min_count >= 2:
            try:
                X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                    X_train, y_train, test_size=val_size,
                    stratify=y_train, random_state=42
                )
                print("Used stratified validation split")
            except:
                X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                    X_train, y_train, test_size=val_size, random_state=42
                )
                print("Used random validation split")
        else:
            # Use small subset as validation
            val_samples = min(10, len(X_train) // 5)
            X_train_fit = X_train
            y_train_fit = y_train
            X_val = X_train[:val_samples]
            y_val = y_train[:val_samples]
            print(f"Used pseudo-validation with {val_samples} samples")

        # Training parameters
        max_epochs = 150
        patience = 25
        batch_size = max(8, min(64, len(X_train_fit) // 4))
        virtual_batch_size = max(4, batch_size // 2)

        # Adjust batch size if needed
        if len(X_train_fit) <= batch_size:
            batch_size = max(2, len(X_train_fit) - 1)
            virtual_batch_size = max(2, batch_size // 2)

        print(f"Training parameters:")
        print(f"  Samples: {len(X_train_fit)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {max_epochs}")

        # Train model
        try:
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
                drop_last=True if len(X_train_fit) > batch_size else False
            )
            print(f"✓ Training completed for {target_name}")
        except Exception as e:
            print(f"⚠ Training error for {target_name}: {e}")
            # Simplified fallback
            history = model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                eval_name=['validation'],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=15,
                batch_size=max(2, len(X_train_fit) // 2),
                virtual_batch_size=2,
                num_workers=0,
                drop_last=False
            )
            print(f"✓ Fallback training completed for {target_name}")

        self.models[target_name] = model
        return model, history

    def evaluate_model(self, X_test, y_test, target_name):
        """Evaluate trained model"""
        print(f"\nEvaluating {target_name} model:")
        print("-" * 30)

        model = self.models[target_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

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
                                       digits=4,
                                       zero_division=0)
        print(report)

        # Store results
        results = {
            'accuracy': accuracy,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'target_names': target_names,
            'unique_classes': unique_classes
        }

        self.results[target_name] = results
        return results

    def save_models_and_preprocessors(self):
        """Save trained models and preprocessing objects"""
        print(f"\nSaving models and preprocessors...")

        saved_files = []

        # Save each trained model
        for target_name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"tabnet_{target_name.lower()}_{self.timestamp}.zip")
            model.save_model(model_path)
            saved_files.append(model_path)
            print(f"✓ Saved {target_name} model: {model_path}")

        # Save preprocessing objects
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'exclude_columns': self.exclude_columns,
            'timestamp': self.timestamp
        }

        # Save feature encoders
        for attr_name in dir(self):
            if attr_name.startswith('feature_encoder_'):
                preprocessor_data[attr_name] = getattr(self, attr_name)

        preprocessor_path = os.path.join(self.model_dir, f"preprocessors_{self.timestamp}.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        saved_files.append(preprocessor_path)
        print(f"✓ Saved preprocessors: {preprocessor_path}")

        # Save model info
        model_info = {
            'target_columns': self.target_columns,
            'feature_columns': self.feature_columns,
            'timestamp': self.timestamp,
            'device': str(self.device),
            'results_summary': {}
        }

        for target in self.target_columns:
            if target in self.results:
                model_info['results_summary'][target] = {
                    'accuracy': self.results[target]['accuracy'],
                    'classes': self.results[target]['target_names']
                }

        info_path = os.path.join(self.model_dir, f"model_info_{self.timestamp}.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        saved_files.append(info_path)
        print(f"✓ Saved model info: {info_path}")

        return saved_files

    def load_models_and_preprocessors(self, timestamp):
        """Load previously saved models and preprocessors"""
        print(f"\nLoading models and preprocessors for timestamp: {timestamp}")

        try:
            # Load preprocessors
            preprocessor_path = os.path.join(self.model_dir, f"preprocessors_{timestamp}.pkl")
            with open(preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)

            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.feature_columns = preprocessor_data['feature_columns']
            self.target_columns = preprocessor_data['target_columns']
            self.exclude_columns = preprocessor_data['exclude_columns']

            # Load feature encoders
            for key, value in preprocessor_data.items():
                if key.startswith('feature_encoder_'):
                    setattr(self, key, value)

            print(f"✓ Loaded preprocessors")

            # Load models
            for target in self.target_columns:
                model_path = os.path.join(self.model_dir, f"tabnet_{target.lower()}_{timestamp}.zip")
                if os.path.exists(model_path):
                    model = TabNetClassifier()
                    model.load_model(model_path)
                    self.models[target] = model
                    print(f"✓ Loaded {target} model")
                else:
                    print(f"⚠ Model file not found for {target}: {model_path}")

            # Load model info
            info_path = os.path.join(self.model_dir, f"model_info_{timestamp}.pkl")
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)

            print(f"✓ Loaded model info")
            print(f"Models loaded: {list(self.models.keys())}")

            return True

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False

    def predict_new_data(self, new_data_path, timestamp=None):
        """Predict on new data using saved models"""
        print(f"\n{'='*80}")
        print("PREDICTING ON NEW DATA")
        print('='*80)

        # Load models if timestamp provided
        if timestamp and not self.models:
            if not self.load_models_and_preprocessors(timestamp):
                print("Failed to load models")
                return None

        if not self.models:
            print("No models available. Train models first or provide timestamp.")
            return None

        # Load new data
        try:
            # Try different sheet names
            try:
                new_data = pd.read_excel(new_data_path, sheet_name='Processed_Data')
                print("✓ New data loaded from 'Processed_Data' sheet")
            except:
                new_data = pd.read_excel(new_data_path)
                print("✓ New data loaded from default sheet")

            print(f"New data shape: {new_data.shape}")

            # Check if required feature columns exist
            missing_features = [col for col in self.feature_columns if col not in new_data.columns]
            if missing_features:
                print(f"⚠ Missing feature columns: {missing_features}")
                print("Prediction may not be accurate")

            # Preprocess new data (without fitting transformers)
            X_new = self.preprocess_data(new_data, fit_transformers=False)
            print(f"Preprocessed data shape: {X_new.shape}")

            # Make predictions for each target
            predictions = {}
            prediction_probabilities = {}

            for target in self.target_columns:
                if target in self.models:
                    print(f"\nPredicting {target}...")

                    model = self.models[target]
                    y_pred = model.predict(X_new)
                    y_pred_proba = model.predict_proba(X_new)

                    # Convert predictions back to original labels
                    le = self.label_encoders[target]
                    predicted_labels = le.inverse_transform(y_pred)

                    predictions[target] = predicted_labels
                    prediction_probabilities[target] = y_pred_proba

                    print(f"✓ {target} predictions completed")

                    # Show prediction distribution
                    unique_preds, counts = np.unique(predicted_labels, return_counts=True)
                    print(f"Prediction distribution:")
                    for pred, count in zip(unique_preds, counts):
                        pct = (count / len(predicted_labels)) * 100
                        print(f"  {pred}: {count} ({pct:.1f}%)")

            # Create results DataFrame
            results_df = new_data.copy()

            # Add predictions
            for target in self.target_columns:
                if target in predictions:
                    results_df[f'Predicted_{target}'] = predictions[target]

                    # Add probability columns
                    le = self.label_encoders[target]
                    for i, class_name in enumerate(le.classes_):
                        if i < prediction_probabilities[target].shape[1]:
                            results_df[f'Prob_{target}_{class_name}'] = prediction_probabilities[target][:, i]

            # Save predictions
            output_path = f"predictions_{timestamp or self.timestamp}_{int(time.time())}.xlsx"
            results_df.to_excel(output_path, index=False)
            print(f"\n✓ Predictions saved to: {output_path}")

            # Summary
            print(f"\nPrediction Summary:")
            print(f"Input samples: {len(new_data)}")
            print(f"Predictions made for: {list(predictions.keys())}")
            print(f"Output file: {output_path}")

            return results_df, predictions, prediction_probabilities

        except Exception as e:
            print(f"✗ Error in prediction: {e}")
            return None

    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("STARTING TABNET TRAINING WITH MODEL SAVING")
        print("="*70)

        # Load and prepare data
        if not self.load_and_prepare_data():
            return False

        # Preprocess data
        X, targets_encoded = self.preprocess_data()

        # Create train/test split
        X_train, X_test, targets_train, targets_test = self.create_stratified_split(
            X, targets_encoded
        )

        # Train models for each target
        print(f"\n{'='*70}")
        print("TRAINING PHASE")
        print('='*70)

        for target in self.target_columns:
            y_train = targets_train[target]
            y_test = targets_test[target]

            try:
                # Train model
                model, history = self.train_target_model(X_train, y_train, target)

                # Evaluate model
                self.evaluate_model(X_test, y_test, target)

            except Exception as e:
                print(f"✗ Failed to train {target}: {e}")
                continue

        # Save models and preprocessors
        if self.models:
            saved_files = self.save_models_and_preprocessors()

            print(f"\n{'='*70}")
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)

            print(f"\nTraining Summary:")
            print(f"Timestamp: {self.timestamp}")
            print(f"Models trained: {list(self.models.keys())}")
            print(f"Files saved: {len(saved_files)}")

            for target in self.target_columns:
                if target in self.results:
                    acc = self.results[target]['accuracy']
                    print(f"- {target}: Accuracy={acc:.4f}")

            print(f"\nTo make predictions on new data, use:")
            print(f"predictor.predict_new_data('path_to_new_data.xlsx', '{self.timestamp}')")

            return True
        else:
            print("✗ No models were trained successfully")
            return False

def main():
    """Example usage"""
    print("TabNet Trainer with Model Saving and Prediction")
    print("="*60)

    # Initialize trainer
    trainer = TabNetTrainerWithPrediction('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Train models and save
    success = trainer.run_training_pipeline()

    if success:
        print(f"\n✓ Training completed! Models saved with timestamp: {trainer.timestamp}")

        # Example of how to use for prediction
        print(f"\n" + "="*60)
        print("PREDICTION EXAMPLE")
        print("="*60)
        print("To predict on new data:")
        print("1. Use the same trainer instance:")
        print("   predictions = trainer.predict_new_data('path_to_new_data.xlsx')")
        print()
        print("2. Or create new instance and load saved models:")
        print("   predictor = TabNetTrainerWithPrediction('data_path')")
        print(f"   predictions = predictor.predict_new_data('new_data.xlsx', '{trainer.timestamp}')")

    else:
        print("\n✗ Training failed!")

# Example function for prediction only
def predict_with_saved_models(new_data_path, timestamp):
    """Standalone function to make predictions using saved models"""
    print("Making predictions with saved models...")

    # Create predictor instance (no training data needed for prediction)
    predictor = TabNetTrainerWithPrediction('')

    # Make predictions
    results = predictor.predict_new_data(new_data_path, timestamp)

    if results:
        results_df, predictions, probabilities = results
        print("✓ Predictions completed successfully!")
        return results_df
    else:
        print("✗ Prediction failed!")
        return None

if __name__ == "__main__":
    main()