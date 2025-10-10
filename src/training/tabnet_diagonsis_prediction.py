import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import pearsonr, chi2_contingency
import warnings
import matplotlib
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')
from pytorch_tabnet.tab_model import TabNetClassifier



class DiagnosisPredictor:
    def __init__(self, data_path):
        """Initialize the Diagnosis Predictor"""
        self.data_path = data_path
        self.data = None
        self.target_column = 'Diagnosis'
        self.exclude_columns = ['US_Performed', 'US_Number', 'Management', 'Severity', 'Diagnosis']
        self.selected_features = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.correlation_results = {}

    def load_data(self):
        """Load data from Excel file"""
        print("=" * 80)
        print("LOADING PROCESSED APPENDICITIS DATA FOR DIAGNOSIS PREDICTION")
        print("=" * 80)

        try:
            # Try to read from Processed_Data sheet first
            try:
                self.data = pd.read_excel(self.data_path, sheet_name='Processed_Data')
                print("✓ Data loaded from 'Processed_Data' sheet")
            except:
                self.data = pd.read_excel(self.data_path)
                print("✓ Data loaded from default sheet")

            print(f"Data shape: {self.data.shape}")
            print(f"Target column '{self.target_column}' distribution:")
            target_counts = self.data[self.target_column].value_counts()
            for value, count in target_counts.items():
                pct = (count / len(self.data)) * 100
                print(f"  {value}: {count} ({pct:.1f}%)")

            return True

        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False

    def calculate_feature_correlation_with_target(self):
        """Calculate correlation of all features with target variable"""
        print(f"\nCalculating correlation with {self.target_column}:")
        print("-" * 60)

        # Prepare candidate features (exclude target and other targets)
        all_exclude = [self.target_column] + self.exclude_columns
        candidate_features = [col for col in self.data.columns if col not in all_exclude]

        print(f"Analyzing {len(candidate_features)} candidate features...")

        correlations = []

        # Encode target variable for correlation calculation
        target_encoded = self.label_encoder.fit_transform(self.data[self.target_column])

        for i, feature in enumerate(candidate_features):
            if i % 20 == 0:
                print(f"Progress: {i}/{len(candidate_features)} features processed")

            try:
                feature_data = self.data[feature].copy()

                # Create valid data (both feature and target non-null)
                valid_mask = feature_data.notnull()
                if valid_mask.sum() < 10:  # Not enough valid data
                    continue

                feature_valid = feature_data[valid_mask]
                target_valid = target_encoded[valid_mask]

                # Determine correlation method based on feature type and unique values
                unique_count = feature_valid.nunique()

                if unique_count <= 6:
                    # Categorical feature - use Chi-square test
                    try:
                        contingency_table = pd.crosstab(feature_valid, target_valid)
                        if contingency_table.size >= 4:
                            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                            # Calculate Cramér's V as effect size
                            n = contingency_table.sum().sum()
                            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                            correlation = cramers_v
                            method = "Chi-square (Cramér's V)"
                        else:
                            correlation, p_value = np.nan, np.nan
                            method = "Chi-square (insufficient data)"
                    except:
                        correlation, p_value = np.nan, np.nan
                        method = "Chi-square (error)"
                else:
                    # Numerical feature - use Pearson correlation
                    try:
                        # Convert to numeric if possible
                        feature_numeric = pd.to_numeric(feature_valid, errors='coerce')
                        valid_numeric_mask = ~np.isnan(feature_numeric)

                        if valid_numeric_mask.sum() < 10:
                            correlation, p_value = np.nan, np.nan
                            method = "Pearson (insufficient data)"
                        else:
                            feature_final = feature_numeric[valid_numeric_mask]
                            target_final = target_valid[valid_numeric_mask]
                            correlation, p_value = pearsonr(feature_final, target_final)
                            method = "Pearson"
                    except:
                        correlation, p_value = np.nan, np.nan
                        method = "Pearson (error)"

                correlations.append({
                    'Feature': feature,
                    'Correlation': correlation,
                    'Abs_Correlation': abs(correlation) if not np.isnan(correlation) else np.nan,
                    'P_Value': p_value,
                    'Method': method,
                    'Unique_Values': unique_count,
                    'Valid_Samples': valid_mask.sum(),
                    'Significant': p_value < 0.05 if not np.isnan(p_value) else False
                })

            except Exception as e:
                print(f"Error processing {feature}: {e}")
                continue

        # Convert to DataFrame and sort by absolute correlation
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False, na_position='last')

        self.correlation_results = corr_df

        print(f"✓ Correlation analysis completed")
        print(f"Valid correlations: {corr_df['Correlation'].notna().sum()}")
        print(f"Significant correlations: {corr_df['Significant'].sum()}")

        return corr_df

    def select_top_features(self, top_n=10):
        """Select top N features with highest correlation to target"""
        print(f"\nSelecting top {top_n} features with highest correlation:")
        print("-" * 60)

        if self.correlation_results is None:
            self.calculate_feature_correlation_with_target()

        # Get top features with valid correlations
        valid_corr = self.correlation_results[self.correlation_results['Correlation'].notna()]
        top_features = valid_corr.head(top_n)

        print(f"{'Rank':<4} {'Feature':<35} {'Correlation':<12} {'Method':<20} {'Significant':<12}")
        print("-" * 85)

        selected_feature_names = []
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_name = row['Feature']
            selected_feature_names.append(feature_name)

            print(
                f"{i:<4} {feature_name[:34]:<35} {row['Correlation']:>11.4f} {row['Method']:<20} {'Yes' if row['Significant'] else 'No':<12}")

        self.selected_features = selected_feature_names

        print(f"\nSelected {len(selected_feature_names)} features for model training")
        return selected_feature_names

    def prepare_model_data(self):
        """Prepare data for model training"""
        print(f"\nPreparing data for model training:")
        print("-" * 40)

        if self.selected_features is None:
            self.select_top_features()

        # Prepare features and target
        X = self.data[self.selected_features].copy()
        y = self.data[self.target_column].copy()

        # Handle missing values in features (should be minimal after preprocessing)
        print(f"Checking for missing values in selected features...")
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}")
            # Fill missing values with mode for categorical, mean for numerical
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if X[col].nunique() <= 6:  # Categorical
                        X[col].fillna(X[col].mode()[0], inplace=True)
                    else:  # Numerical
                        X[col].fillna(X[col].mean(), inplace=True)
        else:
            print("✓ No missing values in selected features")

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        target_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"Target encoding: {target_mapping}")

        # Scale features for TabNet
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        print(f"Final data shape: {X_scaled.shape}")
        print(f"Target classes: {len(np.unique(y_encoded))}")

        return X_scaled, y_encoded

    def create_stratified_splits(self, X, y, n_splits=5, test_size=0.25, random_state=42):
        """Create stratified train-test splits with 5-fold CV"""
        print(f"\nCreating stratified data splits:")
        print("-" * 35)
        print(f"Using {n_splits}-fold cross-validation")
        print(f"Train-test ratio: {1 - test_size:.1f}:{test_size:.1f}")

        # First split into train+val and test sets
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train+Validation set: {X_trainval.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Check class distribution
        print("\nClass distribution in train+val:")
        trainval_counts = np.bincount(y_trainval)
        for i, count in enumerate(trainval_counts):
            pct = (count / len(y_trainval)) * 100
            print(f"  Class {i}: {count} ({pct:.1f}%)")

        print("\nClass distribution in test:")
        test_counts = np.bincount(y_test)
        for i, count in enumerate(test_counts):
            pct = (count / len(y_test)) * 100
            print(f"  Class {i}: {count} ({pct:.1f}%)")

        # Create 5-fold CV splits on train+val set
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = list(skf.split(X_trainval, y_trainval))

        print(f"\n✓ Created {n_splits} CV folds for training")

        return X_trainval, X_test, y_trainval, y_test, cv_splits

    def train_tabnet_model(self, X_train, y_train, X_val, y_val, fold_num=1):
        """Train TabNet model"""
        print(f"\nTraining TabNet model for fold {fold_num}:")
        print("-" * 45)

        # Convert to numpy arrays
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_train_np = y_train.astype(int)
        y_val_np = y_val.astype(int)

        # Initialize TabNet model
        tabnet_model = TabNetClassifier(
            n_d=32,  # Width of the decision prediction layer
            n_a=32,  # Width of the attention embedding for each mask
            n_steps=5,  # Number of steps in the architecture
            gamma=1.5,  # Coefficient for feature reusage in the masks
            cat_idxs=[],  # List of categorical feature indices (empty since all are preprocessed)
            cat_dims=[],  # List of categorical feature dimensions
            cat_emb_dim=1,  # Embedding dimension for categorical features
            n_independent=2,  # Number of independent GLU layers in each GLU block
            n_shared=2,  # Number of shared GLU layers in each GLU block
            epsilon=1e-15,  # Avoid log(0), this should be kept very low
            momentum=0.02,  # Momentum for batch normalization
            lambda_sparse=1e-3,  # Sparsity regularization
            seed=42,
            verbose=1
        )

        print("TabNet Model Configuration:")
        print(f"  - Decision layer width (n_d): 32")
        print(f"  - Attention width (n_a): 32")
        print(f"  - Number of steps: 5")
        print(f"  - Gamma: 1.5")
        print(f"  - Lambda sparse: 1e-3")

        # Train the model
        tabnet_model.fit(
            X_train=X_train_np,
            y_train=y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=['validation'],
            eval_metric=['accuracy'],
            max_epochs=200,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            augmentations=None,
            weights=0  # No class weights
        )

        # Get feature importance
        feature_importance = tabnet_model.feature_importances_

        print(f"✓ TabNet model training completed for fold {fold_num}")

        return tabnet_model, feature_importance

    def evaluate_model(self, model, X_test, y_test, fold_num=1, feature_names=None):
        """Evaluate model performance"""
        print(f"\nEvaluating model performance for fold {fold_num}:")
        print("-" * 50)

        # Convert to numpy if needed
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        # Make predictions
        y_pred = model.predict(X_test_np)
        y_pred_proba = model.predict_proba(X_test_np)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report
        target_names = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            # Convert classes to strings to ensure compatibility with classification_report
            target_names = [str(class_name) for class_name in self.label_encoder.classes_]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        return {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }

    def visualize_feature_importance(self, feature_importances_list, feature_names, fold_results):
        """Visualize feature importance across folds"""
        print(f"\nVisualizing results:")
        print("-" * 25)

        # Average feature importance across folds
        avg_importance = np.mean(feature_importances_list, axis=0)
        std_importance = np.std(feature_importances_list, axis=0)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance,
            'Std': std_importance
        }).sort_values('Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.barh(range(len(importance_df)), importance_df['Importance'],
                 xerr=importance_df['Std'], alpha=0.7)
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('TabNet Feature Importance (Average across folds)')
        plt.gca().invert_yaxis()

        # Plot accuracy across folds
        plt.subplot(2, 2, 2)
        accuracies = [result['accuracy'] for result in fold_results]
        plt.bar(range(1, len(accuracies) + 1), accuracies, alpha=0.7)
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy across CV Folds')
        plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                    label=f'Mean: {np.mean(accuracies):.4f}')
        plt.legend()

        # Plot confusion matrix for best fold
        best_fold_idx = np.argmax(accuracies)
        plt.subplot(2, 2, 3)
        sns.heatmap(fold_results[best_fold_idx]['confusion_matrix'],
                    annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Best Fold: {best_fold_idx + 1})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Plot correlation heatmap for top features
        plt.subplot(2, 2, 4)
        if hasattr(self, 'data') and self.selected_features:
            top_features_data = self.data[self.selected_features[:8]]  # Top 8 features
            corr_matrix = top_features_data.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        square=True, cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix (Top 8)')

        plt.tight_layout()
        plt.savefig('../../pictures/tabnet_diagnosis_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Visualization saved as: pictures/tabnet_diagnosis_prediction_results.png")

        return importance_df

    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        print("STARTING TABNET DIAGNOSIS PREDICTION EXPERIMENT")
        print("=" * 80)

        # Step 1: Load data
        if not self.load_data():
            return False

        # Step 2: Calculate correlations and select features
        self.calculate_feature_correlation_with_target()
        self.select_top_features(top_n=10)

        # Step 3: Prepare data
        X, y = self.prepare_model_data()

        # Step 4: Create stratified splits
        X_trainval, X_test, y_trainval, y_test, cv_splits = self.create_stratified_splits(X, y)

        # Step 5: Train models using 5-fold CV
        print(f"\n" + "=" * 60)
        print("TRAINING TABNET MODELS WITH 5-FOLD CROSS-VALIDATION")
        print("=" * 60)

        fold_results = []
        feature_importances_list = []
        models = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, 1):
            print(f"\n{'=' * 40}")
            print(f"FOLD {fold_idx}/5")
            print('=' * 40)

            # Split data for current fold
            X_train_fold = X_trainval.iloc[train_idx]
            X_val_fold = X_trainval.iloc[val_idx]
            y_train_fold = y_trainval[train_idx]
            y_val_fold = y_trainval[val_idx]

            print(f"Train samples: {len(X_train_fold)}")
            print(f"Validation samples: {len(X_val_fold)}")

            # Train model
            model, feature_importance = self.train_tabnet_model(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_idx
            )

            # Evaluate on validation set
            fold_result = self.evaluate_model(
                model, X_val_fold, y_val_fold, fold_idx, self.selected_features
            )

            fold_results.append(fold_result)
            feature_importances_list.append(feature_importance)
            models.append(model)

        # Step 6: Final evaluation on test set using best model
        print(f"\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Select best model based on validation accuracy
        val_accuracies = [result['accuracy'] for result in fold_results]
        best_fold_idx = np.argmax(val_accuracies)
        best_model = models[best_fold_idx]

        print(f"Best model from fold {best_fold_idx + 1} (accuracy: {val_accuracies[best_fold_idx]:.4f})")

        # Final test evaluation
        test_results = self.evaluate_model(
            best_model, X_test, y_test, "Final Test", self.selected_features
        )

        # Step 7: Generate comprehensive report
        print(f"\n" + "=" * 80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("=" * 80)

        print(f"\nCross-Validation Results:")
        print(f"Mean CV Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
        print(f"Best CV Accuracy: {np.max(val_accuracies):.4f}")
        print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")

        print(f"\nSelected Features (Top 10):")
        for i, feature in enumerate(self.selected_features, 1):
            corr_info = self.correlation_results[self.correlation_results['Feature'] == feature].iloc[0]
            print(f"  {i:2d}. {feature:<30} (corr: {corr_info['Correlation']:>7.4f})")

        # Step 8: Visualize results
        importance_df = self.visualize_feature_importance(
            feature_importances_list, self.selected_features, fold_results
        )

        # Save detailed results
        self.save_results(fold_results, test_results, importance_df)

        print(f"\n✓ Experiment completed successfully!")
        print(f"Results saved to: tabnet_diagnosis_results.csv")

        return {
            'cv_results': fold_results,
            'test_results': test_results,
            'feature_importance': importance_df,
            'best_model': best_model,
            'selected_features': self.selected_features
        }

    def save_results(self, fold_results, test_results, importance_df):
        """Save experiment results to files"""
        # Save feature importance
        importance_df.to_csv('../../output/tabnet_feature_importance.csv', index=False)

        # Save detailed results
        results_summary = {
            'Metric': [
                'Mean_CV_Accuracy', 'Std_CV_Accuracy', 'Best_CV_Accuracy',
                'Final_Test_Accuracy', 'Number_of_Features', 'Number_of_Folds'
            ],
            'Value': [
                np.mean([r['accuracy'] for r in fold_results]),
                np.std([r['accuracy'] for r in fold_results]),
                np.max([r['accuracy'] for r in fold_results]),
                test_results['accuracy'],
                len(self.selected_features),
                len(fold_results)
            ]
        }

        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('../../results/tabnet_diagnosis_results.csv', index=False)

        # Save selected features and their correlations
        feature_details = self.correlation_results[
            self.correlation_results['Feature'].isin(self.selected_features)
        ].copy()
        feature_details.to_csv('../../output/selected_features_correlation.csv', index=False)

        print("✓ Results saved:")
        print("  - tabnet_feature_importance.csv")
        print("  - tabnet_diagnosis_results.csv")
        print("  - selected_features_correlation.csv")


def main():
    """Main execution function"""
    # Initialize predictor
    predictor = DiagnosisPredictor('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run complete experiment
    results = predictor.run_complete_experiment()

    if results:
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final Test Accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"Selected Features: {len(results['selected_features'])}")
        print(f"Best performing features:")
        for i, (_, row) in enumerate(results['feature_importance'].head(5).iterrows(), 1):
            print(f"  {i}. {row['Feature']}: {row['Importance']:.4f}")
    else:
        print("Experiment failed!")


if __name__ == "__main__":
    main()