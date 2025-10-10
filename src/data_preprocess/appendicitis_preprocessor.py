import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class AppendicitisPreprocessor:
    def __init__(self, data_path):
        """Initialize data preprocessor"""
        self.data_path = data_path
        self.original_data = None
        self.processed_data = None
        self.processing_log = []
        self.label_encoders = {}

    def log_step(self, step_name, message):
        """Log processing steps"""
        log_entry = f"{step_name}: {message}"
        self.processing_log.append(log_entry)
        print(f"✓ {log_entry}")

    def load_data(self):
        """Load original data"""
        print("=" * 80)
        print("LOADING APPENDICITIS DATA")
        print("=" * 80)
        try:
            self.original_data = pd.read_excel(self.data_path)
            self.processed_data = self.original_data.copy()
            print(f"Data loaded successfully!")
            print(f"Original data shape: {self.original_data.shape}")
            print(f"Column names: {list(self.original_data.columns)}")
            self.log_step("Data Loading", f"Successfully loaded {self.original_data.shape[0]} rows {self.original_data.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

    def display_initial_info(self):
        """Display initial data information"""
        print(f"\nInitial data analysis:")
        print("-" * 50)
        # Missing data statistics
        missing_info = []
        for col in self.processed_data.columns:
            missing_count = self.processed_data[col].isnull().sum()
            missing_pct = (missing_count / len(self.processed_data)) * 100
            if missing_count > 0:
                missing_info.append((col, missing_count, missing_pct))
        print(f"Missing data overview:")
        for col, count, pct in missing_info[:10]:  # Show first 10columns with missing data
            print(f"  {col}: {count} ({pct:.1f}%)")
        if len(missing_info) > 10:
            print(f"... and {len(missing_info) - 10} more columns with missing data")
        # Data type statistics
        print(f"\nData type distribution:")
        dtype_counts = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")

    def step1_remove_empty_age_rows(self):
        """Step 1: Remove rows with empty Age column"""
        print(f"\nStep 1: Remove rows with empty Age column")
        print("-" * 40)

        initial_rows = len(self.processed_data)

        if 'Age' in self.processed_data.columns:
            age_missing = self.processed_data['Age'].isnull().sum()
            print(f"Age column missing values: {age_missing}")

            if age_missing > 0:
                self.processed_data = self.processed_data.dropna(subset=['Age'])
                final_rows = len(self.processed_data)
                removed_rows = initial_rows - final_rows
                self.log_step("Remove Empty Age Rows", f"Removed {removed_rows} rows, remaining {final_rows} rows")
            else:
                self.log_step("Remove Empty Age Rows", "No missing values in Age column")
        else:
            print("Age column not found")
            self.log_step("Remove Empty Age Rows", "Age column not found")

    def step2_fill_diagnosis_missing(self):
        """Step 2: Fill Diagnosis column missing values with appendicitis"""
        print(f"\nStep 2: Fill Diagnosis column missing values")
        print("-" * 40)

        if 'Diagnosis' in self.processed_data.columns:
            missing_count = self.processed_data['Diagnosis'].isnull().sum()
            print(f"Diagnosis missing values: {missing_count}")

            if missing_count > 0:
                self.processed_data['Diagnosis'].fillna('appendicitis', inplace=True)
                self.log_step("Diagnosis Filling", f"Filled {missing_count} missing values with 'appendicitis'")
            else:
                self.log_step("Diagnosis Filling", "No missing values to fill")
        else:
            print("Diagnosis column not found")
            self.log_step("Diagnosis Filling", "Diagnosis column not found")

    def step3_remove_high_missing_columns(self, threshold=70):
        """Step 3: Remove columns with missing rate greater than 70%"""
        print(f"\nStep 3: Remove columns with missing rate greater than {threshold}%")
        print("-" * 40)

        missing_pct = (self.processed_data.isnull().sum() / len(self.processed_data)) * 100
        high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()

        if high_missing_cols:
            print("High missing rate columns:")
            for col in high_missing_cols:
                print(f"  - {col}: {missing_pct[col]:.1f}%")

            self.processed_data = self.processed_data.drop(columns=high_missing_cols)
            self.log_step("Remove High Missing Columns", f"Removed {len(high_missing_cols)} columns: {high_missing_cols}")
        else:
            self.log_step("Remove High Missing Columns", f"No columns with missing rate greater than {threshold}%")

    def step4_fill_bmi_mean(self):
        """Step 4: Fill BMI column with mean values"""
        print(f"\nStep 4: Fill BMI column with mean values")
        print("-" * 40)

        if 'BMI' in self.processed_data.columns:
            missing_count = self.processed_data['BMI'].isnull().sum()
            if missing_count > 0:
                mean_value = self.processed_data['BMI'].mean()
                self.processed_data['BMI'].fillna(mean_value, inplace=True)
                self.log_step("BMI Mean Filling", f"Filled {missing_count} missing values with mean {mean_value:.2f}")
            else:
                self.log_step("BMI Mean Filling", "No missing values to fill")
        else:
            self.log_step("BMI Mean Filling", "BMI column not found")

    def step5_knn_fill_height_weight(self):
        """Step 5: Fill Height and Weight using KNN with Age dimension"""
        print(f"\nStep 5: KNN filling for Height and Weight")
        print("-" * 40)

        # Find Height and Weight columns
        height_cols = [col for col in self.processed_data.columns if 'height' in col.lower()]
        weight_cols = [col for col in self.processed_data.columns if 'weight' in col.lower()]

        print(f"Found Height columns: {height_cols}")
        print(f"Found Weight columns: {weight_cols}")

        if 'Age' not in self.processed_data.columns:
            self.log_step("KNN Filling", "Age column not found, cannot perform KNN filling")
            return

        for col_type, cols in [('Height', height_cols), ('Weight', weight_cols)]:
            for col in cols:
                if col in self.processed_data.columns:
                    missing_count = self.processed_data[col].isnull().sum()
                    if missing_count > 0:
                        print(f"Processing {col} column, missing values: {missing_count}")

                        # Prepare KNN data
                        knn_data = self.processed_data[['Age', col]].copy()

                        # Use KNN filling
                        imputer = KNNImputer(n_neighbors=5)
                        filled_data = imputer.fit_transform(knn_data)

                        # Update data
                        self.processed_data[col] = filled_data[:, 1]
                        self.log_step(f"{col_type} KNN Filling", f"Filled {missing_count} missing values in {col} column using KNN")
                    else:
                        self.log_step(f"{col_type} KNN Filling", f"No missing values in {col} column")

    def step6_remove_length_of_stay(self):
        """Step 6: Remove Length_of_Stay columns"""
        print(f"\nStep 6: Remove Length_of_Stay related columns")
        print("-" * 40)

        # Find hospital stay related columns
        stay_keywords = ['length', 'stay', 'hospital', 'los', 'duration','diagnosis_presumptive']
        stay_cols = []

        for col in self.processed_data.columns:
            if any(keyword in col.lower() for keyword in stay_keywords):
                stay_cols.append(col)

        if stay_cols:
            print(f"Found hospital stay related columns: {stay_cols}")
            self.processed_data = self.processed_data.drop(columns=stay_cols)
            self.log_step("Remove Hospital Stay Columns", f"Removed {len(stay_cols)} columns: {stay_cols}")
        else:
            self.log_step("Remove Hospital Stay Columns", "No hospital stay related columns found")

    def identify_column_types(self):
        """Identify column data types"""
        print(f"\nData type identification:")
        print("-" * 30)

        categorical_cols = []
        numerical_cols = []
        date_cols = []

        # Special handling columns
        special_cols = ['US_Number']

        for col in self.processed_data.columns:
            if col in special_cols:
                continue

            # Check if date type
            if any(keyword in col.lower() for keyword in ['date', 'time']) or 'datetime' in str(
                    self.processed_data[col].dtype):
                date_cols.append(col)
            # Check if categorical type
            elif (self.processed_data[col].dtype == 'object' or
                  (self.processed_data[col].dtype in ['int64', 'float64'] and
                   self.processed_data[col].nunique() <= 10 and
                   self.processed_data[col].nunique() < len(self.processed_data) * 0.05)):
                categorical_cols.append(col)
            # Numerical type
            elif self.processed_data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                # Default to categorical
                categorical_cols.append(col)

        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        print(f"Date columns ({len(date_cols)}): {date_cols}")

        return categorical_cols, numerical_cols, date_cols

    def step7_fill_categorical_missing(self, categorical_cols):
        """Step 7: Fill categorical data with mode values grouped by same Diagnosis"""
        print(f"\nStep 7: Fill categorical data missing values")
        print("-" * 40)

        if'Diagnosis' not in self.processed_data.columns:
            self.log_step("Categorical Data Filling", "Diagnosis column not found")
            return

        total_filled = 0

        for col in categorical_cols:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                print(f"Processing categorical column {col}, missing values: {missing_count}")

                # Fill by Diagnosis groups
                for diagnosis in self.processed_data['Diagnosis'].unique():
                    if pd.isna(diagnosis):
                        continue

                    # Find rows in this diagnosis group with missing values in this column
                    mask = (self.processed_data['Diagnosis'] == diagnosis) & (self.processed_data[col].isnull())

                    if mask.sum() > 0:
                        # Calculate mode for this diagnosis group
                        group_data = self.processed_data[
                            (self.processed_data['Diagnosis'] == diagnosis) &
                            (self.processed_data[col].notnull())
                            ][col]

                        if len(group_data) > 0:
                            group_mode = group_data.mode()
                            if len(group_mode) > 0:
                                self.processed_data.loc[mask, col] = group_mode.iloc[0]
                                total_filled += mask.sum()

                # If still missing values, fill with global mode
                remaining_missing = self.processed_data[col].isnull().sum()
                if remaining_missing > 0:
                    global_mode = self.processed_data[col].mode()
                    if len(global_mode) > 0:
                        self.processed_data[col].fillna(global_mode.iloc[0], inplace=True)
                        total_filled += remaining_missing

        self.log_step("Categorical Data Filling", f"Filled {total_filled} categorical missing values grouped by Diagnosis")

    def step8_fill_numerical_missing(self, numerical_cols):
        """Step 8: Fill numerical data with mean values grouped by same Diagnosis"""
        print(f"\nStep 8: Fill numerical data missing values")
        print("-" * 40)

        if 'Diagnosis' not in self.processed_data.columns:
            self.log_step("Numerical Data Filling", "Diagnosis column not found")
            return

        total_filled = 0

        for col in numerical_cols:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                print(f"Processing numerical column {col}, missing values: {missing_count}")

                # Fill by Diagnosis groups
                for diagnosis in self.processed_data['Diagnosis'].unique():
                    if pd.isna(diagnosis):
                        continue

                    mask = (self.processed_data['Diagnosis'] == diagnosis) & (self.processed_data[col].isnull())

                    if mask.sum() > 0:
                        # Calculate mean for this diagnosis group
                        group_data = self.processed_data[
                            (self.processed_data['Diagnosis'] == diagnosis) &
                            (self.processed_data[col].notnull())
                            ][col]

                        if len(group_data) > 0:
                            group_mean = group_data.mean()
                            if not pd.isna(group_mean):
                                self.processed_data.loc[mask, col] = group_mean
                                total_filled += mask.sum()

                # If still missing values, fill with global mean
                remaining_missing = self.processed_data[col].isnull().sum()
                if remaining_missing > 0:
                    global_mean = self.processed_data[col].mean()
                    if not pd.isna(global_mean):
                        self.processed_data[col].fillna(global_mean, inplace=True)
                        total_filled += remaining_missing

        self.log_step("Numerical Data Filling", f"Filled {total_filled} numerical missing values grouped by Diagnosis")

    def step9_label_encode_categorical(self, categorical_cols):
        """Step 9: Label encode categorical data"""
        print(f"\nStep 9: Label encode categorical data")
        print("-" * 40)

        # Exclude target variables and special columns
        target_cols = ['US_Number']
        encode_cols = [col for col in categorical_cols if col not in target_cols]

        if encode_cols:
            print(f"Columns to encode: {encode_cols}")

            encoded_count = 0
            for col in encode_cols:
                if self.processed_data[col].dtype == 'object':
                    le = LabelEncoder()

                    # Handle missing values
                    non_null_data = self.processed_data[col].dropna()
                    if len(non_null_data) > 0:
                        # Encode non-null values
                        self.processed_data[col] = self.processed_data[col].astype(str)

                        # Convert NaN to string then encode
                        self.processed_data[col] = self.processed_data[col].replace('nan', np.nan)

                        # Fill NaN with special value
                        self.processed_data[col].fillna('MISSING', inplace=True)

                        # Encode
                        self.processed_data[col] = le.fit_transform(self.processed_data[col])

                        # Save encoder
                        self.label_encoders[col] = le
                        encoded_count += 1
                        print(f"  Encoded {col}: {len(le.classes_)} categories")

            self.log_step("Label Encoding", f"Encoded {encoded_count} categorical columns")
        else:
            self.log_step("Label Encoding", "No categorical columns to encode")

    def generate_comprehensive_report(self):
        """Generate comprehensive processing report"""
        print(f"\n" + "=" * 80)
        print("DATA PREPROCESSING COMPREHENSIVE REPORT")
        print("=" * 80)

        # Basic statistics comparison
        original_shape = self.original_data.shape
        final_shape = self.processed_data.shape

        print(f"\nData shape changes:")
        print("-" * 30)
        print(f"Original data: {original_shape[0]} rows × {original_shape[1]} columns")
        print(f"Processed:{final_shape[0]} rows × {final_shape[1]} columns")
        print(f"Row change:    {final_shape[0] - original_shape[0]:+d}")
        print(f"Column change: {final_shape[1] - original_shape[1]:+d}")

        # Missing values statistics
        original_missing = self.original_data.isnull().sum().sum()
        final_missing = self.processed_data.isnull().sum().sum()

        print(f"\nMissing values changes:")
        print("-" * 20)
        print(f"Original missing values: {original_missing}")
        print(f"Final missing values: {final_missing}")
        print(f"Cleaned missing values: {original_missing - final_missing}")

        # Data type distribution
        print(f"\nFinal data type distribution:")
        print("-" * 25)
        dtype_counts = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{str(dtype):<12}: {count:>3} columns")

        # Target variable distribution
        target_cols = ['Diagnosis', 'Severity', 'Management']
        for col in target_cols:
            if col in self.processed_data.columns:
                print(f"\n{col} distribution:")
                print("-" * 15)
                value_counts = self.processed_data[col].value_counts()
                for value, count in value_counts.head(10).items():  # Show only top 10
                    pct = (count / len(self.processed_data)) * 100
                    print(f"  {str(value):<15}: {count:>4} ({pct:>5.1f}%)")

                if len(value_counts) > 10:
                    print(f"  ... and {len(value_counts) - 10} more values")

        # Processing steps summary
        print(f"\nProcessing steps summary:")
        print("-" * 30)
        for i, step in enumerate(self.processing_log, 1):
            print(f"{i:2d}. {step}")

        # Encoding information
        if self.label_encoders:
            print(f"\nLabel encoding information:")
            print("-" * 20)
            for col, encoder in self.label_encoders.items():
                print(f"{col}: {len(encoder.classes_)} categories")
                if len(encoder.classes_) <= 10:
                    class_mapping = {cls: i for i, cls in enumerate(encoder.classes_)}
                    print(f"  Mapping: {class_mapping}")

        # Data quality check
        remaining_missing = self.processed_data.isnull().sum().sum()
        print(f"\nData quality check:")
        print("-" * 20)
        if remaining_missing == 0:
            print("✓ All missing values have been processed")
        else:
            print(f"⚠ Still have {remaining_missing} missing values")
            missing_cols = self.processed_data.isnull().sum()
            for col in missing_cols[missing_cols > 0].index:
                print(f"  - {col}: {missing_cols[col]} values")

        print("✓ Data preprocessing completed")

    def save_processed_data(self, output_path='../../data/appendicitis/processed_appendicitis_data_final.xlsx'):
        """Save processed data"""
        try:
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Save main data
                self.processed_data.to_excel(writer, sheet_name='Processed_Data', index=False)

                # Save processing report
                report_df = pd.DataFrame({
                    'Step': range(1, len(self.processing_log) + 1),
                    'Processing_Action': self.processing_log
                })
                report_df.to_excel(writer, sheet_name='Processing_Log', index=False)

                # Save data comparison
                comparison_data = {
                    'Metric': ['Rows', 'Columns', 'Missing_Values', 'Total_Cells'],
                    'Original': [
                        self.original_data.shape[0],
                        self.original_data.shape[1],
                        self.original_data.isnull().sum().sum(),
                        self.original_data.shape[0] * self.original_data.shape[1]
                    ],
                    'Processed': [
                        self.processed_data.shape[0],
                        self.processed_data.shape[1],
                        self.processed_data.isnull().sum().sum(),
                        self.processed_data.shape[0] * self.processed_data.shape[1]
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df['Change'] = comparison_df['Processed'] - comparison_df['Original']
                comparison_df.to_excel(writer, sheet_name='Data_Comparison', index=False)

                # Save Label encoding mapping
                if self.label_encoders:
                    encoding_info = []
                    for col, encoder in self.label_encoders.items():
                        for i, cls in enumerate(encoder.classes_):
                            encoding_info.append({
                                'Column': col,
                                'Original_Value': cls,
                                'Encoded_Value': i
                            })

                    if encoding_info:
                        encoding_df = pd.DataFrame(encoding_info)
                        encoding_df.to_excel(writer, sheet_name='Label_Encoding_Map', index=False)

            print(f"\n✓ Processed data has been saved to: {output_path}")
            print("  Contains worksheets:")
            print("    - Processed_Data: Processed data")
            print("    - Processing_Log: Processing steps log")
            print("    - Data_Comparison: Data comparison")
            if self.label_encoders:
                print("    - Label_Encoding_Map: Label encoding mapping table")

            return True

        except Exception as e:
            print(f"\n✗ Save failed: {e}")
            return False

    def run_preprocessing(self):
        """Execute complete preprocessing workflow"""
        #1. Load data
        if not self.load_data():
            return False

        # 2. Display initial information
        self.display_initial_info()

        # 3. Execute preprocessing steps
        self.step1_remove_empty_age_rows()
        self.step2_fill_diagnosis_missing()
        self.step3_remove_high_missing_columns()
        # US_Number column remains unchanged (skip processing)
        self.step4_fill_bmi_mean()
        self.step5_knn_fill_height_weight()
        self.step6_remove_length_of_stay()

        # 4. Identify data types
        categorical_cols, numerical_cols, date_cols = self.identify_column_types()

        # 5. Fill missing values
        self.step7_fill_categorical_missing(categorical_cols)
        self.step8_fill_numerical_missing(numerical_cols)
        # Date type data keep original values (no processing)

        # 6. Label encoding
        self.step9_label_encode_categorical(categorical_cols)

        # 7. Generate report
        self.generate_comprehensive_report()

        # 8. Save data
        success = self.save_processed_data()

        return success


def main():
    """Main function"""
    print("Starting Appendicitis data preprocessing...")

    # Create preprocessor
    preprocessor = AppendicitisPreprocessor('../../data/appendicitis/app_data.xlsx')

    # Execute preprocessing
    success = preprocessor.run_preprocessing()

    if success:
        print(f"\n" + "=" * 80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Output file: processed_appendicitis_data_final.xlsx")
        print("\nMain processing contents:")
        print("1. Remove rows with empty Age")
        print("2. Fill Diagnosis missing values with'appendicitis'")
        print("3. Remove columns with missing rate >70%")
        print("4. Keep US_Number column unchanged")
        print("5. Fill BMI with mean values")
        print("6. Fill Height/Weight using KNN with Age")
        print("7. Remove hospital stay related columns")
        print("8. Fill categorical data with mode grouped by Diagnosis")
        print("9. Fill numerical data with mean grouped by Diagnosis")
        print("10. Label encode categorical data")
        print("11. Keep date data with original values")
    else:
        print("\nPreprocessing failed!")


if __name__ == "__main__":
    main()