print("\n--- Running Preprocessing Script ---\n")
# Imports
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

## Load Dataset (Appendicitis Dataset from UCI repository)
df = pd.read_excel("app_data.xlsx", sheet_name="All cases")

# Check the dataset shape and columns
print("Dataset shape:", df.shape)
print("Columns:", df.columns)   
# Display first few rows of the dataset
print("First few rows:", df.head())
print("\nDataFrame Info:")
print(df.info())

## Data Cleaning

# Check for missing values - summary
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing %': missing_percentage
})
missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
missing_summary = missing_summary.sort_values(by='Missing %', ascending=False)

print("\nMissing Values Summary:")
print(missing_summary)

# Drop columns based on high missingness or weak clinical justification
columns_to_drop = [
    'Abscess_Location', 'Gynecological_Findings', 'Conglomerate_of_Bowel_Loops',
    'Segmented_Neutrophils', 'Ileus', 'Perfusion', 'Enteritis', 'Appendicolith',
    'Coprostasis', 'Perforation', 'Appendicular_Abscess', 'Bowel_Wall_Thickening',
    'Lymph_Nodes_Location', 'Target_Sign', 'Meteorism', 'Pathological_Lymph_Nodes',
    'Appendix_Wall_Layers', 'Surrounding_Tissue_Reaction',
    'RBC_in_Urine', 'Ketones_in_Urine', 'WBC_in_Urine', 'Diagnosis_Presumptive'
]
df.drop(columns=columns_to_drop, inplace=True)

print(f"\nDropped {len(columns_to_drop)} columns with high missingness or poor justification.")
print("New dataset shape:", df.shape)

# Handle the one patient in the Stool column with 'constipation + diarrhea'
df['Stool'] = df['Stool'].replace('constipation, diarrhea', 'constipation')

## Impute remaining missing values

# Median imputation for continuous variables
median_impute_cols = [
    'Appendix_Diameter', 'Neutrophil_Percentage', 'BMI', 'Height', 'RDW',
    'US_Number', 'Thrombocyte_Count', 'Hemoglobin', 'RBC_Count', 'CRP',
    'Body_Temperature', 'WBC_Count', 'Weight', 'Length_of_Stay', 'Age'
]
for col in median_impute_cols:
    df[col] = df[col].fillna(df[col].median())

# Mode imputation for binary/categorical variables
mode_impute_cols = [
    'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Neutrophilia', 'Ipsilateral_Rebound_Tenderness',
    'Free_Fluids', 'Psoas_Sign', 'Dysuria', 'Stool', 'Coughing_Pain',
    'Contralateral_Rebound_Tenderness', 'Loss_of_Appetite', 'Peritonitis', 'Migratory_Pain',
    'Nausea', 'Lower_Right_Abd_Pain', 'Appendix_on_US', 'US_Performed',
    'Diagnosis', 'Sex', 'Severity', 'Management'
]
for col in mode_impute_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing value imputation completed.")
print("Remaining missing values:", df.isnull().sum().sum())

# List of numerical columns after preprocessing
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numeric columns to consider for scaling:", numeric_cols)

## Feature Scaling and Encoding
# Columns to scale (excluding 'US_Number')
scale_cols = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay',
    'Alvarado_Score', 'Paedriatic_Appendicitis_Score',
    'Appendix_Diameter', 'Body_Temperature', 'WBC_Count',
    'Neutrophil_Percentage', 'RBC_Count', 'Hemoglobin',
    'RDW', 'Thrombocyte_Count', 'CRP'
]

# Standardize numeric columns (zero mean, unit variance)
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("\nStandardization complete.")

# Drop the row with Management = 'simultaneous appendectomy' (rare outlier of 1 case)
if 'simultaneous appendectomy' in df['Management'].values:
    df = df[df['Management'] != 'simultaneous appendectomy']

# One-hot encoding for multi-class categorical variables
df = pd.get_dummies(df, columns=[
    'Management', 'Peritonitis', 'Stool'
], dtype=int, drop_first=True)

# Label encoding for binary categorical variables
binary_cols = [
    'Neutrophilia', 'Ipsilateral_Rebound_Tenderness', 'Free_Fluids',
    'Psoas_Sign', 'Dysuria', 'Coughing_Pain', 'Contralateral_Rebound_Tenderness',
    'Loss_of_Appetite', 'Migratory_Pain', 'Nausea', 'Lower_Right_Abd_Pain',
    'Appendix_on_US', 'US_Performed'
]

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Encode Sex and Diagnosis (binary)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Diagnosis'] = df['Diagnosis'].map({'appendicitis': 1, 'no appendicitis': 0})
df['Severity'] = df['Severity'].map({'uncomplicated': 0, 'complicated': 1, 'no appendicitis': -1})

print("\nCategorical encoding complete.")
print("Final dataset shape:", df.shape)

# Show a sample of the transformed dataframe to verify encoding
print("\nSample of transformed DataFrame (first 10 rows):")
print(df.head(10))

# Show new column names to verify one-hot encoding
print("\nColumn names after encoding:")
print(df.columns.tolist())

print("Final missing values check:", df.isnull().sum().sum())

# Save the preprocessed DataFrame to a CSV file
df.to_csv("preprocessed_appendicitis.csv", index=False)

print("\nPreprocessed dataset saved as 'preprocessed_appendicitis.csv'")