import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def analyze_correlation_with_diagnosis(file_path):
    """
    Analyze correlation between each column and Diagnosis in Excel file
    """
    try:
        # Read Excel file
        print("Reading Excel file...")
        df = pd.read_excel(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # Check if Diagnosis column exists
        if 'Diagnosis' not in df.columns:
            # Try to find similar column names
            diagnosis_cols = [col for col in df.columns if 'diagnosis' in col.lower()]
            if diagnosis_cols:
                print(f"'Diagnosis' column not found, but found similar columns: {diagnosis_cols}")
                print("Please confirm the target column name")
                return None
            else:
                print("No Diagnosis-related column found. All column names for reference:")
                for i, col in enumerate(df.columns):
                    print(f"{i}: {col}")
                return None

        # Get target variable
        target = df['Diagnosis']
        features = df.drop('Diagnosis', axis=1)

        print(f"\nDiagnosis column value distribution:")
        print(target.value_counts())

        # Store correlation results
        correlation_results = []

        print("\nStarting correlation analysis for each column with Diagnosis...")

        for col in features.columns:
            print(f"\nAnalyzing column: {col}")

            # Skip completely empty columns
            if features[col].isnull().all():
                print(f"  {col}: All values are null, skipping")
                continue

            # Get non-null data
            col_data = features[col].dropna()
            target_aligned = target[col_data.index]

            if len(col_data) == 0:
                print(f"  {col}: No valid data, skipping")
                continue

            # Show basic data information
            print(f"  Data type: {col_data.dtype}")
            print(f"  Non-null count: {len(col_data)}")
            print(f"  Unique values: {col_data.nunique()}")
            print(f"  Sample values: {list(col_data.head())}")

            result = {
                'feature': col,
                'data_type': str(col_data.dtype),
                'non_null_count': len(col_data),
                'unique_values': col_data.nunique()
            }

            # Try to convert data to numeric
            is_numeric = False
            numeric_col_data = None

            try:
                # Try direct conversion first
                if pd.api.types.is_numeric_dtype(col_data):
                    numeric_col_data = col_data.astype(float)
                    is_numeric = True
                else:
                    # Try converting strings to numeric
                    numeric_col_data = pd.to_numeric(col_data, errors='coerce')
                    # If enough data can be converted to numeric, treat as numeric
                    if numeric_col_data.notna().sum() > len(col_data) * 0.5:  # At least 50% convertible
                        is_numeric = True
                        # Realign target variable
                        valid_indices = numeric_col_data.notna()
                        numeric_col_data = numeric_col_data[valid_indices]
                        target_aligned = target_aligned[valid_indices]
                        print(f"  Successfully converted to numeric, valid data: {len(numeric_col_data)}")
                    else:
                        print(f"  Cannot convert to numeric (only {numeric_col_data.notna().sum()} valid numbers)")
            except Exception as e:
                print(f"  Numeric conversion failed: {e}")
                is_numeric = False

            if is_numeric and len(numeric_col_data) > 1:
                # Numeric data - use Pearson and Spearman correlation
                try:
                    # Ensure target variable is also numeric
                    if not pd.api.types.is_numeric_dtype(target_aligned):
                        # Try encoding target variable as numeric
                        le = LabelEncoder()
                        target_numeric = le.fit_transform(target_aligned)
                        print(f"  Target variable encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
                    else:
                        target_numeric = target_aligned.astype(float)

                    pearson_corr, pearson_p = pearsonr(numeric_col_data, target_numeric)
                    spearman_corr, spearman_p = spearmanr(numeric_col_data, target_numeric)

                    result.update({
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'method': 'numeric'
                    })

                    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
                    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

                except Exception as e:
                    print(f"  Numeric correlation calculation error: {e}")
                    result.update({
                        'pearson_correlation': np.nan,
                        'pearson_p_value': np.nan,
                        'spearman_correlation': np.nan,
                        'spearman_p_value': np.nan,
                        'method': 'numeric_error'
                    })

            else:
                # Categorical data - use Chi-square test
                try:
                    # Ensure data is string type for crosstab analysis
                    categorical_col_data = col_data.astype(str)
                    categorical_target = target_aligned.astype(str)

                    # Create contingency table
                    contingency_table = pd.crosstab(categorical_col_data, categorical_target)
                    print(f"  Contingency table shape: {contingency_table.shape}")

                    # Check if contingency table is valid (at least 2x2)
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                        # Calculate Cramér's V (standardized chi-square statistic)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                        result.update({
                            'chi2_statistic': chi2,
                            'chi2_p_value': p_value,
                            'cramers_v': cramers_v,
                            'method': 'categorical'
                        })

                        print(f"  Chi-square statistic: {chi2:.4f} (p={p_value:.4f})")
                        print(f"  Cramér's V: {cramers_v:.4f}")
                    else:
                        print(f"  Contingency table too small for chi-square test")
                        result.update({
                            'chi2_statistic': np.nan,
                            'chi2_p_value': np.nan,
                            'cramers_v': np.nan,
                            'method': 'categorical_insufficient'
                        })

                except Exception as e:
                    print(f"  Categorical correlation calculation error: {e}")
                    result.update({
                        'chi2_statistic': np.nan,
                        'chi2_p_value': np.nan,
                        'cramers_v': np.nan,
                        'method': 'categorical_error'
                    })

            correlation_results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(correlation_results)

        # Display summary results
        print("\n" + "=" * 60)
        print("Correlation Analysis Summary Results")
        print("=" * 60)

        # Sort numeric features by correlation
        numeric_results = results_df[results_df['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            print("\nNumeric features correlation with Diagnosis (sorted by absolute Pearson correlation):")
            numeric_results['abs_pearson'] = numeric_results['pearson_correlation'].abs()
            numeric_sorted = numeric_results.sort_values('abs_pearson', ascending=False)

            for _, row in numeric_sorted.iterrows():
                print(f"  {row['feature']:<30} | Pearson: {row['pearson_correlation']:8.4f} | "
                      f"Spearman: {row['spearman_correlation']:8.4f} | "
                      f"P-value: {row['pearson_p_value']:8.4f}")

        # Sort categorical features by correlation
        categorical_results = results_df[results_df['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            print("\nCategorical features correlation with Diagnosis (sorted by Cramér's V):")
            categorical_sorted = categorical_results.sort_values('cramers_v', ascending=False)

            for _, row in categorical_sorted.iterrows():
                print(f"  {row['feature']:<30} | Cramér's V: {row['cramers_v']:8.4f} | "
                      f"Chi2 P-value: {row['chi2_p_value']:8.4f}")

        # Save detailed results to CSV
        results_df.to_csv('appendicitis_correlation_results.csv', index=False, encoding='utf-8')
        print(f"\nDetailed results saved to: appendicitis_correlation_results.csv")

        # Create visualizations
        create_visualizations(df, results_df)

        return results_df, df

    except Exception as e:
        print(f"Error during analysis: {e}")
        return None, None


def create_visualizations(df, results_df):
    """
    Create visualizations for correlation analysis results
    """
    try:
        # Set font for better readability
        plt.rcParams['font.size'] = 9
        plt.rcParams['figure.autolayout'] = True

        # Create charts with larger figure size
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Appendicitis Data Correlation Analysis Results', fontsize=16, fontweight='bold')

        # 1. Numeric features correlation bar chart
        numeric_results = results_df[results_df['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            numeric_results = numeric_results.dropna(subset=['pearson_correlation'])
            if not numeric_results.empty:
                ax1 = axes[0, 0]
                # Sort by absolute correlation for better visualization
                numeric_results = numeric_results.reindex(
                    numeric_results['pearson_correlation'].abs().sort_values(ascending=True).index
                )
                y_pos = np.arange(len(numeric_results))
                bars = ax1.barh(y_pos, numeric_results['pearson_correlation'])
                ax1.set_yticks(y_pos)
                # Truncate long feature names to prevent overlap
                feature_labels = [feat[:12] + '...' if len(feat) > 12 else feat
                                  for feat in numeric_results['feature']]
                ax1.set_yticklabels(feature_labels, fontsize=7)
                ax1.set_xlabel('Pearson Correlation Coefficient')
                ax1.set_title('Numeric Features Correlation with Diagnosis', pad=20)
                ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

                # Add color mapping for bars
                for i, bar in enumerate(bars):
                    corr_val = numeric_results.iloc[i]['pearson_correlation']
                    if abs(corr_val) > 0.5:
                        bar.set_color('red')
                    elif abs(corr_val) > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightblue')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Numeric Features', ha='center', va='center',
                                transform=axes[0, 0].transAxes, fontsize=12)
                axes[0, 0].set_title('Numeric Features Correlation with Diagnosis')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Numeric Features', ha='center', va='center',
                            transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_title('Numeric Features Correlation with Diagnosis')

        # 2. Categorical features Cramér's V bar chart
        categorical_results = results_df[results_df['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            categorical_results = categorical_results.dropna(subset=['cramers_v'])
            if not categorical_results.empty:
                ax2 = axes[0, 1]
                # Sort by Cramér's V for better visualization
                categorical_results = categorical_results.sort_values('cramers_v', ascending=True)
                y_pos = np.arange(len(categorical_results))
                bars = ax2.barh(y_pos, categorical_results['cramers_v'])
                ax2.set_yticks(y_pos)
                # Truncate long feature names to prevent overlap
                feature_labels = [feat[:12] + '...' if len(feat) > 12 else feat
                                  for feat in categorical_results['feature']]
                ax2.set_yticklabels(feature_labels, fontsize=7)
                ax2.set_xlabel("Cramér's V")
                ax2.set_title('Categorical Features Correlation with Diagnosis', pad=20)

                # Add color mapping for bars
                for i, bar in enumerate(bars):
                    v_val = categorical_results.iloc[i]['cramers_v']
                    if v_val > 0.5:
                        bar.set_color('red')
                    elif v_val > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightblue')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Categorical Features', ha='center', va='center',
                                transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Categorical Features Correlation with Diagnosis')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Categorical Features', ha='center', va='center',
                            transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Categorical Features Correlation with Diagnosis')

        # 3. Diagnosis distribution pie chart
        ax3 = axes[1, 0]
        diagnosis_counts = df['Diagnosis'].value_counts()
        wedges, texts, autotexts = ax3.pie(diagnosis_counts.values, labels=diagnosis_counts.index,
                                           autopct='%1.1f%%', startangle=90)
        ax3.set_title('Diagnosis Distribution', pad=20)

        # 4. Data quality overview
        ax4 = axes[1, 1]
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage')

        if not missing_df.empty and len(missing_df) <= 20:  # Limit to 20 columns for readability
            y_pos = np.arange(len(missing_df))
            bars = ax4.barh(y_pos, missing_df['Missing_Percentage'])
            ax4.set_yticks(y_pos)
            # Truncate long column names to prevent overlap
            column_labels = [col[:12] + '...' if len(col) > 12 else col
                             for col in missing_df.index]
            ax4.set_yticklabels(column_labels, fontsize=7)
            ax4.set_xlabel('Missing Value Percentage (%)')
            ax4.set_title('Missing Values by Column', pad=20)
        elif len(missing_df) > 20:
            # Show only top 20 columns with most missing values
            missing_df_top = missing_df.tail(20)
            y_pos = np.arange(len(missing_df_top))
            bars = ax4.barh(y_pos, missing_df_top['Missing_Percentage'])
            ax4.set_yticks(y_pos)
            column_labels = [col[:12] + '...' if len(col) > 12 else col
                             for col in missing_df_top.index]
            ax4.set_yticklabels(column_labels, fontsize=7)
            ax4.set_xlabel('Missing Value Percentage (%)')
            ax4.set_title(f'Top 20 Columns with Missing Values (of {len(missing_df)} total)', pad=20)
        else:
            ax4.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Data Quality: Complete')

        plt.tight_layout(pad=3.0)
        plt.savefig('appendicitis_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Visualization chart saved as: appendicitis_correlation_analysis.png")

    except Exception as e:
        print(f"Error creating visualization charts: {e}")


if __name__ == "__main__":
    # Analyze appendicitis data
    file_path = "data/appendicitis/app_data.xlsx"
    results, data = analyze_correlation_with_diagnosis(file_path)

    if results is not None:
        print("\nAnalysis completed!")
        print(f"Analyzed {len(results)} features for correlation with Diagnosis")

        # Show features with strongest correlations
        print("\nStrongest Correlation Features Summary:")

        # Numeric features
        numeric_results = results[results['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            numeric_results = numeric_results.dropna(subset=['pearson_correlation'])
            if not numeric_results.empty:
                top_numeric = numeric_results.loc[numeric_results['pearson_correlation'].abs().idxmax()]
                print(f"Strongest numeric correlation: {top_numeric['feature']} "
                      f"(Pearson correlation: {top_numeric['pearson_correlation']:.4f})")

        # Categorical features
        categorical_results = results[results['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            categorical_results = categorical_results.dropna(subset=['cramers_v'])
            if not categorical_results.empty:
                top_categorical = categorical_results.loc[categorical_results['cramers_v'].idxmax()]
                print(f"Strongest categorical correlation: {top_categorical['feature']} "
                      f"(Cramér's V: {top_categorical['cramers_v']:.4f})")
    else:
        print("Analysis failed, please check file path and data format")