import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import src.data_preprocess

warnings.filterwarnings('ignore')


def analyze_correlation_with_targets(file_path, target_columns=['Diagnosis', 'Severity', 'Management']):
    """
    Analyze correlation between each column and multiple target variables in Excel file
    """
    try:
        # Read Excel file
        print("Reading Excel file...")
        df = pd.read_excel(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        # Check if target columns exist
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_targets:
            print(f"Missing target columns: {missing_targets}")
            available_targets = [col for col in target_columns if col in df.columns]
            if not available_targets:
                print("No target columns found. Available columns:")
                for i, col in enumerate(df.columns):
                    print(f"{i}: {col}")
                return None, None
            else:
                print(f"Will analyze available targets: {available_targets}")
                target_columns = available_targets

        # Display target variable distributions
        for target_col in target_columns:
            print(f"\n{target_col} column value distribution:")
            print(df[target_col].value_counts())

        # Get features (exclude all target columns)
        features = df.drop(target_columns, axis=1)

        # Store all results
        all_results = {}

        # Analyze correlation for each target
        for target_col in target_columns:
            print(f"\n{'=' * 80}")
            print(f"ANALYZING CORRELATIONS WITH {target_col.upper()}")
            print(f"{'=' * 80}")

            target = df[target_col]
            correlation_results = []

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
                    'target': target_col,
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

            # Store results for this target
            all_results[target_col] = pd.DataFrame(correlation_results)

            # Display summary for this target
            print(f"\n{'-' * 60}")
            print(f"SUMMARY FOR {target_col.upper()}")
            print(f"{'-' * 60}")

            results_df = all_results[target_col]

            # Numeric features summary
            numeric_results = results_df[results_df['method'] == 'numeric'].copy()
            if not numeric_results.empty:
                print(f"\nNumeric features correlation with {target_col} (sorted by absolute Pearson correlation):")
                numeric_results['abs_pearson'] = numeric_results['pearson_correlation'].abs()
                numeric_sorted = numeric_results.sort_values('abs_pearson', ascending=False)

                for _, row in numeric_sorted.iterrows():
                    print(f"  {row['feature']:<30} | Pearson: {row['pearson_correlation']:8.4f} | "
                          f"Spearman: {row['spearman_correlation']:8.4f} | "
                          f"P-value: {row['pearson_p_value']:8.4f}")

            # Categorical features summary
            categorical_results = results_df[results_df['method'] == 'categorical'].copy()
            if not categorical_results.empty:
                print(f"\nCategorical features correlation with {target_col} (sorted by Cramér's V):")
                categorical_sorted = categorical_results.sort_values('cramers_v', ascending=False)

                for _, row in categorical_sorted.iterrows():
                    print(f"  {row['feature']:<30} | Cramér's V: {row['cramers_v']:8.4f} | "
                          f"Chi2 P-value: {row['chi2_p_value']:8.4f}")

        # Save all results
        for target_col in target_columns:
            filename = base_path+ f'/output/correlation_results_{target_col.lower()}.csv'
            all_results[target_col].to_csv(filename, index=False, encoding='utf-8')
            print(f"\nDetailed results for {target_col} saved to: {filename}")

        # Create visualizations for all targets
        create_multiple_visualizations(df, all_results, target_columns)

        return all_results, df

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_multiple_visualizations(df, all_results, target_columns):
    """
    Create separate visualizations for each target variable
    """
    try:
        # Set font for better readability
        plt.rcParams['font.size'] = 9
        plt.rcParams['figure.autolayout'] = True

        for target_col in target_columns:
            results_df = all_results[target_col]

            # Create charts for each target
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'Correlation Analysis Results with {target_col}', fontsize=16, fontweight='bold')

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
                    ax1.set_title(f'Numeric Features Correlation with {target_col}', pad=20)
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
                    axes[0, 0].set_title(f'Numeric Features Correlation with {target_col}')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Numeric Features', ha='center', va='center',
                                transform=axes[0, 0].transAxes, fontsize=12)
                axes[0, 0].set_title(f'Numeric Features Correlation with {target_col}')

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
                    ax2.set_title(f'Categorical Features Correlation with {target_col}', pad=20)

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
                    axes[0, 1].set_title(f'Categorical Features Correlation with {target_col}')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Categorical Features', ha='center', va='center',
                                transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title(f'Categorical Features Correlation with {target_col}')

            # 3. Target variable distribution pie chart
            ax3 = axes[1, 0]
            target_counts = df[target_col].value_counts()
            wedges, texts, autotexts = ax3.pie(target_counts.values, labels=target_counts.index,
                                               autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'{target_col} Distribution', pad=20)

            # 4. Top correlations summary (combined numeric and categorical)
            ax4 = axes[1, 1]

            # Get top correlations from both numeric and categorical
            top_correlations = []

            if not numeric_results.empty:
                for _, row in numeric_results.iterrows():
                    top_correlations.append({
                        'feature': row['feature'],
                        'strength': abs(row['pearson_correlation']) if not pd.isna(row['pearson_correlation']) else 0,
                        'type': 'Numeric (Pearson)',
                        'value': row['pearson_correlation'] if not pd.isna(row['pearson_correlation']) else 0
                    })

            if not categorical_results.empty:
                for _, row in categorical_results.iterrows():
                    top_correlations.append({
                        'feature': row['feature'],
                        'strength': row['cramers_v'] if not pd.isna(row['cramers_v']) else 0,
                        'type': "Categorical (Cramér's V)",
                        'value': row['cramers_v'] if not pd.isna(row['cramers_v']) else 0
                    })

            if top_correlations:
                # Sort by strength and take top 10
                top_correlations = sorted(top_correlations, key=lambda x: x['strength'], reverse=True)[:10]

                features = [item['feature'][:12] + '...' if len(item['feature']) > 12 else item['feature']
                            for item in top_correlations]
                strengths = [item['strength'] for item in top_correlations]
                types = [item['type'] for item in top_correlations]

                y_pos = np.arange(len(features))
                colors = ['lightcoral' if 'Numeric' in t else 'lightblue' for t in types]
                bars = ax4.barh(y_pos, strengths, color=colors)
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(features, fontsize=7)
                ax4.set_xlabel('Correlation Strength')
                ax4.set_title(f'Top 10 Correlations with {target_col}', pad=20)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='lightcoral', label='Numeric (Pearson)'),
                                   Patch(facecolor='lightblue', label="Categorical (Cramér's V)")]
                ax4.legend(handles=legend_elements, loc='lower right', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'No Valid Correlations', ha='center', va='center',
                         transform=ax4.transAxes, fontsize=12)
                ax4.set_title(f'Top Correlations with {target_col}')

            plt.tight_layout(pad=3.0)
            filename = base_path + f'/output/correlation_analysis_{target_col.lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Visualization chart for {target_col} saved as: {filename}")

    except Exception as e:
        print(f"Error creating visualization charts: {e}")
        import traceback
        traceback.print_exc()


def print_overall_summary(all_results, target_columns):
    """
    Print overall summary of strongest correlations across all targets
    """
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY - STRONGEST CORRELATIONS")
    print(f"{'=' * 80}")

    for target_col in target_columns:
        results_df = all_results[target_col]
        print(f"\nStrongest correlations with {target_col}:")

        # Numeric features
        numeric_results = results_df[results_df['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            numeric_results = numeric_results.dropna(subset=['pearson_correlation'])
            if not numeric_results.empty:
                top_numeric = numeric_results.loc[numeric_results['pearson_correlation'].abs().idxmax()]
                print(f"  Best numeric feature: {top_numeric['feature']:<25} "
                      f"(Pearson: {top_numeric['pearson_correlation']:7.4f})")

        # Categorical features
        categorical_results = results_df[results_df['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            categorical_results = categorical_results.dropna(subset=['cramers_v'])
            if not categorical_results.empty:
                top_categorical = categorical_results.loc[categorical_results['cramers_v'].idxmax()]
                print(f"  Best categorical feature: {top_categorical['feature']:<25} "
                      f"(Cramér's V: {top_categorical['cramers_v']:7.4f})")


if __name__ == "__main__":
    # Analyze appendicitis data for multiple targets
    base_path = os.path.dirname(os.path.dirname(os.getcwd()))
    file_path = base_path + "/data/appendicitis/app_data_modified.xlsx"
    target_columns = ['Diagnosis', 'Severity', 'Management']

    all_results, data = analyze_correlation_with_targets(file_path, target_columns)
    print("all_results111", all_results)
    print("data 111",data)

    # if all_results is not None:
    #     print("\nAnalysis completed!")
    #
    #     # Print overall summary
    #     print_overall_summary(all_results, target_columns)
    #
    #     # Print feature counts for each target
    #     for target_col in target_columns:
    #         results_df = all_results[target_col]
    #         numeric_count = len(results_df[results_df['method'] == 'numeric'])
    #         categorical_count = len(results_df[results_df['method'] == 'categorical'])
    #         total_count = len(results_df)
    #
    #         print(f"\n{target_col} analysis summary:")
    #         print(f"  Total features analyzed: {total_count}")
    #         print(f"  Numeric features: {numeric_count}")
    #         print(f"  Categorical features: {categorical_count}")
    #
    # else:
    #     print("Analysis failed, please check file path and data format")