import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class SmartCorrelationAnalyzer:
    def __init__(self, data_path):
        """Initialize smart correlation analyzer"""
        self.data_path = data_path
        self.data = None
        self.target_columns = ['Management', 'Severity', 'Diagnosis']
        self.exclude_columns = ['US_Performed', 'US_Number']
        self.correlation_results = {}

    def load_data(self):
        """Load the processed data"""
        print("=" * 80)
        print("LOADING PROCESSED APPENDICITIS DATA FOR SMART CORRELATION ANALYSIS")
        print("=" * 80)

        try:
            # Try to read from different possible sheet names
            sheet_names = ['Processed_Data', None]  # None means default sheet
            print('sheet_names111',sheet_names)
            for sheet in sheet_names:
                try:
                    if sheet:
                        self.data = pd.read_excel(self.data_path, sheet_name=sheet)
                        print(f"✓ Data loaded from '{sheet}' sheet")
                    else:
                        self.data = pd.read_excel(self.data_path)
                        print("✓ Data loaded from default sheet")
                    break
                except:
                    continue

            if self.data is None:
                raise Exception("Could not load data from any sheet")

            print(f"Data shape: {self.data.shape}")
            print(f"Total columns: {self.data.shape[1]}")
            print(f"Total rows: {self.data.shape[0]}")

            # Display column overview
            print(f"\nColumn overview (first 15):")
            for i, col in enumerate(self.data.columns[:15]):
                print(f"  {i + 1:2d}. {col}")
            if len(self.data.columns) > 15:
                print(f"  ... and {len(self.data.columns) - 15} more columns")

            return True

        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False

    def check_target_variables(self):
        """Check target variables availability and characteristics"""
        print(f"\nTarget Variables Analysis:")
        print("-" * 50)

        available_targets = []
        for target in self.target_columns:
            if target in self.data.columns:
                available_targets.append(target)
                unique_vals = self.data[target].nunique()
                missing_vals = self.data[target].isnull().sum()
                data_type = self.data[target].dtype

                print(f"✓ {target}:")
                print(f"    Data type: {data_type}")
                print(f"    Unique values: {unique_vals}")
                print(f"    Missing values: {missing_vals}")

                # Display value distribution
                value_counts = self.data[target].value_counts()
                print(f"    Distribution:")
                for value, count in value_counts.head(8).items():
                    pct = (count / len(self.data)) * 100
                    print(f"      {value}: {count} ({pct:.1f}%)")
                if len(value_counts) > 8:
                    print(f"      ... and {len(value_counts) - 8} more values")
                print()
            else:
                print(f"✗ {target}: Not found")

        self.available_targets = available_targets
        print(f"Available target variables: {available_targets}")
        return available_targets

    def prepare_feature_columns(self):
        """Prepare feature columns and classify them"""
        exclude_all = self.exclude_columns + self.available_targets

        # Find all feature columns
        feature_columns = []
        for col in self.data.columns:
            if col not in exclude_all:
                feature_columns.append(col)

        print(f"\nFeature Columns Preparation:")
        print(f"Total columns: {len(self.data.columns)}")
        print(f"Excluded columns ({len(exclude_all)}): {exclude_all}")
        print(f"Feature columns: {len(feature_columns)}")

        # Classify features by unique value count
        numerical_features = []
        categorical_features = []

        print(f"\nFeature Classification (by unique value count):")
        print("-" * 60)

        for col in feature_columns:
            unique_count = self.data[col].nunique()
            non_null_count = self.data[col].count()

            if unique_count > 6:
                numerical_features.append(col)
                feature_type = "Numerical (Pearson)"
            else:
                categorical_features.append(col)
                feature_type = "Categorical (Chi-square)"

            if len(numerical_features) + len(categorical_features) <= 20:  # Show first 20
                print(f"{col:<35} | Unique: {unique_count:>3} | Non-null: {non_null_count:>4} | {feature_type}")

        if len(feature_columns) > 20:
            print(f"... and {len(feature_columns) - 20} more features")

        print(f"\nFeature Type Summary:")
        print(f"Numerical features (>6 unique values): {len(numerical_features)}")
        print(f"Categorical features (≤6 unique values): {len(categorical_features)}")

        self.feature_columns = feature_columns
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        return feature_columns, numerical_features, categorical_features

    def calculate_pearson_correlation(self, feature_col, target_col):
        """Calculate Pearson correlation for numerical features"""
        try:
            # Get feature and target data
            feature_data = self.data[feature_col].copy()
            target_data = self.data[target_col].copy()

            # Handle target encoding if categorical
            if target_data.dtype == 'object':
                le = LabelEncoder()
                # Create combined dataframe to handle missing values together
                combined_df = pd.DataFrame({
                    'feature': feature_data,
                    'target': target_data
                }).dropna()

                if len(combined_df) < 10:  # Not enough data
                    return np.nan, np.nan, len(combined_df)

                target_encoded = le.fit_transform(combined_df['target'])
                feature_clean = combined_df['feature'].values
            else:
                # Both numerical
                combined_df = pd.DataFrame({
                    'feature': feature_data,
                    'target': target_data
                }).dropna()

                if len(combined_df) < 10:  # Not enough data
                    return np.nan, np.nan, len(combined_df)

                target_encoded = combined_df['target'].values
                feature_clean = combined_df['feature'].values

            # Convert feature to numeric if needed
            try:
                feature_clean = pd.to_numeric(feature_clean, errors='coerce')
                # Remove any remaining NaN after conversion
                valid_mask = ~np.isnan(feature_clean) & ~np.isnan(target_encoded)
                feature_clean = feature_clean[valid_mask]
                target_encoded = target_encoded[valid_mask]

                if len(feature_clean) < 10:
                    return np.nan, np.nan, len(feature_clean)

            except:
                return np.nan, np.nan, 0

            # Calculate Pearson correlation
            correlation, p_value = pearsonr(feature_clean, target_encoded)

            return correlation, p_value, len(feature_clean)

        except Exception as e:
            return np.nan, np.nan, 0

    def calculate_chi_square_association(self, feature_col, target_col):
        """Calculate Chi-square test for categorical features"""
        try:
            # Get feature and target data
            feature_data = self.data[feature_col].copy()
            target_data = self.data[target_col].copy()

            # Create combined dataframe to handle missing values
            combined_df = pd.DataFrame({
                'feature': feature_data,
                'target': target_data
            }).dropna()

            if len(combined_df) < 10:  # Not enough data
                return np.nan, np.nan, len(combined_df)

            # Create contingency table
            contingency_table = pd.crosstab(combined_df['feature'], combined_df['target'])

            # Check if contingency table has enough data
            if contingency_table.size < 4 or (contingency_table < 5).sum().sum() > 0.2 * contingency_table.size:
                # Too many cells with low counts, chi-square may not be reliable
                return np.nan, np.nan, len(combined_df)

            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Calculate Cramér's V as effect size (similar to correlation coefficient)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

            return cramers_v, p_value, len(combined_df)

        except Exception as e:
            return np.nan, np.nan, 0

    def analyze_target_correlation(self, target_col):
        """Analyze correlation with a specific target variable"""
        print(f"\nAnalyzing correlation with {target_col}:")
        print("-" * 60)

        results = []

        # Analyze numerical features
        print(f"Processing {len(self.numerical_features)} numerical features...")
        for i, feature in enumerate(self.numerical_features):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(self.numerical_features)} numerical features")

            correlation, p_value, sample_size = self.calculate_pearson_correlation(feature, target_col)

            results.append({
                'Feature': feature,
                'Test_Type': 'Pearson',
                'Correlation_Effect_Size': correlation,
                'P_Value': p_value,
                'Sample_Size': sample_size,
                'Abs_Effect_Size': abs(correlation) if not np.isnan(correlation) else np.nan,
                'Significance': 'Significant' if not np.isnan(p_value) and p_value < 0.05 else 'Not Significant'
            })

        # Analyze categorical features
        print(f"Processing {len(self.categorical_features)} categorical features...")
        for i, feature in enumerate(self.categorical_features):
            if i % 50 == 0 and i > 0:
                print(f"  Processed {i}/{len(self.categorical_features)} categorical features")

            effect_size, p_value, sample_size = self.calculate_chi_square_association(feature, target_col)

            results.append({
                'Feature': feature,
                'Test_Type': 'Chi-square',
                'Correlation_Effect_Size': effect_size,
                'P_Value': p_value,
                'Sample_Size': sample_size,
                'Abs_Effect_Size': abs(effect_size) if not np.isnan(effect_size) else np.nan,
                'Significance': 'Significant' if not np.isnan(p_value) and p_value < 0.05 else 'Not Significant'
            })

        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Abs_Effect_Size', ascending=False, na_position='last')

        print(f"✓ Completed correlation analysis with {target_col}")
        print(f"Valid results: {results_df['Correlation_Effect_Size'].notna().sum()}/{len(results_df)}")
        print(f"Significant results: {len(results_df[results_df['Significance'] == 'Significant'])}")

        return results_df

    def display_top_results(self, results_df, target_col, top_n=30):
        """Display top correlation results"""
        print(f"\nTop {top_n} features with strongest association to {target_col}:")
        print("=" * 120)
        print(
            f"{'Rank':<4} {'Feature Name':<35} {'Test Type':<10} {'Effect Size':<12} {'P-Value':<10} {'Samples':<8} {'Significance':<15}")
        print("-" * 120)

        valid_results = results_df[results_df['Correlation_Effect_Size'].notna()].head(top_n)

        for i, (_, row) in enumerate(valid_results.iterrows(), 1):
            print(f"{i:<4} {row['Feature'][:34]:<35} {row['Test_Type']:<10} "
                  f"{row['Correlation_Effect_Size']:>11.4f} {row['P_Value']:>9.4f} "
                  f"{row['Sample_Size']:>7} {row['Significance']:<15}")

        # Summary statistics
        print(f"\nSummary for {target_col}:")
        print("-" * 30)
        significant_results = valid_results[valid_results['Significance'] == 'Significant']
        print(f"Total valid results: {len(valid_results)}")
        print(f"Significant results (p<0.05): {len(significant_results)}")
        print(f"Pearson correlations: {len(valid_results[valid_results['Test_Type'] == 'Pearson'])}")
        print(f"Chi-square tests: {len(valid_results[valid_results['Test_Type'] == 'Chi-square'])}")

        if len(valid_results) > 0:
            print(f"Mean absolute effect size: {valid_results['Abs_Effect_Size'].mean():.4f}")
            print(f"Max effect size: {valid_results['Abs_Effect_Size'].max():.4f}")

        return valid_results

    def create_visualizations(self, target_col, results_df, top_n=25):
        """Create visualization for correlation results"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Smart Correlation Analysis with {target_col}', fontsize=16, fontweight='bold')

        valid_results = results_df[results_df['Correlation_Effect_Size'].notna()]

        if len(valid_results) == 0:
            print(f"No valid results to visualize for {target_col}")
            return

        top_results = valid_results.head(top_n)

        # 1. Effect size bar plot (colored by test type)
        ax1 = axes[0, 0]
        colors = ['blue' if test_type == 'Pearson' else 'orange'
                  for test_type in top_results['Test_Type']]
        bars = ax1.barh(range(len(top_results)), top_results['Correlation_Effect_Size'],
                        color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_results)))
        ax1.set_yticklabels([feat[:25] + '...' if len(feat) > 25 else feat
                             for feat in top_results['Feature']], fontsize=8)
        ax1.set_xlabel('Effect Size (Correlation/Cramér\'s V)')
        ax1.set_title(f'Top {top_n} Features - Effect Size with {target_col}')
        ax1.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Pearson Correlation'),
                           Patch(facecolor='orange', alpha=0.7, label='Chi-square (Cramér\'s V)')]
        ax1.legend(handles=legend_elements, loc='lower right')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=7)

        # 2. Significance scatter plot
        ax2 = axes[0, 1]
        significant = valid_results[valid_results['Significance'] == 'Significant']
        not_significant = valid_results[valid_results['Significance'] == 'Not Significant']

        if len(significant) > 0:
            ax2.scatter(significant['Abs_Effect_Size'], -np.log10(significant['P_Value']),
                        c='red', alpha=0.7, label=f'Significant (n={len(significant)})')
        if len(not_significant) > 0:
            ax2.scatter(not_significant['Abs_Effect_Size'], -np.log10(not_significant['P_Value']),
                        c='gray', alpha=0.5, label=f'Not Significant (n={len(not_significant)})')

        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax2.set_xlabel('Absolute Effect Size')
        ax2.set_ylabel('-Log10(P-Value)')
        ax2.set_title(f'Significance vs Effect Size for {target_col}')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Effect size distribution
        ax3 = axes[1, 0]
        valid_effects = valid_results['Abs_Effect_Size'].dropna()
        if len(valid_effects) > 0:
            ax3.hist(valid_effects, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(valid_effects.mean(), color='red', linestyle='--',
                        label=f'Mean: {valid_effects.mean():.3f}')
            ax3.axvline(valid_effects.median(), color='green', linestyle='--',
                        label=f'Median: {valid_effects.median():.3f}')
            ax3.set_xlabel('Absolute Effect Size')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Distribution of Effect Sizes with {target_col}')
            ax3.legend()
            ax3.grid(alpha=0.3)

        # 4. Test type comparison
        ax4 = axes[1, 1]
        test_summary = valid_results.groupby('Test_Type').agg({
            'Abs_Effect_Size': ['count', 'mean', 'std'],
            'Significance': lambda x: (x == 'Significant').sum()
        }).round(4)

        test_types = test_summary.index
        counts = test_summary[('Abs_Effect_Size', 'count')]
        means = test_summary[('Abs_Effect_Size', 'mean')]
        significant_counts = test_summary[('Significance', '<lambda>')]

        x_pos = np.arange(len(test_types))
        width = 0.35

        bars1 = ax4.bar(x_pos - width / 2, counts, width, label='Total Count', alpha=0.7)
        bars2 = ax4.bar(x_pos + width / 2, significant_counts, width, label='Significant Count', alpha=0.7)

        ax4.set_xlabel('Test Type')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Test Type Summary for {target_col}')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(test_types)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        filename = f'smart_correlation_analysis_{target_col}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Visualization saved: {filename}")

    def analyze_all_targets(self):
        """Analyze correlation with all target variables"""
        print(f"\n" + "=" * 80)
        print("SMART CORRELATION ANALYSIS FOR ALL TARGETS")
        print("=" * 80)

        all_results = {}

        for target in self.available_targets:
            print(f"\n{'=' * 70}")
            print(f"Analyzing target variable: {target}")
            print('=' * 70)

            # Calculate correlations
            results_df = self.analyze_target_correlation(target)

            # Display top results
            top_results = self.display_top_results(results_df, target, top_n=30)

            # Create visualizations
            self.create_visualizations(target, results_df, top_n=25)

            # Save results
            all_results[target] = results_df

            # Save to CSV
            filename = f'smart_correlation_{target}.csv'
            results_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\nDetailed results saved: {filename}")

        self.correlation_results = all_results
        return all_results

    def create_final_summary(self):
        """Create final summary report"""
        print(f"\n" + "=" * 80)
        print("FINAL SUMMARY REPORT")
        print("=" * 80)

        summary_data = []

        for target in self.available_targets:
            if target in self.correlation_results:
                results_df = self.correlation_results[target]
                valid_results = results_df[results_df['Correlation_Effect_Size'].notna()]
                significant_results = valid_results[valid_results['Significance'] == 'Significant']

                if len(valid_results) > 0:
                    pearson_results = valid_results[valid_results['Test_Type'] == 'Pearson']
                    chisq_results = valid_results[valid_results['Test_Type'] == 'Chi-square']

                    summary = {
                        'Target': target,
                        'Total_Features_Analyzed': len(results_df),
                        'Valid_Results': len(valid_results),
                        'Significant_Results': len(significant_results),
                        'Pearson_Tests': len(pearson_results),
                        'Chi_Square_Tests': len(chisq_results),
                        'Max_Effect_Size': valid_results['Abs_Effect_Size'].max(),
                        'Mean_Effect_Size': valid_results['Abs_Effect_Size'].mean(),
                        'Significant_Percentage': (len(significant_results) / len(valid_results)) * 100
                    }

                    summary_data.append(summary)

                    print(f"\n{target} Summary:")
                    print("-" * 40)
                    print(f"Features analyzed: {summary['Total_Features_Analyzed']}")
                    print(f"Valid results: {summary['Valid_Results']}")
                    print(
                        f"Significant results: {summary['Significant_Results']} ({summary['Significant_Percentage']:.1f}%)")
                    print(f"Pearson correlations: {summary['Pearson_Tests']}")
                    print(f"Chi-square tests: {summary['Chi_Square_Tests']}")
                    print(f"Maximum effect size: {summary['Max_Effect_Size']:.4f}")
                    print(f"Mean effect size: {summary['Mean_Effect_Size']:.4f}")

                    # Show top 3 significant features
                    top_significant = significant_results.head(3)
                    if len(top_significant) > 0:
                        print(f"\nTop 3 significant associations:")
                        for i, (_, row) in enumerate(top_significant.iterrows(), 1):
                            print(f"  {i}. {row['Feature'][:45]} "
                                  f"({row['Test_Type']}, Effect: {row['Correlation_Effect_Size']:.4f})")

        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('smart_correlation_summary.csv', index=False, encoding='utf-8-sig')
            print(f"\nFinal summary saved: smart_correlation_summary.csv")

        return summary_data

    def run_complete_analysis(self):
        """Run the complete smart correlation analysis"""
        # 1. Load data
        if not self.load_data():
            return False

        # 2. Check target variables
        if not self.check_target_variables():
            print("No target variables found")
            return False

        # 3. Prepare and classify features
        features = self.prepare_feature_columns()
        if not features[0]:  # No features found
            print("No feature columns found")
            return False

        # 4. Analyze all targets
        self.analyze_all_targets()

        # 5. Create final summary
        self.create_final_summary()

        print(f"\n" + "=" * 80)
        print("SMART CORRELATION ANALYSIS COMPLETED!")
        print("=" * 80)
        print("\nGenerated files:")
        for target in self.available_targets:
            print(f"  - smart_correlation_{target}.csv")
            print(f"  - smart_correlation_analysis_{target}.png")
        print("  - smart_correlation_summary.csv")

        print(f"\nAnalysis method summary:")
        print(f"- Numerical features (>6 unique values): Pearson Correlation")
        print(f"- Categorical features (≤6 unique values): Chi-square Test (Cramér's V)")
        print(f"- Significance level: p < 0.05")

        return True


def main():
    """Main function"""
    print("Starting Smart Correlation Analysis of Appendicitis Data...")

    # Create analyzer
    analyzer = SmartCorrelationAnalyzer('../../data/appendicitis/processed_appendicitis_data_final.xlsx')

    # Run analysis
    success = analyzer.run_complete_analysis()

    if success:
        print("\n✓ Smart correlation analysis completed successfully!")
    else:
        print("\n✗ Smart correlation analysis failed!")


if __name__ == "__main__":
    main()