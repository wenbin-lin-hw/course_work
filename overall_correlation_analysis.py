import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def calculate_overall_associations(file_path):
    """
    Calculate overall associations between all features and Management, Severity, Diagnosis
    """
    try:
        # Read Excel file
        print("Reading Excel file...")
        df = pd.read_excel(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Total columns: {len(df.columns)}")

        # Define target variables
        target_columns = ['Management', 'Severity', 'Diagnosis']

        # Check if target columns exist
        available_targets = []
        for target in target_columns:
            if target in df.columns:
                available_targets.append(target)
                print(f"\n{target} value distribution:")
                print(df[target].value_counts())
            else:
                print(f"Warning: {target} column not found in dataset")

        if not available_targets:
            print("No target columns found!")
            return None

        print(f"\nAnalyzing associations with: {available_targets}")

        # Get feature columns (exclude target columns)
        feature_columns = [col for col in df.columns if col not in available_targets]
        print(f"Number of feature columns to analyze: {len(feature_columns)}")

        # Store results for each target
        overall_results = {}

        for target in available_targets:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING ASSOCIATIONS WITH {target.upper()}")
            print(f"{'=' * 60}")

            target_data = df[target]
            feature_associations = []

            for feature in feature_columns:
                print(f"Processing: {feature}")

                # Get non-null data for both feature and target
                mask = df[feature].notna() & df[target].notna()
                if mask.sum() == 0:
                    print(f"  No valid data pairs for {feature}")
                    continue

                feature_data = df[feature][mask]
                target_aligned = target_data[mask]

                association_result = {
                    'feature': feature,
                    'target': target,
                    'valid_pairs': len(feature_data),
                    'feature_unique_values': feature_data.nunique(),
                    'target_unique_values': target_aligned.nunique()
                }

                # Try to determine if feature is numeric or categorical
                is_numeric = False
                try:
                    # Check if already numeric
                    if pd.api.types.is_numeric_dtype(feature_data):
                        numeric_data = feature_data.astype(float)
                        is_numeric = True
                    else:
                        # Try to convert to numeric
                        numeric_data = pd.to_numeric(feature_data, errors='coerce')
                        valid_numeric = numeric_data.notna().sum()
                        if valid_numeric > len(feature_data) * 0.7:  # 70% conversion threshold
                            is_numeric = True
                            # Update mask for valid numeric data
                            numeric_mask = numeric_data.notna()
                            numeric_data = numeric_data[numeric_mask]
                            target_aligned = target_aligned[numeric_mask]
                            association_result['valid_pairs'] = len(numeric_data)

                except Exception as e:
                    print(f"  Error in numeric conversion: {e}")
                    is_numeric = False

                if is_numeric and len(numeric_data) > 1:
                    # Numeric association analysis
                    try:
                        # Encode target if categorical
                        if not pd.api.types.is_numeric_dtype(target_aligned):
                            le = LabelEncoder()
                            target_numeric = le.fit_transform(target_aligned)
                        else:
                            target_numeric = target_aligned.astype(float)

                        # Calculate correlations
                        pearson_corr, pearson_p = pearsonr(numeric_data, target_numeric)
                        spearman_corr, spearman_p = spearmanr(numeric_data, target_numeric)

                        association_result.update({
                            'association_type': 'numeric',
                            'pearson_correlation': pearson_corr,
                            'pearson_p_value': pearson_p,
                            'spearman_correlation': spearman_corr,
                            'spearman_p_value': spearman_p,
                            'association_strength': abs(pearson_corr),
                            'significance': 'significant' if pearson_p < 0.05 else 'not_significant'
                        })

                        print(f"  Numeric - Pearson: {pearson_corr:.4f} (p={pearson_p:.4f})")

                    except Exception as e:
                        print(f"  Error in numeric association: {e}")
                        association_result.update({
                            'association_type': 'numeric_error',
                            'association_strength': 0,
                            'significance': 'error'
                        })

                else:
                    # Categorical association analysis
                    try:
                        # Convert both to strings for contingency table
                        feature_cat = feature_data.astype(str)
                        target_cat = target_aligned.astype(str)

                        # Create contingency table
                        contingency_table = pd.crosstab(feature_cat, target_cat)

                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            # Perform chi-square test
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                            # Calculate Cramér's V
                            n = contingency_table.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                            association_result.update({
                                'association_type': 'categorical',
                                'chi2_statistic': chi2,
                                'chi2_p_value': p_value,
                                'cramers_v': cramers_v,
                                'association_strength': cramers_v,
                                'significance': 'significant' if p_value < 0.05 else 'not_significant',
                                'contingency_shape': f"{contingency_table.shape[0]}x{contingency_table.shape[1]}"
                            })

                            print(f"  Categorical - Cramér's V: {cramers_v:.4f} (p={p_value:.4f})")

                        else:
                            print(f"  Contingency table too small: {contingency_table.shape}")
                            association_result.update({
                                'association_type': 'categorical_insufficient',
                                'association_strength': 0,
                                'significance': 'insufficient_data'
                            })

                    except Exception as e:
                        print(f"  Error in categorical association: {e}")
                        association_result.update({
                            'association_type': 'categorical_error',
                            'association_strength': 0,
                            'significance': 'error'
                        })

                feature_associations.append(association_result)

            # Convert to DataFrame and store
            overall_results[target] = pd.DataFrame(feature_associations)

        return overall_results, df, available_targets

    except Exception as e:
        print(f"Error in overall association analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def analyze_and_visualize_results(overall_results, df, available_targets):
    """
    Analyze results and create visualizations
    """
    print(f"\n{'=' * 80}")
    print("OVERALL ASSOCIATION ANALYSIS SUMMARY")
    print(f"{'=' * 80}")

    # Summary statistics
    for target in available_targets:
        results_df = overall_results[target]

        print(f"\n{target.upper()} ASSOCIATIONS:")
        print(f"Total features analyzed: {len(results_df)}")

        # Count by association type
        type_counts = results_df['association_type'].value_counts()
        print("Association types:")
        for assoc_type, count in type_counts.items():
            print(f"  {assoc_type}: {count}")

        # Significant associations
        significant = results_df[results_df['significance'] == 'significant']
        print(f"Significant associations (p < 0.05): {len(significant)}")

        # Top associations
        valid_results = results_df[results_df['association_strength'] > 0].sort_values(
            'association_strength', ascending=False
        )

        print(f"\nTop 10 strongest associations with {target}:")
        for i, (_, row) in enumerate(valid_results.head(10).iterrows()):
            strength = row['association_strength']
            assoc_type = row['association_type']
            significance = "***" if row['significance'] == 'significant' else ""
            print(
                f"  {i + 1:2d}. {row['feature']:<25} | Strength: {strength:.4f} | Type: {assoc_type:<12} {significance}")

    # Save detailed results
    for target in available_targets:
        filename = f'overall_associations_{target.lower()}.csv'
        overall_results[target].to_csv(filename, index=False)
        print(f"\nDetailed results for {target} saved to: {filename}")

    # Create visualizations
    create_association_visualizations(overall_results, available_targets)

    # Create comparison visualization
    create_comparison_visualization(overall_results, available_targets)


def create_association_visualizations(overall_results, available_targets):
    """
    Create individual visualizations for each target
    """
    plt.rcParams['font.size'] = 9

    for target in available_targets:
        results_df = overall_results[target]
        valid_results = results_df[results_df['association_strength'] > 0].sort_values(
            'association_strength', ascending=False
        )

        if len(valid_results) == 0:
            print(f"No valid associations found for {target}")
            continue

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Association Analysis with {target}', fontsize=16, fontweight='bold')

        # 1. Top associations bar chart
        ax1 = axes[0, 0]
        top_20 = valid_results.head(20)
        if not top_20.empty:
            y_pos = np.arange(len(top_20))
            colors = ['red' if x == 'significant' else 'lightblue' for x in top_20['significance']]
            bars = ax1.barh(y_pos, top_20['association_strength'], color=colors)
            ax1.set_yticks(y_pos)
            feature_labels = [feat[:15] + '...' if len(feat) > 15 else feat for feat in top_20['feature']]
            ax1.set_yticklabels(feature_labels, fontsize=7)
            ax1.set_xlabel('Association Strength')
            ax1.set_title(f'Top 20 Associations with {target}')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='Significant (p<0.05)'),
                               Patch(facecolor='lightblue', label='Not Significant')]
            ax1.legend(handles=legend_elements, loc='lower right')

        # 2. Association strength distribution
        ax2 = axes[0, 1]
        ax2.hist(valid_results['association_strength'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Association Strength')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of Association Strengths with {target}')
        ax2.axvline(valid_results['association_strength'].mean(), color='red', linestyle='--',
                    label=f'Mean: {valid_results["association_strength"].mean():.3f}')
        ax2.legend()

        # 3. Association types pie chart
        ax3 = axes[1, 0]
        type_counts = results_df['association_type'].value_counts()
        wedges, texts, autotexts = ax3.pie(type_counts.values, labels=type_counts.index,
                                           autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Association Types for {target}')

        # 4. Significance overview
        ax4 = axes[1, 1]
        sig_counts = results_df['significance'].value_counts()
        colors_sig = ['green' if 'significant' in x else 'orange' if 'not_significant' in x else 'red'
                      for x in sig_counts.index]
        bars = ax4.bar(range(len(sig_counts)), sig_counts.values, color=colors_sig)
        ax4.set_xticks(range(len(sig_counts)))
        ax4.set_xticklabels(sig_counts.index, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Statistical Significance Overview for {target}')

        # Add value labels on bars
        for bar, value in zip(bars, sig_counts.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value}', ha='center', va='bottom')

        plt.tight_layout()
        filename = f'association_analysis_{target.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization for {target} saved as: {filename}")


def create_comparison_visualization(overall_results, available_targets):
    """
    Create comparison visualization across all targets
    """
    if len(available_targets) < 2:
        print("Need at least 2 targets for comparison visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Target Association Comparison', fontsize=16, fontweight='bold')

    # 1. Average association strength comparison
    ax1 = axes[0, 0]
    target_means = []
    target_names = []
    for target in available_targets:
        results_df = overall_results[target]
        valid_results = results_df[results_df['association_strength'] > 0]
        if not valid_results.empty:
            target_means.append(valid_results['association_strength'].mean())
            target_names.append(target)

    if target_means:
        bars = ax1.bar(target_names, target_means, color=['skyblue', 'lightcoral', 'lightgreen'][:len(target_names)])
        ax1.set_ylabel('Average Association Strength')
        ax1.set_title('Average Association Strength by Target')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, target_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.3f}', ha='center', va='bottom')

    # 2. Number of significant associations
    ax2 = axes[0, 1]
    sig_counts = []
    for target in available_targets:
        results_df = overall_results[target]
        sig_count = len(results_df[results_df['significance'] == 'significant'])
        sig_counts.append(sig_count)

    bars = ax2.bar(available_targets, sig_counts,
                   color=['skyblue', 'lightcoral', 'lightgreen'][:len(available_targets)])
    ax2.set_ylabel('Number of Significant Associations')
    ax2.set_title('Significant Associations (p < 0.05) by Target')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, value in zip(bars, sig_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value}', ha='center', va='bottom')

    # 3. Association type distribution comparison
    ax3 = axes[1, 0]
    all_types = set()
    for target in available_targets:
        all_types.update(overall_results[target]['association_type'].unique())

    type_data = {target: overall_results[target]['association_type'].value_counts()
                 for target in available_targets}

    x = np.arange(len(available_targets))
    width = 0.8 / len(all_types)
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_types)))

    for i, assoc_type in enumerate(all_types):
        values = [type_data[target].get(assoc_type, 0) for target in available_targets]
        ax3.bar(x + i * width, values, width, label=assoc_type, color=colors[i])

    ax3.set_xlabel('Target Variables')
    ax3.set_ylabel('Count')
    ax3.set_title('Association Type Distribution by Target')
    ax3.set_xticks(x + width * (len(all_types) - 1) / 2)
    ax3.set_xticklabels(available_targets)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 4. Top shared features (features with high associations across targets)
    ax4 = axes[1, 1]

    # Find features that appear in top 10 for multiple targets
    shared_features = {}
    for target in available_targets:
        results_df = overall_results[target]
        top_features = results_df.nlargest(10, 'association_strength')['feature'].tolist()
        for feature in top_features:
            if feature not in shared_features:
                shared_features[feature] = []
            shared_features[feature].append(target)

    # Find features appearing in multiple targets
    multi_target_features = {k: v for k, v in shared_features.items() if len(v) > 1}

    if multi_target_features:
        features = list(multi_target_features.keys())[:10]  # Top 10
        counts = [len(multi_target_features[f]) for f in features]

        bars = ax4.barh(range(len(features)), counts, color='lightseagreen')
        ax4.set_yticks(range(len(features)))
        feature_labels = [f[:15] + '...' if len(f) > 15 else f for f in features]
        ax4.set_yticklabels(feature_labels, fontsize=8)
        ax4.set_xlabel('Number of Targets')
        ax4.set_title('Features with High Associations Across Multiple Targets')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height() / 2.,
                     f'{value}', ha='left', va='center')
    else:
        ax4.text(0.5, 0.5, 'No features found in\nmultiple target top-10s',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Shared High-Association Features')

    plt.tight_layout()
    plt.savefig('cross_target_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Cross-target comparison saved as: cross_target_comparison.png")


if __name__ == "__main__":
    # Analyze overall associations
    file_path = "data/appendicitis/app_data.xlsx"

    print("Starting overall association analysis...")
    overall_results, df, available_targets = calculate_overall_associations(file_path)

    if overall_results is not None:
        print("\nAnalysis completed successfully!")
        analyze_and_visualize_results(overall_results, df, available_targets)

        # Final summary
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Dataset analyzed: {file_path}")
        print(f"Total features: {len(df.columns) - len(available_targets)}")
        print(f"Target variables: {', '.join(available_targets)}")
        print(f"Visualizations created: {len(available_targets) + 1}")
        print("Analysis files saved:")
        for target in available_targets:
            print(f"  - overall_associations_{target.lower()}.csv")
            print(f"  - association_analysis_{target.lower()}.png")
        print("  - cross_target_comparison.png")

    else:
        print("Analysis failed. Please check the file path and data format.")