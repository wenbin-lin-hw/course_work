import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data(file_path):
    """
    Analyze missing data counts for every column in the Excel file
    """
    try:
        # Read the Excel file
        print("Reading Excel file...")
        df = pd.read_excel(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")

        # Calculate missing data statistics
        missing_stats = []

        for column in df.columns:
            total_count = len(df)
            missing_count = df[column].isnull().sum()
            non_missing_count = total_count - missing_count
            missing_percentage = (missing_count / total_count) * 100

            # Get data type
            dtype = str(df[column].dtype)

            # Get unique values count (for non-missing values only)
            unique_count = df[column].nunique()

            missing_stats.append({
                'Column': column,
                'Total_Count': total_count,
                'Missing_Count': missing_count,
                'Non_Missing_Count': non_missing_count,
                'Missing_Percentage': missing_percentage,
                'Data_Type': dtype,
                'Unique_Values': unique_count
            })

        # Create DataFrame with missing data statistics
        missing_df = pd.DataFrame(missing_stats)

        # Sort by missing count (descending)
        missing_df_sorted = missing_df.sort_values('Missing_Count', ascending=False)

        # Display the complete table
        print("\n" + "="*100)
        print("MISSING DATA ANALYSIS - COMPLETE TABLE")
        print("="*100)

        # Set pandas display options to show all rows
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)

        print(missing_df_sorted.to_string(index=False))

        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

        # Summary statistics
        print(f"\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Columns with no missing data: {len(missing_df[missing_df['Missing_Count'] == 0])}")
        print(f"Columns with missing data: {len(missing_df[missing_df['Missing_Count'] > 0])}")
        print(f"Columns with >50% missing data: {len(missing_df[missing_df['Missing_Percentage'] > 50])}")
        print(f"Columns with >90% missing data: {len(missing_df[missing_df['Missing_Percentage'] > 90])}")

        print(f"\nMissing data statistics:")
        print(f"Average missing percentage: {missing_df['Missing_Percentage'].mean():.2f}%")
        print(f"Median missing percentage: {missing_df['Missing_Percentage'].median():.2f}%")
        print(f"Max missing percentage: {missing_df['Missing_Percentage'].max():.2f}%")
        print(f"Min missing percentage: {missing_df['Missing_Percentage'].min():.2f}%")

        # Columns with highest missing data
        print(f"\nTOP 10 COLUMNS WITH MOST MISSING DATA:")
        print("-" * 60)
        top_missing = missing_df_sorted.head(10)
        for _, row in top_missing.iterrows():
            print(f"{row['Column']:<30} | Missing: {row['Missing_Count']:4d} ({row['Missing_Percentage']:5.1f}%)")

        # Columns with no missing data
        no_missing = missing_df[missing_df['Missing_Count'] == 0]
        if not no_missing.empty:
            print(f"\nCOLUMNS WITH NO MISSING DATA ({len(no_missing)} columns):")
            print("-" * 60)
            for _, row in no_missing.iterrows():
                print(f"{row['Column']:<30} | Data Type: {row['Data_Type']:<12} | Unique: {row['Unique_Values']:4d}")

        # Save results to CSV
        missing_df_sorted.to_csv('missing_data_analysis.csv', index=False)
        print(f"\nDetailed missing data analysis saved to: missing_data_analysis.csv")

        # Create visualizations
        create_missing_data_visualizations(missing_df_sorted, df)

        return missing_df_sorted, df

    except Exception as e:
        print(f"Error analyzing missing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_missing_data_visualizations(missing_df, original_df):
    """
    Create visualizations for missing data analysis
    """
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Missing Data Analysis - Appendicitis Dataset', fontsize=16, fontweight='bold')

        # 1. Bar chart of missing counts (top 20 columns)
        ax1 = axes[0, 0]
        top_20_missing = missing_df[missing_df['Missing_Count'] > 0].head(20)
        if not top_20_missing.empty:
            y_pos = np.arange(len(top_20_missing))
            bars = ax1.barh(y_pos, top_20_missing['Missing_Count'])
            ax1.set_yticks(y_pos)
            # Truncate long column names
            labels = [col[:20] + '...' if len(col) > 20 else col for col in top_20_missing['Column']]
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('Missing Count')
            ax1.set_title('Top 20 Columns with Missing Data (Count)')

            # Color bars based on missing percentage
            for i, bar in enumerate(bars):
                pct = top_20_missing.iloc[i]['Missing_Percentage']
                if pct > 90:
                    bar.set_color('red')
                elif pct > 50:
                    bar.set_color('orange')
                else:
                    bar.set_color('lightblue')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data Found', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Missing Data Count')

        # 2. Bar chart of missing percentages (top 20 columns)
        ax2 = axes[0, 1]
        if not top_20_missing.empty:
            bars = ax2.barh(y_pos, top_20_missing['Missing_Percentage'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('Missing Percentage (%)')
            ax2.set_title('Top 20 Columns with Missing Data (Percentage)')

            # Color bars and add percentage labels
            for i, bar in enumerate(bars):
                pct = top_20_missing.iloc[i]['Missing_Percentage']
                if pct > 90:
                    bar.set_color('red')
                elif pct > 50:
                    bar.set_color('orange')
                else:
                    bar.set_color('lightblue')

                # Add percentage label
                ax2.text(pct + 1, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=7)
        else:
            ax2.text(0.5, 0.5, 'No Missing Data Found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Missing Data Percentage')

        # 3. Histogram of missing percentages
        ax3 = axes[1, 0]
        missing_percentages = missing_df['Missing_Percentage']
        ax3.hist(missing_percentages, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Missing Percentage (%)')
        ax3.set_ylabel('Number of Columns')
        ax3.set_title('Distribution of Missing Data Percentages')
        ax3.axvline(missing_percentages.mean(), color='red', linestyle='--',
                   label=f'Mean: {missing_percentages.mean():.1f}%')
        ax3.axvline(missing_percentages.median(), color='green', linestyle='--',
                   label=f'Median: {missing_percentages.median():.1f}%')
        ax3.legend()

        # 4. Summary pie chart
        ax4 = axes[1, 1]
        no_missing = len(missing_df[missing_df['Missing_Count'] == 0])
        low_missing = len(missing_df[(missing_df['Missing_Percentage'] > 0) & (missing_df['Missing_Percentage'] <= 25)])
        medium_missing = len(missing_df[(missing_df['Missing_Percentage'] > 25) & (missing_df['Missing_Percentage'] <= 50)])
        high_missing = len(missing_df[(missing_df['Missing_Percentage'] > 50) & (missing_df['Missing_Percentage'] <= 90)])
        very_high_missing = len(missing_df[missing_df['Missing_Percentage'] > 90])

        categories = ['No Missing\n(0%)', 'Low Missing\n(0-25%)', 'Medium Missing\n(25-50%)',
                     'High Missing\n(50-90%)', 'Very High Missing\n(>90%)']
        counts = [no_missing, low_missing, medium_missing, high_missing, very_high_missing]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']

        # Only include categories with non-zero counts
        non_zero_categories = []
        non_zero_counts = []
        non_zero_colors = []

        for cat, count, color in zip(categories, counts, colors):
            if count > 0:
                non_zero_categories.append(f'{cat}\n({count} cols)')
                non_zero_counts.append(count)
                non_zero_colors.append(color)

        if non_zero_counts:
            wedges, texts, autotexts = ax4.pie(non_zero_counts, labels=non_zero_categories,
                                              colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Column Distribution by Missing Data Level')
        else:
            ax4.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Missing Data Distribution')

        plt.tight_layout()
        plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Missing data visualization saved as: missing_data_analysis.png")

        # Create a heatmap for missing data pattern (if dataset is not too large)
        if len(missing_df) <= 50:  # Only for reasonable number of columns
            plt.figure(figsize=(12, max(8, len(missing_df) * 0.3)))

            # Create missing data matrix
            missing_matrix = original_df.isnull().astype(int)

            # Plot heatmap
            sns.heatmap(missing_matrix.T, cbar=True, cmap='YlOrRd',
                       yticklabels=True, xticklabels=False)
            plt.title('Missing Data Pattern Heatmap')
            plt.ylabel('Columns')
            plt.xlabel('Rows (Samples)')

            plt.tight_layout()
            plt.savefig('missing_data_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("Missing data heatmap saved as: missing_data_heatmap.png")

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Analyze missing data in appendicitis dataset
    file_path = "data/appendicitis/app_data.xlsx"

    print("Starting missing data analysis...")
    missing_analysis, original_data = analyze_missing_data(file_path)

    if missing_analysis is not None:
        print(f"\nMissing data analysis completed successfully!")
        print(f"Results saved to: missing_data_analysis.csv")
        print(f"Visualizations saved to: missing_data_analysis.png")

        # Additional detailed table output
        print(f"\n{'='*120}")
        print("DETAILED MISSING DATA TABLE")
        print('='*120)
        print(f"{'Column Name':<30} {'Missing Count':<15} {'Missing %':<12} {'Non-Missing':<12} {'Data Type':<15} {'Unique Values':<12}")
        print('-'*120)

        for _, row in missing_analysis.iterrows():
            print(f"{row['Column']:<30} {row['Missing_Count']:<15} {row['Missing_Percentage']:<12.2f} "
                  f"{row['Non_Missing_Count']:<12} {row['Data_Type']:<15} {row['Unique_Values']:<12}")

    else:
        print("Missing data analysis failed. Please check the file path and format.")