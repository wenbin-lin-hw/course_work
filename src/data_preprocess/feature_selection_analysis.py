import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    f_classif,
    chi2,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, data_path):
        """Initialize feature selector with data path"""
        self.data_path = data_path
        self.data = None
        self.target_columns = ['Diagnosis', 'Severity', 'Management']
        self.feature_scores = {}

    def load_data(self):
        """Load and preprocess the data"""
        print("Loading data...")
        self.data = pd.read_excel(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

        # Display basic info about target variables
        for target in self.target_columns:
            if target in self.data.columns:
                print(f"\n{target} distribution:")
                print(self.data[target].value_counts())

        return self.data

    def preprocess_features(self):
        """Preprocess features for analysis"""
        # Get feature columns (exclude targets and ID columns)
        exclude_cols = self.target_columns + ['Patient_ID', 'ID','US_Number','Appendix_on_US']
        self.feature_columns = [col for col in self.data.columns
                               if col not in exclude_cols and col in self.data.columns]

        print(f"\nTotal feature columns for analysis: {len(self.feature_columns)}")

        # Handle missing values
        feature_data = self.data[self.feature_columns].copy()

        # Fill missing values with appropriate strategies
        for col in feature_data.columns:
            if feature_data[col].dtype in ['int64', 'float64']:
                # Numerical: fill with median
                feature_data[col].fillna(feature_data[col].median(), inplace=True)
            else:
                # Categorical: fill with mode
                feature_data[col].fillna(feature_data[col].mode()[0], inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for col in feature_data.columns:
            if feature_data[col].dtype == 'object':
                le = LabelEncoder()
                feature_data[col] = le.fit_transform(feature_data[col].astype(str))
                label_encoders[col] = le

        self.processed_features = feature_data
        self.label_encoders = label_encoders

        return feature_data

    def mutual_information_selection(self, target):
        """Select features using mutual information"""
        print(f"\nCalculating mutual information for {target}...")

        # Prepare target variable
        y = self.data[target].copy()
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Remove missing target values
        valid_idx = ~pd.isna(y)
        X = self.processed_features[valid_idx]
        y = y[valid_idx]

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Create feature importance dataframe
        mi_df = pd.DataFrame({
            'feature': self.feature_columns,
            'mutual_info_score': mi_scores
        }).sort_values('mutual_info_score', ascending=False)

        return mi_df

    def correlation_selection(self, target):
        """Select features using correlation analysis"""
        print(f"\nCalculating correlations for {target}...")

        # Prepare target variable
        y = self.data[target].copy()
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Remove missing target values
        valid_idx = ~pd.isna(y)
        X = self.processed_features[valid_idx]
        y = y[valid_idx]

        # Calculate correlations
        correlations = []
        for col in self.feature_columns:
            try:
                # Pearson correlation
                pearson_corr, _ = pearsonr(X[col], y)
                correlations.append({
                    'feature': col,
                    'pearson_correlation': abs(pearson_corr)
                })
            except:
                correlations.append({
                    'feature': col,
                    'pearson_correlation': 0
                })

        corr_df = pd.DataFrame(correlations).sort_values(
            'pearson_correlation', ascending=False
        )

        return corr_df

    def random_forest_selection(self, target):
        """Select features using Random Forest importance"""
        print(f"\nCalculating Random Forest importance for {target}...")

        # Prepare target variable
        y = self.data[target].copy()
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Remove missing target values
        valid_idx = ~pd.isna(y)
        X = self.processed_features[valid_idx]
        y = y[valid_idx]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importances
        rf_df = pd.DataFrame({
            'feature': self.feature_columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)

        return rf_df

    def univariate_selection(self, target):
        """Select features using univariate statistical tests"""
        print(f"\nCalculating univariate scores for {target}...")

        # Prepare target variable
        y = self.data[target].copy()
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Remove missing target values
        valid_idx = ~pd.isna(y)
        X = self.processed_features[valid_idx]
        y = y[valid_idx]

        # Use f_classif for numerical features
        f_scores, f_pvalues = f_classif(X, y)

        univariate_df = pd.DataFrame({
            'feature': self.feature_columns,
            'f_score': f_scores,
            'f_pvalue': f_pvalues
        }).sort_values('f_score', ascending=False)

        return univariate_df

    def aggregate_feature_scores(self):
        """Aggregate feature scores from all methods"""
        print("\n" + "="*60)
        print("AGGREGATING FEATURE SCORES FROM ALL METHODS")
        print("="*60)

        aggregated_scores = {}

        for target in self.target_columns:
            if target not in self.data.columns:
                continue

            print(f"\nProcessing target: {target}")

            # Get scores from all methods
            mi_scores = self.mutual_information_selection(target)
            corr_scores = self.correlation_selection(target)
            rf_scores = self.random_forest_selection(target)
            univariate_scores = self.univariate_selection(target)

            # Normalize scores to [0, 1] range
            mi_scores['mi_normalized'] = (mi_scores['mutual_info_score'] - mi_scores['mutual_info_score'].min()) / (mi_scores['mutual_info_score'].max() - mi_scores['mutual_info_score'].min())
            corr_scores['corr_normalized'] = corr_scores['pearson_correlation']  # Already in [0, 1]
            rf_scores['rf_normalized'] = (rf_scores['rf_importance'] - rf_scores['rf_importance'].min()) / (rf_scores['rf_importance'].max() - rf_scores['rf_importance'].min())
            univariate_scores['f_normalized'] = (univariate_scores['f_score'] - univariate_scores['f_score'].min()) / (univariate_scores['f_score'].max() - univariate_scores['f_score'].min())

            # Merge all scores
            merged = mi_scores[['feature', 'mi_normalized']].copy()
            merged = merged.merge(corr_scores[['feature', 'corr_normalized']], on='feature')
            merged = merged.merge(rf_scores[['feature', 'rf_normalized']], on='feature')
            merged = merged.merge(univariate_scores[['feature', 'f_normalized']], on='feature')

            # Calculate weighted average (you can adjust weights)
            weights = {
                'mi_normalized': 0.3,
                'corr_normalized': 0.2,
                'rf_normalized': 0.3,
                'f_normalized': 0.2
            }

            merged['weighted_score'] = (
                merged['mi_normalized'] * weights['mi_normalized'] +
                merged['corr_normalized'] * weights['corr_normalized'] +
                merged['rf_normalized'] * weights['rf_normalized'] +
                merged['f_normalized'] * weights['f_normalized']
            )

            merged = merged.sort_values('weighted_score', ascending=False)
            aggregated_scores[target] = merged

            # Display top 10 for this target
            print(f"\nTop 10 features for {target}:")
            for i, (_, row) in enumerate(merged.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:<30} | Score: {row['weighted_score']:.4f}")

        self.aggregated_scores = aggregated_scores
        return aggregated_scores

    def select_top_features(self, n_features=14):
        """Select top N features across all targets"""
        print(f"\n" + "="*60)
        print(f"SELECTING TOP {n_features} FEATURES ACROSS ALL TARGETS")
        print("="*60)

        # Calculate overall importance by averaging across targets
        feature_overall_scores = {}

        for feature in self.feature_columns:
            scores = []
            for target in self.target_columns:
                if target in self.aggregated_scores:
                    target_scores = self.aggregated_scores[target]
                    if feature in target_scores['feature'].values:
                        score = target_scores[target_scores['feature'] == feature]['weighted_score'].iloc[0]
                        scores.append(score)

            if scores:
                feature_overall_scores[feature] = np.mean(scores)
            else:
                feature_overall_scores[feature] = 0

        # Sort features by overall score
        sorted_features = sorted(
            feature_overall_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top N features
        top_features = [feat[0] for feat in sorted_features[:n_features]]
        top_scores = [feat[1] for feat in sorted_features[:n_features]]

        print(f"\nTOP {n_features} SELECTED FEATURES:")
        print("-" * 60)
        for i, (feature, score) in enumerate(zip(top_features, top_scores)):
            print(f"{i+1:2d}. {feature:<35} | Overall Score: {score:.4f}")

        return top_features, top_scores

    def create_visualizations(self, top_features, top_scores):
        """Create visualizations for feature selection results"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Selection Analysis Results', fontsize=16, fontweight='bold')

        # 1. Top features bar plot
        ax1 = axes[0, 0]
        y_pos = np.arange(len(top_features))
        bars = ax1.barh(y_pos, top_scores, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([feat[:20] + '...' if len(feat) > 20 else feat for feat in top_features], fontsize=8)
        ax1.set_xlabel('Overall Importance Score')
        ax1.set_title('Top 14 Selected Features')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'{top_scores[i]:.3f}', ha='left', va='center', fontsize=7)

        # 2. Feature importance by target
        ax2 = axes[0, 1]
        target_means = []
        target_names = []
        for target in self.target_columns:
            if target in self.aggregated_scores:
                target_df = self.aggregated_scores[target]
                mean_score = target_df['weighted_score'].mean()
                target_means.append(mean_score)
                target_names.append(target)

        bars = ax2.bar(target_names, target_means, color=['lightcoral', 'lightgreen', 'lightsalmon'])
        ax2.set_ylabel('Average Feature Importance')
        ax2.set_title('Average Feature Importance by Target')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, target_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')

        # 3. Method comparison heatmap
        ax3 = axes[1, 0]
        # Create heatmap data for top 10 features
        heatmap_data = []
        methods = ['mi_normalized', 'corr_normalized', 'rf_normalized', 'f_normalized']
        method_labels = ['Mutual Info', 'Correlation', 'Random Forest', 'F-statistic']

        # Use first target's data for demonstration
        first_target = list(self.aggregated_scores.keys())[0]
        top_10_data = self.aggregated_scores[first_target].head(10)

        for _, row in top_10_data.iterrows():
            heatmap_data.append([row[method] for method in methods])

        im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(len(method_labels)))
        ax3.set_xticklabels(method_labels, rotation=45, ha='right')
        ax3.set_yticks(range(len(top_10_data)))
        ax3.set_yticklabels([feat[:15] + '...' if len(feat) > 15 else feat
                           for feat in top_10_data['feature']], fontsize=8)
        ax3.set_title(f'Feature Selection Methods Comparison\n(Top 10 for {first_target})')
        plt.colorbar(im, ax=ax3)

        # 4. Score distribution
        ax4 = axes[1, 1]
        ax4.hist(top_scores, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.set_xlabel('Importance Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Top 14 Feature Scores')
        ax4.axvline(np.mean(top_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(top_scores):.3f}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nVisualization saved as: feature_selection_analysis.png")

    def save_results(self, top_features, top_scores):
        """Save feature selection results"""
        # Create results dataframe
        results_df = pd.DataFrame({
            'rank': range(1, len(top_features) + 1),
            'feature': top_features,
            'overall_score': top_scores
        })

        # Add detailed scores for each method
        first_target = list(self.aggregated_scores.keys())[0]
        detailed_scores = self.aggregated_scores[first_target]

        results_df = results_df.merge(
            detailed_scores[['feature', 'mi_normalized', 'corr_normalized',
                           'rf_normalized', 'f_normalized']],
            on='feature',
            how='left'
        )

        # Save to CSV
        results_df.to_csv('top_14_selected_features.csv', index=False)
        print(f"\nDetailed results saved to: top_14_selected_features.csv")

        # Save feature list for easy import
        with open('selected_features_list.py', 'w') as f:
            f.write("# Top 14 selected features for CNN processing\n")
            f.write("# Generated by feature_selection_analysis.py\n\n")
            f.write("SELECTED_FEATURES = [\n")
            for feature in top_features:
                f.write(f"    '{feature}',\n")
            f.write("]\n")

        print(f"Feature list saved to: selected_features_list.py")

        return results_df

def main():
    """Main execution function"""
    print("="*80)
    print("APPENDICITIS DATASET - FEATURE SELECTION ANALYSIS")
    print("="*80)
    base_path = os.getcwd()

    file_path = base_path + "/data/appendicitis/app_data.xlsx"
    print(file_path)

    # Initialize feature selector
    selector = FeatureSelector(file_path)

    # Load and preprocess data
    data = selector.load_data()
    processed_features = selector.preprocess_features()

    # Perform feature selection analysis
    aggregated_scores = selector.aggregate_feature_scores()

    # Select top 14 features
    top_features, top_scores = selector.select_top_features(n_features=14)

    # Create visualizations
    selector.create_visualizations(top_features, top_scores)

    # Save results
    results_df = selector.save_results(top_features, top_scores)

    print("\n" + "="*80)
    print("FEATURE SELECTION ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Selected {len(top_features)} features for CNN processing")
    print("Files generated:")
    print("  - feature_selection_analysis.png (visualization)")
    print("  - top_14_selected_features.csv (detailed results)")
    print("  - selected_features_list.py (Python list)")

    return top_features, results_df

if __name__ == "__main__":
    selected_features, results = main()