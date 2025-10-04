import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MissingDataAnalyzer:
    def __init__(self, data_path):
        """初始化缺失数据分析器"""
        self.data_path = data_path
        self.data = None
        self.missing_stats = None

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        try:
            self.data = pd.read_excel(self.data_path)
            print(f"数据加载成功！")
            print(f"数据形状: {self.data.shape}")
            print(f"总行数: {self.data.shape[0]}")
            print(f"总列数: {self.data.shape[1]}")
            return self.data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def analyze_missing_data(self):
        """分析每列的缺失数据情况"""
        print("\n" + "="*80)
        print("缺失数据分析")
        print("="*80)

        missing_stats = []

        for column in self.data.columns:
            total_count = len(self.data)
            missing_count = self.data[column].isnull().sum()
            non_missing_count = total_count - missing_count
            missing_percentage = (missing_count / total_count) * 100

            # 获取数据类型
            dtype = str(self.data[column].dtype)

            # 获取唯一值数量（仅计算非缺失值）
            unique_count = self.data[column].nunique()

            # 判断数据类型类别
            if self.data[column].dtype in ['int64', 'float64']:
                data_category = 'Numerical'
                # 计算统计信息
                if missing_count < total_count:
                    mean_val = self.data[column].mean()
                    median_val = self.data[column].median()
                    std_val = self.data[column].std()
                else:
                    mean_val = median_val = std_val = np.nan
            else:
                data_category = 'Categorical'
                mean_val = median_val = std_val = np.nan
                # 获取最常见的值
                if missing_count < total_count:
                    mode_val = self.data[column].mode()
                    most_frequent = mode_val.iloc[0] if len(mode_val) > 0 else None
                else:
                    most_frequent = None

            missing_stats.append({
                'Column': column,
                'Total_Count': total_count,
                'Missing_Count': missing_count,
                'Non_Missing_Count': non_missing_count,
                'Missing_Percentage': missing_percentage,
                'Data_Type': dtype,
                'Data_Category': data_category,
                'Unique_Values': unique_count,
                'Mean': mean_val if data_category == 'Numerical' else None,
                'Median': median_val if data_category == 'Numerical' else None,
                'Std': std_val if data_category == 'Numerical' else None,
                'Most_Frequent': most_frequent if data_category == 'Categorical' else None
            })

        # 创建DataFrame
        self.missing_stats = pd.DataFrame(missing_stats)

        # 按缺失百分比排序
        self.missing_stats_sorted = self.missing_stats.sort_values('Missing_Percentage', ascending=False)

        return self.missing_stats_sorted

    def display_missing_summary(self):
        """显示缺失数据汇总"""
        print(f"\n{'='*120}")
        print("缺失数据详细统计表")
        print('='*120)
        print(f"{'列名':<30} {'缺失数量':<10} {'缺失率%':<10} {'非缺失':<10} {'数据类型':<15} {'唯一值':<10} {'处理建议':<25}")
        print('-'*120)

        for _, row in self.missing_stats_sorted.iterrows():
            # 生成处理建议
            suggestion = self.get_treatment_suggestion(row)

            print(f"{row['Column']:<30} {row['Missing_Count']:<10} {row['Missing_Percentage']:<10.2f} "
                  f"{row['Non_Missing_Count']:<10} {row['Data_Category']:<15} {row['Unique_Values']:<10} {suggestion:<25}")

    def get_treatment_suggestion(self, row):
        """根据缺失情况生成处理建议"""
        missing_pct = row['Missing_Percentage']
        data_category = row['Data_Category']

        if missing_pct == 0:
            return "无需处理"
        elif missing_pct > 70:
            return "考虑删除列"
        elif missing_pct > 50:
            return "重点关注/专门处理"
        elif missing_pct > 20:
            if data_category == 'Numerical':
                return "KNN插补/回归插补"
            else:
                return "众数插补/新类别"
        elif missing_pct > 5:
            if data_category == 'Numerical':
                return "中位数插补"
            else:
                return "众数插补"
        else:
            if data_category == 'Numerical':
                return "均值/中位数插补"
            else:
                return "众数插补"

    def create_visualizations(self):
        """创建缺失数据可视化"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Appendicitis数据集缺失数据分析', fontsize=16, fontweight='bold')

        # 1. 缺失数据柱状图（前20列）
        ax1 = axes[0, 0]
        top_20_missing = self.missing_stats_sorted[self.missing_stats_sorted['Missing_Count'] > 0].head(20)
        if not top_20_missing.empty:
            y_pos = np.arange(len(top_20_missing))
            bars = ax1.barh(y_pos, top_20_missing['Missing_Count'])
            ax1.set_yticks(y_pos)
            # 截断长列名
            labels = [col[:25] + '...' if len(col) > 25 else col for col in top_20_missing['Column']]
            ax1.set_yticklabels(labels, fontsize=8)
            ax1.set_xlabel('缺失数据数量')
            ax1.set_title('前20列缺失数据数量')

            # 根据缺失百分比着色
            for i, bar in enumerate(bars):
                pct = top_20_missing.iloc[i]['Missing_Percentage']
                if pct > 50:
                    bar.set_color('red')
                elif pct > 20:
                    bar.set_color('orange')
                elif pct > 5:
                    bar.set_color('yellow')
                else:
                    bar.set_color('lightblue')
        else:
            ax1.text(0.5, 0.5, '没有发现缺失数据', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('缺失数据数量')

        # 2. 缺失百分比柱状图
        ax2 = axes[0, 1]
        if not top_20_missing.empty:
            bars = ax2.barh(y_pos, top_20_missing['Missing_Percentage'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('缺失百分比 (%)')
            ax2.set_title('前20列缺失数据百分比')

            # 添加百分比标签和着色
            for i, bar in enumerate(bars):
                pct = top_20_missing.iloc[i]['Missing_Percentage']
                if pct > 50:
                    bar.set_color('red')
                elif pct > 20:
                    bar.set_color('orange')
                elif pct > 5:
                    bar.set_color('yellow')
                else:
                    bar.set_color('lightblue')

                # 添加百分比标签
                ax2.text(pct + 1, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=7)
        else:
            ax2.text(0.5, 0.5, '没有发现缺失数据', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('缺失数据百分比')

        # 3. 缺失百分比分布直方图
        ax3 = axes[1, 0]
        missing_percentages = self.missing_stats['Missing_Percentage']
        ax3.hist(missing_percentages, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.set_xlabel('缺失百分比 (%)')
        ax3.set_ylabel('列数量')
        ax3.set_title('缺失数据百分比分布')
        ax3.axvline(missing_percentages.mean(), color='red', linestyle='--',
                   label=f'平均值: {missing_percentages.mean():.1f}%')
        ax3.axvline(missing_percentages.median(), color='green', linestyle='--',
                   label=f'中位数: {missing_percentages.median():.1f}%')
        ax3.legend()

        # 4. 数据完整性概览饼图
        ax4 = axes[1, 1]
        no_missing = len(self.missing_stats[self.missing_stats['Missing_Count'] == 0])
        low_missing = len(self.missing_stats[(self.missing_stats['Missing_Percentage'] > 0) &
                                           (self.missing_stats['Missing_Percentage'] <= 5)])
        medium_missing = len(self.missing_stats[(self.missing_stats['Missing_Percentage'] > 5) &
                                              (self.missing_stats['Missing_Percentage'] <= 20)])
        high_missing = len(self.missing_stats[(self.missing_stats['Missing_Percentage'] > 20) &
                                            (self.missing_stats['Missing_Percentage'] <= 50)])
        very_high_missing = len(self.missing_stats[self.missing_stats['Missing_Percentage'] > 50])

        categories = ['无缺失\n(0%)', '低缺失\n(0-5%)', '中等缺失\n(5-20%)',
                     '高缺失\n(20-50%)', '极高缺失\n(>50%)']
        counts = [no_missing, low_missing, medium_missing, high_missing, very_high_missing]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']

        # 只包含非零类别
        non_zero_categories = []
        non_zero_counts = []
        non_zero_colors = []

        for cat, count, color in zip(categories, counts, colors):
            if count > 0:
                non_zero_categories.append(f'{cat}\n({count}列)')
                non_zero_counts.append(count)
                non_zero_colors.append(color)

        if non_zero_counts:
            wedges, texts, autotexts = ax4.pie(non_zero_counts, labels=non_zero_categories,
                                              colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('列按缺失数据级别分布')
        else:
            ax4.text(0.5, 0.5, '无数据可显示', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('缺失数据分布')

        plt.tight_layout()
        plt.savefig('appendicitis_missing_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("缺失数据可视化已保存为: appendicitis_missing_data_analysis.png")

    def generate_treatment_recommendations(self):
        """生成详细的处理建议"""
        print(f"\n{'='*80}")
        print("缺失数据处理建议")
        print('='*80)

        # 按处理优先级分组
        critical_missing = self.missing_stats[self.missing_stats['Missing_Percentage'] > 50]
        high_missing = self.missing_stats[(self.missing_stats['Missing_Percentage'] > 20) &
                                        (self.missing_stats['Missing_Percentage'] <= 50)]
        medium_missing = self.missing_stats[(self.missing_stats['Missing_Percentage'] > 5) &
                                          (self.missing_stats['Missing_Percentage'] <= 20)]
        low_missing = self.missing_stats[(self.missing_stats['Missing_Percentage'] > 0) &
                                       (self.missing_stats['Missing_Percentage'] <= 5)]

        recommendations = {
            "立即处理（>50%缺失）": critical_missing,
            "重点关注（20-50%缺失）": high_missing,
            "需要处理（5-20%缺失）": medium_missing,
            "简单处理（0-5%缺失）": low_missing
        }

        for category, data in recommendations.items():
            if not data.empty:
                print(f"\n{category}:")
                print("-" * 40)
                for _, row in data.iterrows():
                    suggestion = self.get_detailed_suggestion(row)
                    print(f"• {row['Column']}: {row['Missing_Percentage']:.1f}% 缺失")
                    print(f"  处理方案: {suggestion}")
                    print()

        return recommendations

    def get_detailed_suggestion(self, row):
        """获取详细的处理建议"""
        missing_pct = row['Missing_Percentage']
        data_category = row['Data_Category']
        unique_values = row['Unique_Values']

        suggestions = []

        if missing_pct > 70:
            suggestions.append("考虑删除该列（缺失率过高）")
        elif missing_pct > 50:
            if data_category == 'Numerical':
                suggestions.append("使用KNN插补或创建缺失指示器")
            else:
                suggestions.append("创建'Unknown'类别或使用高级插补方法")
        elif missing_pct > 20:
            if data_category == 'Numerical':
                suggestions.append("使用KNN插补或回归插补")
                suggestions.append("考虑创建缺失值指示器变量")
            else:
                if unique_values < 10:
                    suggestions.append("使用众数插补或创建新类别")
                else:
                    suggestions.append("使用KNN插补或创建'Other'类别")
        elif missing_pct > 5:
            if data_category == 'Numerical':
                suggestions.append("使用中位数插补（对异常值鲁棒）")
            else:
                suggestions.append("使用众数插补")
        else:
            if data_category == 'Numerical':
                suggestions.append("使用均值或中位数插补")
            else:
                suggestions.append("使用众数插补")

        return " | ".join(suggestions)

    def create_imputation_strategy(self):
        """创建插补策略代码"""
        print(f"\n{'='*80}")
        print("推荐的插补策略代码")
        print('='*80)

        strategy_code = """
# 推荐的缺失数据处理策略
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

def handle_missing_data(df):
    \"\"\"处理appendicitis数据集的缺失数据\"\"\"
    df_processed = df.copy()
    
    # 1. 删除缺失率过高的列（>70%）
    high_missing_cols = []
"""

        # 添加需要删除的列
        critical_cols = self.missing_stats[self.missing_stats['Missing_Percentage'] > 70]['Column'].tolist()
        if critical_cols:
            for col in critical_cols:
                strategy_code += f"    high_missing_cols.append('{col}')\n"
            strategy_code += "    df_processed = df_processed.drop(columns=high_missing_cols)\n"
        else:
            strategy_code += "    # 没有发现需要删除的高缺失列\n"

        strategy_code += """
    # 2. 数值型特征插补
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    missing_numerical = []
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            missing_pct = (df_processed[col].isnull().sum() / len(df_processed)) * 100
            if missing_pct > 20:
                # 高缺失率：使用KNN插补
                missing_numerical.append((col, 'knn'))
            elif missing_pct > 5:
                # 中等缺失率：使用中位数
                missing_numerical.append((col, 'median'))
            else:
                # 低缺失率：使用均值
                missing_numerical.append((col, 'mean'))
    
    # 执行数值型插补
    for col, method in missing_numerical:
        if method == 'knn':
            # KNN插补（较复杂，这里用中位数代替）
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        elif method == 'median':
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:  # mean
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
    
    # 3. 分类特征插补
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            missing_pct = (df_processed[col].isnull().sum() / len(df_processed)) * 100
            if missing_pct > 20:
                # 高缺失率：创建新类别
                df_processed[col].fillna('Unknown', inplace=True)
            else:
                # 低缺失率：使用众数
                mode_val = df_processed[col].mode()
                if len(mode_val) > 0:
                    df_processed[col].fillna(mode_val[0], inplace=True)
                else:
                    df_processed[col].fillna('Unknown', inplace=True)
    
    return df_processed

# 使用示例
# df_clean = handle_missing_data(df_original)
"""

        print(strategy_code)

        # 保存策略代码到文件
        with open('missing_data_strategy.py', 'w', encoding='utf-8') as f:
            f.write(strategy_code)

        print("插补策略代码已保存到: missing_data_strategy.py")

    def save_results(self):
        """保存分析结果"""
        # 保存详细的缺失数据统计
        self.missing_stats_sorted.to_csv('appendicitis_missing_data_report.csv',
                                        index=False, encoding='utf-8-sig')

        # 创建简化的汇总报告
        summary_stats = self.missing_stats_sorted[['Column', 'Missing_Count', 'Missing_Percentage',
                                                 'Data_Category', 'Unique_Values']].copy()
        summary_stats['Treatment_Suggestion'] = summary_stats.apply(
            lambda row: self.get_treatment_suggestion(row), axis=1
        )

        summary_stats.to_csv('appendicitis_missing_summary.csv',
                           index=False, encoding='utf-8-sig')

        print(f"\n分析结果已保存:")
        print(f"  - appendicitis_missing_data_report.csv (详细报告)")
        print(f"  - appendicitis_missing_summary.csv (汇总报告)")

def main():
    """主执行函数"""
    print("="*80)
    print("APPENDICITIS数据集缺失数据分析")
    print("="*80)

    # 初始化分析器
    analyzer = MissingDataAnalyzer('data/appendicitis/app_data.xlsx')

    # 加载数据
    data = analyzer.load_data()
    if data is None:
        return

    # 分析缺失数据
    missing_stats = analyzer.analyze_missing_data()

    # 显示汇总信息
    print(f"\n汇总统计:")
    print(f"- 总列数: {len(missing_stats)}")
    print(f"- 有缺失数据的列: {len(missing_stats[missing_stats['Missing_Count'] > 0])}")
    print(f"- 完整列数: {len(missing_stats[missing_stats['Missing_Count'] == 0])}")
    print(f"- 平均缺失率: {missing_stats['Missing_Percentage'].mean():.2f}%")
    print(f"- 最高缺失率: {missing_stats['Missing_Percentage'].max():.2f}%")

    # 显示详细统计
    analyzer.display_missing_summary()

    # 生成处理建议
    recommendations = analyzer.generate_treatment_recommendations()

    # 创建可视化
    analyzer.create_visualizations()

    # 创建插补策略
    analyzer.create_imputation_strategy()

    # 保存结果
    analyzer.save_results()

    print(f"\n{'='*80}")
    print("缺失数据分析完成！")
    print("="*80)
    print("生成的文件:")
    print("  - appendicitis_missing_data_analysis.png (可视化图表)")
    print("  - appendicitis_missing_data_report.csv (详细报告)")
    print("  - appendicitis_missing_summary.csv (汇总报告)")
    print("  - missing_data_strategy.py (处理策略代码)")

if __name__ == "__main__":
    main()