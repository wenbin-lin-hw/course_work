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
    分析Excel文件中每一列与Diagnosis的相关性
    """
    try:
        # 读取Excel文件
        print("正在读取Excel文件...")
        df = pd.read_excel(file_path)
        
        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())
        
        # 检查是否存在Diagnosis列
        if 'Diagnosis' not in df.columns:
            # 尝试寻找类似的列名
            diagnosis_cols = [col for col in df.columns if 'diagnosis' in col.lower()]
            if diagnosis_cols:
                print(f"未找到'Diagnosis'列，但找到相似列: {diagnosis_cols}")
                print("请确认目标列名")
                return None
            else:
                print("未找到Diagnosis相关列，显示所有列名供参考:")
                for i, col in enumerate(df.columns):
                    print(f"{i}: {col}")
                return None
        
        # 获取目标变量
        target = df['Diagnosis']
        features = df.drop('Diagnosis', axis=1)
        
        print(f"\nDiagnosis列的值分布:")
        print(target.value_counts())
        
        # 存储相关性结果
        correlation_results = []
        
        print("\n开始分析各列与Diagnosis的相关性...")

        for col in features.columns:
            print(f"\n分析列: {col}")

            # 跳过完全为空的列
            if features[col].isnull().all():
                print(f"  {col}: 所有值都为空，跳过")
                continue

            # 检查数据类型
            col_data = features[col].dropna()
            target_aligned = target[col_data.index]

            if len(col_data) == 0:
                print(f"  {col}: 没有有效数据，跳过")
                continue

            result = {
                'feature': col,
                'data_type': str(col_data.dtype),
                'non_null_count': len(col_data),
                'unique_values': col_data.nunique()
            }

            # 根据数据类型选择相关性分析方法
            if pd.api.types.is_numeric_dtype(col_data):
                # 数值型数据 - 使用Pearson和Spearman相关系数
                try:
                    pearson_corr, pearson_p = pearsonr(col_data, target_aligned)
                    spearman_corr, spearman_p = spearmanr(col_data, target_aligned)

                    result.update({
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'method': 'numeric'
                    })

                    print(f"  Pearson相关系数: {pearson_corr:.4f} (p={pearson_p:.4f})")
                    print(f"  Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.4f})")

                except Exception as e:
                    print(f"  数值相关性计算出错: {e}")
                    result.update({
                        'pearson_correlation': np.nan,
                        'pearson_p_value': np.nan,
                        'spearman_correlation': np.nan,
                        'spearman_p_value': np.nan,
                        'method': 'numeric_error'
                    })

            else:
                # 分类数据 - 使用卡方检验
                try:
                    # 创建交叉表
                    contingency_table = pd.crosstab(col_data, target_aligned)
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                    # 计算Cramér's V (标准化的卡方统计量)
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                    result.update({
                        'chi2_statistic': chi2,
                        'chi2_p_value': p_value,
                        'cramers_v': cramers_v,
                        'method': 'categorical'
                    })

                    print(f"  卡方统计量: {chi2:.4f} (p={p_value:.4f})")
                    print(f"  Cramér's V: {cramers_v:.4f}")
                    print(f"  交叉表形状: {contingency_table.shape}")

                except Exception as e:
                    print(f"  分类相关性计算出错: {e}")
                    result.update({
                        'chi2_statistic': np.nan,
                        'chi2_p_value': np.nan,
                        'cramers_v': np.nan,
                        'method': 'categorical_error'
                    })

            correlation_results.append(result)

        for col in features.columns:
            print(f"\n分析列: {col}")

            # 跳过完全为空的列
            if features[col].isnull().all():
                print(f"  {col}: 所有值都为空，跳过")
                continue

            # 获取非空数据
            col_data = features[col].dropna()
            target_aligned = target[col_data.index]

            if len(col_data) == 0:
                print(f"  {col}: 没有有效数据，跳过")
                continue

            # 显示数据的基本信息
            print(f"  数据类型: {col_data.dtype}")
            print(f"  非空值数量: {len(col_data)}")
            print(f"  唯一值数量: {col_data.nunique()}")
            print(f"  前几个值: {list(col_data.head())}")

            result = {
                'feature': col,
                'data_type': str(col_data.dtype),
                'non_null_count': len(col_data),
                'unique_values': col_data.nunique()
            }

            # 尝试将数据转换为数值型
            is_numeric = False
            numeric_col_data = None

            try:
                # 先尝试直接转换
                if pd.api.types.is_numeric_dtype(col_data):
                    numeric_col_data = col_data.astype(float)
                    is_numeric = True
                else:
                    # 尝试转换字符串为数值
                    numeric_col_data = pd.to_numeric(col_data, errors='coerce')
                    # 如果转换后还有足够的非空数值，则认为是数值型
                    if numeric_col_data.notna().sum() > len(col_data) * 0.5:  # 至少50%的数据可以转换为数值
                        is_numeric = True
                        # 重新对齐目标变量
                        valid_indices = numeric_col_data.notna()
                        numeric_col_data = numeric_col_data[valid_indices]
                        target_aligned = target_aligned[valid_indices]
                        print(f"  成功转换为数值型，有效数据: {len(numeric_col_data)}")
                    else:
                        print(f"  无法转换为数值型 (只有{numeric_col_data.notna().sum()}个有效数值)")
            except Exception as e:
                print(f"  数值转换失败: {e}")
                is_numeric = False

            if is_numeric and len(numeric_col_data) > 1:
                # 数值型数据 - 使用Pearson和Spearman相关系数
                try:
                    # 确保目标变量也是数值型
                    if not pd.api.types.is_numeric_dtype(target_aligned):
                        # 尝试将目标变量编码为数值
                        le = LabelEncoder()
                        target_numeric = le.fit_transform(target_aligned)
                        print(f"  目标变量编码: {dict(zip(le.classes_, range(len(le.classes_))))}")
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

                    print(f"  Pearson相关系数: {pearson_corr:.4f} (p={pearson_p:.4f})")
                    print(f"  Spearman相关系数: {spearman_corr:.4f} (p={spearman_p:.4f})")

                except Exception as e:
                    print(f"  数值相关性计算出错: {e}")
                    result.update({
                        'pearson_correlation': np.nan,
                        'pearson_p_value': np.nan,
                        'spearman_correlation': np.nan,
                        'spearman_p_value': np.nan,
                        'method': 'numeric_error'
                    })

            else:
                # 分类数据 - 使用卡方检验
                try:
                    # 确保数据为字符串类型以进行交叉表分析
                    categorical_col_data = col_data.astype(str)
                    categorical_target = target_aligned.astype(str)

                    # 创建交叉表
                    contingency_table = pd.crosstab(categorical_col_data, categorical_target)
                    print(f"  交叉表形状: {contingency_table.shape}")

                    # 检查交叉表是否有效（至少2x2）
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                        # 计算Cramér's V (标准化的卡方统计量)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

                        result.update({
                            'chi2_statistic': chi2,
                            'chi2_p_value': p_value,
                            'cramers_v': cramers_v,
                            'method': 'categorical'
                        })

                        print(f"  卡方统计量: {chi2:.4f} (p={p_value:.4f})")
                        print(f"  Cramér's V: {cramers_v:.4f}")
                    else:
                        print(f"  交叉表太小，无法进行卡方检验")
                        result.update({
                            'chi2_statistic': np.nan,
                            'chi2_p_value': np.nan,
                            'cramers_v': np.nan,
                            'method': 'categorical_insufficient'
                        })

                except Exception as e:
                    print(f"  分类相关性计算出错: {e}")
                    result.update({
                        'chi2_statistic': np.nan,
                        'chi2_p_value': np.nan,
                        'cramers_v': np.nan,
                        'method': 'categorical_error'
                    })

            correlation_results.append(result)

        # 创建结果DataFrame
        results_df = pd.DataFrame(correlation_results)
        
        # 显示汇总结果
        print("\n" + "="*60)
        print("相关性分析汇总结果")
        print("="*60)
        
        # 数值型特征的相关性排序
        numeric_results = results_df[results_df['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            print("\n数值型特征与Diagnosis的相关性 (按Pearson相关系数绝对值排序):")
            numeric_results['abs_pearson'] = numeric_results['pearson_correlation'].abs()
            numeric_sorted = numeric_results.sort_values('abs_pearson', ascending=False)
            
            for _, row in numeric_sorted.iterrows():
                print(f"  {row['feature']:<30} | Pearson: {row['pearson_correlation']:8.4f} | "
                      f"Spearman: {row['spearman_correlation']:8.4f} | "
                      f"P-value: {row['pearson_p_value']:8.4f}")
        
        # 分类型特征的相关性排序
        categorical_results = results_df[results_df['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            print("\n分类型特征与Diagnosis的相关性 (按Cramér's V排序):")
            categorical_sorted = categorical_results.sort_values('cramers_v', ascending=False)
            
            for _, row in categorical_sorted.iterrows():
                print(f"  {row['feature']:<30} | Cramér's V: {row['cramers_v']:8.4f} | "
                      f"Chi2 P-value: {row['chi2_p_value']:8.4f}")
        
        # 保存详细结果到CSV
        results_df.to_csv('appendicitis_correlation_results.csv', index=False, encoding='utf-8')
        print(f"\n详细结果已保存到: appendicitis_correlation_results.csv")
        
        # 创建可视化图表
        create_visualizations(df, results_df)
        
        return results_df, df
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return None, None

def create_visualizations(df, results_df):
    """
    创建相关性分析的可视化图表
    """
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('阑尾炎数据相关性分析结果', fontsize=16, fontweight='bold')
        
        # 1. 数值型特征相关系数条形图
        numeric_results = results_df[results_df['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            numeric_results = numeric_results.dropna(subset=['pearson_correlation'])
            if not numeric_results.empty:
                ax1 = axes[0, 0]
                y_pos = np.arange(len(numeric_results))
                bars = ax1.barh(y_pos, numeric_results['pearson_correlation'])
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(numeric_results['feature'], fontsize=10)
                ax1.set_xlabel('Pearson相关系数')
                ax1.set_title('数值型特征与Diagnosis的相关性')
                ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # 为条形图添加颜色映射
                for i, bar in enumerate(bars):
                    corr_val = numeric_results.iloc[i]['pearson_correlation']
                    if abs(corr_val) > 0.5:
                        bar.set_color('red')
                    elif abs(corr_val) > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightblue')
        
        # 2. 分类型特征Cramér's V条形图
        categorical_results = results_df[results_df['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            categorical_results = categorical_results.dropna(subset=['cramers_v'])
            if not categorical_results.empty:
                ax2 = axes[0, 1]
                y_pos = np.arange(len(categorical_results))
                bars = ax2.barh(y_pos, categorical_results['cramers_v'])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(categorical_results['feature'], fontsize=10)
                ax2.set_xlabel("Cramér's V")
                ax2.set_title('分类型特征与Diagnosis的相关性')
                
                # 为条形图添加颜色映射
                for i, bar in enumerate(bars):
                    v_val = categorical_results.iloc[i]['cramers_v']
                    if v_val > 0.5:
                        bar.set_color('red')
                    elif v_val > 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightblue')
        
        # 3. Diagnosis分布饼图
        ax3 = axes[1, 0]
        diagnosis_counts = df['Diagnosis'].value_counts()
        ax3.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%')
        ax3.set_title('Diagnosis分布')
        
        # 4. 数据质量概览
        ax4 = axes[1, 1]
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage')
        
        if not missing_df.empty:
            y_pos = np.arange(len(missing_df))
            ax4.barh(y_pos, missing_df['Missing_Percentage'])
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(missing_df.index, fontsize=8)
            ax4.set_xlabel('缺失值百分比 (%)')
            ax4.set_title('各列缺失值情况')
        else:
            ax4.text(0.5, 0.5, '无缺失值', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('数据质量：完整')
        
        plt.tight_layout()
        plt.savefig('appendicitis_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存为: appendicitis_correlation_analysis.png")
        
    except Exception as e:
        print(f"创建可视化图表时出现错误: {e}")

if __name__ == "__main__":
    # 分析appendicitis数据
    file_path = "data/appendicitis/app_data.xlsx"
    results, data = analyze_correlation_with_diagnosis(file_path)
    
    if results is not None:
        print("\n分析完成！")
        print(f"共分析了 {len(results)} 个特征与Diagnosis的相关性")
        
        # 显示最强相关性的特征
        print("\n最强相关性特征总结:")
        
        # 数值型特征
        numeric_results = results[results['method'] == 'numeric'].copy()
        if not numeric_results.empty:
            numeric_results = numeric_results.dropna(subset=['pearson_correlation'])
            if not numeric_results.empty:
                top_numeric = numeric_results.loc[numeric_results['pearson_correlation'].abs().idxmax()]
                print(f"数值型最强相关: {top_numeric['feature']} "
                      f"(Pearson相关系数: {top_numeric['pearson_correlation']:.4f})")
        
        # 分类型特征
        categorical_results = results[results['method'] == 'categorical'].copy()
        if not categorical_results.empty:
            categorical_results = categorical_results.dropna(subset=['cramers_v'])
            if not categorical_results.empty:
                top_categorical = categorical_results.loc[categorical_results['cramers_v'].idxmax()]
                print(f"分类型最强相关: {top_categorical['feature']} "
                      f"(Cramér's V: {top_categorical['cramers_v']:.4f})")
    else:
        print("分析失败，请检查文件路径和数据格式")