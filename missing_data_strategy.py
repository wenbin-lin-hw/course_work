
# 推荐的缺失数据处理策略
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

def handle_missing_data():
    """处理appendicitis数据集的缺失数据"""
    df = pd.read_excel('data/appendicitis/app_data.xlsx', names=['All cases'])
    df_processed = df.copy()
    
    # 1. 删除缺失率过高的列（>70%）
    high_missing_cols = []
    # high_missing_cols.append('Segmented_Neutrophils')
    # high_missing_cols.append('Appendix_Wall_Layers')
    # high_missing_cols.append('Target_Sign')
    # high_missing_cols.append('Appendicolith')
    # high_missing_cols.append('Perfusion')
    # high_missing_cols.append('Perforation')
    # high_missing_cols.append('Appendicular_Abscess')
    # high_missing_cols.append('Abscess_Location')
    # high_missing_cols.append('Pathological_Lymph_Nodes')
    # high_missing_cols.append('Lymph_Nodes_Location')
    # high_missing_cols.append('Bowel_Wall_Thickening')
    # high_missing_cols.append('Conglomerate_of_Bowel_Loops')
    high_missing_cols.append('Ileus')
    # high_missing_cols.append('Coprostasis')
    # high_missing_cols.append('Meteorism')
    # high_missing_cols.append('Enteritis')
    # high_missing_cols.append('Gynecological_Findings')
    df_processed = df_processed.drop(high_missing_cols,axis=1)

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

    df_processed.to_excel("data/appendicitis/app_data_processed.xlsx")
    return df_processed


handle_missing_data()

# 使用示例
# df_clean = handle_missing_data(df_original)
