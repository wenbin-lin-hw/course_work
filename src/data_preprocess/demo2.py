import pandas as pd
import numpy as np


def identify_categorical_columns(data_path):
    """识别appendicitis数据中的分类型列"""

    print("=" * 80)
    print("APPENDICITIS数据分类型列识别")
    print("=" * 80)

    # 读取数据
    try:
        df = pd.read_excel(data_path)
        print(f"✓ 数据加载成功！")
        print(f"数据形状: {df.shape}")
        print(f"总列数: {df.shape[1]}")
        print()
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None

    # 显示前几行数据
    print("数据预览 (前5行):")
    print("-" * 50)
    print(df.head())
    print()

    # 分析每列的数据类型和特征
    print("所有列的基本信息:")
    print("-" * 80)
    print(f"{'列名':<30} {'数据类型':<12} {'唯一值数量':<12} {'缺失值':<8} {'分类判断':<10}")
    print("-" * 80)

    categorical_columns = []
    numerical_columns = []
    date_columns = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()

        # 判断是否为分类型数据的逻辑
        is_categorical = False
        category_reason = ""

        # 1. 明确的object类型（字符串）
        if df[col].dtype == 'object':
            is_categorical = True
            category_reason = "字符串类型"

        # 2. 数值型但唯一值很少（通常≤10被认为是分类）
        elif df[col].dtype in ['int64', 'float64'] and unique_count <= 10:
            is_categorical = True
            category_reason = f"数值型但只有{unique_count}个唯一值"

        # 3. 布尔型
        elif df[col].dtype == 'bool':
            is_categorical = True
            category_reason = "布尔类型"

        # 4. 日期时间类型
        elif 'datetime' in str(df[col].dtype).lower() or 'date' in col.lower():
            date_columns.append(col)
            category_reason = "日期类型"

        # 5. 其他数值型
        else:
            numerical_columns.append(col)
            category_reason = "数值型"

        if is_categorical and col not in date_columns:
            categorical_columns.append(col)

        # 显示信息
        status = "分类型" if is_categorical and col not in date_columns else (
            "日期型" if col in date_columns else "数值型")
        print(f"{col:<30} {dtype:<12} {unique_count:<12} {missing_count:<8} {status:<10}")

    print()

    # 详细分析分类型列
    print("=" * 80)
    print(f"分类型列详细分析 (共{len(categorical_columns)}列)")
    print("=" * 80)

    for i, col in enumerate(categorical_columns, 1):
        print(f"\n{i}. 列名: {col}")
        print(f"   数据类型: {df[col].dtype}")
        print(f"   唯一值数量: {df[col].nunique()}")
        print(f"   缺失值数量: {df[col].isnull().sum()}")
        print(f"   缺失率: {(df[col].isnull().sum() / len(df)) * 100:.1f}%")

        # 显示唯一值（如果不太多的话）
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= 20:
            print(f"   唯一值: {list(unique_values)}")
        else:
            print(f"   唯一值太多，显示前10个: {list(unique_values[:10])}")

        # 显示值分布
        value_counts = df[col].value_counts()
        print(f"   值分布:")
        for value, count in value_counts.head(5).items():
            percentage = (count / len(df)) * 100
            print(f"     {value}: {count} ({percentage:.1f}%)")

        if len(value_counts) > 5:
            print(f"     ... 还有 {len(value_counts) - 5} 个其他值")

    # 汇总统计
    print("\n" + "=" * 80)
    print("列类型汇总统计")
    print("=" * 80)
    print(f"总列数:     {df.shape[1]}")
    print(f"分类型列:   {len(categorical_columns)} 个")
    print(f"数值型列:   {len(numerical_columns)} 个")
    print(f"日期型列:   {len(date_columns)} 个")

    print(f"\n分类型列名列表:")
    print("-" * 30)
    for i, col in enumerate(categorical_columns, 1):
        print(f"{i:2d}. {col}")

    print(f"\n数值型列名列表:")
    print("-" * 30)
    for i, col in enumerate(numerical_columns, 1):
        print(f"{i:2d}. {col}")

    if date_columns:
        print(f"\n日期型列名列表:")
        print("-" * 30)
        for i, col in enumerate(date_columns, 1):
            print(f"{i:2d}. {col}")

    # 保存结果
    result_dict = {
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'date_columns': date_columns,
        'total_columns': df.shape[1],
        'total_rows': df.shape[0]
    }

    # 保存到CSV文件
    column_analysis = []
    for col in df.columns:
        col_type = 'categorical' if col in categorical_columns else (
            'numerical' if col in numerical_columns else 'date')
        column_analysis.append({
            'Column_Name': col,
            'Data_Type': str(df[col].dtype),
            'Column_Category': col_type,
            'Unique_Values': df[col].nunique(),
            'Missing_Values': df[col].isnull().sum(),
            'Missing_Percentage': (df[col].isnull().sum() / len(df)) * 100
        })

    analysis_df = pd.DataFrame(column_analysis)
    analysis_df.to_csv('appendicitis_column_analysis.csv', index=False, encoding='utf-8-sig')

    print(f"\n详细分析结果已保存到: appendicitis_column_analysis.csv")

    return result_dict


# 主程序
if __name__ == "__main__":
    data_path = '../../data/appendicitis/app_data.xlsx'
    result = identify_categorical_columns(data_path)

    if result:
        print(f"\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        print(f"共识别出 {len(result['categorical_columns'])} 个分类型列")

        # 生成Python代码格式的列名列表
        print(f"\n可复制的Python列表格式:")
        print("categorical_columns = [")
        for col in result['categorical_columns']:
            print(f"    '{col}',")
        print("]")