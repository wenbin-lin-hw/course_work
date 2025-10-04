import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_appendicitis_data():
    """
    读取appendicitis文件夹中的数据文件，分析Diagnosis、Severity、Management列的分布
    """
    try:
        # 读取Excel文件
        file_path = 'data/appendicitis/app_data.xlsx'

        # 尝试读取Excel文件
        if os.path.exists(file_path):
            print(f"正在读取文件: {file_path}")
            df = pd.read_excel(file_path)
            print(f"数据形状: {df.shape}")
            print("\n列名:")
            print(df.columns.tolist())
            print("\n前5行数据:")
            print(df.head())

        else:
            print(f"文件 {file_path} 不存在")
            return

        # 检查目标列是否存在
        target_columns = ['Diagnosis', 'Severity', 'Management']
        existing_columns = []

        for col in target_columns:
            if col in df.columns:
                existing_columns.append(col)
            else:
                # 尝试寻找相似的列名（忽略大小写）
                similar_cols = [c for c in df.columns if col.lower() in c.lower()]
                if similar_cols:
                    print(f"未找到列 '{col}'，但找到相似列: {similar_cols}")
                    existing_columns.append(similar_cols[0])
                else:
                    print(f"未找到列 '{col}'")

        if not existing_columns:
            print("没有找到任何目标列，显示所有列的信息:")
            for col in df.columns:
                print(f"\n列 '{col}' 的唯一值:")
                print(df[col].value_counts())
            return

        # 分析每列的类别分布
        category_counts = {}
        for col in existing_columns:
            print(f"\n分析列: {col}")
            print(f"唯一值数量: {df[col].nunique()}")
            print(f"缺失值数量: {df[col].isnull().sum()}")

            # 计算每个类别的数量
            counts = df[col].value_counts()
            category_counts[col] = counts
            print(f"类别分布:\n{counts}")

        # 创建柱状图
        create_bar_charts(category_counts, existing_columns)

        return df, category_counts

    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()


def create_bar_charts(category_counts, columns):
    """
    创建柱状图显示各列的类别分布
    """
    # 设置图形大小
    fig_width = max(12, len(columns) * 4)
    fig, axes = plt.subplots(1, len(columns), figsize=(fig_width, 6))

    # 如果只有一列，axes不是数组
    if len(columns) == 1:
        axes = [axes]

    # 为每个列创建柱状图
    for i, col in enumerate(columns):
        counts = category_counts[col]

        # 创建柱状图
        bars = axes[i].bar(range(len(counts)), counts.values,
                           color=plt.cm.Set3(np.linspace(0, 1, len(counts))))

        # 设置标题和标签
        axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Categories', fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)

        # 设置x轴标签
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index, rotation=45, ha='right')

        # 在柱子上添加数值标签
        for bar, count in zip(bars, counts.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom', fontweight='bold')

        # 添加网格
        axes[i].grid(axis='y', alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_filename = 'pictures/appendicitis_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n图表已保存为: {output_filename}")

    # 显示图片
    plt.show()


def create_combined_chart(category_counts, columns):
    """
    创建一个组合柱状图，将所有列的数据显示在一张图中
    """
    # 准备数据
    all_categories = []
    all_counts = []
    all_columns = []

    for col in columns:
        counts = category_counts[col]
        for category, count in counts.items():
            all_categories.append(f"{col}_{category}")
            all_counts.append(count)
            all_columns.append(col)

    # 创建图形
    plt.figure(figsize=(15, 8))

    # 为每个原始列分配不同颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(columns)))
    color_map = {col: colors[i] for i, col in enumerate(columns)}
    bar_colors = [color_map[col] for col in all_columns]

    # 创建柱状图
    bars = plt.bar(range(len(all_categories)), all_counts, color=bar_colors)

    # 设置标题和标签
    plt.title('Appendicitis Data Analysis - Combined Distribution',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # 设置x轴标签
    plt.xticks(range(len(all_categories)), all_categories,
               rotation=45, ha='right')

    # 在柱子上添加数值标签
    for bar, count in zip(bars, all_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    # 添加图例
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_map[col], label=col)
                       for col in columns]
    plt.legend(handles=legend_elements, loc='upper right')

    # 添加网格
    plt.grid(axis='y', alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_filename = 'pictures/appendicitis_combined_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"组合图表已保存为: {output_filename}")

    plt.show()


if __name__ == "__main__":
    print("开始分析阑尾炎数据...")
    print("=" * 50)

    # 分析数据
    result = analyze_appendicitis_data()

    if result:
        df, category_counts = result
        columns = list(category_counts.keys())

        if len(columns) > 1:
            # 创建组合图表
            create_combined_chart(category_counts, columns)

        print("\n分析完成!")
        print("=" * 50)

        # 打印总结信息
        print("\n数据总结:")
        for col in columns:
            counts = category_counts[col]
            print(f"\n{col}:")
            print(f"  - 总类别数: {len(counts)}")
            print(f"  - 总样本数: {counts.sum()}")
            print(f"  - 最常见类别: {counts.index[0]} ({counts.iloc[0]} 个)")