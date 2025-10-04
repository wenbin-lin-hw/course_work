import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class AppendicitisPreprocessor:
    def __init__(self, data_path):
        """初始化数据预处理器"""
        self.data_path = data_path
        self.original_data = None
        self.processed_data = None
        self.processing_log = []
        self.label_encoders = {}

    def log_step(self, step_name, message):
        """记录处理步骤"""
        log_entry = f"{step_name}: {message}"
        self.processing_log.append(log_entry)
        print(f"✓ {log_entry}")

    def load_data(self):
        """加载原始数据"""
        print("=" * 80)
        print("开始加载APPENDICITIS数据")
        print("=" * 80)

        try:
            self.original_data = pd.read_excel(self.data_path)
            self.processed_data = self.original_data.copy()

            print(f"数据加载成功！")
            print(f"原始数据形状: {self.original_data.shape}")
            print(f"列名: {list(self.original_data.columns)}")

            self.log_step("数据加载", f"成功加载 {self.original_data.shape[0]} 行 {self.original_data.shape[1]} 列")
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def display_initial_info(self):
        """显示初始数据信息"""
        print(f"\n初始数据分析:")
        print("-" * 50)

        # 缺失数据统计
        missing_info = []
        for col in self.processed_data.columns:
            missing_count = self.processed_data[col].isnull().sum()
            missing_pct = (missing_count / len(self.processed_data)) * 100
            if missing_count > 0:
                missing_info.append((col, missing_count, missing_pct))

        print(f"缺失数据概览:")
        for col, count, pct in missing_info[:10]:  # 显示前10个有缺失的列
            print(f"  {col}: {count} ({pct:.1f}%)")

        if len(missing_info) > 10:
            print(f"  ... 还有 {len(missing_info) - 10} 个列有缺失数据")

        # 数据类型统计
        print(f"\n数据类型分布:")
        dtype_counts = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} 列")

    def step1_remove_empty_age_rows(self):
        """步骤1: 删除Age列为空的行"""
        print(f"\n步骤1: 删除Age列为空的行")
        print("-" * 40)

        initial_rows = len(self.processed_data)

        if 'Age' in self.processed_data.columns:
            age_missing = self.processed_data['Age'].isnull().sum()
            print(f"Age列缺失值: {age_missing}")

            if age_missing > 0:
                self.processed_data = self.processed_data.dropna(subset=['Age'])
                final_rows = len(self.processed_data)
                removed_rows = initial_rows - final_rows
                self.log_step("删除空Age行", f"删除 {removed_rows} 行，剩余 {final_rows} 行")
            else:
                self.log_step("删除空Age行", "Age列无缺失值")
        else:
            print("未找到Age列")
            self.log_step("删除空Age行", "未找到Age列")

    def step2_fill_diagnosis_missing(self):
        """步骤2: Diagnosis列空值用appendicitis填充"""
        print(f"\n步骤2: Diagnosis列空值填充")
        print("-" * 40)

        if 'Diagnosis' in self.processed_data.columns:
            missing_count = self.processed_data['Diagnosis'].isnull().sum()
            print(f"Diagnosis缺失值: {missing_count}")

            if missing_count > 0:
                self.processed_data['Diagnosis'].fillna('appendicitis', inplace=True)
                self.log_step("Diagnosis填充", f"用'appendicitis'填充 {missing_count} 个缺失值")
            else:
                self.log_step("Diagnosis填充", "无缺失值需要填充")
        else:
            print("未找到Diagnosis列")
            self.log_step("Diagnosis填充", "未找到Diagnosis列")

    def step3_remove_high_missing_columns(self, threshold=70):
        """步骤3: 删除缺失率大于70%的列"""
        print(f"\n步骤3: 删除缺失率大于{threshold}%的列")
        print("-" * 40)

        missing_pct = (self.processed_data.isnull().sum() / len(self.processed_data)) * 100
        high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()

        if high_missing_cols:
            print("高缺失率列:")
            for col in high_missing_cols:
                print(f"  - {col}: {missing_pct[col]:.1f}%")

            self.processed_data = self.processed_data.drop(columns=high_missing_cols)
            self.log_step("删除高缺失率列", f"删除 {len(high_missing_cols)} 列: {high_missing_cols}")
        else:
            self.log_step("删除高缺失率列", f"无缺失率大于{threshold}%的列")

    def step4_fill_bmi_mean(self):
        """步骤4: BMI列用均值填充"""
        print(f"\n步骤4: BMI列均值填充")
        print("-" * 40)

        if 'BMI' in self.processed_data.columns:
            missing_count = self.processed_data['BMI'].isnull().sum()
            if missing_count > 0:
                mean_value = self.processed_data['BMI'].mean()
                self.processed_data['BMI'].fillna(mean_value, inplace=True)
                self.log_step("BMI均值填充", f"用均值 {mean_value:.2f} 填充 {missing_count} 个缺失值")
            else:
                self.log_step("BMI均值填充", "无缺失值需要填充")
        else:
            self.log_step("BMI均值填充", "未找到BMI列")

    def step5_knn_fill_height_weight(self):
        """步骤5: Height和Weight用Age的KNN填充"""
        print(f"\n步骤5: Height和Weight的KNN填充")
        print("-" * 40)

        # 查找Height和Weight列
        height_cols = [col for col in self.processed_data.columns if 'height' in col.lower()]
        weight_cols = [col for col in self.processed_data.columns if 'weight' in col.lower()]

        print(f"找到Height列: {height_cols}")
        print(f"找到Weight列: {weight_cols}")

        if 'Age' not in self.processed_data.columns:
            self.log_step("KNN填充", "未找到Age列，无法进行KNN填充")
            return

        for col_type, cols in [('Height', height_cols), ('Weight', weight_cols)]:
            for col in cols:
                if col in self.processed_data.columns:
                    missing_count = self.processed_data[col].isnull().sum()
                    if missing_count > 0:
                        print(f"处理 {col} 列，缺失值: {missing_count}")

                        # 准备KNN数据
                        knn_data = self.processed_data[['Age', col]].copy()

                        # 使用KNN填充
                        imputer = KNNImputer(n_neighbors=5)
                        filled_data = imputer.fit_transform(knn_data)

                        # 更新数据
                        self.processed_data[col] = filled_data[:, 1]
                        self.log_step(f"{col_type} KNN填充", f"{col}列用KNN填充 {missing_count} 个缺失值")
                    else:
                        self.log_step(f"{col_type} KNN填充", f"{col}列无缺失值")

    def step6_remove_length_of_stay(self):
        """步骤6: 删除Length_of_Stay列"""
        print(f"\n步骤6: 删除Length_of_Stay相关列")
        print("-" * 40)

        # 查找住院时间相关列
        stay_keywords = ['length', 'stay', 'hospital', 'los', 'duration','diagnosis_presumptive']
        stay_cols = []

        for col in self.processed_data.columns:
            if any(keyword in col.lower() for keyword in stay_keywords):
                stay_cols.append(col)

        if stay_cols:
            print(f"找到住院时间相关列: {stay_cols}")
            self.processed_data = self.processed_data.drop(columns=stay_cols)
            self.log_step("删除住院时间列", f"删除 {len(stay_cols)} 列: {stay_cols}")
        else:
            self.log_step("删除住院时间列", "未找到住院时间相关列")

    def identify_column_types(self):
        """识别列的数据类型"""
        print(f"\n数据类型识别:")
        print("-" * 30)

        categorical_cols = []
        numerical_cols = []
        date_cols = []

        # 特殊处理的列
        special_cols = ['US_Number']

        for col in self.processed_data.columns:
            if col in special_cols:
                continue

            # 检查是否为日期类型
            if any(keyword in col.lower() for keyword in ['date', 'time']) or 'datetime' in str(
                    self.processed_data[col].dtype):
                date_cols.append(col)
            # 检查是否为分类类型
            elif (self.processed_data[col].dtype == 'object' or
                  (self.processed_data[col].dtype in ['int64', 'float64'] and
                   self.processed_data[col].nunique() <= 10 and
                   self.processed_data[col].nunique() < len(self.processed_data) * 0.05)):
                categorical_cols.append(col)
            # 数值类型
            elif self.processed_data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                # 默认归为分类
                categorical_cols.append(col)

        print(f"分类列 ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        print(f"数值列 ({len(numerical_cols)}): {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        print(f"日期列 ({len(date_cols)}): {date_cols}")

        return categorical_cols, numerical_cols, date_cols

    def step7_fill_categorical_missing(self, categorical_cols):
        """步骤7: 分类数据用相同Diagnosis的众数填充"""
        print(f"\n步骤7: 分类数据缺失值填充")
        print("-" * 40)

        if 'Diagnosis' not in self.processed_data.columns:
            self.log_step("分类数据填充", "未找到Diagnosis列")
            return

        total_filled = 0

        for col in categorical_cols:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                print(f"处理分类列 {col}，缺失值: {missing_count}")

                # 按Diagnosis分组填充
                for diagnosis in self.processed_data['Diagnosis'].unique():
                    if pd.isna(diagnosis):
                        continue

                    # 找到该诊断组中该列为空的行
                    mask = (self.processed_data['Diagnosis'] == diagnosis) & (self.processed_data[col].isnull())

                    if mask.sum() > 0:
                        # 计算该诊断组的众数
                        group_data = self.processed_data[
                            (self.processed_data['Diagnosis'] == diagnosis) &
                            (self.processed_data[col].notnull())
                            ][col]

                        if len(group_data) > 0:
                            group_mode = group_data.mode()
                            if len(group_mode) > 0:
                                self.processed_data.loc[mask, col] = group_mode.iloc[0]
                                total_filled += mask.sum()

                # 如果还有缺失值，用全局众数填充
                remaining_missing = self.processed_data[col].isnull().sum()
                if remaining_missing > 0:
                    global_mode = self.processed_data[col].mode()
                    if len(global_mode) > 0:
                        self.processed_data[col].fillna(global_mode.iloc[0], inplace=True)
                        total_filled += remaining_missing

        self.log_step("分类数据填充", f"按Diagnosis分组填充 {total_filled} 个分类缺失值")

    def step8_fill_numerical_missing(self, numerical_cols):
        """步骤8: 数值数据用相同Diagnosis的均值填充"""
        print(f"\n步骤8: 数值数据缺失值填充")
        print("-" * 40)

        if 'Diagnosis' not in self.processed_data.columns:
            self.log_step("数值数据填充", "未找到Diagnosis列")
            return

        total_filled = 0

        for col in numerical_cols:
            missing_count = self.processed_data[col].isnull().sum()
            if missing_count > 0:
                print(f"处理数值列 {col}，缺失值: {missing_count}")

                # 按Diagnosis分组填充
                for diagnosis in self.processed_data['Diagnosis'].unique():
                    if pd.isna(diagnosis):
                        continue

                    mask = (self.processed_data['Diagnosis'] == diagnosis) & (self.processed_data[col].isnull())

                    if mask.sum() > 0:
                        # 计算该诊断组的均值
                        group_data = self.processed_data[
                            (self.processed_data['Diagnosis'] == diagnosis) &
                            (self.processed_data[col].notnull())
                            ][col]

                        if len(group_data) > 0:
                            group_mean = group_data.mean()
                            if not pd.isna(group_mean):
                                self.processed_data.loc[mask, col] = group_mean
                                total_filled += mask.sum()

                # 如果还有缺失值，用全局均值填充
                remaining_missing = self.processed_data[col].isnull().sum()
                if remaining_missing > 0:
                    global_mean = self.processed_data[col].mean()
                    if not pd.isna(global_mean):
                        self.processed_data[col].fillna(global_mean, inplace=True)
                        total_filled += remaining_missing

        self.log_step("数值数据填充", f"按Diagnosis分组填充 {total_filled} 个数值缺失值")

    def step9_label_encode_categorical(self, categorical_cols):
        """步骤9: 分类数据Label编码"""
        print(f"\n步骤9: 分类数据Label编码")
        print("-" * 40)

        # 排除目标变量和特殊列
        target_cols = [ 'US_Number']
        encode_cols = [col for col in categorical_cols if col not in target_cols]

        if encode_cols:
            print(f"需要编码的列: {encode_cols}")

            encoded_count = 0
            for col in encode_cols:
                if self.processed_data[col].dtype == 'object':
                    le = LabelEncoder()

                    # 处理缺失值
                    non_null_data = self.processed_data[col].dropna()
                    if len(non_null_data) > 0:
                        # 对非空值进行编码
                        self.processed_data[col] = self.processed_data[col].astype(str)

                        # 将NaN转换为字符串再编码
                        self.processed_data[col] = self.processed_data[col].replace('nan', np.nan)

                        # 填充NaN为一个特殊值
                        self.processed_data[col].fillna('MISSING', inplace=True)

                        # 编码
                        self.processed_data[col] = le.fit_transform(self.processed_data[col])

                        # 保存编码器
                        self.label_encoders[col] = le
                        encoded_count += 1

                        print(f"  编码 {col}: {len(le.classes_)} 个类别")

            self.log_step("Label编码", f"编码 {encoded_count} 个分类列")
        else:
            self.log_step("Label编码", "无需要编码的分类列")

    def generate_comprehensive_report(self):
        """生成综合处理报告"""
        print(f"\n" + "=" * 80)
        print("数据预处理综合报告")
        print("=" * 80)

        # 基本统计对比
        original_shape = self.original_data.shape
        final_shape = self.processed_data.shape

        print(f"\n数据形状变化:")
        print("-" * 30)
        print(f"原始数据: {original_shape[0]} 行 × {original_shape[1]} 列")
        print(f"处理后:   {final_shape[0]} 行 × {final_shape[1]} 列")
        print(f"行变化:   {final_shape[0] - original_shape[0]:+d}")
        print(f"列变化:   {final_shape[1] - original_shape[1]:+d}")

        # 缺失值统计
        original_missing = self.original_data.isnull().sum().sum()
        final_missing = self.processed_data.isnull().sum().sum()

        print(f"\n缺失值变化:")
        print("-" * 20)
        print(f"原始缺失值: {original_missing}")
        print(f"最终缺失值: {final_missing}")
        print(f"清理缺失值: {original_missing - final_missing}")

        # 数据类型分布
        print(f"\n最终数据类型分布:")
        print("-" * 25)
        dtype_counts = self.processed_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{str(dtype):<12}: {count:>3} 列")

        # 目标变量分布
        target_cols = ['Diagnosis', 'Severity', 'Management']
        for col in target_cols:
            if col in self.processed_data.columns:
                print(f"\n{col} 分布:")
                print("-" * 15)
                value_counts = self.processed_data[col].value_counts()
                for value, count in value_counts.head(10).items():  # 只显示前10个
                    pct = (count / len(self.processed_data)) * 100
                    print(f"  {str(value):<15}: {count:>4} ({pct:>5.1f}%)")

                if len(value_counts) > 10:
                    print(f"  ... 还有 {len(value_counts) - 10} 个其他值")

        # 处理步骤汇总
        print(f"\n处理步骤汇总:")
        print("-" * 30)
        for i, step in enumerate(self.processing_log, 1):
            print(f"{i:2d}. {step}")

        # 编码信息
        if self.label_encoders:
            print(f"\nLabel编码信息:")
            print("-" * 20)
            for col, encoder in self.label_encoders.items():
                print(f"{col}: {len(encoder.classes_)} 个类别")
                if len(encoder.classes_) <= 10:
                    class_mapping = {cls: i for i, cls in enumerate(encoder.classes_)}
                    print(f"  映射: {class_mapping}")

        # 数据质量检查
        remaining_missing = self.processed_data.isnull().sum().sum()
        print(f"\n数据质量检查:")
        print("-" * 20)
        if remaining_missing == 0:
            print("✓ 所有缺失值已处理")
        else:
            print(f"⚠ 仍有 {remaining_missing} 个缺失值")
            missing_cols = self.processed_data.isnull().sum()
            for col in missing_cols[missing_cols > 0].index:
                print(f"  - {col}: {missing_cols[col]} 个")

        print("✓ 数据预处理完成")

    def save_processed_data(self, output_path='processed_appendicitis_data_final.xlsx'):
        """保存处理后的数据"""
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 保存主要数据
                self.processed_data.to_excel(writer, sheet_name='Processed_Data', index=False)

                # 保存处理报告
                report_df = pd.DataFrame({
                    'Step': range(1, len(self.processing_log) + 1),
                    'Processing_Action': self.processing_log
                })
                report_df.to_excel(writer, sheet_name='Processing_Log', index=False)

                # 保存数据对比
                comparison_data = {
                    'Metric': ['Rows', 'Columns', 'Missing_Values', 'Total_Cells'],
                    'Original': [
                        self.original_data.shape[0],
                        self.original_data.shape[1],
                        self.original_data.isnull().sum().sum(),
                        self.original_data.shape[0] * self.original_data.shape[1]
                    ],
                    'Processed': [
                        self.processed_data.shape[0],
                        self.processed_data.shape[1],
                        self.processed_data.isnull().sum().sum(),
                        self.processed_data.shape[0] * self.processed_data.shape[1]
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df['Change'] = comparison_df['Processed'] - comparison_df['Original']
                comparison_df.to_excel(writer, sheet_name='Data_Comparison', index=False)

                # 保存Label编码映射
                if self.label_encoders:
                    encoding_info = []
                    for col, encoder in self.label_encoders.items():
                        for i, cls in enumerate(encoder.classes_):
                            encoding_info.append({
                                'Column': col,
                                'Original_Value': cls,
                                'Encoded_Value': i
                            })

                    if encoding_info:
                        encoding_df = pd.DataFrame(encoding_info)
                        encoding_df.to_excel(writer, sheet_name='Label_Encoding_Map', index=False)

            print(f"\n✓ 处理后的数据已保存至: {output_path}")
            print("  包含的工作表:")
            print("    - Processed_Data: 处理后的数据")
            print("    - Processing_Log: 处理步骤日志")
            print("    - Data_Comparison: 数据对比")
            if self.label_encoders:
                print("    - Label_Encoding_Map: Label编码映射表")

            return True

        except Exception as e:
            print(f"\n✗ 保存失败: {e}")
            return False

    def run_preprocessing(self):
        """执行完整的预处理流程"""
        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 显示初始信息
        self.display_initial_info()

        # 3. 执行预处理步骤
        self.step1_remove_empty_age_rows()
        self.step2_fill_diagnosis_missing()
        self.step3_remove_high_missing_columns()
        # US_Number列不做任何处理（跳过）
        self.step4_fill_bmi_mean()
        self.step5_knn_fill_height_weight()
        self.step6_remove_length_of_stay()

        # 4. 识别数据类型
        categorical_cols, numerical_cols, date_cols = self.identify_column_types()

        # 5. 填充缺失值
        self.step7_fill_categorical_missing(categorical_cols)
        self.step8_fill_numerical_missing(numerical_cols)
        # 日期型数据保留原值（不处理）

        # 6. Label编码
        self.step9_label_encode_categorical(categorical_cols)

        # 7. 生成报告
        self.generate_comprehensive_report()

        # 8. 保存数据
        success = self.save_processed_data()

        return success


def main():
    """主函数"""
    print("开始Appendicitis数据预处理...")

    # 创建预处理器
    preprocessor = AppendicitisPreprocessor('../../data/appendicitis/app_data.xlsx')

    # 执行预处理
    success = preprocessor.run_preprocessing()

    if success:
        print(f"\n" + "=" * 80)
        print("预处理成功完成！")
        print("=" * 80)
        print("输出文件: processed_appendicitis_data_final.xlsx")
        print("\n主要处理内容:")
        print("1. 删除Age为空的行")
        print("2. Diagnosis空值填充为'appendicitis'")
        print("3. 删除缺失率>70%的列")
        print("4. US_Number列保持不变")
        print("5. BMI用均值填充")
        print("6. Height/Weight用Age的KNN填充")
        print("7. 删除住院时间相关列")
        print("8. 分类数据按Diagnosis分组众数填充")
        print("9. 数值数据按Diagnosis分组均值填充")
        print("10. 分类数据Label编码")
        print("11. 日期数据保留原值")
    else:
        print("\n预处理失败！")


if __name__ == "__main__":
    main()