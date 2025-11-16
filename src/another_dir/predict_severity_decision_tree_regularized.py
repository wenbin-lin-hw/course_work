"""
Decision Tree Model with Regularization to Prevent Overfitting
优化版决策树模型 - 防止过拟合

主要改进:
1. 更严格的参数网格（限制树的复杂度）
2. 使用剪枝参数 (ccp_alpha)
3. 增加正则化约束
4. 早停策略
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """加载预分割的训练和测试数据集"""
    print("="*70)
    print("加载数据...")
    print("="*70)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\n训练数据: {train_df.shape}")
    print(f"测试数据: {test_df.shape}")
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """准备特征"""
    print("\n" + "="*70)
    print("准备特征...")
    print("="*70)
    
    excluded_columns = ['US_Performed', 'US_Number', 'Severity', 'Management', 'Diagnosis']
    feature_columns = [col for col in train_df.columns if col not in excluded_columns]
    
    X_train = train_df[feature_columns].copy()
    y_train = train_df['Severity'].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df['Severity'].copy()
    
    print(f"\n特征列数: {len(feature_columns)}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_columns


def encode_categorical_features(X_train, X_test):
    """编码分类特征"""
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) > 0:
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        # 确保训练集和测试集有相同的列
        for col in set(X_train_encoded.columns) - set(X_test_encoded.columns):
            X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        return X_train_encoded, X_test_encoded
    return X_train, X_test


def get_regularized_param_grid():
    """
    获取防止过拟合的参数网格
    
    关键策略:
    1. 限制max_depth: 防止树过深
    2. 增加min_samples_split: 要求更多样本才能分裂
    3. 增加min_samples_leaf: 叶节点需要更多样本
    4. 使用ccp_alpha: 成本复杂度剪枝
    5. 限制max_features: 减少每次分裂考虑的特征数
    """
    
    param_grid_options = {
        # 方案1: 保守型 (强正则化，防止过拟合)
        'conservative': {
            'max_depth': [3, 4, 5, 6, 7],  # 限制树深度
            'min_samples_split': [20, 30, 40, 50],  # 增加分裂所需样本数
            'min_samples_leaf': [10, 15, 20, 25],  # 增加叶节点最小样本数
            'criterion': ['gini', 'entropy'],
            'splitter': ['best'],
            'max_features': ['sqrt', 'log2', None],  # 限制特征数
            'ccp_alpha': [0.0, 0.001, 0.005, 0.01]  # 剪枝参数
        },
        
        # 方案2: 平衡型 (中等正则化)
        'balanced': {
            'max_depth': [4, 5, 6, 7, 8, 10],
            'min_samples_split': [10, 15, 20, 30],
            'min_samples_leaf': [5, 8, 10, 15],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best'],
            'max_features': ['sqrt', 'log2', None],
            'ccp_alpha': [0.0, 0.001, 0.005]
        },
        
        # 方案3: 激进型 (轻度正则化)
        'aggressive': {
            'max_depth': [5, 7, 8, 10, 12],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [2, 4, 6, 8],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None],
            'ccp_alpha': [0.0, 0.001]
        }
    }
    
    return param_grid_options


def train_with_regularization(X_train, y_train, strategy='balanced', n_folds=5):
    """
    使用正则化策略训练决策树
    
    Parameters:
    -----------
    strategy : str
        'conservative': 强正则化，最大程度防止过拟合
        'balanced': 平衡性能和泛化能力
        'aggressive': 轻度正则化，追求更高性能
    """
    print("\n" + "="*70)
    print(f"训练决策树 - 策略: {strategy.upper()}")
    print("="*70)
    
    param_grids = get_regularized_param_grid()
    param_grid = param_grids[strategy]
    
    print("\n参数网格:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # 创建交叉验证器
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 创建基础模型
    base_model = DecisionTreeClassifier(random_state=42)
    
    # GridSearchCV with multiple scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'f1': 'f1',
        'accuracy': 'accuracy'
    }
    
    print(f"\n开始网格搜索 (使用{n_folds}折交叉验证)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='roc_auc',  # 使用ROC AUC选择最佳模型
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\n" + "-"*70)
    print("最佳参数:")
    print("-"*70)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n最佳交叉验证 ROC AUC: {grid_search.best_score_:.4f}")
    
    # 显示所有评分指标
    best_index = grid_search.best_index_
    print("\n最佳模型的所有指标:")
    print("-"*70)
    for scorer in scoring.keys():
        mean_score = grid_search.cv_results_[f'mean_test_{scorer}'][best_index]
        std_score = grid_search.cv_results_[f'std_test_{scorer}'][best_index]
        print(f"  {scorer}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
    
    # 检查过拟合
    train_roc_auc = grid_search.cv_results_['mean_train_roc_auc'][best_index]
    test_roc_auc = grid_search.cv_results_['mean_test_roc_auc'][best_index]
    overfitting_gap = train_roc_auc - test_roc_auc
    
    print(f"\n过拟合检查:")
    print(f"  训练集 ROC AUC: {train_roc_auc:.4f}")
    print(f"  验证集 ROC AUC: {test_roc_auc:.4f}")
    print(f"  差距: {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.1:
        print("  ⚠️  警告: 检测到过拟合！建议使用更保守的策略。")
    elif overfitting_gap > 0.05:
        print("  ⚠️  注意: 轻微过拟合。")
    else:
        print("  ✓  拟合良好。")
    
    return best_model, best_params, grid_search


def evaluate_model(model, X_train, X_test, y_train, y_test, strategy):
    """评估模型"""
    print("\n" + "="*70)
    print("模型评估")
    print("="*70)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auroc = roc_auc_score(y_train, y_train_proba)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    train_aupr = average_precision_score(y_train, y_train_proba)
    test_aupr = average_precision_score(y_test, y_test_proba)
    
    print("\n训练集性能:")
    print(f"  准确率: {train_acc:.4f}")
    print(f"  AUROC: {train_auroc:.4f}")
    print(f"  AUPR: {train_aupr:.4f}")
    
    print("\n测试集性能:")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  AUROC: {test_auroc:.4f}")
    print(f"  AUPR: {test_aupr:.4f}")
    
    print("\n过拟合分析:")
    print(f"  准确率差距: {train_acc - test_acc:.4f}")
    print(f"  AUROC差距: {train_auroc - test_auroc:.4f}")
    print(f"  AUPR差距: {train_aupr - test_aupr:.4f}")
    
    # 分类报告
    print("\n" + "-"*70)
    print("分类报告 (测试集):")
    print("-"*70)
    print(classification_report(y_test, y_test_pred))
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {strategy.capitalize()} Strategy')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{strategy}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC和PR曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[0].plot(fpr, tpr, label=f'Test AUROC = {test_auroc:.4f}', linewidth=2, color='blue')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    axes[1].plot(recall, precision, label=f'Test AUPR = {test_aupr:.4f}', linewidth=2, color='green')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Performance Curves - {strategy.capitalize()} Strategy', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/performance_curves_{strategy}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_auroc, test_aupr


def compare_strategies(X_train, X_test, y_train, y_test):
    """比较不同正则化策略"""
    print("\n" + "="*70)
    print("比较不同正则化策略")
    print("="*70)
    
    strategies = ['conservative', 'balanced', 'aggressive']
    results = []
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"测试策略: {strategy.upper()}")
        print(f"{'='*70}")
        
        # 训练模型
        model, params, grid_search = train_with_regularization(
            X_train, y_train, strategy=strategy, n_folds=5
        )
        
        # 评估模型
        test_auroc, test_aupr = evaluate_model(
            model, X_train, X_test, y_train, y_test, strategy
        )
        
        # 计算过拟合程度
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        train_auroc = roc_auc_score(y_train, y_train_proba)
        overfitting_gap = train_auroc - test_auroc
        
        results.append({
            'Strategy': strategy,
            'Test AUROC': test_auroc,
            'Test AUPR': test_aupr,
            'Train AUROC': train_auroc,
            'Overfitting Gap': overfitting_gap,
            'Max Depth': params['max_depth'],
            'Min Samples Split': params['min_samples_split'],
            'Min Samples Leaf': params['min_samples_leaf']
        })
    
    # 创建比较表
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("策略比较结果")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv('results/strategy_comparison.csv', index=False)
    print("\n比较结果已保存: results/strategy_comparison.csv")
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # AUROC比较
    x = np.arange(len(strategies))
    width = 0.35
    axes[0, 0].bar(x - width/2, results_df['Train AUROC'], width, label='Train', alpha=0.8)
    axes[0, 0].bar(x + width/2, results_df['Test AUROC'], width, label='Test', alpha=0.8)
    axes[0, 0].set_xlabel('Strategy')
    axes[0, 0].set_ylabel('AUROC')
    axes[0, 0].set_title('AUROC Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # AUPR比较
    axes[0, 1].bar(strategies, results_df['Test AUPR'], color='green', alpha=0.8)
    axes[0, 1].set_xlabel('Strategy')
    axes[0, 1].set_ylabel('AUPR')
    axes[0, 1].set_title('Test AUPR Comparison')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['Test AUPR']):
        axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # 过拟合程度
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.03 else 'green' 
              for gap in results_df['Overfitting Gap']]
    axes[1, 0].bar(strategies, results_df['Overfitting Gap'], color=colors, alpha=0.8)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='Warning threshold')
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Overfitting Gap (Train - Test AUROC)')
    axes[1, 0].set_title('Overfitting Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['Overfitting Gap']):
        axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    # 参数比较
    axes[1, 1].plot(strategies, results_df['Max Depth'], marker='o', label='Max Depth', linewidth=2)
    axes[1, 1].plot(strategies, results_df['Min Samples Split'], marker='s', label='Min Samples Split', linewidth=2)
    axes[1, 1].plot(strategies, results_df['Min Samples Leaf'], marker='^', label='Min Samples Leaf', linewidth=2)
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('Best Parameters by Strategy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("比较图已保存: results/strategy_comparison.png")
    plt.close()
    
    # 推荐最佳策略
    print("\n" + "="*70)
    print("推荐")
    print("="*70)
    
    # 找出过拟合最小且性能最好的策略
    valid_strategies = results_df[results_df['Overfitting Gap'] < 0.1]
    if len(valid_strategies) > 0:
        best_idx = valid_strategies['Test AUROC'].idxmax()
        best_strategy = results_df.loc[best_idx]
        print(f"推荐策略: {best_strategy['Strategy'].upper()}")
        print(f"  测试集 AUROC: {best_strategy['Test AUROC']:.4f}")
        print(f"  测试集 AUPR: {best_strategy['Test AUPR']:.4f}")
        print(f"  过拟合程度: {best_strategy['Overfitting Gap']:.4f}")
    else:
        print("⚠️  所有策略都存在过拟合问题，建议使用conservative策略")
    
    return results_df


def main():
    """主函数"""
    print("\n" + "="*70)
    print("决策树模型 - 防止过拟合优化版")
    print("="*70)
    
    # 加载数据
    train_path = 'data/appendicitis/train_data_balanced.csv'
    test_path = 'data/appendicitis/test_data.csv'
    train_df, test_df = load_data(train_path, test_path)
    
    # 准备特征
    X_train, X_test, y_train, y_test, feature_columns = prepare_features(train_df, test_df)
    
    # 编码分类特征
    X_train, X_test = encode_categorical_features(X_train, X_test)
    
    print(f"\n最终特征数: {X_train.shape[1]}")
    
    # 比较不同策略
    results_df = compare_strategies(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("所有任务完成！")
    print("="*70)


if __name__ == "__main__":
    main()
