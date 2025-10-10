"""
TabNet Model Parameter Tuning Guide
====================================

This guide provides comprehensive information on how to adjust TabNet parameters
for optimal performance in your appendicitis diagnosis prediction task.
"""

import torch
from pytorch_tabnet.tab_model import TabNetClassifier


def create_optimized_tabnet_configs():
    """
    Different TabNet configurations for various scenarios
    """

    # 1. CONSERVATIVE CONFIGURATION (Safe, stable training)
    conservative_config = {
        'n_d': 32,  # Decision layer width - smaller for stability
        'n_a': 32,  # Attention layer width - smaller for stability
        'n_steps': 4,  # Number of decision steps - fewer for simplicity
        'gamma': 1.5,  # Feature reusage coefficient - higher for more reuse
        'n_independent': 2,  # Independent GLU layers per step
        'n_shared': 2,  # Shared GLU layers per step
        'lambda_sparse': 1e-4,  # Sparsity regularization - light regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=1e-2, weight_decay=1e-5),  # Lower learning rate
        'mask_type': 'sparsemax',  # More stable than entmax
        'scheduler_params': {"step_size": 30, "gamma": 0.8},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'seed': 42,
        'verbose': 1
    }

    # 2. AGGRESSIVE CONFIGURATION (Higher capacity, may need more data)
    aggressive_config = {
        'n_d': 128,  # Larger decision layer
        'n_a': 128,  # Larger attention layer
        'n_steps': 8,  # More decision steps
        'gamma': 1.2,  # Less feature reusage (more diverse)
        'n_independent': 3,  # More independent layers
        'n_shared': 3,  # More shared layers
        'lambda_sparse': 1e-3,  # Stronger sparsity regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2, weight_decay=1e-4),  # Higher learning rate
        'mask_type': 'entmax',  # More expressive masking
        'scheduler_params': {"step_size": 50, "gamma": 0.9},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'seed': 42,
        'verbose': 1
    }

    # 3. BALANCED CONFIGURATION (Good starting point)
    balanced_config = {
        'n_d': 64,  # Moderate decision layer width
        'n_a': 64,  # Moderate attention layer width
        'n_steps': 6,  # Moderate number of steps
        'gamma': 1.3,  # Balanced feature reusage
        'n_independent': 2,  # Standard independent layers
        'n_shared': 2,  # Standard shared layers
        'lambda_sparse': 5e-4,  # Moderate sparsity regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=1.5e-2, weight_decay=5e-5),
        'mask_type': 'entmax',
        'scheduler_params': {"step_size": 40, "gamma": 0.85},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'seed': 42,
        'verbose': 1
    }

    return conservative_config, balanced_config, aggressive_config


def get_training_parameter_options():
    """
    Training parameter options and their effects
    """

    training_configs = {
        # FAST TRAINING (Quick experiments)
        'fast': {
            'max_epochs': 100,
            'patience': 15,
            'batch_size': 1024,  # Larger batches for speed
            'virtual_batch_size': 512,
            'num_workers': 4,  # Use multiple workers
            'drop_last': False
        },

        # THOROUGH TRAINING (Best performance)
        'thorough': {
            'max_epochs': 300,
            'patience': 50,
            'batch_size': 256,  # Smaller batches for better gradients
            'virtual_batch_size': 128,
            'num_workers': 0,  # Single worker for stability
            'drop_last': False
        },

        # BALANCED TRAINING
        'balanced': {
            'max_epochs': 200,
            'patience': 30,
            'batch_size': 512,
            'virtual_batch_size': 256,
            'num_workers': 2,  # Moderate parallelism
            'drop_last': False
        },

        # SMALL DATASET (For datasets < 1000 samples)
        'small_data': {
            'max_epochs': 150,
            'patience': 25,
            'batch_size': 64,  # Much smaller batches
            'virtual_batch_size': 32,
            'num_workers': 0,
            'drop_last': True  # Drop incomplete batches
        }
    }

    return training_configs


def explain_parameters():
    """
    Detailed explanation of each parameter and how to adjust them
    """

    parameter_guide = {
        'MODEL ARCHITECTURE PARAMETERS': {
            'n_d': {
                'description': 'Width of the decision prediction layer',
                'typical_range': '8-256',
                'adjustment_tips': [
                    'Increase for complex datasets (more features/patterns)',
                    'Decrease if overfitting or with small datasets',
                    'Should be equal to n_a for balanced attention/decision'
                ],
                'your_current': 'Not specified (likely default 8)',
                'recommended': '32-64 for your appendicitis dataset'
            },

            'n_a': {
                'description': 'Width of the attention embedding for each mask',
                'typical_range': '8-256',
                'adjustment_tips': [
                    'Controls attention mechanism capacity',
                    'Higher values = more complex feature interactions',
                    'Keep equal to n_d for balance'
                ],
                'your_current': 'Not specified (likely default 8)',
                'recommended': '32-64 for your appendicitis dataset'
            },

            'n_steps': {
                'description': 'Number of sequential decision steps',
                'typical_range': '3-10',
                'adjustment_tips': [
                    'More steps = more complex decision process',
                    'Too many steps can cause overfitting',
                    'Start with 4-6 steps'
                ],
                'your_current': 'Not specified (likely default 3)',
                'recommended': '5-6 for medical diagnosis'
            },

            'gamma': {
                'description': 'Coefficient for feature reusage in masks',
                'typical_range': '1.0-2.0',
                'adjustment_tips': [
                    'Higher values = more feature reuse across steps',
                    'Lower values = more diverse feature selection',
                    'Medical data often benefits from moderate reuse'
                ],
                'your_current': 'Not specified (likely default 1.3)',
                'recommended': '1.2-1.5 for your dataset'
            },

            'lambda_sparse': {
                'description': 'Sparsity regularization coefficient',
                'typical_range': '1e-6 to 1e-2',
                'adjustment_tips': [
                    'Higher values = more sparse feature selection',
                    'Helps with interpretability and overfitting',
                    'Medical applications benefit from sparsity'
                ],
                'your_current': 'Not specified (likely default 1e-3)',
                'recommended': '1e-4 to 1e-3 for medical data'
            }
        },

        'TRAINING PARAMETERS': {
            'max_epochs': {
                'description': 'Maximum number of training epochs',
                'your_current': 200,
                'adjustment_tips': [
                    'Increase if loss is still decreasing',
                    'Decrease if early stopping triggers often',
                    'Medical datasets often need 150-300 epochs'
                ],
                'recommended_adjustment': '150-250 depending on convergence'
            },

            'patience': {
                'description': 'Early stopping patience (epochs without improvement)',
                'your_current': 30,
                'adjustment_tips': [
                    'Increase if training is unstable',
                    'Decrease for faster experimentation',
                    'Should be 10-20% of max_epochs'
                ],
                'recommended_adjustment': '20-40 for your setup'
            },

            'batch_size': {
                'description': 'Number of samples per batch',
                'your_current': 512,
                'adjustment_tips': [
                    'Larger batches = more stable gradients, faster training',
                    'Smaller batches = better generalization, more updates',
                    'Adjust based on dataset size and memory'
                ],
                'recommended_adjustment': '256-512 for your dataset size'
            },

            'virtual_batch_size': {
                'description': 'Virtual batch size for batch normalization',
                'your_current': 256,
                'adjustment_tips': [
                    'Should be smaller than batch_size',
                    'Typically batch_size/2 or batch_size/4',
                    'Affects normalization stability'
                ],
                'recommended_adjustment': '128-256 (keep current or halve)'
            }
        }
    }

    return parameter_guide


def generate_optimized_training_code():
    """
    Generate optimized training code with different parameter sets
    """

    code_template = '''
# OPTION 1: Conservative Training (Stable, good for small datasets)
history = self.model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    eval_name=['validation'],
    eval_metric=['accuracy', 'logloss'],
    max_epochs=150,              # Reduced for stability
    patience=25,                 # Moderate patience
    batch_size=256,              # Smaller batches for better gradients
    virtual_batch_size=128,      # Half of batch_size
    num_workers=0,               # Single worker for stability
    drop_last=False,
    # Additional parameters you can add:
    weights=1,                   # Can use class weights here
    from_unsupervised=None,      # Pre-trained weights if available
    warm_start=False,            # Whether to continue from previous training
    augmentations=None,          # Data augmentation (if implemented)
)

# OPTION 2: Aggressive Training (Higher performance potential)
history = self.model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    eval_name=['validation'],
    eval_metric=['accuracy', 'logloss', 'auc'],  # Added AUC metric
    max_epochs=250,              # More epochs for complex learning
    patience=40,                 # Higher patience for complex patterns
    batch_size=256,              # Optimal batch size for most datasets
    virtual_batch_size=128,      # Standard virtual batch size
    num_workers=2,               # Moderate parallelism
    drop_last=False,
)

# OPTION 3: Your Current Setup (Optimized)
history = self.model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    eval_name=['validation'],
    eval_metric=['accuracy', 'logloss'],
    max_epochs=200,              # Your current - good
    patience=30,                 # Your current - good
    batch_size=256,              # Reduced from 512 for better gradients
    virtual_batch_size=128,      # Reduced from 256 proportionally
    num_workers=1,               # Reduced from 0 for slight speedup
    drop_last=False,             # Keep as is
)
'''

    return code_template


def create_adaptive_training_function():
    """
    Create an adaptive training function that adjusts based on dataset size
    """

    adaptive_code = '''
def adaptive_train_model(self, X_train, y_train, validation_split=0.2):
    """Adaptively adjust training parameters based on dataset characteristics"""

    dataset_size = len(X_train)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Adaptive parameter selection
    if dataset_size < 500:
        # Small dataset
        max_epochs = 100
        patience = 20
        batch_size = min(64, dataset_size // 4)
        virtual_batch_size = max(16, batch_size // 2)
        print("Using SMALL dataset configuration")

    elif dataset_size < 2000:
        # Medium dataset
        max_epochs = 150
        patience = 25
        batch_size = min(128, dataset_size // 8)
        virtual_batch_size = batch_size // 2
        print("Using MEDIUM dataset configuration")

    else:
        # Large dataset
        max_epochs = 200
        patience = 30
        batch_size = min(512, dataset_size // 10)
        virtual_batch_size = batch_size // 2
        print("Using LARGE dataset configuration")

    # Adjust for number of features
    if n_features > 100:
        max_epochs += 50  # More epochs for high-dimensional data
        patience += 10

    print(f"Adaptive parameters:")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Patience: {patience}")
    print(f"  Batch size: {batch_size}")
    print(f"  Virtual batch size: {virtual_batch_size}")

    # Create validation split (same as your current code)
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, 
        stratify=y_train if len(np.unique(y_train)) > 1 and np.min(np.bincount(y_train)) >= 2 else None,
        random_state=42
    )

    # Train with adaptive parameters
    history = self.model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        eval_name=['validation'],
        eval_metric=['accuracy', 'logloss'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,  # Keep stable
        drop_last=False
    )

    return history
'''

    return adaptive_code


def main():
    """
    Main function to display parameter tuning guide
    """
    print("=" * 80)
    print("TabNet Parameter Tuning Guide for Appendicitis Diagnosis")
    print("=" * 80)

    # Get configurations
    conservative, balanced, aggressive = create_optimized_tabnet_configs()
    training_configs = get_training_parameter_options()
    parameter_guide = explain_parameters()

    print("\n1. MODEL ARCHITECTURE RECOMMENDATIONS:")
    print("-" * 50)
    print("Based on your appendicitis dataset, here are recommended TabNet model parameters:")
    print("\nCONSERVATIVE (Safe, stable):")
    for key, value in conservative.items():
        if key not in ['optimizer_fn', 'scheduler_fn']:
            print(f"  {key}: {value}")

    print("\nBALANCED (Recommended starting point):")
    for key, value in balanced.items():
        if key not in ['optimizer_fn', 'scheduler_fn']:
            print(f"  {key}: {value}")

    print("\n2. TRAINING PARAMETER RECOMMENDATIONS:")
    print("-" * 50)
    print("For your current training setup, consider these adjustments:")

    print("\nCURRENT vs RECOMMENDED:")
    current_vs_recommended = [
        ("max_epochs", "200", "150-250 (depending on convergence)"),
        ("patience", "30", "25-40 (increase if unstable)"),
        ("batch_size", "512", "256-384 (smaller for better gradients)"),
        ("virtual_batch_size", "256", "128-192 (half of batch_size)"),
        ("num_workers", "0", "1-2 (slight speedup)"),
    ]

    for param, current, recommended in current_vs_recommended:
        print(f"  {param:<20}: {current:<10} -> {recommended}")

    print("\n3. QUICK OPTIMIZATION SUGGESTIONS:")
    print("-" * 50)
    print("IMMEDIATE IMPROVEMENTS you can make:")
    print("1. Reduce batch_size from 512 to 256 for better gradient updates")
    print("2. Reduce virtual_batch_size from 256 to 128 accordingly")
    print("3. Add 'auc' to eval_metric for better medical classification assessment")
    print("4. Consider adding class weights for imbalanced data")
    print("5. Set num_workers=1 for slight performance improvement")

    print("\n4. PARAMETER ADJUSTMENT STRATEGY:")
    print("-" * 50)
    print("STEP-BY-STEP tuning approach:")
    print("1. Start with BALANCED configuration")
    print("2. Monitor validation curves - if overfitting, reduce n_d/n_a")
    print("3. If underfitting, increase n_steps or n_d/n_a")
    print("4. Adjust lambda_sparse for feature selection control")
    print("5. Fine-tune learning rate if convergence is slow/unstable")

    print("\n5. OPTIMIZED CODE EXAMPLE:")
    print("-" * 50)
    optimized_code = '''
# Recommended optimization for your current setup:
history = self.model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_val, y_val)],
    eval_name=['validation'],
    eval_metric=['accuracy', 'logloss', 'auc'],  # Added AUC
    max_epochs=200,                              # Keep current
    patience=35,                                 # Slightly increased
    batch_size=256,                              # Reduced for better gradients
    virtual_batch_size=128,                      # Reduced proportionally
    num_workers=1,                               # Added for performance
    drop_last=False,                             # Keep current
    weights=1                                    # Can add class weights
)'''

    print(optimized_code)

    print("\n" + "=" * 80)
    print("Remember: Start with small changes and monitor the impact!")
    print("=" * 80)


if __name__ == "__main__":
    main()