# LoRAPrune Unlearning Modification Plan

## Overview
This document outlines the necessary modifications to adapt LoRAPrune for unlearning tasks using a dual-importance pruning strategy. The goal is to enable selective forgetting of specific data while retaining performance on desired data.

## 1. Dataset Handling Modifications
### File: `/root/autodl-tmp/loraprune/prune.py`
- Modify the `train()` function to accept two datasets:
  ```python
  def train(
      forget_dataset: str = "",  # Dataset to unlearn
      retain_dataset: str = "",  # Dataset to retain
      ...
  )
  ```
- Add data loading logic for both datasets
- Create separate DataLoaders with appropriate batch sizes
- Implement dataset validation to ensure no overlap

## 2. Importance Score Computation
### File: `/root/autodl-tmp/loraprune/loraprune/utils.py`

#### 2.1 Sensitivity Computation
Modify `compute_sensitivity()`:
```python
def compute_sensitivity(
    layer, 
    is_attn, 
    dataset_type,  # 'forget' or 'retain'
    prune_metric='lora', 
    epsilon=1e-6
):
```

#### 2.2 New Dual Sensitivity Function
```python
def compute_dual_sensitivity(
    layer,
    forget_grad,
    retain_grad,
    epsilon=1e-6
):
    """
    Compute unlearning importance score:
    S = I_{ij}^{forget}/(I_{ij}^{retain} + epsilon)
    """
```

#### 2.3 Update Sensitivity Dictionary
Modify `update_sensitivity_dict()`:
- Track both forget and retain sensitivities
- Implement score computation
- Add threshold-based decisions

## 3. Pruning Logic Modifications
### File: `/root/autodl-tmp/loraprune/loraprune/utils.py`

#### 3.1 Local Pruning Updates
Modify `local_prune()`:
- Use unlearning scores for pruning decisions
- Implement threshold-based pruning
- Add early stopping based on forget set performance

#### 3.2 New Unlearning Pruning Function
```python
def unlearning_prune(
    model,
    forget_dict,
    retain_dict,
    threshold,
    min_retain_performance
):
    """
    Prune weights based on unlearning scores while
    maintaining minimum performance on retain set
    """
```

## 4. Training Loop Modifications
### File: `/root/autodl-tmp/loraprune/loraprune/trainer.py`

#### 4.1 Trainer Class Updates
Modify `LoRAPruneTrainer`:
- Add dual dataset evaluation support
- Implement forget/retain loss computation
- Add unlearning metrics tracking
- Implement validation on both datasets

#### 4.2 New Methods
```python
def compute_unlearning_metrics(self):
    """Track unlearning progress and performance trade-offs"""

def evaluate_forget_retain(self):
    """Evaluate model on both datasets"""
```

## 5. Configuration Updates
### File: `/root/autodl-tmp/loraprune/loraprune/lora.py`

Update `LoraConfig`:
```python
@dataclass
class LoraConfig(PeftConfig):
    # Add new fields
    unlearning_threshold: float = field(default=0.5)
    unlearning_epsilon: float = field(default=1e-6)
    min_retain_performance: float = field(default=0.9)
    forget_retain_ratio: float = field(default=1.0)
```

## 6. Evaluation Modifications
### File: `/root/autodl-tmp/loraprune/inference.py`

- Add separate evaluation for forget and retain sets
- Implement metrics for unlearning effectiveness:
  - Performance drop on forget set
  - Performance retention on retain set
  - Unlearning efficiency metrics

## Implementation Priority
1. Dataset handling modifications
2. Dual sensitivity computation
3. Unlearning score calculation
4. Pruning logic updates
5. Training loop modifications
6. Configuration updates
7. Evaluation modifications

## Notes
- Maintain compatibility with existing LoRAPrune functionality
- Add comprehensive logging for unlearning metrics
- Implement validation steps for unlearning effectiveness
- Preserve group-wise pruning structure
- Consider adding safeguards against catastrophic forgetting
- Add documentation for unlearning-specific parameters and usage