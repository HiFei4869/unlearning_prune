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
      forget_subset: str = None,  # Optional subset identifier
      retain_subset: str = None,  # Optional subset identifier
      ...
  )
  ```
- Add data loading logic for both datasets
- Support partial dataset loading:
  ```python
  def load_dataset_split(dataset_path: str, subset: str = None):
      """
      Load full dataset or specific subset:
      Example: 
      - Full: load_dataset("locuslab/TOFU")
      - Partial: load_dataset("locuslab/TOFU", "forget01")
      """
      if subset:
          return load_dataset(dataset_path, subset)
      return load_dataset(dataset_path)
  ```
- Create separate DataLoaders with appropriate batch sizes
- Implement dataset validation to ensure no overlap

## 2. Importance Score Computation
### File: `/root/autodl-tmp/loraprune/loraprune/utils.py`



#### 2.1 New Dual Sensitivity Function
```python
def compute_dual_sensitivity(
    layer,
    forget_sensitivity,
    retain_sensitivity,
    epsilon=1e-6
):
    """
    1. Calculate unlearning importance score:
       S = s_{ij}^{forget}/(s_{ij}^{retain} + epsilon)
    2. Return the computed score
    """
```


## 3. Training Loop Modifications
### File: `/root/autodl-tmp/loraprune/loraprune/trainer.py`

#### 3.1 Trainer Class Updates
Modify `LoRAPruneTrainer`:


## 4. Pruning Logic Modifications
### File: `/root/autodl-tmp/loraprune/loraprune/utils.py`

#### 4.1 Local Pruning Updates
Create new function `unlearning_prune()`:
- Use unlearning importance score for pruning decisions
- Implement threshold-based pruning

```python
def unlearning_prune(
    model,
    dual_sensitivity_dict, # function compute_dual_sensitivity  
    ratio,
    unlearning_threshold
):
    """
    Prune weights based on unlearning scores while
    maintaining minimum performance on retain set
    """
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
2. Separate gradient computation implementation
3. Dual sensitivity computation
4. Training loop modifications for separate gradient tracking
5. Pruning logic updates
6. Configuration updates
7. Evaluation modifications

## File Organization
All modifications for unlearning functionality should be stored in new files with '_ul' suffix:
- `prune.py` → `prune_ul.py`
- `utils.py` → `utils_ul.py`
- `trainer.py` → `trainer_ul.py`
- `lora.py` → `lora_ul.py`
- `inference.py` → `inference_ul.py`

This separation ensures:
- Clean distinction between original and unlearning implementations
- Easy maintenance and rollback if needed
- Clear tracking of unlearning-specific changes
- Ability to use either version based on needs

## Notes
- Maintain compatibility with existing LoRAPrune functionality
- Add comprehensive logging for unlearning metrics
- Implement validation steps for unlearning effectiveness
- Preserve group-wise pruning structure
- Consider adding safeguards against catastrophic forgetting
- Add documentation for unlearning-specific parameters and usage
- Document all dependencies between components explicitly
- Changes to one component must cascade appropriately to all dependent components
- **Important Implementation Notes**:
  - Always clear gradients between forget and retain computations
  - Use separate optimizers for importance computation and training
  - Consider gradient accumulation for larger batch sizes
  - Monitor memory usage when storing separate gradients
  - Implement gradient checkpointing if memory becomes a constraint