"""
Loggers for dual training experiments (comparing Shampoo with different order_multipliers, Muon, etc.)

This module provides loggers specifically designed for the dual/triple model training setup,
following the same pattern as src/loggers.py but adapted for torch.nn.Module models
rather than functional models.

Also includes console logging utilities for pretty-printing training progress.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelLogger:
    """Base class for loggers that record properties of a single model during training."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log properties of the model and optimizer.
        
        Args:
            model: The neural network model
            optimizer: The optimizer being used
            **kwargs: Additional context (e.g., data loaders, device)
            
        Returns:
            Dictionary of logged values
        """
        raise NotImplementedError()


class GroupLogger:
    """Base class for loggers that record collective properties of multiple models."""
    
    def log(self, models: Dict[str, nn.Module], **kwargs) -> Dict[str, Any]:
        """Log collective properties across multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            **kwargs: Additional context
            
        Returns:
            Dictionary of logged values
        """
        raise NotImplementedError()


@dataclass
class LossAndAccuracyLogger(ModelLogger):
    """Logs loss and accuracy on train or test set."""
    
    split: Literal["train", "test"]  # Which split to evaluate on
    label_smoothing: float = 0.0     # Label smoothing for loss computation
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Compute and log loss and accuracy."""
        loader = kwargs.get(f'{self.split}_loader')
        device = kwargs.get('device', 'cuda')
        batch_size = kwargs.get('batch_size', 2000)

        
        if loader is None:
            return {}
        
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            # For CifarLoader-style loaders
            if hasattr(loader, 'normalize') and hasattr(loader, 'images'):
                images = loader.normalize(loader.images)
                labels = loader.labels
                
                # Process in batches
                for i in range(0, len(images), batch_size):
                    inputs = images[i:i+batch_size]
                    targets = labels[i:i+batch_size]
                    
                    outputs = model(inputs)
                    loss = F.cross_entropy(
                        outputs, targets,
                        label_smoothing=self.label_smoothing,
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_correct += (outputs.argmax(1) == targets).float().sum().item()
                    total_samples += len(inputs)
            else:
                # For standard PyTorch DataLoader
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(
                        outputs, targets,
                        label_smoothing=self.label_smoothing,
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_correct += (outputs.argmax(1) == targets).float().sum().item()
                    total_samples += len(inputs)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {
            f'{self.split}_loss': avg_loss,
            f'{self.split}_acc': avg_acc,
        }


class OptimizerStateLogger(ModelLogger):
    """Logs optimizer state information."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log optimizer-specific state."""
        out = {}
        
        # Log learning rates
        lrs = [group['lr'] for group in optimizer.param_groups]
        out['lr'] = lrs[0] if len(lrs) == 1 else lrs
        
        # Log Shampoo-specific info
        if hasattr(optimizer, 'order_multiplier'):
            out['order_multiplier'] = optimizer.order_multiplier
            
        # Count parameters with state
        params_with_state = len([p for p in model.parameters() if p in optimizer.state])
        out['params_with_state'] = params_with_state
        
        # Log step count (if available)
        if optimizer.state:
            steps = [state.get('step', 0) for state in optimizer.state.values() if 'step' in state]
            if steps:
                out['optimizer_step'] = max(steps)
        
        return {'opt': out}


class SingularValueLogger(ModelLogger):
    """Logs singular values of weight matrices."""
    
    def __init__(self, param_name_filter=None):
        """Initialize singular value logger.
        
        Args:
            param_name_filter: Optional function to filter which parameters to track.
                             Should return True for parameters to track.
        """
        self.param_name_filter = param_name_filter or (lambda n, p: p.ndim >= 2)
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Compute singular values for tracked parameters."""
        singular_values = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not self.param_name_filter(name, param):
                    continue
                
                # Reshape to 2D if needed
                if param.ndim == 2:
                    matrix = param.data
                else:
                    matrix = param.data.reshape(param.size(0), -1)
                
                try:
                    # Compute SVD (only singular values)
                    matrix_cpu = matrix.cpu().float()
                    _, s, _ = torch.svd(matrix_cpu)
                    singular_values[name] = s.numpy()
                except Exception as e:
                    # Skip if SVD fails
                    continue
        
        return {'singular_values': singular_values}


class GradientNormLogger(ModelLogger):
    """Logs gradient norms."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log gradient statistics."""
        total_norm = 0.0
        param_norms = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    total_norm += grad_norm ** 2
                    param_norms[name] = grad_norm
        
        total_norm = total_norm ** 0.5
        
        return {
            'grad_norm': total_norm,
            'param_grad_norms': param_norms,
        }


class ParameterStatsLogger(ModelLogger):
    """Logs parameter statistics (norms, means, stds)."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log parameter statistics."""
        stats = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                stats[f'{name}_norm'] = param.data.norm().item()
                stats[f'{name}_mean'] = param.data.mean().item()
                stats[f'{name}_std'] = param.data.std().item()
        
        return {'param_stats': stats}


class ModelDistanceLogger(GroupLogger):
    """Logs distances between models' parameters."""
    
    def log(self, models: Dict[str, nn.Module], **kwargs) -> Dict[str, Any]:
        """Compute pairwise parameter distances between models."""
        distances = {}
        names = sorted(models.keys())
        
        with torch.no_grad():
            for i in range(len(names)):
                for j in range(i):
                    model_i, model_j = models[names[i]], models[names[j]]
                    
                    # Compute L2 distance between all parameters
                    total_dist = 0.0
                    for (name_i, param_i), (name_j, param_j) in zip(
                        model_i.named_parameters(),
                        model_j.named_parameters()
                    ):
                        if name_i == name_j:  # Make sure parameters correspond
                            total_dist += (param_i - param_j).norm().item() ** 2
                    
                    total_dist = total_dist ** 0.5
                    distances[f'{names[j]}_to_{names[i]}'] = total_dist
        
        return {'model_distances': distances}


class TrainingProgressLogger(ModelLogger):
    """Logs training progress metrics (epoch, step, etc.)."""
    
    def log(self, model: nn.Module, optimizer: torch.optim.Optimizer, **kwargs) -> Dict[str, Any]:
        """Log training progress information."""
        return {
            'epoch': kwargs.get('epoch', 0),
            'step': kwargs.get('step', 0),
            'total_steps': kwargs.get('total_steps', 0),
        }


def create_default_loggers(
    label_smoothing: float = 0.0,
    track_singular_values: bool = True,
    singular_value_freq: int = 1,
) -> Dict[str, list]:
    """Create a default set of loggers for dual training.
    
    Args:
        label_smoothing: Label smoothing parameter
        track_singular_values: Whether to track singular values
        singular_value_freq: How often to log singular values (every N calls)
        
    Returns:
        Dictionary mapping 'process' and 'group' to lists of loggers
    """
    process_loggers = [
        LossAndAccuracyLogger(split='train', label_smoothing=label_smoothing),
        LossAndAccuracyLogger(split='test', label_smoothing=label_smoothing),
        OptimizerStateLogger(),
        GradientNormLogger(),
        TrainingProgressLogger(),
    ]
    
    if track_singular_values:
        # Filter for filter/weight parameters (not biases or embeddings)
        def filter_params(name, param):
            return (param.ndim >= 2 and 
                   "embed" not in name and 
                   "head" not in name and
                   "bias" not in name)
        
        process_loggers.append(SingularValueLogger(param_name_filter=filter_params))
    
    group_loggers = [
        ModelDistanceLogger(),
    ]
    
    return {
        'process': process_loggers,
        'group': group_loggers,
    }


#############################################
#         Console Logging Utilities         #
#############################################

# Default columns for training log table
DEFAULT_LOGGING_COLUMNS = ['epoch', 'opt', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'lr']


def print_columns(columns_list: List[str], is_head: bool = False, is_final_entry: bool = False):
    """
    Print a row of a formatted table with columns.
    
    Args:
        columns_list: List of strings to print as columns
        is_head: If True, print a separator line before the row (for table header)
        is_final_entry: If True, print a separator line after the row (for table footer)
        
    Example:
        >>> print_columns(['epoch', 'loss', 'acc'], is_head=True)
        ------------------------
        |  epoch  |  loss  |  acc  |
        >>> print_columns(['1', '0.234', '0.856'])
        |  1  |  0.234  |  0.856  |
    """
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    
    if is_head:
        print('-' * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-' * len(print_string))


def print_training_details(
    variables: Dict[str, Any],
    is_final_entry: bool = False,
    columns: List[str] = None
):
    """
    Print training metrics in a formatted table row.
    
    Args:
        variables: Dictionary mapping column names to values
        is_final_entry: If True, print a separator line after the row
        columns: List of column names to print (default: DEFAULT_LOGGING_COLUMNS)
        
    Example:
        >>> print_training_details({
        ...     'epoch': 1,
        ...     'opt': 'Shampoo',
        ...     'train_loss': 0.234,
        ...     'train_acc': 0.856,
        ...     'test_loss': 0.289,
        ...     'test_acc': 0.823,
        ...     'lr': 0.001
        ... })
        |  1  |  Shampoo  |  0.234000  |  0.856000  |  0.289000  |  0.823000  |  0.001000  |
    """
    if columns is None:
        columns = DEFAULT_LOGGING_COLUMNS
    
    formatted = []
    for col in columns:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.6f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    
    print_columns(formatted, is_final_entry=is_final_entry)


def compute_singular_values(model: nn.Module, param_names: List[str] = None) -> Dict[str, Any]:
    """
    Compute singular values for all 2D+ parameters in the model.

    This is a utility function for tracking weight matrix properties during training.

    Args:
        model: PyTorch model
        param_names: Optional list of parameter names to track. If None, track all 2D+ params.

    Returns:
        Dictionary mapping parameter names to their singular values (as numpy arrays)

    Example:
        >>> model = MyModel()
        >>> sv = compute_singular_values(model)
        >>> for name, values in sv.items():
        ...     print(f"{name}: condition number = {values[0] / values[-1]:.2f}")
    """
    singular_values = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            if param_names is not None and name not in param_names:
                continue

            # Reshape to 2D if needed
            if param.ndim == 2:
                matrix = param.data
            else:
                # For higher dimensional tensors, reshape to 2D
                matrix = param.data.reshape(param.size(0), -1)

            # Compute SVD (only singular values)
            try:
                # Move to CPU for SVD computation
                matrix_cpu = matrix.cpu().float()
                _, s, _ = torch.svd(matrix_cpu)
                singular_values[name] = s.numpy()
            except Exception as e:
                print(f"Warning: Could not compute SVD for {name}: {e}")
                continue

    return singular_values


def get_important_conv_layer(model: nn.Module, method : str = None) -> str:
    """
    Automatically identify an important layer for gradient tracking.

    For VIT: selects transformer.layers.3.0.to_qkv.weight
    For CifarNet: selects the first conv in the first ConvGroup (layers.0.conv1)
    For MLP: selects the first 2D layer (typically layers.0.weight)
    For other architectures: selects the first Conv2d or Linear layer found

    Args:
        model: PyTorch model
        method: Optional method hint (not currently used)

    Returns:
        Parameter name of the important layer, or None if no appropriate layer found
    """
    # For VIT: use transformer.layers.3.0.to_qkv.weight
    vit_param_name = 'transformer.layers.3.0.to_qkv.weight'
    if any(n == vit_param_name for n, _ in model.named_parameters()):
        return vit_param_name

    # For CifarNet: use first conv in first block
    for name in ['layers.0.conv1.weight', 'layers.1.conv1.weight']:
        if any(n == name for n, _ in model.named_parameters()):
            return name

    # For MLP: use first linear layer (Sequential uses numeric indices)
    for name in ['1.weight', '0.weight', 'layers.0.weight', 'layers.1.weight', 'fc1.weight', 'linear1.weight']:
        param = None
        for n, p in model.named_parameters():
            if n == name:
                param = p
                break
        # Check if it's a 2D parameter (Linear layer)
        if param is not None and param.ndim == 2:
            return name

    # Fallback: find first Conv2d or Linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            param_name = name + '.weight'
            if any(n == param_name for n, _ in model.named_parameters()):
                return param_name

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_name = name + '.weight'
            if any(n == param_name for n, _ in model.named_parameters()):
                return param_name

    return None


def compute_gradient_statistics(
    model: nn.Module,
    param_name: str = None,
    optimizers: List[torch.optim.Optimizer] = None,
    lr: float = 1.0,
    scale_by_lr: bool = False,
    use_momentum: bool = False
) -> Dict[str, Any]:
    """
    Compute comprehensive gradient statistics for a specific parameter.

    Tracks gradient spectrum (singular values), norms, and trace - useful for understanding
    optimization dynamics and gradient flow. Can analyze either raw gradients or momentum buffers.

    Args:
        model: PyTorch model
        param_name: Name of parameter to track. If None, auto-detects important conv layer.
        optimizers: List of optimizers (required if use_momentum=True)
        lr: Learning rate to scale by (used if scale_by_lr=True)
        scale_by_lr: If True, scale all statistics by learning rate
        use_momentum: If True, compute statistics of momentum buffers instead of raw gradients

    Returns:
        Dictionary containing:
            - 'spectrum': Singular values of gradient or momentum (numpy array)
            - 'frobenius_norm': Frobenius norm ||∇||_F
            - 'spectral_norm': Spectral norm (largest singular value) ||∇||_2
            - 'trace': Trace of gradient Gram matrix Tr(∇^T∇)
            - 'param_name': Name of the tracked parameter

    Example:
        >>> # Raw gradient statistics
        >>> grad_stats = compute_gradient_statistics(model, 'layers.0.conv1.weight')

        >>> # Momentum buffer statistics
        >>> grad_stats = compute_gradient_statistics(
        ...     model, 'layers.0.conv1.weight', optimizers=opts, use_momentum=True
        ... )
    """
    # Auto-detect parameter if not specified
    if param_name is None:
        param_name = get_important_conv_layer(model)
        if param_name is None:
            return {}

    # Find the parameter
    param = None
    for name, p in model.named_parameters():
        if name == param_name:
            param = p
            break

    if param is None:
        return {}

    # Get the data to analyze (gradient or momentum buffer)
    if use_momentum:
        if optimizers is None:
            raise ValueError("optimizers must be provided when use_momentum=True")

        # Build mapping from parameter id to optimizer state
        param_to_state = {}
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    if p in optimizer.state:
                        param_to_state[id(p)] = optimizer.state[p]

        # Get momentum buffer
        state = param_to_state.get(id(param))
        if state is None:
            return {}

        # Different optimizers store momentum under different keys
        if 'momentum_buffer' in state:
            data = state['momentum_buffer']
        elif 'exp_avg' in state:
            data = state['exp_avg']
        else:
            return {}
    else:
        if param.grad is None:
            return {}
        data = param.grad.data

    # Reshape to 2D if needed (conv layers are typically 4D: out_ch, in_ch, h, w)
    if data.ndim == 2:
        data_2d = data
    else:
        data_2d = data.reshape(data.size(0), -1)

    stats = {'param_name': param_name}

    try:
        # Move to CPU and convert to float32 for numerical stability
        data_cpu = data_2d.cpu().float()

        # Compute SVD to get spectrum
        _, s, _ = torch.svd(data_cpu)

        # Optionally scale by learning rate
        if scale_by_lr:
            s = s * lr
            scale_factor = lr
        else:
            scale_factor = 1.0

        stats['spectrum'] = s.numpy()

        # Spectral norm (largest singular value)
        stats['spectral_norm'] = s[0].item()

        # Frobenius norm: ||G||_F = sqrt(sum of squared entries)
        stats['frobenius_norm'] = torch.linalg.norm(data_cpu, ord='fro').item() * scale_factor

        # Trace of Gram matrix: Tr(G^T G) = sum of squared singular values
        stats['trace'] = (s ** 2).sum().item()

    except Exception as e:
        print(f"Warning: Could not compute gradient statistics for {param_name}: {e}")
        return {}

    return stats


def compute_gradient_spectra(
    model: nn.Module,
    param_names: List[str],
    optimizers: List[torch.optim.Optimizer] = None,
    lr: float = 1.0,
    scale_by_lr: bool = False,
    use_momentum: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Compute gradient spectra for multiple parameters.

    This function computes the spectrum (singular values) of gradients or momentum buffers
    for specified parameters, optionally scaled by learning rate.

    Args:
        model: PyTorch model
        param_names: List of parameter names to track
        optimizers: List of optimizers (required if use_momentum=True or scale_by_lr=True)
        lr: Learning rate to scale by (used if scale_by_lr=True)
        scale_by_lr: If True, scale all statistics by learning rate
        use_momentum: If True, compute spectra of momentum buffers instead of raw gradients
                      (requires optimizers to be provided)

    Returns:
        Dictionary mapping parameter names to their gradient/momentum statistics:
            - 'spectrum': Singular values of gradient or momentum (numpy array)
            - 'frobenius_norm': Frobenius norm
            - 'spectral_norm': Spectral norm (largest singular value)
            - 'trace': Trace of Gram matrix

    Example:
        >>> # Raw gradient spectra
        >>> grad_spectra = compute_gradient_spectra(model, param_names)

        >>> # Momentum buffer spectra, scaled by lr
        >>> grad_spectra = compute_gradient_spectra(
        ...     model, param_names, optimizers=opts, lr=0.001,
        ...     scale_by_lr=True, use_momentum=True
        ... )
    """
    gradient_spectra = {}

    # Build mapping from parameter id to optimizer state (if using momentum)
    param_to_state = {}
    if use_momentum:
        if optimizers is None:
            raise ValueError("optimizers must be provided when use_momentum=True")
        for optimizer in optimizers:
            # Iterate through ALL parameter groups, not just the first one
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param in optimizer.state:
                        param_to_state[id(param)] = optimizer.state[param]

    with torch.no_grad():
        for param_name in param_names:
            # Find the parameter
            param = None
            for name, p in model.named_parameters():
                if name == param_name:
                    param = p
                    break

            if param is None:
                continue

            # Get the data to analyze (gradient or momentum buffer)
            if use_momentum:
                state = param_to_state.get(id(param))
                if state is None:
                    continue

                # Different optimizers store momentum under different keys
                # Muon/Shampoo/SGD: 'momentum_buffer'
                # Adam/AdamW: 'exp_avg' (first moment)
                if 'momentum_buffer' in state:
                    data = state['momentum_buffer']
                elif 'exp_avg' in state:
                    data = state['exp_avg']
                else:
                    continue
            else:
                if param.grad is None:
                    continue
                data = param.grad.data

            # Reshape to 2D if needed
            if data.ndim == 2:
                data_2d = data
            else:
                data_2d = data.reshape(data.size(0), -1)

            try:
                # Move to CPU and convert to float32 for numerical stability
                data_cpu = data_2d.cpu().float()

                # Compute SVD to get spectrum
                _, s, _ = torch.svd(data_cpu)

                # Optionally scale by learning rate
                if scale_by_lr:
                    s = s * lr
                    scale_factor = lr
                else:
                    scale_factor = 1.0

                # Store statistics
                gradient_spectra[param_name] = {
                    'spectrum': s.numpy(),
                    'spectral_norm': s[0].item(),
                    'frobenius_norm': torch.linalg.norm(data_cpu, ord='fro').item() * scale_factor,
                    'trace': (s ** 2).sum().item()
                }

            except Exception as e:
                print(f"Warning: Could not compute gradient spectrum for {param_name}: {e}")
                continue

    return gradient_spectra

