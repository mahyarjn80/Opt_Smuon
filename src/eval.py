"""
Evaluation utilities for model testing.

This module provides functions for evaluating trained models on test/validation sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(model: nn.Module, loader, device: str = 'cuda', batch_size: int = 2000) -> tuple[float, float]:
    """
    Evaluate model on a data loader.
    
    This function computes loss and accuracy for a model on the given dataset.
    It's optimized for CifarLoader-style loaders that keep data on GPU.
    
    Args:
        model: PyTorch model to evaluate
        loader: Data loader (CifarLoader or standard PyTorch DataLoader)
        device: Device to run evaluation on ('cuda' or 'cpu')
        batch_size: Batch size for processing (to avoid memory issues)
        
    Returns:
        Tuple of (average_loss, average_accuracy)
        
    Example:
        >>> model = MyModel()
        >>> loader = CifarLoader('cifar10', train=False)
        >>> test_loss, test_acc = evaluate(model, loader)
        >>> print(f"Test accuracy: {test_acc:.4f}")
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # For CifarLoader-style loaders (images already on device)
        if hasattr(loader, 'normalize') and hasattr(loader, 'images'):
            test_images = loader.normalize(loader.images)
            labels = loader.labels
            
            # Process in batches to avoid memory issues
            for i in range(0, len(test_images), batch_size):
                inputs = test_images[i:i+batch_size]
                targets = labels[i:i+batch_size]
                
                outputs = model(inputs)
                
                # Compute loss
                loss = F.cross_entropy(outputs, targets, reduction='sum')
                total_loss += loss.item()
                
                # Compute accuracy
                total_correct += (outputs.argmax(1) == targets).float().sum().item()
                total_samples += len(inputs)
        else:
            # For standard PyTorch DataLoader
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Compute loss
                loss = F.cross_entropy(outputs, targets, reduction='sum')
                total_loss += loss.item()
                
                # Compute accuracy
                total_correct += (outputs.argmax(1) == targets).float().sum().item()
                total_samples += len(inputs)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def evaluate_with_label_smoothing(
    model: nn.Module,
    loader,
    label_smoothing: float = 0.0,
    device: str = 'cuda',
    batch_size: int = 2000
) -> tuple[float, float]:
    """
    Evaluate model with label smoothing support.
    
    Args:
        model: PyTorch model to evaluate
        loader: Data loader (CifarLoader or standard PyTorch DataLoader)
        label_smoothing: Label smoothing parameter (0.0 = no smoothing)
        device: Device to run evaluation on ('cuda' or 'cpu')
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # For CifarLoader-style loaders
        if hasattr(loader, 'normalize') and hasattr(loader, 'images'):
            test_images = loader.normalize(loader.images)
            labels = loader.labels
            
            for i in range(0, len(test_images), batch_size):
                inputs = test_images[i:i+batch_size]
                targets = labels[i:i+batch_size]
                
                outputs = model(inputs)
                
                # Compute loss with label smoothing
                loss = F.cross_entropy(
                    outputs, targets,
                    label_smoothing=label_smoothing,
                    reduction='sum'
                )
                total_loss += loss.item()
                
                # Compute accuracy (always uses hard labels)
                total_correct += (outputs.argmax(1) == targets).float().sum().item()
                total_samples += len(inputs)
        else:
            # For standard PyTorch DataLoader
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                loss = F.cross_entropy(
                    outputs, targets,
                    label_smoothing=label_smoothing,
                    reduction='sum'
                )
                total_loss += loss.item()
                
                total_correct += (outputs.argmax(1) == targets).float().sum().item()
                total_samples += len(inputs)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc

