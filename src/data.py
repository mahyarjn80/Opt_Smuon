"""
Data loading utilities for CIFAR-10 training.

This module provides:
- CifarLoader: Efficient data loader for CIFAR-10 with augmentation support
- Augmentation functions: batch_flip_lr, batch_crop
- CIFAR-10 normalization constants
"""

import os
from math import ceil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# CIFAR-10 normalization constants
CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    """
    Randomly flip images horizontally in a batch.
    
    Args:
        inputs: Batch of images [B, C, H, W]
        
    Returns:
        Batch of images with random horizontal flips applied
    """
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    """
    Apply random crops to a batch of images.
    
    Args:
        images: Batch of images [B, C, H, W]
        crop_size: Size of the square crop
        
    Returns:
        Batch of randomly cropped images [B, C, crop_size, crop_size]
    """
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    
    return images_out


class CifarLoader:
    """
    High-performance CIFAR-10 data loader with optional augmentation.
    
    Features:
    - Loads entire dataset to GPU for fast access
    - Supports horizontal flipping and random cropping
    - Uses half precision (float16) for memory efficiency
    - Implements efficient every-other-epoch flipping scheme
    
    Args:
        path: Directory to store/load CIFAR-10 data
        train: If True, load training set; otherwise load test set
        batch_size: Number of samples per batch
        aug: Dictionary of augmentation options. Supported keys:
            - 'flip': bool, enable horizontal flipping
            - 'translate': int, amount of random translation (in pixels)
    
    Example:
        >>> loader = CifarLoader('cifar10', train=True, batch_size=128, 
        ...                      aug={'flip': True, 'translate': 2})
        >>> for images, labels in loader:
        ...     # images: [batch_size, 3, 32, 32]
        ...     # labels: [batch_size]
        ...     pass
    """
    
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        
        # Download and save dataset if not exists
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)
        
        # Load dataset to GPU (or CPU if CUDA not available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = torch.load(data_path, map_location=torch.device(device))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        
        # Convert to half precision and channels_last format for efficiency
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        
        # Normalization transform
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        
        # Cached processed images (populated on first epoch)
        self.proc_images = {}
        self.epoch = 0
        
        # Validate and store augmentation config
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], f'Unrecognized augmentation key: {k}'
        
        # Batch configuration
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
    
    def __len__(self):
        """Return number of batches in an epoch."""
        if self.drop_last:
            return len(self.images) // self.batch_size
        else:
            return ceil(len(self.images) / self.batch_size)
    
    def __iter__(self):
        """Iterate over batches of (images, labels)."""
        
        # First epoch: preprocess images
        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')
        
        # Select appropriate image source based on augmentation
        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        
        # Flip all images together every other epoch
        # This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)
        
        self.epoch += 1
        
        # Generate batch indices (shuffled or sequential)
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        
        # Yield batches
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

