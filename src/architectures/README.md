This package houses the PyTorch architectures that can be used with our code.

## Available Architectures

- **CifarNet**: High-performance CNN from airbench with whitening preprocessing, optimized for CIFAR-10
- **CNN**: Simple convolutional neural network
- **MLP**: Multi-layer perceptron
- **VIT**: Vision Transformer
- **LSTM**: Long Short-Term Memory network
- **Mamba**: Mamba architecture
- **Transformer**: Transformer architecture
- **Resnet**: Residual network

## Usage

All architectures follow the same interface:

```python
from src.architectures import CifarNet

# Create architecture specification
arch = CifarNet(block1_width=64, block2_width=256, block3_width=256)

# Instantiate the model
model = arch.create(input_shape=(3, 32, 32), output_dim=10)
```

With the training scripts:

```bash
python main_dual_training.py --arch cifarnet
python main_dual_training.py --arch mlp --arch.hidden-dim 512
```
