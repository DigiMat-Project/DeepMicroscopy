import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_norm_layer(norm_type, dimensions, num_features):
    """Get a normalization layer appropriate for the specified dimensions.

    Args:
        norm_type: Type of normalization ('bn', 'in', 'adain', 'ln', 'none')
        dimensions: Number of dimensions (2 or 3)
        num_features: Number of feature channels

    Returns:
        Normalization layer
    """
    if norm_type == 'bn':
        if dimensions == 2:
            return nn.BatchNorm2d(num_features)
        else:
            return nn.BatchNorm3d(num_features)
    elif norm_type == 'in':
        if dimensions == 2:
            return nn.InstanceNorm2d(num_features)
        else:
            return nn.InstanceNorm3d(num_features)
    elif norm_type == 'ln':
        from .norms import LayerNorm
        return LayerNorm(num_features)
    elif norm_type == 'adain':
        # Import AdaptiveInstanceNorm from norms
        if dimensions == 2:
            from .norms import AdaptiveInstanceNorm2d
            return AdaptiveInstanceNorm2d(num_features)
        else:
            from .norms import AdaptiveInstanceNorm3d
            return AdaptiveInstanceNorm3d(num_features)
    elif norm_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


def get_activation(activation_type):
    """Get an activation function.

    Args:
        activation_type: Type of activation ('relu', 'lrelu', 'prelu', 'tanh', 'none')

    Returns:
        Activation function
    """
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation_type == 'prelu':
        return nn.PReLU()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def get_padding_layer(pad_type, padding, dimensions):
    """Get a padding layer appropriate for the specified dimensions.

    Args:
        pad_type: Type of padding ('reflect', 'replicate', 'zero')
        padding: Padding size
        dimensions: Number of dimensions (2 or 3)

    Returns:
        Padding layer
    """
    if pad_type == 'reflect':
        if dimensions == 2:
            return nn.ReflectionPad2d(padding)
        else:
            return nn.ReflectionPad3d(padding)
    elif pad_type == 'replicate':
        if dimensions == 2:
            return nn.ReplicationPad2d(padding)
        else:
            return nn.ReplicationPad3d(padding)
    elif pad_type == 'zero':
        if dimensions == 2:
            return nn.ZeroPad2d(padding)
        else:
            return nn.ConstantPad3d(padding, 0)
    else:
        raise ValueError(f"Unsupported padding type: {pad_type}")


def get_conv_layer(dimensions, in_channels, out_channels, kernel_size, stride=1, bias=True):
    """Get a convolution layer appropriate for the specified dimensions.

    Args:
        dimensions: Number of dimensions (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Convolution stride
        bias: Whether to include bias

    Returns:
        Convolution layer
    """
    if dimensions == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=bias)


def get_upsample_layer(dimensions, scale_factor=2, mode='nearest'):
    """Get an upsampling layer appropriate for the specified dimensions.

    Args:
        dimensions: Number of dimensions (2 or 3)
        scale_factor: Factor to scale by
        mode: Upsampling mode

    Returns:
        Upsampling layer
    """
    if dimensions == 2:
        return nn.Upsample(scale_factor=scale_factor, mode=mode)
    else:
        return nn.Upsample(scale_factor=scale_factor, mode=mode)


def get_avgpool_layer(dimensions, kernel_size, stride=None, padding=0):
    """Get an average pooling layer appropriate for the specified dimensions.

    Args:
        dimensions: Number of dimensions (2 or 3)
        kernel_size: Size of the pooling kernel
        stride: Pooling stride
        padding: Padding size

    Returns:
        Average pooling layer
    """
    if dimensions == 2:
        return nn.AvgPool2d(kernel_size, stride, padding)
    else:
        return nn.AvgPool3d(kernel_size, stride, padding)


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style transfer.

    This module applies instance normalization with learnable affine parameters
    that are dynamically computed from a style code. It works for both 2D and 3D inputs.
    """

    def __init__(self, num_features, dimensions, eps=1e-5, momentum=0.1):
        """Initialize adaptive instance normalization.

        Args:
            num_features: Number of feature channels
            dimensions: Number of dimensions (2 or 3)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.dimensions = dimensions
        self.weight = None
        self.bias = None

        # Register buffers for running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """Apply adaptive instance normalization.

        Args:
            x: Input tensor (B, C, ...) where ... represents spatial dimensions

        Returns:
            Normalized tensor
        """
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"

        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Reshape for batch normalization
        if self.dimensions == 2:
            x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        else:
            x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        # Apply batch normalization
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        # Restore original shape
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, dimensions={self.dimensions})"


class ConvBlock(nn.Module):
    """Convolution block with optional normalization and activation.

    This module combines padding, convolution, normalization, and activation
    into a single block that works for both 2D and 3D inputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero',
                 dimensions=3):
        """Initialize convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Convolution stride
            padding: Padding size
            norm: Normalization type ('bn', 'in', 'none')
            activation: Activation type ('relu', 'lrelu', 'prelu', 'tanh', 'none')
            pad_type: Padding type ('reflect', 'replicate', 'zero')
            dimensions: Number of dimensions (2 or 3)
        """
        super(ConvBlock, self).__init__()

        self.dimensions = dimensions
        self.use_bias = norm == 'none'

        # Add padding layer if needed
        if padding > 0:
            self.pad = get_padding_layer(pad_type, padding, dimensions)
        else:
            self.pad = None

        # Add convolution layer
        self.conv = get_conv_layer(dimensions, in_channels, out_channels, kernel_size, stride, self.use_bias)

        # Add normalization layer if needed
        self.norm = get_norm_layer(norm, dimensions, out_channels) if norm != 'none' else None

        # Add activation layer if needed
        self.activation = get_activation(activation) if activation != 'none' else None

    def forward(self, x):
        """Apply convolution block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.pad:
            x = self.pad(x)

        x = self.conv(x)

        if self.norm:
            x = self.norm(x)

        if self.activation:
            x = self.activation(x)

        return x


class ResBlock(nn.Module):
    """Residual block with normalization.

    This module implements a residual block that works for both 2D and 3D inputs.
    """

    def __init__(self, channels, norm='in', activation='relu', pad_type='zero', dimensions=3):
        """Initialize residual block.

        Args:
            channels: Number of channels
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(ResBlock, self).__init__()

        # Create a sequence of two convolution blocks
        self.model = nn.Sequential(
            ConvBlock(channels, channels, 3, 1, 1, norm, activation, pad_type, dimensions),
            ConvBlock(channels, channels, 3, 1, 1, norm, 'none', pad_type, dimensions)
        )

    def forward(self, x):
        """Apply residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    """Sequential container for multiple residual blocks.

    This module stacks multiple residual blocks for both 2D and 3D inputs.
    """

    def __init__(self, num_blocks, channels, norm='in', activation='relu', pad_type='zero', dimensions=3):
        """Initialize residual blocks.

        Args:
            num_blocks: Number of residual blocks
            channels: Number of channels
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(ResBlocks, self).__init__()

        # Create a sequence of residual blocks
        blocks = [
            ResBlock(channels, norm, activation, pad_type, dimensions)
            for _ in range(num_blocks)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """Apply residual blocks.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.model(x)


class UpBlock(nn.Module):
    """Upsampling block with convolution.

    This module combines upsampling and convolution for both 2D and 3D inputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero',
                 dimensions=3, scale_factor=2, mode='nearest'):
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Convolution stride
            padding: Padding size
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
            scale_factor: Upsampling scale factor
            mode: Upsampling mode
        """
        super(UpBlock, self).__init__()

        # Create upsampling layer
        self.upsample = get_upsample_layer(dimensions, scale_factor, mode)

        # Create convolution block
        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, stride,
            padding, norm, activation, pad_type, dimensions
        )

    def forward(self, x):
        """Apply upsampling block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    This module implements a multi-layer perceptron for style encoding.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, norm='none', activation='relu'):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            n_blocks: Number of hidden layers
            norm: Normalization type
            activation: Activation type
        """
        super(MLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if norm == 'bn':
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(get_activation(activation))

        # Hidden layers
        for _ in range(n_blocks - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm == 'bn':
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Apply MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.model(x.view(x.size(0), -1))