import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization for both 2D and 3D inputs.

    This module applies normalization over all features of each sample,
    independently for each sample in a batch.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        """Initialize layer normalization.

        Args:
            num_features: Number of feature channels
            eps: Small constant for numerical stability
            affine: Whether to apply learnable affine transformation
        """
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x: Input tensor (B, C, ...)

        Returns:
            Normalized tensor
        """
        # Calculate mean and standard deviation
        shape = [-1] + [1] * (x.dim() - 1)

        if x.size(0) == 1:
            # Special case for batch size 1
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            # Regular case
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        # Normalize
        x = (x - mean) / (std + self.eps)

        # Apply affine transformation if enabled
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps}, affine={self.affine})"


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization for 2D inputs.

    This module applies instance normalization with learnable affine parameters
    that are dynamically computed from a style code.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """Initialize adaptive instance normalization.

        Args:
            num_features: Number of feature channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None

        # Register buffers for running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """Apply adaptive instance normalization.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Normalized tensor
        """
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"

        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Reshape for batch normalization
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        # Apply batch normalization
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        # Restore original shape
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps})"


class AdaptiveInstanceNorm3d(nn.Module):
    """Adaptive Instance Normalization for 3D inputs.

    This module applies instance normalization with learnable affine parameters
    that are dynamically computed from a style code.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """Initialize adaptive instance normalization.

        Args:
            num_features: Number of feature channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super(AdaptiveInstanceNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None

        # Register buffers for running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """Apply adaptive instance normalization.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Normalized tensor
        """
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"

        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Reshape for batch normalization
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        # Apply batch normalization
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        # Restore original shape
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps})"


def get_adain_layer(dimensions, num_features, eps=1e-5, momentum=0.1):
    """Get an adaptive instance normalization layer appropriate for the specified dimensions.

    Args:
        dimensions: Number of dimensions (2 or 3)
        num_features: Number of feature channels
        eps: Small constant for numerical stability
        momentum: Momentum for running statistics

    Returns:
        AdaIN layer
    """
    if dimensions == 2:
        return AdaptiveInstanceNorm2d(num_features, eps, momentum)
    else:
        return AdaptiveInstanceNorm3d(num_features, eps, momentum)