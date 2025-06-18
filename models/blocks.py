import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    get_conv_layer, get_norm_layer, get_activation, get_padding_layer,
    get_upsample_layer, get_avgpool_layer,
    ConvBlock, ResBlock, ResBlocks, AdaptiveInstanceNorm
)


class ContentEncoder(nn.Module):
    """Content encoder for style transfer networks.

    This module extracts content features from an image/volume,
    working for both 2D and 3D inputs.
    """

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activation, pad_type, dimensions):
        """Initialize content encoder.

        Args:
            n_downsample: Number of downsampling layers
            n_res: Number of residual blocks
            input_dim: Number of input channels
            dim: Base feature channels
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(ContentEncoder, self).__init__()

        self.model = []

        # Initial convolution
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm, activation, pad_type, dimensions)]

        # Downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm, activation, pad_type, dimensions)]
            dim *= 2

        # Residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activation, pad_type, dimensions)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        """Extract content features.

        Args:
            x: Input tensor

        Returns:
            Content features
        """
        return self.model(x)


class StyleEncoder(nn.Module):
    """Style encoder for style transfer networks.

    This module extracts style features from an image/volume,
    working for both 2D and 3D inputs.
    """

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activation, pad_type, dimensions):
        """Initialize style encoder.

        Args:
            n_downsample: Number of downsampling layers
            input_dim: Number of input channels
            dim: Base feature channels
            style_dim: Dimension of style code
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(StyleEncoder, self).__init__()

        self.model = []

        # Initial convolution
        self.model += [ConvBlock(input_dim, dim, 7, 1, 3, norm, activation, pad_type, dimensions)]

        # Downsampling blocks
        for i in range(2):
            self.model += [ConvBlock(dim, 2 * dim, 4, 2, 1, norm, activation, pad_type, dimensions)]
            dim *= 2

        # Additional downsampling
        for i in range(n_downsample - 2):
            self.model += [ConvBlock(dim, dim, 4, 2, 1, norm, activation, pad_type, dimensions)]

        # Global pooling and final convolution
        if dimensions == 2:
            self.model += [nn.AdaptiveAvgPool2d(1)]
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        else:
            self.model += [nn.AdaptiveAvgPool3d(1)]
            self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        """Extract style features.

        Args:
            x: Input tensor

        Returns:
            Style code
        """
        return self.model(x)


class Decoder(nn.Module):
    """Decoder for style transfer networks.

    This module generates an image/volume from content and style features,
    working for both 2D and 3D inputs.
    """

    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm, activation, pad_type, dimensions):
        """Initialize decoder.

        Args:
            n_upsample: Number of upsampling layers
            n_res: Number of residual blocks
            dim: Input feature channels
            output_dim: Number of output channels
            res_norm: Normalization type for residual blocks
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(Decoder, self).__init__()

        self.model = []

        # Residual blocks with AdaIN
        self.model += [ResBlocks(n_res, dim, res_norm, activation, pad_type, dimensions)]

        # Upsampling blocks
        for i in range(n_upsample):
            if dimensions == 2:
                self.model += [nn.Upsample(scale_factor=2, mode='nearest')]
                self.model += [ConvBlock(dim, dim // 2, 5, 1, 2, 'ln', activation, pad_type, dimensions)]
            else:
                self.model += [nn.Upsample(scale_factor=2, mode='nearest')]
                self.model += [ConvBlock(dim, dim // 2, 5, 1, 2, 'ln', activation, pad_type, dimensions)]
            dim //= 2

        # Final convolution
        self.model += [ConvBlock(dim, output_dim, 7, 1, 3, 'none', 'tanh', pad_type, dimensions)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """Generate output from features.

        Args:
            x: Input features

        Returns:
            Generated image/volume
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for GANs.

    This module implements a multi-scale discriminator that operates
    at different resolutions, working for both 2D and 3D inputs.
    """

    def __init__(self, input_dim, n_layers, dim, num_scales, norm, activation, pad_type, dimensions):
        """Initialize multi-scale discriminator.

        Args:
            input_dim: Number of input channels
            n_layers: Number of layers in each discriminator
            dim: Base feature channels
            num_scales: Number of scales
            norm: Normalization type
            activation: Activation type
            pad_type: Padding type
            dimensions: Number of dimensions (2 or 3)
        """
        super(MultiScaleDiscriminator, self).__init__()

        self.n_layers = n_layers
        self.dim = dim
        self.norm = norm
        self.activation = activation
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.dimensions = dimensions
        self.input_dim = input_dim

        # Create downsampling module
        if dimensions == 2:
            self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        else:
            self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)

        # Create discriminators at different scales
        self.discriminators = nn.ModuleList()
        for _ in range(self.num_scales):
            self.discriminators.append(self._make_net())

    def _make_net(self):
        """Create a single discriminator network.

        Returns:
            Discriminator network
        """
        dim = self.dim
        net = []

        # Initial convolution
        net += [ConvBlock(self.input_dim, dim, 4, 2, 1, 'none', self.activation, self.pad_type, self.dimensions)]

        # Additional layers with normalization
        for i in range(self.n_layers - 1):
            net += [ConvBlock(dim, dim * 2, 4, 2, 1, self.norm, self.activation, self.pad_type, self.dimensions)]
            dim *= 2

        # Final convolution
        if self.dimensions == 2:
            net += [nn.Conv2d(dim, 1, 1, 1, 0)]
        else:
            net += [nn.Conv3d(dim, 1, 1, 1, 0)]

        return nn.Sequential(*net)

    def forward(self, x):
        """Apply discriminator at multiple scales.

        Args:
            x: Input tensor

        Returns:
            List of outputs at different scales
        """
        outputs = []

        # Apply discriminator at each scale
        for model in self.discriminators:
            outputs.append(model(x))

            # Downsample for next scale
            if self.num_scales > 1:
                x = self.downsample(x)

        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        """Calculate discriminator loss.

        Args:
            input_fake: Fake input tensor
            input_real: Real input tensor

        Returns:
            Discriminator loss
        """
        # Forward pass for fake and real inputs
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)

        # Calculate loss
        loss = 0
        for out0, out1 in zip(outs0, outs1):
            # LSGAN loss
            loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)

        return loss

    def calc_gen_loss(self, input_fake):
        """Calculate generator loss.

        Args:
            input_fake: Fake input tensor

        Returns:
            Generator loss
        """
        # Forward pass for fake inputs
        outs0 = self.forward(input_fake)

        # Calculate loss
        loss = 0
        for out0 in outs0:
            # LSGAN loss
            loss += torch.mean((out0 - 1) ** 2)

        return loss