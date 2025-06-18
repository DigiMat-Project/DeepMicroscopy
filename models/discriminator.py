import torch.nn as nn
from models.blocks import MultiScaleDiscriminator


class Discriminator(nn.Module):
    """Dimension-agnostic discriminator for GAN training.

    This discriminator works for both 2D and 3D data and implements
    a multi-scale patch-based architecture for better stability.
    """

    def __init__(self, config):
        """Initialize discriminator.

        Args:
            config: Configuration object
        """
        super(Discriminator, self).__init__()

        self.config = config
        self.dimensions = config.dimensions

        # Get discriminator parameters from config
        input_nc = 1  # Single channel input
        n_layers = config.get('model.discriminator.n_layers', 4)
        last_nf = config.get('model.discriminator.last_nf', 8)
        num_scales = config.get('model.discriminator.num_scales', 3)
        norm = config.get('model.discriminator.norm', 'none')
        activation = config.get('model.discriminator.activation', 'lrelu')
        pad_type = config.get('model.discriminator.pad_type', 'reflect')

        # Create multi-scale discriminator
        self.model = MultiScaleDiscriminator(
            input_dim=input_nc,
            n_layers=n_layers,
            dim=last_nf,
            num_scales=num_scales,
            norm=norm,
            activation=activation,
            pad_type=pad_type,
            dimensions=self.dimensions
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights by properly handling ConvBlock modules."""
        for module in self.modules():
            classname = module.__class__.__name__
            
            # Initialize direct Conv layers
            if classname.find('Conv') != -1 and hasattr(module, 'weight'):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            
            # Initialize Conv layers inside ConvBlock modules
            elif classname == 'ConvBlock':
                if hasattr(module, 'conv') and hasattr(module.conv, 'weight'):
                    nn.init.normal_(module.conv.weight.data, 0.0, 0.02)
                    if hasattr(module.conv, 'bias') and module.conv.bias is not None:
                        nn.init.constant_(module.conv.bias.data, 0.0)
            
            # Initialize BatchNorm layers
            elif classname.find('BatchNorm') != -1 and hasattr(module, 'weight'):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        """Apply discriminator.

        Args:
            x: Input tensor (B, C, ...) where ... are the spatial dimensions

        Returns:
            List of outputs at different scales
        """
        return self.model(x)

    def calc_dis_loss(self, input_fake, input_real):
        """Calculate discriminator loss.

        Args:
            input_fake: Fake input tensor
            input_real: Real input tensor

        Returns:
            Discriminator loss
        """
        return self.model.calc_dis_loss(input_fake, input_real)

    def calc_gen_loss(self, input_fake):
        """Calculate generator loss based on discriminator output.

        Args:
            input_fake: Fake input tensor

        Returns:
            Generator loss
        """
        return self.model.calc_gen_loss(input_fake)