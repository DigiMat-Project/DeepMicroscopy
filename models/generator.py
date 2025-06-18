import torch.nn as nn
from models.layers import MLP
from models.blocks import ContentEncoder, StyleEncoder, Decoder


class Generator(nn.Module):
    """Dimension-agnostic generator for image-to-image translation.

    This generator implements the MUNIT architecture with separate
    content and style encoders, working for both 2D and 3D data.
    """

    def __init__(self, config):
        """Initialize generator.

        Args:
            config: Configuration object
        """
        super(Generator, self).__init__()

        self.config = config
        self.dimensions = config.dimensions

        # Get parameters from config
        input_nc = 1  # Single channel input
        style_dim = config.get('model.style_dim', 8)

        # Content encoder parameters
        n_downsample = config.get('model.content_encoder.n_downsample', 3)
        n_res = config.get('model.content_encoder.n_res', 4)
        last_nf = config.get('model.content_encoder.last_nf', 8)
        content_norm = config.get('model.content_encoder.norm', 'in')
        content_activation = config.get('model.content_encoder.activation', 'relu')
        content_pad_type = config.get('model.content_encoder.pad_type', 'reflect')

        # Style encoder parameters
        style_n_downsample = config.get('model.style_encoder.n_downsample', 4)
        style_norm = config.get('model.style_encoder.norm', 'none')
        style_activation = config.get('model.style_encoder.activation', 'relu')
        style_pad_type = config.get('model.style_encoder.pad_type', 'reflect')

        # Decoder parameters
        n_upsample = config.get('model.decoder.n_upsample', 3)
        decoder_n_res = config.get('model.decoder.n_res', 4)
        res_norm = config.get('model.decoder.res_norm', 'adain')
        decoder_activation = config.get('model.decoder.activation', 'relu')
        decoder_pad_type = config.get('model.decoder.pad_type', 'reflect')

        # MLP parameters
        mlp_dim = config.get('model.style_encoder.mlp_dim', 256)
        mlp_n_blocks = config.get('model.style_encoder.mlp_n_blocks', 3)
        mlp_norm = config.get('model.style_encoder.mlp_norm', 'none')
        mlp_activation = config.get('model.style_encoder.mlp_activation', 'relu')

        # Create content encoder
        self.content_encoder = ContentEncoder(
            n_downsample=n_downsample,
            n_res=n_res,
            input_dim=input_nc,
            dim=last_nf,
            norm=content_norm,
            activation=content_activation,
            pad_type=content_pad_type,
            dimensions=self.dimensions
        )

        # Create style encoder
        self.style_encoder = StyleEncoder(
            n_downsample=style_n_downsample,
            input_dim=input_nc,
            dim=last_nf,
            style_dim=style_dim,
            norm=style_norm,
            activation=style_activation,
            pad_type=style_pad_type,
            dimensions=self.dimensions
        )

        # Create decoder
        self.decoder = Decoder(
            n_upsample=n_upsample,
            n_res=decoder_n_res,
            dim=self.content_encoder.output_dim,
            output_dim=input_nc,
            res_norm=res_norm,
            activation=decoder_activation,
            pad_type=decoder_pad_type,
            dimensions=self.dimensions
        )
        
        # Create MLP for AdaIN parameters
        self.mlp = MLP(
            input_dim=style_dim,
            output_dim=self._get_num_adain_params(self.decoder),
            hidden_dim=mlp_dim,
            n_blocks=mlp_n_blocks,
            norm=mlp_norm,
            activation=mlp_activation
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights properly for nested modules."""
        for module in self.modules():
            if hasattr(module, 'weight') and 'Conv' in module.__class__.__name__:
                # Direct weight initialization for convolution layers
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
            elif hasattr(module, 'weight') and 'BatchNorm' in module.__class__.__name__:
                # Direct weight initialization for batch norm layers
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x):
        """Apply generator.

        Args:
            x: Input tensor

        Returns:
            Reconstructed output
        """
        # Extract content and style
        content, style_fake = self.encode(x)

        # Reconstruct image
        images_recon = self.decode(content, style_fake)

        return images_recon

    def encode(self, x):
        """Encode input into content and style codes.

        Args:
            x: Input tensor

        Returns:
            Tuple of (content_code, style_code)
        """
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style

    def decode(self, content, style):
        """Decode content and style codes into an image.

        Args:
            content: Content code
            style: Style code

        Returns:
            Reconstructed image
        """
        # Generate AdaIN parameters
        adain_params = self.mlp(style)

        # Assign AdaIN parameters to decoder
        self._assign_adain_params(adain_params, self.decoder)

        # Generate image
        images = self.decoder(content)
        return images

    def sample(self, content, style):
        """Sample a new image with given content and style.

        Args:
            content: Content code
            style: Style code

        Returns:
            Generated image
        """
        # Generate AdaIN parameters
        adain_params = self.mlp(style)

        # Assign AdaIN parameters to decoder
        self._assign_adain_params(adain_params, self.decoder)

        # Generate image
        images = self.decoder(content)
        return images

    def _get_num_adain_params(self, model):
        """Get the number of AdaIN parameters in the model.

        Args:
            model: Model to count AdaIN parameters

        Returns:
            Number of AdaIN parameters
        """
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d" or m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def _assign_adain_params(self, adain_params, model):
        """Assign AdaIN parameters to the model.

        Args:
            adain_params: AdaIN parameters
            model: Model to assign parameters to
        """
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d" or m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)

                # Move to next parameters
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]