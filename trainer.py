import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from models.generator import Generator
from models.discriminator import Discriminator
from utils import get_model_list


class Trainer(nn.Module):
    """Trainer for MUNIT model.

    This class manages the training process for the MUNIT model,
    including optimization, loss calculation, and checkpointing.
    It supports both 2D and 3D data.
    """

    def __init__(self, config):
        """Initialize trainer.

        Args:
            config: Configuration object
        """
        super(Trainer, self).__init__()

        self.config = config
        self.dimensions = config.dimensions

        # Style dimension
        self.style_dim = config.get('model.style_dim', 8)

        # Loss weights
        self.gan_w = config.get('training.loss_weights.gan', 1.0)
        self.recon_x_w = config.get('training.loss_weights.recon_x', 1.0)
        self.recon_s_w = config.get('training.loss_weights.recon_s', 1.0)
        self.recon_c_w = config.get('training.loss_weights.recon_c', 1.0)
        self.recon_x_cyc_w = config.get('training.loss_weights.recon_x_cyc', 1.0)

        # Initialize generators for domain X and Y
        self.gen_x = Generator(config)
        self.gen_y = Generator(config)

        # Initialize discriminators for domain X and Y
        self.dis_x = Discriminator(config)
        self.dis_y = Discriminator(config)

        # Initialize display style codes
        display_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.dimensions == 2:
            self.s_x = torch.randn(display_size, self.style_dim, 1, 1).to(device)
            self.s_y = torch.randn(display_size, self.style_dim, 1, 1).to(device)
        else:
            self.s_x = torch.randn(display_size, self.style_dim, 1, 1, 1).to(device)
            self.s_y = torch.randn(display_size, self.style_dim, 1, 1, 1).to(device)

        # Setup optimizers
        self._setup_optimizers()

        # Initialize loss tracking attributes
        self.loss_gen_total = 0
        self.loss_gen_recon_x_x = 0
        self.loss_gen_recon_x_y = 0
        self.loss_gen_recon_s_x = 0
        self.loss_gen_recon_s_y = 0
        self.loss_gen_recon_c_x = 0
        self.loss_gen_recon_c_y = 0
        self.loss_gen_cycrecon_x_x = 0
        self.loss_gen_cycrecon_x_y = 0
        self.loss_gen_adv_x = 0
        self.loss_gen_adv_y = 0
        self.loss_dis_x = 0
        self.loss_dis_y = 0
        self.loss_dis_total = 0

    def _setup_optimizers(self):
        """Setup optimizers and learning rate schedulers."""
        # Get optimizer parameters
        lr = self.config.get('training.optimizer.lr', 0.0001)
        beta1 = self.config.get('training.optimizer.beta1', 0.5)
        beta2 = self.config.get('training.optimizer.beta2', 0.999)
        weight_decay = self.config.get('training.optimizer.weight_decay', 0.0001)

        # Collect discriminator and generator parameters
        dis_params = list(self.dis_x.parameters()) + list(self.dis_y.parameters())
        gen_params = list(self.gen_x.parameters()) + list(self.gen_y.parameters())

        # Create optimizers
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )

        # Get scheduler parameters
        scheduler_step = self.config.get('training.scheduler.step_size', 10000)
        scheduler_gamma = self.config.get('training.scheduler.gamma', 0.8)

        # Create schedulers
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_opt, step_size=scheduler_step, gamma=scheduler_gamma)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_opt, step_size=scheduler_step, gamma=scheduler_gamma)

    def recon_criterion(self, input, target):
        """Reconstruction loss criterion (L1 loss).

        Args:
            input: Input tensor
            target: Target tensor

        Returns:
            L1 loss
        """
        return torch.mean(torch.abs(input - target))

    def forward(self, x_x, x_y):
        """Forward pass for inference.

        Args:
            x_x: Input from domain X
            x_y: Input from domain Y

        Returns:
            Tuple of (x_xy, x_yx) - translations between domains
        """
        self.eval()

        # Use fixed style codes for visualization
        s_x = torch.autograd.Variable(self.s_x)
        s_y = torch.autograd.Variable(self.s_y)

        # Extract content codes
        c_x, _ = self.gen_x.encode(x_x)
        c_y, _ = self.gen_y.encode(x_y)

        # Translate domains
        x_yx = self.gen_x.decode(c_y, s_x)
        x_xy = self.gen_y.decode(c_x, s_y)

        self.train()
        return x_xy, x_yx

    def gen_update(self, x_x, x_y):
        """Update generators.

        Args:
            x_x: Input from domain X
            x_y: Input from domain Y
        """
        self.gen_opt.zero_grad()

        # Get device and create random style codes
        device = x_x.device
        if self.dimensions == 2:
            s_x = torch.randn(x_x.size(0), self.style_dim, 1, 1).to(device)
            s_y = torch.randn(x_y.size(0), self.style_dim, 1, 1).to(device)
        else:
            s_x = torch.randn(x_x.size(0), self.style_dim, 1, 1, 1).to(device)
            s_y = torch.randn(x_y.size(0), self.style_dim, 1, 1, 1).to(device)

        # ================== Encode ==================
        c_x, s_x_prime = self.gen_x.encode(x_x)
        c_y, s_y_prime = self.gen_y.encode(x_y)

        # ================== Reconstruct ==================
        x_x_recon = self.gen_x.decode(c_x, s_x_prime)
        x_y_recon = self.gen_y.decode(c_y, s_y_prime)

        # ================== Translate ==================
        x_yx = self.gen_x.decode(c_y, s_x)
        x_xy = self.gen_y.decode(c_x, s_y)

        # ================== Cycle ==================
        c_y_recon, s_x_recon = self.gen_x.encode(x_yx)
        c_x_recon, s_y_recon = self.gen_y.encode(x_xy)

        x_xyx = None
        x_yxy = None
        if self.recon_x_cyc_w > 0:
            x_xyx = self.gen_x.decode(c_x_recon, s_x_prime)
            x_yxy = self.gen_y.decode(c_y_recon, s_y_prime)

        # ================== Calculate losses ==================
        # Auto-encoder reconstruction loss
        self.loss_gen_recon_x_x = self.recon_criterion(x_x_recon, x_x)
        self.loss_gen_recon_x_y = self.recon_criterion(x_y_recon, x_y)

        # Style reconstruction loss
        self.loss_gen_recon_s_x = self.recon_criterion(s_x_recon, s_x)
        self.loss_gen_recon_s_y = self.recon_criterion(s_y_recon, s_y)

        # Content reconstruction loss
        self.loss_gen_recon_c_x = self.recon_criterion(c_x_recon, c_x)
        self.loss_gen_recon_c_y = self.recon_criterion(c_y_recon, c_y)

        # Cycle-consistency loss
        self.loss_gen_cycrecon_x_x = 0
        self.loss_gen_cycrecon_x_y = 0
        if self.recon_x_cyc_w > 0:
            self.loss_gen_cycrecon_x_x = self.recon_criterion(x_xyx, x_x)
            self.loss_gen_cycrecon_x_y = self.recon_criterion(x_yxy, x_y)

        # Adversarial loss
        self.loss_gen_adv_x = self.dis_x.calc_gen_loss(x_yx)
        self.loss_gen_adv_y = self.dis_y.calc_gen_loss(x_xy)

        # Total loss
        self.loss_gen_total = (
                self.gan_w * (self.loss_gen_adv_x + self.loss_gen_adv_y) +
                self.recon_x_w * (self.loss_gen_recon_x_x + self.loss_gen_recon_x_y) +
                self.recon_s_w * (self.loss_gen_recon_s_x + self.loss_gen_recon_s_y) +
                self.recon_c_w * (self.loss_gen_recon_c_x + self.loss_gen_recon_c_y) +
                self.recon_x_cyc_w * (self.loss_gen_cycrecon_x_x + self.loss_gen_cycrecon_x_y)
        )

        # Update generators
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_x, x_y):
        """Sample translations for visualization.

        Args:
            x_x: Input batch from domain X
            x_y: Input batch from domain Y

        Returns:
            List of tensors for visualization
        """
        self.eval()

        # Get device
        device = x_x.device

        # Generate random style codes
        batch_size = x_x.size(0)
        if self.dimensions == 2:
            s_x1 = torch.autograd.Variable(self.s_x)
            s_y1 = torch.autograd.Variable(self.s_y)
            s_x2 = torch.randn(batch_size, self.style_dim, 1, 1).to(device)
            s_y2 = torch.randn(batch_size, self.style_dim, 1, 1).to(device)
        else:
            s_x1 = torch.autograd.Variable(self.s_x)
            s_y1 = torch.autograd.Variable(self.s_y)
            s_x2 = torch.randn(batch_size, self.style_dim, 1, 1, 1).to(device)
            s_y2 = torch.randn(batch_size, self.style_dim, 1, 1, 1).to(device)

        # Initialize output lists
        x_x_recon, x_y_recon = [], []
        x_yx1, x_yx2 = [], []
        x_xy1, x_xy2 = [], []

        # Process each sample in the batch
        for i in range(batch_size):
            # Get individual samples
            x_x_i = x_x[i].unsqueeze(0)
            x_y_i = x_y[i].unsqueeze(0)

            # Encode content
            c_x, s_x_fake = self.gen_x.encode(x_x_i)
            c_y, s_y_fake = self.gen_y.encode(x_y_i)

            # Reconstruct
            x_x_recon.append(self.gen_x.decode(c_x, s_x_fake))
            x_y_recon.append(self.gen_y.decode(c_y, s_y_fake))

            # Translate with fixed styles
            x_yx1.append(self.gen_x.decode(c_y, s_x1[i].unsqueeze(0)))
            x_xy1.append(self.gen_y.decode(c_x, s_y1[i].unsqueeze(0)))

            # Translate with random styles
            x_yx2.append(self.gen_x.decode(c_y, s_x2[i].unsqueeze(0)))
            x_xy2.append(self.gen_y.decode(c_x, s_y2[i].unsqueeze(0)))

        # Concatenate outputs
        x_x_recon = torch.cat(x_x_recon)
        x_y_recon = torch.cat(x_y_recon)
        x_yx1 = torch.cat(x_yx1)
        x_yx2 = torch.cat(x_yx2)
        x_xy1 = torch.cat(x_xy1)
        x_xy2 = torch.cat(x_xy2)

        self.train()

        # Return outputs for visualization
        return x_x, x_x_recon, x_xy1, x_xy2, x_y, x_y_recon, x_yx1, x_yx2

    def dis_update(self, x_x, x_y):
        """Update discriminators.

        Args:
            x_x: Input from domain X
            x_y: Input from domain Y
        """
        self.dis_opt.zero_grad()

        # Get device and create random style codes
        device = x_x.device
        if self.dimensions == 2:
            s_x = torch.randn(x_x.size(0), self.style_dim, 1, 1).to(device)
            s_y = torch.randn(x_y.size(0), self.style_dim, 1, 1).to(device)
        else:
            s_x = torch.randn(x_x.size(0), self.style_dim, 1, 1, 1).to(device)
            s_y = torch.randn(x_y.size(0), self.style_dim, 1, 1, 1).to(device)

        # Encode content
        c_x, _ = self.gen_x.encode(x_x)
        c_y, _ = self.gen_y.encode(x_y)

        # Translate
        x_yx = self.gen_x.decode(c_y, s_x)
        x_xy = self.gen_y.decode(c_x, s_y)

        # Calculate discriminator losses
        self.loss_dis_x = self.dis_x.calc_dis_loss(x_yx.detach(), x_x)
        self.loss_dis_y = self.dis_y.calc_dis_loss(x_xy.detach(), x_y)

        # Total loss
        self.loss_dis_total = self.gan_w * (self.loss_dis_x + self.loss_dis_y)

        # Update discriminators
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        """Update learning rates for both optimizers."""
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, device):
        """Resume training from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoints
            device: Device to load the model on

        Returns:
            Iteration to resume from
        """
        # Load generator checkpoint
        last_model_name = get_model_list(checkpoint_dir, "gen")
        if last_model_name is None:
            return 0

        state_dict = torch.load(last_model_name, map_location=device)
        self.gen_x.load_state_dict(state_dict['x'])
        self.gen_y.load_state_dict(state_dict['y'])

        # Parse iteration from filename
        iterations = int(last_model_name[-11:-3])

        # Load discriminator checkpoint
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name, map_location=device)
        self.dis_x.load_state_dict(state_dict['x'])
        self.dis_y.load_state_dict(state_dict['y'])

        # Load optimizer checkpoint
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'), map_location=device)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        # Update learning rate schedulers
        self.dis_scheduler = lr_scheduler.StepLR(
            self.dis_opt,
            step_size=self.config.get('training.scheduler.step_size', 10000),
            gamma=self.config.get('training.scheduler.gamma', 0.8),
            last_epoch=iterations
        )
        self.gen_scheduler = lr_scheduler.StepLR(
            self.gen_opt,
            step_size=self.config.get('training.scheduler.step_size', 10000),
            gamma=self.config.get('training.scheduler.gamma', 0.8),
            last_epoch=iterations
        )

        print(f'Resumed from iteration {iterations}')
        return iterations

    def save(self, snapshot_dir, iterations):
        """Save model checkpoint.

        Args:
            snapshot_dir: Directory to save the checkpoint
            iterations: Current iteration
        """
        # Create filenames
        gen_name = os.path.join(snapshot_dir, f'gen_{iterations + 1:08d}.pt')
        dis_name = os.path.join(snapshot_dir, f'dis_{iterations + 1:08d}.pt')
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')

        # Save generators
        torch.save({'x': self.gen_x.state_dict(), 'y': self.gen_y.state_dict()}, gen_name)

        # Save discriminators
        torch.save({'x': self.dis_x.state_dict(), 'y': self.dis_y.state_dict()}, dis_name)

        # Save optimizers
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)