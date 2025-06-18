import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ================= HELPER FUNCTIONS =================

def prepare_directories(output_dir):
    """Prepare output directories for training.

    Args:
        output_dir: Base output directory

    Returns:
        Tuple of (checkpoint_dir, image_dir)
    """
    # Create output directories
    image_dir = os.path.join(output_dir, 'images')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Directories prepared:")
    print(f"  - Images: {image_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")

    return checkpoint_dir, image_dir


def log_progress(iteration, trainer, log_file=None):
    """Log training progress to console and optionally to file.

    Args:
        iteration: Current iteration
        trainer: Trainer object with loss attributes
        log_file: Path to log file (optional)
    """
    # Format timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Prepare log message
    log_message = f"[{timestamp}] Iteration: {iteration + 1:08d}"

    # Find all loss attributes
    loss_values = {}
    for attr_name in dir(trainer):
        if (not callable(getattr(trainer, attr_name)) and
                not attr_name.startswith("__") and
                ('loss' in attr_name)):

            value = getattr(trainer, attr_name)
            if isinstance(value, torch.Tensor):
                value = value.item()

            loss_values[attr_name] = value
            log_message += f", {attr_name}: {value:.4f}"

    # Print to console
    print(log_message)

    # Write to log file if provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_message + "\n")


def write_images(image_outputs, display_size, image_dir, filename):
    """Save image grids for model outputs.

    Args:
        image_outputs: List of image tensors
        display_size: Number of images to display
        image_dir: Directory to save images
        filename: Output filename prefix
    """
    # Total number of output sets
    n = len(image_outputs)

    # Handle x->y translation (first half of outputs)
    image_tensor = torch.cat([images[:display_size] for images in image_outputs[0:n // 2]], 0)

    # If data is 3D, extract middle slice
    if len(image_tensor.shape) == 5:  # (B, C, D, H, W)
        depth_mid = image_tensor.size(2) // 2
        image_tensor = image_tensor[:, :, depth_mid, :, :]

    # Create and save grid for x->y
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_size, padding=2, normalize=True)
    vutils.save_image(image_grid, f'{image_dir}/{filename}_x2y.jpg', nrow=1)

    # Handle y->x translation (second half of outputs)
    image_tensor = torch.cat([images[:display_size] for images in image_outputs[n // 2:n]], 0)

    # If data is 3D, extract middle slice
    if len(image_tensor.shape) == 5:  # (B, C, D, H, W)
        depth_mid = image_tensor.size(2) // 2
        image_tensor = image_tensor[:, :, depth_mid, :, :]

    # Create and save grid for y->x
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_size, padding=2, normalize=True)
    vutils.save_image(image_grid, f'{image_dir}/{filename}_y2x.jpg', nrow=1)


def get_model_list(dirname, key):
    """Get the latest model file based on key.

    Args:
        dirname: Directory containing models
        key: Key to match in filename (e.g., 'gen', 'dis')

    Returns:
        Path to latest model file or None if no models found
    """
    if not os.path.exists(dirname):
        return None

    model_files = [os.path.join(dirname, f) for f in os.listdir(dirname)
                   if os.path.isfile(os.path.join(dirname, f)) and key in f and f.endswith('.pt')]

    if not model_files:
        return None

    model_files.sort()
    return model_files[-1]  # Return latest model


def weights_init(init_type='gaussian'):
    """Get a weight initialization function.

    Args:
        init_type: Type of initialization ('gaussian', 'kaiming', 'xavier', 'orthogonal')

    Returns:
        Initialization function
    """

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise ValueError(f"Unsupported initialization: {init_type}")

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def tensor_to_numpy(tensor):
    """Convert a tensor to numpy array.

    Args:
        tensor: Input tensor

    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def normalize_tensor(tensor, min_val=0, max_val=1):
    """Normalize a tensor to a specific range.

    Args:
        tensor: Input tensor
        min_val: Minimum value in output range
        max_val: Maximum value in output range

    Returns:
        Normalized tensor
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    if tensor_min == tensor_max:
        return torch.ones_like(tensor) * min_val

    normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized * (max_val - min_val) + min_val


def count_parameters(model):
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get the best available device.

    Returns:
        Device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_config(config, save_path):
    """Save configuration to file.

    Args:
        config: Configuration object
        save_path: Path to save the configuration
    """
    config.save(save_path)
    print(f"Configuration saved to {save_path}")


# ================= VISUALIZATION FUNCTIONS =================

def visualize_slice(volume, slice_idx=None, axis=0, cmap='gray', title=None, output_path=None):
    """Visualize a slice from a 3D volume.

    Args:
        volume: Volume tensor (C, D, H, W) or (D, H, W)
        slice_idx: Index of the slice to visualize (default: middle slice)
        axis: Axis to slice along (0=depth, 1=height, 2=width)
        cmap: Colormap for visualization
        title: Plot title
        output_path: Path to save the visualization

    Returns:
        Figure object
    """
    # Handle different input formats
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    # Remove channel dimension if present
    if volume.ndim == 4:  # (C, D, H, W)
        volume = volume[0]  # Take first channel

    # Set default slice index to middle
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    # Extract the slice
    if axis == 0:
        slice_data = volume[slice_idx, :, :]
        slice_name = f"Depth Slice (z={slice_idx})"
    elif axis == 1:
        slice_data = volume[:, slice_idx, :]
        slice_name = f"Height Slice (y={slice_idx})"
    else:
        slice_data = volume[:, :, slice_idx]
        slice_name = f"Width Slice (x={slice_idx})"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(slice_data, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set title
    if title:
        ax.set_title(f"{title} - {slice_name}")
    else:
        ax.set_title(slice_name)

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return fig


def visualize_multi_slice(volume, n_slices=3, axis=0, cmap='gray', title=None, output_path=None):
    """Visualize multiple slices from a 3D volume.

    Args:
        volume: Volume tensor (C, D, H, W) or (D, H, W)
        n_slices: Number of slices to visualize
        axis: Axis to slice along (0=depth, 1=height, 2=width)
        cmap: Colormap for visualization
        title: Plot title
        output_path: Path to save the visualization

    Returns:
        Figure object
    """
    # Handle different input formats
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    # Remove channel dimension if present
    if volume.ndim == 4:  # (C, D, H, W)
        volume = volume[0]  # Take first channel

    # Calculate slice indices
    axis_size = volume.shape[axis]
    indices = np.linspace(0, axis_size - 1, n_slices, dtype=int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))

    # Handle case with single slice
    if n_slices == 1:
        axes = [axes]

    # Plot each slice
    for i, idx in enumerate(indices):
        # Extract the slice
        if axis == 0:
            slice_data = volume[idx, :, :]
            slice_name = f"z={idx}"
        elif axis == 1:
            slice_data = volume[:, idx, :]
            slice_name = f"y={idx}"
        else:
            slice_data = volume[:, :, idx]
            slice_name = f"x={idx}"

        # Plot
        im = axes[i].imshow(slice_data, cmap=cmap)
        axes[i].set_title(slice_name)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return fig


def visualize_orthogonal(volume, slice_indices=None, cmap='gray', title=None, output_path=None):
    """Visualize orthogonal slices from a 3D volume.

    Args:
        volume: Volume tensor (C, D, H, W) or (D, H, W)
        slice_indices: Tuple of (z, y, x) indices (default: middle slices)
        cmap: Colormap for visualization
        title: Plot title
        output_path: Path to save the visualization

    Returns:
        Figure object
    """
    # Handle different input formats
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    # Remove channel dimension if present
    if volume.ndim == 4:  # (C, D, H, W)
        volume = volume[0]  # Take first channel

    # Set default indices to middle slices
    if slice_indices is None:
        d, h, w = volume.shape
        slice_indices = (d // 2, h // 2, w // 2)

    z_idx, y_idx, x_idx = slice_indices

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot axial slice (xy plane)
    axial = volume[z_idx, :, :]
    im0 = axes[0].imshow(axial, cmap=cmap)
    axes[0].set_title(f"Axial (z={z_idx})")
    axes[0].axhline(y=y_idx, color='r', linestyle='--')
    axes[0].axvline(x=x_idx, color='r', linestyle='--')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot coronal slice (xz plane)
    coronal = volume[:, y_idx, :]
    im1 = axes[1].imshow(coronal, cmap=cmap)
    axes[1].set_title(f"Coronal (y={y_idx})")
    axes[1].axhline(y=z_idx, color='r', linestyle='--')
    axes[1].axvline(x=x_idx, color='r', linestyle='--')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot sagittal slice (yz plane)
    sagittal = volume[:, :, x_idx]
    im2 = axes[2].imshow(sagittal, cmap=cmap)
    axes[2].set_title(f"Sagittal (x={x_idx})")
    axes[2].axhline(y=z_idx, color='r', linestyle='--')
    axes[2].axvline(x=y_idx, color='r', linestyle='--')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return fig


def create_volume_animation(volume, axis=0, interval=50, cmap='gray', title=None, output_path=None):
    """Create an animation of slices through a 3D volume.

    Args:
        volume: Volume tensor (C, D, H, W) or (D, H, W)
        axis: Axis to slice along (0=depth, 1=height, 2=width)
        interval: Delay between frames in milliseconds
        cmap: Colormap for visualization
        title: Animation title
        output_path: Path to save the animation (must be .gif or .mp4)

    Returns:
        Animation object
    """
    # Handle different input formats
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    # Remove channel dimension if present
    if volume.ndim == 4:  # (C, D, H, W)
        volume = volume[0]  # Take first channel

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Function to update the plot for each frame
    def update(frame):
        ax.clear()

        # Extract the slice
        if axis == 0:
            slice_data = volume[frame, :, :]
            ax.set_title(f"{title or 'Volume Animation'} - Depth Slice {frame}")
        elif axis == 1:
            slice_data = volume[:, frame, :]
            ax.set_title(f"{title or 'Volume Animation'} - Height Slice {frame}")
        else:
            slice_data = volume[:, :, frame]
            ax.set_title(f"{title or 'Volume Animation'} - Width Slice {frame}")

        # Plot
        im = ax.imshow(slice_data, cmap=cmap, animated=True)
        return [im]

    # Create animation
    frames = volume.shape[axis]
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    # Save if requested
    if output_path:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', dpi=100)
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', dpi=100)
        plt.close(fig)

    return anim


def visualize_comparison(image1, image2, titles=None, cmap='gray', output_path=None):
    """Visualize a comparison between two images or slices.

    Args:
        image1: First image/slice
        image2: Second image/slice
        titles: Tuple of (title1, title2)
        cmap: Colormap for visualization
        output_path: Path to save the visualization

    Returns:
        Figure object
    """
    # Handle different input formats
    if isinstance(image1, torch.Tensor):
        image1 = image1.detach().cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.detach().cpu().numpy()

    # Remove channel dimension if present
    if image1.ndim == 3:  # (C, H, W)
        image1 = image1[0]
    if image2.ndim == 3:  # (C, H, W)
        image2 = image2[0]

    # Handle 3D volumes by taking middle slice
    if image1.ndim == 3:  # (D, H, W)
        image1 = image1[image1.shape[0] // 2]
    if image2.ndim == 3:  # (D, H, W)
        image2 = image2[image2.shape[0] // 2]

    # Set default titles
    if titles is None:
        titles = ('Image 1', 'Image 2')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot images
    im1 = axes[0].imshow(image1, cmap=cmap)
    axes[0].set_title(titles[0])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(image2, cmap=cmap)
    axes[1].set_title(titles[1])
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return fig


def visualize_batch(batch, max_samples=16, is_3d=None, axis=0, cmap='gray', output_path=None):
    """Visualize a batch of images or volumes.

    Args:
        batch: Batch tensor (B, C, H, W) or (B, C, D, H, W)
        max_samples: Maximum number of samples to visualize
        is_3d: Whether the batch contains 3D volumes (inferred if None)
        axis: Axis to slice along for 3D volumes
        cmap: Colormap for visualization
        output_path: Path to save the visualization

    Returns:
        Figure object
    """
    # Handle different input formats
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu()

    # Determine if the batch contains 3D volumes
    if is_3d is None:
        is_3d = batch.ndim == 5

    # Limit number of samples
    batch_size = batch.shape[0]
    n_samples = min(batch_size, max_samples)
    batch = batch[:n_samples]

    if is_3d:
        # For 3D volumes, extract middle slices
        middle_slices = []
        for i in range(n_samples):
            volume = batch[i].squeeze(0).numpy() if batch[i].ndim > 4 else batch[i].numpy()

            # Extract slice
            if axis == 0:
                slice_idx = volume.shape[0] // 2
                slice_data = volume[slice_idx]
            elif axis == 1:
                slice_idx = volume.shape[1] // 2
                slice_data = volume[:, slice_idx]
            else:
                slice_idx = volume.shape[2] // 2
                slice_data = volume[:, :, slice_idx]

            middle_slices.append(slice_data)

        # Create grid of slices
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # Handle case with single sample
        if n_samples == 1:
            axes = np.array([axes])

        # Flatten axes for easier indexing
        axes = axes.flatten()

        # Plot each slice
        for i in range(n_samples):
            im = axes[i].imshow(middle_slices[i], cmap=cmap)
            axes[i].set_title(f"Sample {i}")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        # Hide unused axes
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

    else:
        # For 2D images, use torchvision's make_grid
        grid = vutils.make_grid(batch, nrow=4, normalize=True, pad_value=1)
        grid = grid.permute(1, 2, 0).numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12 * n_samples / 8))

        # Handle grayscale
        if grid.shape[2] == 1:
            ax.imshow(grid.squeeze(2), cmap=cmap)
        else:
            ax.imshow(grid)

        ax.axis('off')

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    return fig