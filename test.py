import os
import sys
import random
import argparse
import glob
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from trainer import Trainer

# Set random seeds for reproducibility
torch.manual_seed(20250616)
random.seed(20250616)
np.random.seed(20250616)
cudnn.benchmark = True


class VolumeProcessor:
    """Base class for volume processing"""
    
    def __init__(self, model, device, scale_factor=6.4):
        self.model = model
        self.device = device
        self.scale_factor = scale_factor
        
    def generate_style_codes(self, num_styles, style_dim, is_3d=True):
        """Generate random style codes."""
        if is_3d:
            return torch.randn(num_styles, style_dim, 1, 1, 1).to(self.device)
        else:
            return torch.randn(num_styles, style_dim, 1, 1).to(self.device)


class OverlapInference3D(VolumeProcessor):
    """
    3D overlap inference for large volumes using sliding window approach
    """
    
    def __init__(self, model, device, patch_size=(60, 60, 60), overlap=10, crop_border=5, scale_factor=6.4):
        super().__init__(model, device, scale_factor)
        self.patch_size = patch_size
        self.overlap = overlap
        self.crop_border = crop_border
        
        # Calculate scaled dimensions
        self.scaled_patch_size = tuple(int(s * scale_factor) for s in patch_size)
        self.scaled_crop_border = int(crop_border * scale_factor)
        self.stride = tuple(s - overlap for s in patch_size)
        self.output_effective_size = tuple(s - 2 * self.scaled_crop_border for s in self.scaled_patch_size)
        
        self._print_config()
        
    def _print_config(self):
        print(f"\n=== 3D Overlap Inference Configuration ===")
        print(f"Original patch size: {self.patch_size}")
        print(f"Scale factor: {self.scale_factor}")
        print(f"Scaled patch size: {self.scaled_patch_size}")
        print(f"Overlap: {self.overlap}")
        print(f"Crop border: {self.crop_border} -> {self.scaled_crop_border}")
        print(f"Stride: {self.stride}")
        print(f"Output effective size per patch: {self.output_effective_size}")
        
    def compute_patch_positions(self, volume_shape):
        """Compute patch positions for sliding window inference"""
        d, h, w = volume_shape
        pd, ph, pw = self.patch_size
        stride_d, stride_h, stride_w = self.stride
        
        positions = []
        
        # Calculate number of patches
        n_patches_d = max(1, (d - pd) // stride_d + 1)
        n_patches_h = max(1, (h - ph) // stride_h + 1)
        n_patches_w = max(1, (w - pw) // stride_w + 1)
        
        for iz in range(n_patches_d):
            z = min(iz * stride_d, d - pd)
            for iy in range(n_patches_h):
                y = min(iy * stride_h, h - ph)
                for ix in range(n_patches_w):
                    x = min(ix * stride_w, w - pw)
                    positions.append((z, y, x))
        
        print(f"\nPatch positions for {volume_shape} volume:")
        print(f"  Patches per dimension: {n_patches_d} × {n_patches_h} × {n_patches_w}")
        print(f"  Total patches: {len(positions)}")
        
        return positions
    
    def process_volume(self, volume, style_code, encode_fn, decode_fn):
        """Process entire volume using overlap inference"""
        input_shape = volume.shape
        
        # Calculate output shape
        effective_input_size = tuple(s - 2 * self.crop_border for s in input_shape)
        output_shape = tuple(int(s * self.scale_factor) for s in effective_input_size)
        
        positions = self.compute_patch_positions(input_shape)
        
        # Initialize output volume
        output_volume = np.zeros(output_shape, dtype=np.float32)
        weight_volume = np.zeros(output_shape, dtype=np.float32)
        
        print(f"\nProcessing {len(positions)} patches...")
        
        with torch.no_grad():
            for position in tqdm(positions, desc="Processing patches"):
                # Extract patch
                z_start, y_start, x_start = position
                pd, ph, pw = self.patch_size
                patch = volume[z_start:z_start+pd, y_start:y_start+ph, x_start:x_start+pw]
                
                # Scale patch
                scaled_patch = ndimage.zoom(patch, self.scale_factor, order=0)
                
                # Convert to tensor
                patch_tensor = torch.FloatTensor(scaled_patch / 127.5 - 1.0)
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Process through model
                content, _ = encode_fn(patch_tensor)
                prediction = decode_fn(content, style_code)
                
                # Crop borders
                if self.scaled_crop_border > 0:
                    prediction = prediction[:, :, 
                        self.scaled_crop_border:-self.scaled_crop_border,
                        self.scaled_crop_border:-self.scaled_crop_border,
                        self.scaled_crop_border:-self.scaled_crop_border]
                
                # Convert back to numpy
                pred_np = prediction.squeeze().cpu().numpy()
                pred_np = ((pred_np + 1) * 127.5).clip(0, 255)
                
                # Calculate output position
                z_out = int(z_start * self.scale_factor)
                y_out = int(y_start * self.scale_factor)
                x_out = int(x_start * self.scale_factor)
                
                # Get prediction size
                pd, ph, pw = pred_np.shape
                
                # Place prediction with blending
                z_end = min(z_out + pd, output_shape[0])
                y_end = min(y_out + ph, output_shape[1])
                x_end = min(x_out + pw, output_shape[2])
                
                # Update output and weight volumes
                output_volume[z_out:z_end, y_out:y_end, x_out:x_end] += pred_np[:z_end-z_out, :y_end-y_out, :x_end-x_out]
                weight_volume[z_out:z_end, y_out:y_end, x_out:x_end] += 1.0
        
        # Normalize by weights
        mask = weight_volume > 0
        output_volume[mask] /= weight_volume[mask]
        
        return output_volume.astype(np.uint8)


class SliceBySliceProcessor2D(VolumeProcessor):
    """
    Process 3D volume slice by slice using 2D model
    """
    
    def __init__(self, model, device, scale_factor=5.0):
        super().__init__(model, device, scale_factor)
        print(f"\n=== 2D Slice-by-Slice Configuration ===")
        print(f"Scale factor: {self.scale_factor}")
        print(f"Processing mode: 2D slices")
        
    def process_slice(self, slice_2d, style_code, encode_fn, decode_fn):
        """Process a single 2D slice"""
        # Scale the slice
        scaled_slice = ndimage.zoom(slice_2d, self.scale_factor, order=1)
        
        # Convert to tensor
        slice_tensor = torch.FloatTensor(scaled_slice / 127.5 - 1.0)
        slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Process through model
        with torch.no_grad():
            content, _ = encode_fn(slice_tensor)
            generated = decode_fn(content, style_code)
        
        # Convert back to numpy
        output = generated.squeeze().cpu().numpy()
        output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        return output
        
    def process_volume(self, volume, style_code, encode_fn, decode_fn):
        """Process 3D volume slice by slice"""
        d, h, w = volume.shape
        
        # First, scale in z-direction
        print(f"Scaling volume in z-direction by {self.scale_factor}x...")
        volume_z_scaled = ndimage.zoom(volume, (self.scale_factor, 1, 1), order=1)
        new_d = volume_z_scaled.shape[0]
        
        print(f"Volume after z-scaling: {volume.shape} → {volume_z_scaled.shape}")
        
        # Output dimensions
        out_d = new_d
        out_h = int(h * self.scale_factor)
        out_w = int(w * self.scale_factor)
        
        # Initialize output volume
        output_volume = np.zeros((out_d, out_h, out_w), dtype=np.uint8)
        
        # Process each slice
        for i in tqdm(range(out_d), desc="Processing slices"):
            slice_2d = volume_z_scaled[i]
            hr_slice = self.process_slice(slice_2d, style_code, encode_fn, decode_fn)
            output_volume[i] = hr_slice
        
        return output_volume


def save_as_raw(data, output_path):
    """Save data as RAW file."""
    data.tofile(output_path)
    print(f"Saved: {output_path} (shape: {data.shape})")


def save_slice_visualizations(volume, output_path, volume_name=""):
    """Save slice visualizations for volume."""
    d, h, w = volume.shape
    
    # Save middle slices from each axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (xy plane)
    axes[0].imshow(volume[d//2], cmap='gray')
    axes[0].set_title(f'Axial (z={d//2})')
    axes[0].axis('off')
    
    # Coronal (xz plane)
    axes[1].imshow(volume[:, h//2, :], cmap='gray')
    axes[1].set_title(f'Coronal (y={h//2})')
    axes[1].axis('off')
    
    # Sagittal (yz plane)
    axes[2].imshow(volume[:, :, w//2], cmap='gray')
    axes[2].set_title(f'Sagittal (x={w//2})')
    axes[2].axis('off')
    
    plt.suptitle(volume_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified test script for DeepMicroscopy')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input RAW file path')
    parser.add_argument('--input-size', type=int, nargs=3, required=True,
                        metavar=('D', 'H', 'W'),
                        help='Size of input volume [depth, height, width]')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    
    # Processing mode
    parser.add_argument('--mode', type=str, choices=['3d', '2d', 'auto'], default='auto',
                        help='Processing mode: 3d (overlap), 2d (slice-by-slice), auto (detect from config)')
    
    # 3D processing options
    parser.add_argument('--patch-size', type=int, nargs=3, default=[60, 60, 60],
                        metavar=('D', 'H', 'W'),
                        help='Patch size for 3D inference (default: 60 60 60)')
    parser.add_argument('--overlap', type=int, default=10,
                        help='Overlap between patches in pixels (default: 10)')
    parser.add_argument('--crop-border', type=int, default=5,
                        help='Border pixels to crop from each patch (default: 5)')
    
    # Common options
    parser.add_argument('--scale-factor', type=float, default=None,
                        help='Scale factor (default: auto-detect from config)')
    parser.add_argument('--num-styles', type=int, default=5,
                        help='Number of style variations (default: 5)')
    parser.add_argument('--x2y', action='store_true', default=True,
                        help='Translate from X to Y domain')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')
    parser.add_argument('--save-slices', action='store_true', default=True,
                        help='Save slice visualizations')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    style_dim = config.get('model.style_dim', 8)
    
    # Auto-detect mode from config
    if args.mode == 'auto':
        mode = '3d' if config.dimensions == 3 else '2d'
        print(f"Auto-detected mode from config: {mode}")
    else:
        mode = args.mode
    
    # Auto-detect scale factor if not provided
    if args.scale_factor is None:
        if mode == '3d':
            args.scale_factor = config.get('data.dataset.scale_factor', [6.4, 6.4, 6.4])[0]
        else:
            args.scale_factor = config.get('data.dataset.scale_factor', [5.0, 5.0])[0]
        print(f"Auto-detected scale factor: {args.scale_factor}")
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Setup model
    trainer = Trainer(config)
    trainer.to(device)
    trainer.eval()
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    trainer.gen_x.load_state_dict(state_dict['x'])
    trainer.gen_y.load_state_dict(state_dict['y'])
    
    # Set up encode/decode functions
    if args.x2y:
        encode_fn = trainer.gen_x.encode
        decode_fn = trainer.gen_y.decode
        print("Translation direction: X → Y")
    else:
        encode_fn = trainer.gen_y.encode
        decode_fn = trainer.gen_x.decode
        print("Translation direction: Y → X")
    
    # Load input volume
    print(f"\nLoading input volume...")
    input_volume = np.fromfile(args.input_file, np.uint8).reshape(args.input_size)
    print(f"Input volume shape: {input_volume.shape}")
    
    # Extract sample name
    sample_name = os.path.basename(args.input_file).split('_')[0].split('-')[0]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save LR volume
    if mode == '3d':
        # For 3D mode, save effective area only
        border = args.crop_border
        lr_volume = input_volume[border:-border, border:-border, border:-border]
    else:
        lr_volume = input_volume
    
    lr_size_str = "x".join(map(str, lr_volume.shape))
    lr_filename = f"{sample_name}_LR_{args.scale_factor}x_8bu_{lr_size_str}.raw"
    lr_path = os.path.join(args.output_dir, lr_filename)
    save_as_raw(lr_volume, lr_path)
    
    # Save LR visualization
    if args.save_slices:
        lr_vis_path = os.path.join(args.output_dir, f"{sample_name}_LR_{args.scale_factor}x_8bu_{lr_size_str}.png")
        save_slice_visualizations(lr_volume, lr_vis_path, f"{sample_name} LR {lr_size_str}")
    
    # Create processor based on mode
    if mode == '3d':
        processor = OverlapInference3D(
            model=trainer,
            device=device,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            crop_border=args.crop_border,
            scale_factor=args.scale_factor
        )
    else:
        processor = SliceBySliceProcessor2D(
            model=trainer,
            device=device,
            scale_factor=args.scale_factor
        )
    
    # Generate style codes
    is_3d = (mode == '3d')
    style_codes = processor.generate_style_codes(args.num_styles, style_dim, is_3d)
    
    # Process with different styles
    for style_idx in range(args.num_styles):
        print(f"\n\nProcessing style {style_idx}/{args.num_styles-1}...")
        
        style = style_codes[style_idx].unsqueeze(0)
        
        # Process volume
        predicted_volume = processor.process_volume(
            input_volume, style, encode_fn, decode_fn
        )
        
        # Save HR volume
        hr_size_str = "x".join(map(str, predicted_volume.shape))
        hr_filename = f"{sample_name}_HR_style{style_idx}_8bu_{hr_size_str}.raw"
        hr_path = os.path.join(args.output_dir, hr_filename)
        save_as_raw(predicted_volume, hr_path)
        
        # Save HR visualization
        if args.save_slices:
            hr_vis_path = os.path.join(args.output_dir, f"{sample_name}_HR_style{style_idx}_8bu_{hr_size_str}.png")
            save_slice_visualizations(predicted_volume, hr_vis_path, f"{sample_name} HR Style {style_idx} {hr_size_str}")
    
    # Save processing info
    info_path = os.path.join(args.output_dir, "processing_info.txt")
    with open(info_path, 'w') as f:
        f.write("DeepMicroscopy Volume Processing\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mode: {mode.upper()}\n")
        f.write(f"Input: {args.input_file}\n")
        f.write(f"Input size: {input_volume.shape}\n")
        f.write(f"Scale factor: {args.scale_factor}x\n")
        f.write(f"Number of styles: {args.num_styles}\n")
        f.write(f"Translation direction: {'X→Y' if args.x2y else 'Y→X'}\n")
        if mode == '3d':
            f.write(f"Patch size: {args.patch_size}\n")
            f.write(f"Overlap: {args.overlap}\n")
            f.write(f"Crop border: {args.crop_border}\n")
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()