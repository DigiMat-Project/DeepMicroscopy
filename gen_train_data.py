import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import Config
from data import PatchGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate microscopy image training patches')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (overrides config)')
    parser.add_argument('--raw-dir', type=str,
                        help='Raw data directory (overrides config)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify patches after generation')
    parser.add_argument('--dimensions', type=int, choices=[2, 3],
                        help='Override dimensions setting (2D or 3D)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for patch extraction')
    return parser.parse_args()


def verify_patches(patch_dir, block_size, dimensions):
    """Verify that generated patches have expected size.
    
    Args:
        patch_dir: Directory containing patches
        block_size: Expected patch size
        dimensions: Data dimensions (2 or 3)
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    files = sorted(os.listdir(patch_dir))
    if not files:
        print(f"No files found in {patch_dir}")
        return False
    
    # Calculate expected file size based on block size
    if dimensions == 2:
        if isinstance(block_size, (list, tuple)):
            expected_size = block_size[0] * block_size[1]
        else:
            expected_size = block_size * block_size
    else:  # 3D
        expected_size = block_size[0] * block_size[1] * block_size[2]
    
    # Verify file sizes
    for file in tqdm(files[:min(100, len(files))], desc="Verifying patches"):
        file_path = os.path.join(patch_dir, file)
        file_size = os.path.getsize(file_path)
        
        if file_size != expected_size:
            print(f"Size mismatch: {file_path} is {file_size} bytes, expected {expected_size} bytes")
            return False
    
    return True


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = args.config
    print(f"Loading configuration: {config_path}")
    config = Config(config_path)
    
    # Override dimensions if specified
    if args.dimensions:
        config.dimensions = args.dimensions
        print(f"Overridden dimensions: {args.dimensions}D")
    
    # Override random seed if specified
    if args.seed:
        config.set('data.seed', args.seed)
        print(f"Overridden random seed: {args.seed}")
    
    print(f"Configuration loaded, operating in {config.dimensions}D mode")
    
    # Override directories if specified
    if args.output_dir:
        config.set('data.paths.output_dir', args.output_dir)
        print(f"Overridden output directory: {args.output_dir}")
    
    if args.raw_dir:
        config.set('data.paths.raw_dir', args.raw_dir)
        print(f"Overridden raw data directory: {args.raw_dir}")
    
    # Create output directory
    output_dir = config.get('data.paths.output_dir')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create PatchGenerator instance
    print("\nInitializing PatchGenerator...")
    patch_generator = PatchGenerator(config)
    
    # Display extraction parameters
    x_file = os.path.join(
        config.get('data.paths.raw_dir'),
        config.get('data.x_domain.filename')
    )
    y_file = os.path.join(
        config.get('data.paths.raw_dir'),
        config.get('data.y_domain.filename')
    )
    x_shape = config.get('data.x_domain.shape')
    y_shape = config.get('data.y_domain.shape')
    x_block_size = config.get('data.x_domain.block_size')
    y_block_size = config.get('data.y_domain.block_size')
    x_num_patches = config.get('data.x_domain.num_patches')
    y_num_patches = config.get('data.y_domain.num_patches')
    
    # Format shape and block size strings based on dimensions
    if config.dimensions == 2:
        x_shape_str = f"{x_shape} (D×H×W)" if len(x_shape) == 3 else f"{x_shape} (H×W)"
        y_shape_str = f"{y_shape} (H×W)"
        x_block_str = f"{x_block_size} (H×W)"
        y_block_str = f"{y_block_size} (H×W)"
    else:  # 3D
        x_shape_str = f"{x_shape} (D×H×W)"
        y_shape_str = f"{y_shape} (D×H×W)"
        x_block_str = f"{x_block_size} (D×H×W)"
        y_block_str = f"{y_block_size} (D×H×W)"
    
    print(f"Source domain file: {x_file}")
    print(f"Source domain shape: {x_shape_str}")
    print(f"Source domain block size: {x_block_str}")
    print(f"Source domain number of patches: {x_num_patches}")
    print(f"Target domain file: {y_file}")
    print(f"Target domain shape: {y_shape_str}")
    print(f"Target domain block size: {y_block_str}")
    print(f"Target domain number of patches: {y_num_patches}")
    print(f"Output format: RAW (np.uint8)")
    
    # Check if source files exist
    if not os.path.exists(x_file):
        print(f"ERROR: Source file not found: {x_file}")
        sys.exit(1)
    if not os.path.exists(y_file):
        print(f"ERROR: Target file not found: {y_file}")
        sys.exit(1)
    
    # Generate training set
    print("\nStarting patch generation...")
    try:
        patch_generator.generate()
    except Exception as e:
        print(f"ERROR during patch generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify patches if requested
    if args.verify:
        print("\nVerifying generated patches...")
        
        # Verify source domain patches
        print("Verifying source domain patches...")
        x_verified = verify_patches(patch_generator.x_output_dir, x_block_size, config.dimensions)
        print(f"Source domain verification result: {'Passed' if x_verified else 'Failed'}")
        
        # Verify target domain patches
        print("Verifying target domain patches...")
        y_verified = verify_patches(patch_generator.y_output_dir, y_block_size, config.dimensions)
        print(f"Target domain verification result: {'Passed' if y_verified else 'Failed'}")
    
    # Count generated patches
    x_files = [f for f in os.listdir(patch_generator.x_output_dir) if f.endswith('.raw')]
    y_files = [f for f in os.listdir(patch_generator.y_output_dir) if f.endswith('.raw')]
    
    x_count = len(x_files)
    y_count = len(y_files)
    
    print(f"\nSummary:")
    print(f"Generated {x_count}/{x_num_patches} source domain patches, size: {x_block_str}")
    print(f"Generated {y_count}/{y_num_patches} target domain patches, size: {y_block_str}")
    print(f"Patches saved at:")
    print(f"  - Source domain: {patch_generator.x_output_dir}")
    print(f"  - Target domain: {patch_generator.y_output_dir}")
    
    # Sample verification
    print("\nSample Data Verification:")
    if x_count > 0:
        x_sample = os.path.join(patch_generator.x_output_dir, x_files[0])
        file_size = os.path.getsize(x_sample)
        
        # Calculate expected size
        if config.dimensions == 2:
            if isinstance(x_block_size, (list, tuple)):
                expected_size = x_block_size[0] * x_block_size[1]
            else:
                expected_size = x_block_size * x_block_size
        else:  # 3D
            expected_size = x_block_size[0] * x_block_size[1] * x_block_size[2]
        
        print(f"  - Source sample: {x_sample}")
        print(f"    Size: {file_size} bytes (Expected: {expected_size} bytes)")
        print(f"    Valid: {file_size == expected_size}")
        
        # Additional verification for 3D data: read and check dimensions
        if config.dimensions == 3:
            try:
                data = np.fromfile(x_sample, np.uint8)
                actual_shape = None
                try:
                    if isinstance(x_block_size, (list, tuple)) and len(x_block_size) == 3:
                        reshaped = data.reshape(x_block_size)
                        actual_shape = reshaped.shape
                    print(f"    Actual shape: {actual_shape}")
                    print(f"    Shape valid: {actual_shape == tuple(x_block_size)}")
                except ValueError as e:
                    print(f"    ERROR: {e}")
            except Exception as e:
                print(f"    ERROR reading data: {e}")
    
    if y_count > 0:
        y_sample = os.path.join(patch_generator.y_output_dir, y_files[0])
        file_size = os.path.getsize(y_sample)
        
        # Calculate expected size
        if config.dimensions == 2:
            if isinstance(y_block_size, (list, tuple)):
                expected_size = y_block_size[0] * y_block_size[1]
            else:
                expected_size = y_block_size * y_block_size
        else:  # 3D
            expected_size = y_block_size[0] * y_block_size[1] * y_block_size[2]
        
        print(f"  - Target sample: {y_sample}")
        print(f"    Size: {file_size} bytes (Expected: {expected_size} bytes)")
        print(f"    Valid: {file_size == expected_size}")
        
        # Additional verification for 3D data: read and check dimensions
        if config.dimensions == 3:
            try:
                data = np.fromfile(y_sample, np.uint8)
                actual_shape = None
                try:
                    if isinstance(y_block_size, (list, tuple)) and len(y_block_size) == 3:
                        reshaped = data.reshape(y_block_size)
                        actual_shape = reshaped.shape
                    print(f"    Actual shape: {actual_shape}")
                    print(f"    Shape valid: {actual_shape == tuple(y_block_size)}")
                except ValueError as e:
                    print(f"    ERROR: {e}")
            except Exception as e:
                print(f"    ERROR reading data: {e}")
    
    # Suggest next steps
    print("\nGeneration process completed!")
    print("Next step: You can start training the model with the following command:")
    train_cmd = f"python train.py --config {config_path}"
    print(f"  {train_cmd}")


if __name__ == '__main__':
    main()