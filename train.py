import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from data import MicroscopyDataset
from trainer import Trainer
from utils import (
    prepare_directories, log_progress, write_images,
    save_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MUNIT model for microscopy translation')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')

    parser.add_argument('--output-dir', type=str,
                        help='Override output directory in config')

    parser.add_argument('--data-dir', type=str,
                        help='Override data directory in config')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 for CPU)')

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Mode: {config.dimensions}D")

    # Override directories if specified
    if args.output_dir:
        output_dir = args.output_dir
        print(f"Overriding output directory: {output_dir}")
    else:
        output_dir = config.get('data.paths.output_dir', 'experiments/default')

    if args.data_dir:
        config.set('data.paths.output_dir', args.data_dir)
        print(f"Overriding data directory: {args.data_dir}")

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Prepare directories
    checkpoint_dir, image_dir = prepare_directories(output_dir)

    # Save configuration for reference
    save_config(config, os.path.join(output_dir, 'config.yaml'))

    # Create dataset and dataloader
    dataset = MicroscopyDataset(config, mode='train')
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('training.batch_size', 1),
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create log file
    log_file = os.path.join(output_dir, 'training_log.txt')

    # Create trainer
    trainer = Trainer(config)
    trainer.to(device)

    # Resume from checkpoint if requested
    start_iter = 0
    if args.resume:
        start_iter = trainer.resume(checkpoint_dir, device)
        print(f"Resumed from iteration {start_iter}")

    # Create display data for visualization
    display_size = 2
    train_display_x = []
    train_display_y = []

    # Sample up to display_size items from the dataset
    for i in range(min(display_size, len(dataset))):
        sample = dataset[i]
        train_display_x.append(sample['X'].unsqueeze(0))  # Changed from 'LR' to 'X'
        train_display_y.append(sample['Y'].unsqueeze(0))  # Changed from 'HR' to 'Y'

    # Concatenate samples
    if train_display_x:
        train_display_x = torch.cat(train_display_x).to(device)
        train_display_y = torch.cat(train_display_y).to(device)

    # Get training parameters
    n_iters = config.get('training.n_iters', 200000)
    log_step = config.get('training.logging.log_step', 100)
    image_save_iter = config.get('training.logging.image_save_iter', 100)
    image_display_iter = config.get('training.logging.image_display_iter', 100)
    snapshot_save_iter = config.get('training.logging.snapshot_save_iter', 100)

    print("\nStarting training...")
    print(f"Total iterations: {n_iters}")

    # Training loop
    iter_idx = start_iter
    while iter_idx < n_iters:
        for batch in dataloader:
            # Move data to device
            x = batch['X'].to(device)
            y = batch['Y'].to(device)
            # Update discriminator
            trainer.dis_update(x, y)

            # Update generator
            trainer.gen_update(x, y)

            # Update learning rate
            trainer.update_learning_rate()

            # Log progress
            if (iter_idx + 1) % log_step == 0:
                log_progress(iter_idx, trainer, log_file)

            # Save images
            if (iter_idx + 1) % image_save_iter == 0:
                with torch.no_grad():
                    train_image_outputs = trainer.sample(train_display_x, train_display_y)

                write_images(train_image_outputs, display_size, image_dir, f'train_{iter_idx + 1:08d}')

            # Display current results
            if (iter_idx + 1) % image_display_iter == 0:
                with torch.no_grad():
                    train_image_outputs = trainer.sample(train_display_x, train_display_y)

                write_images(train_image_outputs, display_size, image_dir, 'train_current')

            # Save snapshot
            if (iter_idx + 1) % snapshot_save_iter == 0:
                trainer.save(checkpoint_dir, iter_idx)
                print(f"Saved checkpoint at iteration {iter_idx + 1}")

            # Increment iteration counter
            iter_idx += 1

            # Break if we've reached the desired number of iterations
            if iter_idx >= n_iters:
                break

    # Save final model
    trainer.save(checkpoint_dir, iter_idx - 1)
    print(f"\nTraining completed after {iter_idx} iterations!")
    print(f"Final model saved to {checkpoint_dir}")


if __name__ == '__main__':
    main()