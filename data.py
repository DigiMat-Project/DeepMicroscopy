import os
import numpy as np
import random
import glob
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from tqdm import tqdm
from PIL import Image


class MicroscopyDataset(Dataset):
    """Dataset for paired microscopy images/volumes.

    This dataset supports both 2D and 3D data and handles loading of
    RAW format files with appropriate reshaping.
    """

    def __init__(self, config, mode='train', unaligned=None):
        """Initialize the dataset.

        Args:
            config: Configuration object
            mode: Dataset mode ('train', 'test', 'val')
            unaligned: Whether to use unaligned data (overrides config value if provided)
        """
        self.config = config
        self.dimensions = config.dimensions
        self.mode = mode

        # Path setup
        data_dir = config.get('data.paths.output_dir')

        # Use unaligned from config unless explicitly overridden
        if unaligned is None:
            self.unaligned = config.get('data.dataset.unaligned', True)
        else:
            self.unaligned = unaligned

        # Get file lists (only RAW files)
        x_dir = os.path.join(data_dir, 'x_domain')
        y_dir = os.path.join(data_dir, 'y_domain')
        
        self.files_x = sorted(glob.glob(os.path.join(x_dir, '*.raw')))
        self.files_y = sorted(glob.glob(os.path.join(y_dir, '*.raw')))

        if not self.files_x:
            raise ValueError(f"No RAW files found in {x_dir}")
        if not self.files_y:
            raise ValueError(f"No RAW files found in {y_dir}")

        # Shuffle for training
        if mode == 'train':
            random.shuffle(self.files_x)
            random.shuffle(self.files_y)

        # Get block shapes
        if self.dimensions == 2:
            # For 2D, use height and width from config
            if config.get('data.x_domain.block_size') and isinstance(config.get('data.x_domain.block_size'), list) and len(config.get('data.x_domain.block_size')) == 3:
                # Extract 2D shape from 3D configuration
                self.x_shape = tuple(config.get('data.x_domain.block_size')[1:])
                self.y_shape = tuple(config.get('data.y_domain.block_size')[1:])
            else:
                self.x_shape = tuple(config.get('data.x_domain.block_size'))
                self.y_shape = tuple(config.get('data.y_domain.block_size'))
        else:
            # For 3D, use full shape
            self.x_shape = tuple(config.get('data.x_domain.block_size'))
            self.y_shape = tuple(config.get('data.y_domain.block_size'))

        # Get scaling parameters
        self.scale_factor = config.get('data.dataset.scale_factor', None)
        self.scale_mode = config.get('data.dataset.scale_mode', 'nearest')

        print(f"Initialized dataset with {len(self.files_x)} source files and {len(self.files_y)} target files")
        print(f"Source shape: {self.x_shape}, Target shape: {self.y_shape}")

    def __len__(self):
        """Get dataset length."""
        return max(len(self.files_x), len(self.files_y))

    def __getitem__(self, index):
        """Get a data item.

        Args:
            index: Item index

        Returns:
            Dictionary with 'X' and 'Y' tensors
        """
        # Load source (X) data
        x_index = index % len(self.files_x)
        item_x = self._read_raw(self.files_x[x_index], self.x_shape)

        # Scale if needed
        if self.scale_factor:
            if self.dimensions == 2:
                # For 2D data
                scale_factor = self.scale_factor[:2] if len(self.scale_factor) > 2 else self.scale_factor
                item_x = ndimage.zoom(item_x, scale_factor, mode=self.scale_mode)
            else:
                # For 3D data
                item_x = ndimage.zoom(item_x, self.scale_factor, mode=self.scale_mode)

        # Load target (Y) data
        if self.unaligned:
            y_index = random.randint(0, len(self.files_y) - 1)
        else:
            y_index = index % len(self.files_y)

        item_y = self._read_raw(self.files_y[y_index], self.y_shape)

        # Convert to tensors and normalize to [-1, 1]
        item_x = torch.Tensor(item_x / 127.5 - 1.0)
        item_y = torch.Tensor(item_y / 127.5 - 1.0)

        # Add channel dimension if needed
        item_x = item_x.unsqueeze(0)
        item_y = item_y.unsqueeze(0)

        return {'X': item_x, 'Y': item_y}

    def _read_raw(self, file_path, shape):
        """Read a RAW file and reshape it.

        Args:
            file_path: Path to the RAW file
            shape: Shape to reshape the data to

        Returns:
            Numpy array with the data
        """
        try:
            data = np.fromfile(file_path, np.uint8)
            
            # Calculate expected size based on shape
            expected_size = np.prod(shape)
            
            # Handle size mismatch
            if data.size != expected_size:
                print(f"Warning: RAW file size ({data.size}) doesn't match expected shape {shape} (size {expected_size})")
                
                # Try to detect the actual shape
                file_size = data.size
                
                # For 3D shape
                if len(shape) == 3:
                    if shape[0] * shape[1] * shape[2] != file_size:
                        # Try to adjust depth dimension
                        pd = file_size // (shape[1] * shape[2])
                        if pd * shape[1] * shape[2] == file_size:
                            new_shape = (pd, shape[1], shape[2])
                            print(f"Adjusted shape to {new_shape}")
                            return data.reshape(new_shape)
                
                # For 2D shape
                if len(shape) == 2:
                    if shape[0] * shape[1] != file_size:
                        # Try square shape
                        side = int(np.sqrt(file_size))
                        if side * side == file_size:
                            new_shape = (side, side)
                            print(f"Adjusted shape to {new_shape}")
                            return data.reshape(new_shape)
                
                # Return zeros as last resort
                print(f"Could not determine correct shape for {file_path}")
                return np.zeros(shape, dtype=np.uint8)
            
            return data.reshape(shape)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Return zeros as a fallback
            return np.zeros(shape, dtype=np.uint8)
            

class TestDataset(Dataset):
    """Dataset for testing, loading only one domain.

    This dataset is used for inference when only source or target domain
    data is needed.
    """

    def __init__(self, config, is_target=False):
        """Initialize the test dataset.

        Args:
            config: Configuration object
            is_target: Whether to load target domain (True) or source domain (False)
        """
        self.config = config
        self.dimensions = config.dimensions
        self.is_target = is_target

        # Path setup
        data_dir = config.get('data.paths.output_dir')
        subdir = 'y_domain' if is_target else 'x_domain'

        # Get file list (only RAW files)
        self.files = sorted(glob.glob(os.path.join(data_dir, subdir, '*.raw')))

        if not self.files:
            raise ValueError(f"No RAW files found in {os.path.join(data_dir, subdir)}")

        # Get shape
        if is_target:
            if self.dimensions == 2:
                # For 2D, use height and width
                if isinstance(config.get('data.y_domain.block_size'), list) and len(config.get('data.y_domain.block_size')) == 3:
                    # Extract 2D shape from 3D configuration
                    self.shape = tuple(config.get('data.y_domain.block_size')[1:])
                else:
                    self.shape = tuple(config.get('data.y_domain.block_size'))
            else:
                # For 3D, use full shape
                self.shape = tuple(config.get('data.y_domain.block_size'))
        else:
            if self.dimensions == 2:
                # For 2D, use height and width
                if isinstance(config.get('data.x_domain.block_size'), list) and len(config.get('data.x_domain.block_size')) == 3:
                    # Extract 2D shape from 3D configuration
                    self.shape = tuple(config.get('data.x_domain.block_size')[1:])
                else:
                    self.shape = tuple(config.get('data.x_domain.block_size'))
            else:
                # For 3D, use full shape
                self.shape = tuple(config.get('data.x_domain.block_size'))

        print(f"Initialized test dataset with {len(self.files)} RAW files")
        print(f"Shape: {self.shape}")

    def __len__(self):
        """Get dataset length."""
        return len(self.files)

    def __getitem__(self, index):
        """Get a data item.

        Args:
            index: Item index

        Returns:
            Tensor with the data
        """
        # Load data
        data = self._read_raw(self.files[index], self.shape)

        # Convert to tensor and normalize to [-1, 1]
        data = torch.Tensor(data / 127.5 - 1.0)

        # Add channel dimension if needed
        if len(data.shape) == 2:
            data = data.unsqueeze(0)

        return data

    def _read_raw(self, file_path, shape):
        """Read a RAW file and reshape it.

        Args:
            file_path: Path to the RAW file
            shape: Shape to reshape the data to

        Returns:
            Numpy array with the data
        """
        try:
            data = np.fromfile(file_path, np.uint8)
            
            # Calculate expected size based on shape
            expected_size = np.prod(shape)
            
            # Handle size mismatch
            if data.size != expected_size:
                print(f"Warning: RAW file size ({data.size}) doesn't match expected shape {shape} (size {expected_size})")
                
                # Try to detect the actual shape
                file_size = data.size
                
                # For 3D shape
                if len(shape) == 3:
                    if shape[0] * shape[1] * shape[2] != file_size:
                        # Try to adjust depth dimension
                        pd = file_size // (shape[1] * shape[2])
                        if pd * shape[1] * shape[2] == file_size:
                            new_shape = (pd, shape[1], shape[2])
                            print(f"Adjusted shape to {new_shape}")
                            return data.reshape(new_shape)
                
                # For 2D shape
                if len(shape) == 2:
                    if shape[0] * shape[1] != file_size:
                        # Try square shape
                        side = int(np.sqrt(file_size))
                        if side * side == file_size:
                            new_shape = (side, side)
                            print(f"Adjusted shape to {new_shape}")
                            return data.reshape(new_shape)
                
                # Return zeros as last resort
                print(f"Could not determine correct shape for {file_path}")
                return np.zeros(shape, dtype=np.uint8)
            
            return data.reshape(shape)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Return zeros as a fallback
            return np.zeros(shape, dtype=np.uint8)


class PatchGenerator:
    """Generate training patches from large raw files.

    This class extracts smaller patches from large source and target domain
    files for training. It supports both 2D and 3D data formats.
    """

    def __init__(self, config):
        """Initialize the patch generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.dimensions = config.dimensions
        self._continue_from = 0  # ADD THIS LINE - for multi-modal support

        # Create output directories
        output_dir = config.get('data.paths.output_dir')
        self.x_output_dir = os.path.join(output_dir, 'x_domain')
        self.y_output_dir = os.path.join(output_dir, 'y_domain')

        os.makedirs(self.x_output_dir, exist_ok=True)
        os.makedirs(self.y_output_dir, exist_ok=True)
        
        # Random seed
        self.seed = config.get('data.seed', 42)

        print(f"Output directories created: {self.x_output_dir}, {self.y_output_dir}")
        print(f"Using RAW format for all outputs")

    def generate(self):
        """Generate patches for both domains."""
        if self.dimensions == 3:
            # Generate 3D patches for both domains
            self._generate_3d_patches()
        else:
            # Generate 2D patches for both domains
            if len(self.config.get('data.x_domain.shape')) == 3:
                # X domain is 3D, extract 2D slices
                # MODIFY: Only generate x patches if _continue_from == 0
                if self._continue_from == 0:
                    self._generate_x_domain_2d_from_3d()
            else:
                # X domain is already 2D
                # MODIFY: Only generate x patches if _continue_from == 0
                if self._continue_from == 0:
                    self._generate_x_domain_2d()
                
            if len(self.config.get('data.y_domain.shape')) == 3:
                # Y domain is 3D, extract 2D slices
                self._generate_y_domain_2d_from_3d()
            else:
                # Y domain is already 2D
                self._generate_y_domain_2d()
                
    def _generate_3d_patches(self):
        """Generate 3D patches for both domains."""
        # Process source domain (x)
        x_file = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.x_domain.filename')
        )
        x_shape = self.config.get('data.x_domain.shape')
        x_block_size = self.config.get('data.x_domain.block_size')
        x_num_patches = self.config.get('data.x_domain.num_patches')
        
        # Process target domain (y)
        y_file = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.y_domain.filename')
        )
        y_shape = self.config.get('data.y_domain.shape')
        y_block_size = self.config.get('data.y_domain.block_size')
        y_num_patches = self.config.get('data.y_domain.num_patches')
        
        # Extract patches
        # MODIFY: Only generate x patches if _continue_from == 0
        if self._continue_from == 0:
            self._extract_3d_patches_from_file(x_file, x_shape, x_block_size, x_num_patches, self.x_output_dir)
        self._extract_3d_patches_from_file(y_file, y_shape, y_block_size, y_num_patches, self.y_output_dir)
                
    def _generate_x_domain_2d(self):
        """Generate patches from 2D X domain."""
        x_file_path = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.x_domain.filename')
        )
        
        x_shape = self.config.get('data.x_domain.shape')
        patch_size = self.config.get('data.x_domain.block_size')
        num_patches = self.config.get('data.x_domain.num_patches')
        
        try:
            data = np.fromfile(x_file_path, np.uint8).reshape(x_shape)
            self._extract_2d_patches(data, patch_size, num_patches, self.x_output_dir)
        except Exception as e:
            print(f"Error processing X domain data: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_y_domain_2d(self):
        """Generate patches from 2D Y domain."""
        y_file_path = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.y_domain.filename')
        )
        
        y_shape = self.config.get('data.y_domain.shape')
        patch_size = self.config.get('data.y_domain.block_size')
        num_patches = self.config.get('data.y_domain.num_patches')
        
        try:
            data = np.fromfile(y_file_path, np.uint8).reshape(y_shape)
            self._extract_2d_patches(data, patch_size, num_patches, self.y_output_dir, start_idx=self._continue_from)  # MODIFY
        except Exception as e:
            print(f"Error processing Y domain data: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_x_domain_2d_from_3d(self):
        """Generate 2D patches from 3D X domain volume."""
        # Get path to raw file
        x_file_path = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.x_domain.filename')
        )
        
        print(f"Loading 3D X domain data from {x_file_path}...")
        # Get shape and parameters
        x_shape = self.config.get('data.x_domain.shape')
        patch_size = self.config.get('data.x_domain.block_size')
        num_patches = self.config.get('data.x_domain.num_patches')
        
        # Load 3D data
        try:
            nz, nx, ny = x_shape
            data = np.fromfile(x_file_path, np.uint8).reshape(nz, nx, ny)
            print(f"Successfully loaded 3D data with shape: {data.shape}")
            
            # If patch_size is 2D, use it directly; if 3D, extract 2D size
            if isinstance(patch_size, (list, tuple)):
                if len(patch_size) == 2:
                    ph, pw = patch_size
                elif len(patch_size) == 3:
                    ph, pw = patch_size[1], patch_size[2]
                else:
                    ph = pw = patch_size[0]
            else:
                ph = pw = patch_size
            
            # Generate random slice indices
            np.random.seed(self.seed)
            slice_indices = np.random.randint(0, nz, num_patches)
            
            # Extract 2D patches from random slices
            for i, slice_idx in enumerate(tqdm(slice_indices, desc="Extracting 2D patches from 3D volume")):
                # Get 2D slice
                slice_2d = data[slice_idx]
                
                # Generate random position in the slice
                y = np.random.randint(ph//2, nx - ph//2)
                x = np.random.randint(pw//2, ny - pw//2)
                
                # Extract patch
                patch = slice_2d[y-ph//2:y+ph//2, x-pw//2:x+pw//2]
                
                # Verify patch size
                if patch.shape != (ph, pw):
                    print(f"Warning: Patch {i} has wrong shape: {patch.shape}, expected: ({ph}, {pw})")
                    continue
                
                # Normalize if needed
                if self.config.get('data.normalize', False):
                    patch = 255 * (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
                    patch = patch.astype(np.uint8)
                
                # Save the patch
                self._save_as_raw(patch, os.path.join(self.x_output_dir, f"{i:06d}.raw"))
                    
        except Exception as e:
            print(f"Error processing 3D X domain data: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_y_domain_2d_from_3d(self):
        """Generate 2D patches from 3D Y domain volume."""
        # Get path to raw file
        y_file_path = os.path.join(
            self.config.get('data.paths.raw_dir'),
            self.config.get('data.y_domain.filename')
        )
        
        print(f"Loading 3D Y domain data from {y_file_path}...")
        # Get shape and parameters
        y_shape = self.config.get('data.y_domain.shape')
        patch_size = self.config.get('data.y_domain.block_size')
        num_patches = self.config.get('data.y_domain.num_patches')
        
        # Load 3D data
        try:
            nz, nx, ny = y_shape
            data = np.fromfile(y_file_path, np.uint8).reshape(nz, nx, ny)
            print(f"Successfully loaded 3D data with shape: {data.shape}")
            
            # If patch_size is 2D, use it directly; if 3D, extract 2D size
            if isinstance(patch_size, (list, tuple)):
                if len(patch_size) == 2:
                    ph, pw = patch_size
                elif len(patch_size) == 3:
                    ph, pw = patch_size[1], patch_size[2]
                else:
                    ph = pw = patch_size[0]
            else:
                ph = pw = patch_size
            
            # Generate random slice indices
            np.random.seed(self.seed + 1)  # Different seed for Y domain
            slice_indices = np.random.randint(0, nz, num_patches)
            
            # Extract 2D patches from random slices
            for i, slice_idx in enumerate(tqdm(slice_indices, desc="Extracting 2D patches from 3D volume")):
                # Get 2D slice
                slice_2d = data[slice_idx]
                
                # Generate random position in the slice
                y = np.random.randint(ph//2, nx - ph//2)
                x = np.random.randint(pw//2, ny - pw//2)
                
                # Extract patch
                patch = slice_2d[y-ph//2:y+ph//2, x-pw//2:x+pw//2]
                
                # Verify patch size
                if patch.shape != (ph, pw):
                    print(f"Warning: Patch {i} has wrong shape: {patch.shape}, expected: ({ph}, {pw})")
                    continue
                
                # Normalize if needed
                if self.config.get('data.normalize', False):
                    patch = 255 * (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
                    patch = patch.astype(np.uint8)
                
                # Save the patch - MODIFY: use _continue_from for numbering
                self._save_as_raw(patch, os.path.join(self.y_output_dir, f"{self._continue_from + i:06d}.raw"))
                    
        except Exception as e:
            print(f"Error processing 3D Y domain data: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_3d_patches_from_file(self, file_path, volume_shape, patch_size, num_patches, output_dir):
        """Extract 3D patches from a 3D volume file.
        
        Args:
            file_path: Path to the volume file
            volume_shape: Shape of the volume (d, h, w)
            patch_size: Size of patches to extract
            num_patches: Number of patches to extract
            output_dir: Directory to save patches
        """
        print(f"Loading 3D data from {file_path}...")
        
        try:
            # Load 3D data
            data = np.fromfile(file_path, np.uint8).reshape(volume_shape)
            print(f"Successfully loaded 3D data with shape: {data.shape}")
            
            # Extract patches - MODIFY: add start_idx for y_domain
            if output_dir == self.y_output_dir:
                self._extract_3d_patches(data, patch_size, num_patches, output_dir, start_idx=self._continue_from)
            else:
                self._extract_3d_patches(data, patch_size, num_patches, output_dir)
        except Exception as e:
            print(f"Error processing 3D data from {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_2d_patches(self, data, patch_size, num_patches, output_dir, start_idx=0):  # MODIFY: add start_idx
        """Extract 2D patches from 2D data.
        
        Args:
            data: 2D data array
            patch_size: Size of patches to extract
            num_patches: Number of patches to extract
            output_dir: Directory to save patches
            start_idx: Starting index for patch numbering (for multi-modal support)
        """
        print(f"Extracting {num_patches} 2D patches, size: {patch_size}...")
        h, w = data.shape
        
        # If patch_size is a list/tuple, extract the relevant dimensions
        if isinstance(patch_size, (list, tuple)):
            if len(patch_size) == 2:  # Already 2D
                ph, pw = patch_size
            elif len(patch_size) == 3:  # Extract from 3D specification
                ph, pw = patch_size[1], patch_size[2]
            else:
                ph = pw = patch_size[0]  # Assume square if single value
        else:
            ph = pw = patch_size  # Assume square if single value
        
        # For even-sized patches, adjust to get the right size
        if ph % 2 == 0:
            hs1, hs2 = ph//2, ph//2
        else:
            hs1, hs2 = ph//2, ph//2 + 1
            
        if pw % 2 == 0:
            ws1, ws2 = pw//2, pw//2
        else:
            ws1, ws2 = pw//2, pw//2 + 1
        
        # Generate random coordinates
        np.random.seed(self.seed)  # For reproducibility
        I = np.random.randint(hs1 + 10, h - hs2 - 10, num_patches)
        J = np.random.randint(ws1 + 10, w - ws2 - 10, num_patches)
        
        # Extract and save patches
        for i, (ii, jj) in enumerate(tqdm(zip(I, J), total=num_patches, desc="Extracting 2D patches")):
            patch = data[ii-hs1:ii+hs2, jj-ws1:jj+ws2]
            
            # Verify patch shape
            if patch.shape != (ph, pw):
                print(f"Warning: Patch {i} has wrong shape: {patch.shape}, expected: ({ph}, {pw})")
                # Resize to match expected dimensions if necessary
                try:
                    from PIL import Image
                    img = Image.fromarray(patch)
                    img = img.resize((pw, ph))
                    patch = np.array(img)
                except Exception as e:
                    print(f"Error resizing patch: {e}")
                    continue
            
            # Normalize if needed
            if self.config.get('data.normalize', False):
                patch = 255 * (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
                patch = patch.astype(np.uint8)
            
            # Save the patch as RAW - MODIFY: use start_idx
            self._save_as_raw(patch, os.path.join(output_dir, f"{start_idx + i:06d}.raw"))
    
    def _extract_3d_patches(self, data, patch_size, num_patches, output_dir, start_idx=0):  # MODIFY: add start_idx
        """Extract 3D patches from 3D data with exact patch size.

        Args:
            data: 3D data array
            patch_size: Size of patches to extract [d, h, w]
            num_patches: Number of patches to extract
            output_dir: Directory to save patches
            start_idx: Starting index for patch numbering (for multi-modal support)
        """
        print(f"Extracting {num_patches} 3D patches, size: {patch_size}...")
        d, h, w = data.shape
        pd, ph, pw = patch_size

        if not (pd <= d and ph <= h and pw <= w):
            raise ValueError(f"Patch size {patch_size} exceeds data volume size {data.shape}")

        np.random.seed(self.seed)
        # Modified index generation to ensure correct depth - remove the +1
        Z = np.random.randint(0, d - pd, num_patches)  
        Y = np.random.randint(0, h - ph, num_patches)
        X = np.random.randint(0, w - pw, num_patches)

        for i, (z, y, x) in enumerate(tqdm(zip(Z, Y, X), total=num_patches, desc="Extracting 3D patches")):
            # Extract the patch
            patch = data[z:z+pd, y:y+ph, x:x+pw]
            
            # Verify patch dimensions
            if patch.shape != (pd, ph, pw):
                print(f"Warning: Patch {i} has wrong shape: {patch.shape}, expected: ({pd}, {ph}, {pw})")
                print(f"Extraction coordinates: z={z}, y={y}, x={x}")
                
                # Try to fix the patch by ensuring exact dimensions
                z_end = min(z + pd, d)
                y_end = min(y + ph, h)
                x_end = min(x + pw, w)
                
                # If we're hitting a boundary and the patch is smaller than expected
                # Create a new patch with zeros and copy the available data
                fixed_patch = np.zeros((pd, ph, pw), dtype=data.dtype)
                fixed_patch[:z_end-z, :y_end-y, :x_end-x] = data[z:z_end, y:y_end, x:x_end]
                patch = fixed_patch
                print(f"Fixed patch shape: {patch.shape}")
            
            # Normalize if needed
            if self.config.get('data.normalize', False):
                patch = 255 * (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
                patch = patch.astype(np.uint8)

            # Save the patch - MODIFY: use start_idx
            self._save_as_raw(patch, os.path.join(output_dir, f"{start_idx + i:06d}.raw"))
            
        # Verify all saved patches have the correct size
        print("Verifying saved patch sizes...")
        for i in range(num_patches):
            file_path = os.path.join(output_dir, f"{start_idx + i:06d}.raw")  # MODIFY
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                expected_size = pd * ph * pw
                if file_size != expected_size:
                    print(f"Warning: File {file_path} has size {file_size}, expected {expected_size}")

    def _save_as_raw(self, data, output_path):
        """Save data as RAW file.
        
        Args:
            data: Data to save
            output_path: Path to save the data
        """
        # Ensure data is uint8
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
            
        # Save data
        data.tofile(output_path)