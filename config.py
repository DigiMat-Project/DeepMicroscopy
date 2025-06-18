import os
import yaml
from pathlib import Path
import copy


class Config:
    """Configuration manager for DeepMicroscopy.

    This class handles loading, accessing, and modifying configuration values
    from YAML files. It provides attribute-style access to nested configuration
    values and supports default values for missing keys.
    """

    def __init__(self, config_path):
        """Initialize configuration from a file path or dictionary.

        Args:
            config_path: Either a path to a YAML file or a configuration dictionary
        """
        if isinstance(config_path, dict):
            self.config = copy.deepcopy(config_path)
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Validate and set dimensions
        self.dimensions = self.config.get('dimensions', 3)
        assert self.dimensions in [2, 3], f"Dimensions must be 2 or 3, got {self.dimensions}"

    def get(self, key_path, default=None):
        """Get a configuration value using a dot-separated path.

        Args:
            key_path: Dot-separated path to the configuration value
            default: Value to return if the path doesn't exist

        Returns:
            The configuration value or the default value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path, value):
        """Set a configuration value using a dot-separated path.

        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to the correct position
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

        # Update dimensions if changed
        if key_path == 'dimensions':
            self.dimensions = value

    def __getattr__(self, key):
        """Provide attribute-style access to config values.

        Args:
            key: Configuration key

        Returns:
            Configuration value or a nested Config object

        Raises:
            AttributeError: If the key doesn't exist
        """
        if key in self.config:
            val = self.config[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"No configuration value for '{key}'")

    def save(self, save_path):
        """Save the configuration to a YAML file.

        Args:
            save_path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_path):
        """Create a Config object from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            Config object
        """
        return cls(yaml_path)

    @classmethod
    def merge(cls, base_config, override_config):
        """Merge two configurations, with override_config taking precedence.

        Args:
            base_config: Base configuration
            override_config: Configuration to override values in base_config

        Returns:
            Merged Config object
        """
        if isinstance(base_config, cls):
            base_config = base_config.config
        if isinstance(override_config, cls):
            override_config = override_config.config

        merged = copy.deepcopy(base_config)

        # Recursive merge
        def merge_dict(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value

        merge_dict(merged, override_config)
        return cls(merged)

    def __str__(self):
        """String representation of the configuration.

        Returns:
            YAML string of the configuration
        """
        return yaml.dump(self.config, default_flow_style=False)

    def is_2d(self):
        """Check if the configuration is for 2D mode.

        Returns:
            True if dimensions=2, False otherwise
        """
        return self.dimensions == 2

    def is_3d(self):
        """Check if the configuration is for 3D mode.

        Returns:
            True if dimensions=3, False otherwise
        """
        return self.dimensions == 3


# Functions for working with configurations
def load_config(config_path):
    """Load a configuration from a YAML file or dictionary.

    Args:
        config_path: Path to YAML file or configuration dictionary

    Returns:
        Config object
    """
    return Config(config_path)


def save_config(config, save_path):
    """Save a configuration to a YAML file.

    Args:
        config: Config object or dictionary
        save_path: Path to save the configuration
    """
    if isinstance(config, dict):
        config = Config(config)
    config.save(save_path)


def merge_configs(base_config, override_config):
    """Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration (Config object or path)
        override_config: Configuration to override values (Config object or path)

    Returns:
        Merged Config object
    """
    return Config.merge(base_config, override_config)
