"""Configuration management utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Hierarchical configuration with dot notation access.

    Supports nested access via attributes or dictionary-style access.

    Example:
        config = Config({"model": {"latent_dim": 32}})
        print(config.model.latent_dim)  # 32
        print(config["model"]["latent_dim"])  # 32
    """

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """Initialize configuration.

        Args:
            config_dict: Dictionary with configuration values.
        """
        self._config = {}
        for key, value in (config_dict or {}).items():
            if isinstance(value, dict):
                self._config[key] = Config(value)
            else:
                self._config[key] = self._convert_value(value)

    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert string values that look like numbers to proper types.

        Handles scientific notation (1e-3, 1e6) that YAML may parse as strings.
        """
        if not isinstance(value, str):
            return value

        # Try to convert to int or float
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value

    def __getattr__(self, name: str) -> Any:
        """Get config value via attribute access."""
        if name.startswith("_"):
            return super().__getattribute__(name)

        if name not in self._config:
            raise AttributeError(f"Config has no attribute '{name}'")

        return self._config[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set config value via attribute access."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._config[name] = self._to_config(value)

    def __getitem__(self, key: str) -> Any:
        """Get config value via dictionary access."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config value via dictionary access."""
        self._config[key] = self._to_config(value)

    @classmethod
    def _to_config(cls, value: Any) -> Any:
        """Convert dict to Config, convert numeric strings to numbers."""
        if isinstance(value, dict):
            return Config(value)
        return cls._convert_value(value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self.to_dict()})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default fallback.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        return self._config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self._config.items()
        }

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply.
        """
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(self._config.get(key), Config):
                self._config[key].update(value)
            else:
                self._config[key] = self._to_config(value)

    def copy(self) -> Config:
        """Create a deep copy of the configuration.

        Returns:
            New Config instance with copied values.
        """
        return Config(copy.deepcopy(self.to_dict()))


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Optional dictionary of values to override.

    Returns:
        Configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = Config(yaml.safe_load(f))

    if overrides:
        config.update(overrides)

    return config


def save_config(config: Config, save_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        save_path: Path to save YAML file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
