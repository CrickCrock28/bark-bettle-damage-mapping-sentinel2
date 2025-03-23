import yaml

class Config:
    """Configuration class to load configuration from a YAML file."""
    def __init__(self, path="config/config.yaml"):
        """Load configuration from a YAML file."""
        with open(path, 'r') as file:
            self._config_data = yaml.safe_load(file)

    def __getattr__(self, name):
        """Get attribute from configuration data."""
        if name in self._config_data:
            return self._config_data[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")