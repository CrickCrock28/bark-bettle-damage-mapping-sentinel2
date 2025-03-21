import yaml

class Config:
    def __init__(self, path="config/config.yaml"):
        with open(path, 'r') as file:
            self._config_data = yaml.safe_load(file)

    def __getattr__(self, name):
        if name in self._config_data:
            return self._config_data[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")