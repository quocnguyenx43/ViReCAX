import yaml
from pathlib import Path

def load_config(level, path=''):
    config_folder = Path(__file__)
    if path == '':
        for i in range(level):
            config_folder = config_folder.parent
    else:
        config_folder = Path(__file__).parent + path

    with open(f"{config_folder}\config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_parent_path(level):
    path = Path(__file__)
    for i in range(level):
        path = path.parent
    return path