import json
from pathlib import Path

proj_dir = Path(__file__).resolve().parent.parent

def get_config():
    with open(f'{proj_dir}/config.json') as f:
        config = json.load(f)
    return config

print(get_config())