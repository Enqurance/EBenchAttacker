import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    
    config.model_paths = [
        "/home/LAB/gaoshiqi/models/vicuna-7b-v1.5",
    ]
    config.tokenizer_paths = [
        "/home/LAB/gaoshiqi/models/vicuna-7b-v1.5",
    ]
    
    return config