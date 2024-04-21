import os

os.sys.path.append("..")
os.sys.path.append("./Attacks")

from configs.template import get_config as default_config
from assets.ModelPath import GEMMA_PATH

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_gemma'
    
    config.tokenizer_paths=[GEMMA_PATH]
    config.model_paths=[GEMMA_PATH]
    config.conversation_templates=['gemma']

    return config