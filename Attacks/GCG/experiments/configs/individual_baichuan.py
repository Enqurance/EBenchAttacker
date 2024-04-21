import os

os.sys.path.append("..")
os.sys.path.append("./Attacks")

from configs.template import get_config as default_config
from assets.ModelPath import BAICHUAN_PATH

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_baichuan'
    
    config.tokenizer_paths=[BAICHUAN_PATH]
    config.model_paths=[BAICHUAN_PATH]
    config.conversation_templates=['baichuan']

    return config