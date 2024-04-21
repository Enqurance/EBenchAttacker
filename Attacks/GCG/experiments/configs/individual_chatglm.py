import os

os.sys.path.append("..")
os.sys.path.append("./Attacks")

from configs.template import get_config as default_config
from assets.ModelPath import CHATGLM_PATH

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_chatglm'
    
    config.tokenizer_paths=[CHATGLM_PATH]
    config.model_paths=[CHATGLM_PATH]
    config.conversation_templates=['chatglm']

    return config