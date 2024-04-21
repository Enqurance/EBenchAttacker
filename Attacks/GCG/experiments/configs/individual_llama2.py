import os

os.sys.path.append("..")
os.sys.path.append("./Attacks")

from configs.template import get_config as default_config
from assets.ModelPath import LLAMA_PATH


def get_config():
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=[LLAMA_PATH]
    config.model_paths=[LLAMA_PATH]
    config.conversation_templates=['llama-2']

    return config