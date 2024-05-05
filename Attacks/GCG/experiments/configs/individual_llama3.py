import os

os.sys.path.append("..")
os.sys.path.append("./Attacks")

from configs.template import get_config as default_config
from assets.ModelPath import LLAMA3_PATH


def get_config():
    config = default_config()

    config.result_prefix = 'results/individual_llama3'

    config.tokenizer_paths=[LLAMA3_PATH]
    config.model_paths=[LLAMA3_PATH]
    config.conversation_templates=['llama-3']

    return config