import argparse
from datetime import datetime
import torch
from Attacks.assets.ModelPath import LLAMA3_PATH, LLAMA_PATH, GEMMA_PATH, CHATGLM_PATH, BAICHUAN_PATH
from Attacks.assets.ModelPath import LLAVA_PATH, QWEN_PATH

import Attacks.Default.Runner as Runner
import Attacks.PAIR.PairRunner as PairRunner
import Attacks.Tools.InfoTool as InfoTool
import Attacks.GPTFuzz.FuzzRunner as FuzzRunner
import Attacks.GCG.GCGRunner as GCGRunner
import Attacks.GCGTransfer.GCGTransferRunner as GCGTransferRunner
import Attacks.VisualAttack.VisualAttackRunner as VisualAttackRunner
import Attacks.AutoDAN.AutoDANRunner as AutoDANRunner

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='This is a attacking benchmark for LLMs')

parser.add_argument('--attacks',type=str,
                    nargs='+', choices=['direct', 'multilingual', 'pair', 'gptfuzz', 'gcg', 'gcg_transfer', 'autodan', 'visual'],
                    help='This list assigns the methods for attacking')
parser.add_argument('--dataset', type=str, choices=['small', 'medium', 'origin', 'test'],
                    help='This arg. assigns the dataset to use')
parser.add_argument('--transfer_dataset', type=str, choices=['LLaMA-2-7B-chat-hf', 'Gemma-2B', 'Baichuan2-7B-Chat'], 
                    help='This arg. assigns the dataset to use when using GCG transfer attack')

parser.add_argument('--test-api', type=bool, default=False, help='This arg. determines whether to test APIs or not')
parser.add_argument('--api-use-proxy', type=bool, default=True, help='This arg. determines whether to use APIs proxy or not')
parser.add_argument('--api-info-path', type=str, default="./Attacks/assets/API_info.json", help='This arg. determines the path for APIs\' information')

# ----------------------------------------------------------------------------------------------
# | 'model_list' includes models to be testes. If you want to add a open source model, please  |
# |  add it in 'model_list' and add the local path of that model in file 'LLMAttacks/Attacks/  |
# |  assets/ModelPath.py'. If you want to test a additional api service, please also add it in |
# | 'model_list' and edit the file 'LLMAttacks/Attacks/assets/API_info.json'. Please note that |
# | only GPT and Claude APIs are supported by the current benchmark.                           |
# ----------------------------------------------------------------------------------------------

open_model_list = ["Baichuan2-7B-Chat", "ChatGLM3-6B", "Gemma-2B", "LLaMA-2-7B-chat-hf"]

api_model_paths = {
    "GPT-3.5-Turbo-0125": "GPT-3.5-Turbo-0125",
    "GPT-4": "GPT-4",
    "Claude-3-Haiku-20240307": "Claude-3-Haiku-20240307"
}

model_paths = {
    # "Baichuan2-7B-Chat": BAICHUAN_PATH,
    # "ChatGLM3-6B": CHATGLM_PATH,
    # "LLaMA-2-7B-chat-hf" : LLAMA_PATH,
    # "Gemma-2B": GEMMA_PATH,
    "LLaMA-3-8B-Instruct": LLAMA3_PATH
}

multimodal_model_paths = {
    "Qwen-VL-Chat": QWEN_PATH,
    "LLaVA-v1.5-7B": LLAVA_PATH
}

data_paths = {
    None: {"json": "./data/EBench_test.json", "csv": "./data/EBench.csv"},
    "origin": {"json": "./data/EBench.json", "csv": "./data/EBench.csv"},
    "medium": {"json": "./data/EBench_medium.json", "csv": "./data/EBench_medium.csv"},
    "small": {"json": "./data/EBench_small.json", "csv": "./data/EBench_small.csv"},
    "test": {"json": "./data/EBench_test.json", "csv": "./data/EBench_test.csv"}
}

transfer_data_paths = {
    None: "./data/EBench_GCG_LLaMA-2-7B.json",
    'LLaMA-2-7B-chat-hf': "./data/EBench_GCG_LLaMA-2-7B.json",
    'Gemma-2B': "./data/EBench_GCG_Gemma-2B.json",
    'Baichuan2-7B-Chat': "./data/EBench_GCG_Baichuan2-7B.json"
}

def Attacking(params):
    attacks = params["attacks"]
    dataset = params["dataset"]
    test_api = params["test_api"]
    
    if attacks is None:
        raise ValueError("Argument '--attacks' is None, please assign attacking methods")
    
    # Get the path for dataset
    if dataset not in data_paths.keys():
        raise ValueError("Argument '--dataset' shall be one of None, medium or small")
    
    transfer_dataset = params["transfer_dataset"]
    transfer_data_path = transfer_data_paths[transfer_dataset]
    selected_paths = data_paths.get(dataset, data_paths[None])
    data_path_json = selected_paths["json"]
    data_path_csv = selected_paths["csv"]
    
    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    for a in attacks:
        match a:
            case "direct":
                Runner.DirectAttack(model_paths, data_path_json, time_str)
                if test_api:
                    Runner.DirectAPIAttack(params, data_path_json, time_str)
            case "multilingual":
                Runner.MultilingualAttack(model_paths, data_path_json, time_str)
                if test_api:
                    Runner.MultilingualAPIAttack(params, data_path_json, time_str)
            case "pair":
                PairRunner.PAIRAttack("LLaMA-2-7B-chat-hf", model_paths, data_path_json, params)
                if test_api:
                    PairRunner.PAIRAttack("LLaMA-2-7B-chat-hf", api_model_paths, data_path_json, params)
            case "gptfuzz":
                FuzzRunner.FuzzAttack(model_paths, data_path_json, params, time_str)
                if test_api:
                    FuzzRunner.FuzzAttack(api_model_paths, data_path_json, params, time_str)
            case "gcg":
                GCGRunner.GCGAttack(model_paths, data_path_csv, time_str)
            case "gcg_transfer":
                if test_api:
                    GCGTransferRunner.GCGTransferAPIAttack(params, transfer_data_path, time_str)
                GCGTransferRunner.GCGTransferAttack(model_paths, transfer_data_path, time_str)
            case "autodan":
                AutoDANRunner.AutoDANAttack(model_paths, data_path_csv, time_str)
            case "visual":
                VisualAttackRunner.VisualAttack(multimodal_model_paths, time_str)
            case _:
                pass
            
            


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    InfoTool.PrintWithBorders(f"Using: {device}")
    args = parser.parse_args()
    params = vars(args)
    Attacking(params)
    


if __name__ == "__main__":
    main()