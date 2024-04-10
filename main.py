import argparse
import Attacks.Direct.Runner as DirectRunner
import Attacks.Multilingual.Runner as MultilingualRunner

parser = argparse.ArgumentParser(description='This is a attacking benchmark for LLMs')
parser.add_argument('--attacks',type=str,
                    nargs='+', help='This list assigns the methods for attacking')
parser.add_argument('--dataset', type=str, help='This arg assigns the dataset to use')


model_paths = {
    "Baichuan-7B-Chat" : "/home/LAB/gaoshiqi/models/Baichuan2-7B-Chat",
    "ChatGLM3-6B" : "/home/LAB/gaoshiqi/models/chatglm3-6b",
    "Gemma-2B" : "/home/LAB/gaoshiqi/models/gemma-2b",
    "LLaMA-2-7B-chat-hf" : "/home/LAB/gongtx/models/llama-2-7b-chat-hf",
}

data_paths = {
    None: {"json": "./data/EBench.json", "csv": "./data/EBench.csv"},
    "medium": {"json": "./data/EBench_medium.json", "csv": "./data/EBench_medium.csv"},
    "small": {"json": "./data/EBench_small.json", "csv": "./data/EBench_small.csv"}
}

def Attacking(params):
    attacks = params["attacks"]
    dataset = params["dataset"]
    
    
    if attacks is None:
        raise ValueError("Argument '--attacks' is None, please assign attacking methods")
    
    # Get the path for dataset
    if dataset not in data_paths.keys():
        raise ValueError("Argument '--dataset' shall be one of None, medium or small")
    selected_paths = data_paths.get(dataset, data_paths[None])
    data_path_json = selected_paths["json"]
    data_path_csv = selected_paths["csv"]
    
    for a in attacks:
        match a:
            case "direct":
                DirectRunner.DirectAttack(model_paths, data_path_json)
            case "multilingual":
                MultilingualRunner.MultilingualAttack(model_paths, data_path_json)
            case _:
                pass
            
            


def main():
    args = parser.parse_args()
    params = vars(args)
    Attacking(params)
    


if __name__ == "__main__":
    main()