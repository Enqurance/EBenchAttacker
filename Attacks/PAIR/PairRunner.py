import json
import subprocess
from tqdm import tqdm


def LoadJson(path):
    with open(path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data
        

def Attack(attack_model, target_model, data_path, params):
    data = LoadJson(data_path)
    judge_model = "GPT-3.5-Turbo-0125"
    for i in tqdm(range(len(data)), desc="Processing"):
        item = data[i]
        command = "python3 ./Attacks/PAIR/main.py --attack-model " + attack_model + \
            " --target-model " + target_model + \
            " --judge-model " + judge_model + \
            " --goal \"" + item["English"] + \
            "\" --target-str \"" + item["Goal"] + \
            "\" --n-streams 2 --n-iterations 25" + \
            " --api-use-proxy " + str(params["api_use_proxy"]) + \
            " --api-info-path " + params["api_info_path"]
        subprocess.run(command, shell=True, text=True)
        
        
def PAIRAttack(attack_model, models, data_path, params):
    for model in models.keys():
        Attack(attack_model, model, data_path, params)