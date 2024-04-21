import json
import os
import sys
import subprocess

sys.path.append("./Attacks")

from assets.ModelPath import LLAMA_PATH, BAICHUAN_PATH, GEMMA_PATH, CHATGLM_PATH, GPTFUZZ_PATH

model_dict={
    "LLaMA-2-7B-chat-hf":{
        "path":LLAMA_PATH,
    },
    "Gemma-2B":{
        "path":GEMMA_PATH,
    },
    "ChatGLM3-6B":{
        "path":CHATGLM_PATH,
    },
    "Baichuan2-7B-Chat":{
        "path":BAICHUAN_PATH,
    },
    "GPT-3.5-Turbo-0125":{
        "path":"GPT-3.5-Turbo-0125"
    },
    "GPT-4":{
        "path":"GPT-4"
    }
}

def LoadJson(path):
    with open(path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data
        

def Attack(target_model, data_path, params, time_str):
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if openai_key == None:
        raise ValueError("Please export OPENAI_API_KEY in you environmental variables")
    elif anthropic_key == None:
        raise ValueError("Please export ANTHROPIC_API_KEY in you environmental variables")
    
    api_info_path = params["api_info_path"]
    with open(api_info_path, "r") as file:
        api_info_dict = json.load(file)
        
    mutate_model = "GPT-3.5-Turbo-0125"
    command = "python3 ./Attacks/GPTFuzz/main.py " + \
            "--openai_key=\"" + openai_key + "\" " + \
            "--claude_key=\"" + anthropic_key + "\" " + \
            "--model_path=\"" + mutate_model.lower() + "\" " + \
            "--target_model=\"" + model_dict[target_model]["path"] + "\" " + \
            "--target_model_name=\"" + target_model + "\" " + \
            "--judge_model=\"" + GPTFUZZ_PATH + "\" " + \
            "--use_proxy=" + str(params["api_use_proxy"]) + " "\
            "--api_proxy=\"" + api_info_dict["GPT"]["Proxy"] + "\" " + \
            "--data_path=\"" + data_path + "\" " + \
            "--attack_time=\"" + time_str + "\""
    subprocess.run(command, shell=True, text=True)
        
        
def FuzzAttack(models, data_path, params, time_str):
    for model in models.keys():
        Attack(model, data_path, params, time_str)