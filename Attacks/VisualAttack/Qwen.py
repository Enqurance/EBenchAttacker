from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
import json
from tqdm import tqdm


def GetPrompt(path, prompts):
    for key in prompts.keys():
        if key in path:
            return prompts[key]
    return prompts["Default"]


def AttackQwen(model_path, img_path, prompts, attack_times, time_str):
    torch.manual_seed(1234)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    
    res = []
    
    for root, dirs, files in os.walk(img_path):
        for file in tqdm(files, desc="Processing"):
            for i in range(attack_times):
                path = os.path.join(root, file)
                p = GetPrompt(path, prompts)
                query = tokenizer.from_list_format([
                    {'image': path},
                    {'text': p},
                ])
                output, history = model.chat(tokenizer, query=query, history=None)
                res.append({"Image": path, "Output": output})
    
    
    output_path = './result/Visual/' + time_str + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + "Qwen-VL-Chat.json", "w") as file:
        json.dump(res, file, indent=4)
