import sys
import os
import requests
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO

sys.path.append('./Attacks/VisualAttack/')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



def GetPrompt(path, prompts):
    for key in prompts.keys():
        if key in path:
            return prompts[key]
    return prompts["Default"]


def AttackLLaVA(model_path, img_path, prompts, attack_times, time_str):
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="LLaVA-1.5-7B"
    )


    res = []
    for root, dirs, files in os.walk(img_path):
        for file in tqdm(files, desc="Processing"):
            path = os.path.join(root, file)
            p = GetPrompt(path, prompts)
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + p
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            p = conv.get_prompt()
            input_ids = tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            for i in range(attack_times):
                image_data = load_image(str(path))
                image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.9,
                    max_new_tokens=256,
                )
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            res.append({"Image": path, "Output": output})
    
    output_path = './result/Visual/' + time_str + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + "LLaVA-1.5-7B.json", "w") as file:
        json.dump(res, file, indent=4)