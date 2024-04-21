import openai
import anthropic
import time
import json
import os
from tqdm import tqdm
import Attacks.Tools.InfoTool as InfoTool

used_token = 0

def Query(content, client, api_version):
    global used_token
    time.sleep(0.1)
    if str(type(client)) == "<class 'openai.OpenAI'>":
        response = client.chat.completions.create(
            model=api_version,
            max_tokens=256,
            messages=[
                {"role": "user", "content": content},
            ]
        )
        used_token += response.usage.total_tokens
        return response.choices[0].message.content
    elif str(type(client)) == "<class 'anthropic.Anthropic'>":
        response = client.messages.create(
            model=api_version,
            max_tokens=256,
            messages=[
            {"role": "user", "content": content}
            ]
        )
        used_token += response.usage.input_tokens
        used_token += response.usage.output_tokens
        return response.content[0].text
    else:
        raise ValueError("Unrecognized client")


def Attack(params, data, language, api_version, client): 
    batch_res = []
    for i in tqdm(range(len(data)), desc="Processing"):
        temp = {}
        res = Query(data[i][language], client, api_version.lower())
        temp["id"] = data[i]["id"]
        temp["Question"] = data[i][language]
        temp["Output"] = res
        temp["Answer"] = res
        batch_res.append(temp)
    return batch_res

        

def AttackGPTs(params, apis, data, language, attack, now, path):
    use_proxy = params["api_use_proxy"]
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("Env. variable OPENAI_API_KEY is not given")
    
    if use_proxy:
        openai_client = openai.OpenAI(
            base_url=apis["Proxy"],
            api_key=api_key
        )
    else:
        openai_client = openai.OpenAI(
            api_key=api_key
        )
        
    for api in apis["APIs"]:
        info = "Start " + attack + " Attack on " + api + " with language " + language + ". Using dataset " + path
        InfoTool.PrintWithBorders(info)
        output_file = api + "_result_" + language + ".json"
        res = Attack(params, data, language, api, openai_client)
        DumpFile(res, output_file, attack, now)
    InfoTool.PrintWithBorders("Token used:" + str(used_token))
        

def AttackClaudes(params, apis, data, language, attack, now, path):
    use_proxy = params["api_use_proxy"]
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key is None:
        raise ValueError("Env. variable ANTHROPIC_API_KEY is not given")
    
    if use_proxy:
        anthropic_client = anthropic.Anthropic(
            base_url=apis["Proxy"],
            api_key=api_key
        )
    else:
        anthropic_client = anthropic.Anthropic(
            api_key=api_key
        )
        
    for api in apis["APIs"]:
        info = "Start " + attack + " Attack on " + api + " with language " + language + ". Using dataset " + path
        InfoTool.PrintWithBorders(info)
        output_file = api + "_result_" + language + ".json"
        res = Attack(params, data, language, api, anthropic_client)
        DumpFile(res, output_file, attack, now)
        
        
def DumpFile(res, output_file, attack, now):
    file_dir = "./result/" + attack + "/" + now + "/"
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    with open(file_dir + output_file, "w") as file:
        json.dump(res, file, indent=4)


