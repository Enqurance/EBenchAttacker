import Attacks.Tools.DataLoader as DL
import Attacks.Tools.ModelLoader as ML
import Attacks.Tools.InfoTool as InfoTool
import Attacks.Default.APIQuery as APIQuery
import json

languages = ["Chinese", "Javanese", "Urdu", "Igbo", "Hausa", "Lithuanian"]

def Attack(model_path, model_name, path, lan, attack, now):
    info = "Start " + attack + " Attack on " + model_name + " with language " + lan + ". Using dataset " + path
    InfoTool.PrintWithBorders(info)
    data = DL.DataLoader(path)
    model = ML.ModelLoader(model_path, model_name, now)
    model.batch_query(data.json_data, lan)
    model.dump_file(attack)
    

def DirectAttack(model_info, data_path, now):
    model_names = model_info.keys()
    for model in model_names:
        Attack(model_info[model], model, data_path, "English", "Direct", now)
        

def DirectAPIAttack(params, path, now):
    api_info_path = params["api_info_path"]
    data = DL.DataLoader(path)
    with open(api_info_path, "r") as file:
        api_info = json.load(file)
    for item in api_info.keys():
        match item:
            case "GPT":
                APIQuery.AttackGPTs(params, api_info["GPT"], data.json_data, "English", "Direct", now, path)
            case "Claude":
                APIQuery.AttackClaudes(params, api_info["Claude"], data.json_data, "English", "Direct", now, path)
            case _:
                pass
            
            
def MultilingualAPIAttack(params, path, now):
    api_info_path = params["api_info_path"]
    data = DL.DataLoader(path)
    with open(api_info_path, "r") as file:
        api_info = json.load(file)
    for item in api_info.keys():
        match item:
            case "GPT":
                for l in languages:
                    APIQuery.AttackGPTs(params, api_info["GPT"], data.json_data, l, "Multilingual", now, path)
            case "Claude":
                for l in languages:
                    APIQuery.AttackClaudes(params, api_info["Claude"], data.json_data, l,  "Multilingual", now, path)
            case _:
                pass
        

def MultilingualAttack(model_info, data_path, now):
    model_names = model_info.keys()
    for model in model_names:
        for l in languages:
            Attack(model_info[model], model, data_path, l, "Multilingual", now)
