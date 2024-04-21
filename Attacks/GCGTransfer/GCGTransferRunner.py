
import Attacks.Tools.DataLoader as DL
import Attacks.Tools.ModelLoader as ML
import Attacks.Tools.InfoTool as InfoTool
import Attacks.Default.APIQuery as APIQuery
import json

def GCGTransferAttack(model_info, data_path, time_str):
    model_names = model_info.keys()
    for model in model_names:
        if model.split("-")[0] not in data_path:
            info = "Start GCG Transfer Attack on " + model + ". Using dataset " + data_path
            InfoTool.PrintWithBorders(info)
            data = DL.DataLoader(data_path)
            model = ML.ModelLoader(model_info[model], model, time_str)
            model.batch_query(data.json_data, "WithPrefix")
            model.dump_file("GCGTansfer")
        

def GCGTransferAPIAttack(params, path, time_str):
    api_info_path = params["api_info_path"]
    data = DL.DataLoader(path)
    with open(api_info_path, "r") as file:
        api_info = json.load(file)
    for item in api_info.keys():
        match item:
            case "GPT":
                APIQuery.AttackGPTs(params, api_info["GPT"], data.json_data, "WithPrefix", "GCGTansfer", time_str, path)
            case "Claude":
                APIQuery.AttackClaudes(params, api_info["Claude"], data.json_data, "WithPrefix", "GCGTansfer", time_str, path)
            case _:
                pass