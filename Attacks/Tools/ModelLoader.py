from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os
from tqdm import tqdm


class ModelLoader():
    
    def __init__(self, model_path, model_name, now):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to(self.device)
        self.model_name = model_name
        self.res = []
        self.language = ""
        self.now = now
        

    def query_model(self, query):
        input_ids = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=192, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    def batch_query(self, batch, language):
        self.language = language;
        batch_res = []
        for i in tqdm(range(len(batch)), desc="Processing"):
            temp = {}
            result = self.query_model(batch[i][language] + "#")
            temp["id"] = batch[i]["id"]
            temp["Question"] = batch[i][language]
            temp["Output"] = result
            if "#" in result:
                temp["Answer"] = result.split('#', 1)[1]
            else:
                temp["Answer"] = result
            batch_res.append(temp)
        for i in batch_res:
            self.res.append(i)
    
    def dump_file(self, attack):
        file_dir = "./result/" + attack + "/" + self.now + "/"
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        output_file = self.model_name + "_result_" + self.language + ".json"
        with open(file_dir + output_file, "w") as file:
            json.dump(self.res, file, indent=4)
