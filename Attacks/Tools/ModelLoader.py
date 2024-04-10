from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch


class ModelLoader():
    
    def __init__(self, model_path, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to(self.device)
        self.model_name = model_name
        self.res = []
        self.language = ""
        

    def query_model(self, query):
        input_ids = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=128, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    def batch_query(self, batch, language):
        self.language = language;
        batch_res = []
        for item in batch:
            temp = {}
            result = self.query_model(item[language] + "#")
            temp["id"] = item["id"]
            temp["Question"] = item[language]
            temp["Output"] = result
            if "#" in result:
                temp["Answer"] = result.split('#', 1)[1]
            else:
                temp["Answer"] = result
            print("----- " + item["id"] +  " -----")
            batch_res.append(temp)
        for i in batch_res:
            self.res.append(i)
    
    def dump_file(self):
        output_file = "./result/" + self.model_name + "_result_" + self.language + ".json"
        with open(output_file, "w") as file:
            json.dump(self.res, file, indent=4)
