
import Attacks.Tools.DataLoader as DL
import Attacks.Tools.ModelLoader as ML


def Attack(model_path, model_name, path):
    language = "English"
    data = DL.DataLoader(path)
    model = ML.ModelLoader(model_path, model_name)
    for cate in DL.Category:
        batch = data.load_cate(cate.value)
        model.batch_query(batch, language)
    model.dump_file()
    

def DirectAttack(model_info, data_path):
    model_names = model_info.keys()
    for model in model_names:
        Attack(model_info[model], model, data_path)
