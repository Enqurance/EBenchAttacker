
import Attacks.Tools.DataLoader as DL
import Attacks.Tools.ModelLoader as ML


def Attack(model_path, model_name, path, lan):
    data = DL.DataLoader(path)
    model = ML.ModelLoader(model_path, model_name)
    for cate in DL.Category:
        batch = data.load_cate(cate.value)
        model.batch_query(batch, lan)
    model.dump_file()
    

def MultilingualAttack(model_info, data_path):
    model_names = model_info.keys()
    languages = ["Chinese", "Javanese", "Urdu", "Igbo", "Hausa", "Lithuanian"]
    for model in model_names:
        for l in languages:
            Attack(model_info[model], model, data_path, l)
