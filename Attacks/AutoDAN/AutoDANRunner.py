import subprocess

def Attack(target_model, target_model_path, data_path, time_str):
    command = "python3 ./Attacks/AutoDAN/autodan_hga_eval.py --model " + target_model + \
        " --model_path " + target_model_path + \
        " --dataset_path " + data_path + \
        " --time_str " + time_str
    subprocess.run(command, shell=True, text=True)
        
        
def AutoDANAttack(models, data_path, time_str):
    for model in models.keys():
        Attack(model, models[model], data_path, time_str)