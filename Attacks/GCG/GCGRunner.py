import subprocess

model_dict={
    "LLaMA-2-7B-chat-hf":{
        "name": "llama2",
    },
    "Gemma-2B":{
        "name": "gemma",
    },
    "ChatGLM3-6B":{
        "name": "chatglm",
    },
    "Baichuan2-7B-Chat":{
        "name": "baichuan",
    },
    "LLaMA-3-8B":{
        "name": "llama3",
    }
}

def GCGAttack(models, dataset, time_str):
    for model in models.keys():
        script_path = './Attacks/GCG/experiments/launch_scripts/run_gcg.sh'
        subprocess.run(['bash', script_path, model, time_str, model_dict[model]["name"], dataset])


if __name__ == "__main__":
    GCGAttack("LLaMA-2-7B-chat-hf", "2002_06_24")