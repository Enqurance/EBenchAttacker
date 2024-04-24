
import common
from language_models import GPT, Claude, PaLM, HuggingFace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json

sys.path.append("./Attacks")

from assets.ModelPath import LLAMA_PATH, BAICHUAN_PATH, GEMMA_PATH, CHATGLM_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P   

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        use_proxy = args.api_use_proxy,
                        api_info_path = args.api_info_path,
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        use_proxy = args.api_use_proxy,
                        api_info_path = args.api_info_path,
                        )
    return attackLM, targetLM

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float,
                use_proxy: bool,
                api_info_path: str,
                ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.use_proxy = use_proxy
        self.api_info_path = api_info_path
        self.model, self.template = load_indiv_model(model_name, self.use_proxy, self.api_info_path)
        
        if "vicuna" in model_name.lower() or "llama" in model_name.lower():
            self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "GPT" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "GPT" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            use_proxy: bool,
            api_info_path: str,
            preloaded_model: object = None,
            ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.use_proxy = use_proxy
        self.api_info_path = api_info_path
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name, self.use_proxy, self.api_info_path)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "GPT" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        outputs_list = self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
        return outputs_list



def load_indiv_model(model_name, use_proxy, api_info, device=None):
    model_path, template = get_model_path_and_template(model_name)
    with open(api_info, "r") as file:
        api_dict = json.load(file)
    if model_name in api_dict["GPT"]["APIs"]:
        lm = GPT(model_name, use_proxy, api_dict["GPT"]["Proxy"])
    elif model_name in api_dict["Claude"]["APIs"]:
        lm = Claude(model_name, use_proxy, api_dict["Claude"]["Proxy"])
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    # "Baichuan2-7B-Chat", "ChatGLM3-6B", "Gemma-2B", "LLaMA-2-7B-chat-hf",
    # "GPT-3.5-Turbo-0125", "GPT-4", "Claude-Instant"
    full_model_dict={
        "GPT-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "GPT-3.5-Turbo-0125": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "LLaMA-2-7B-chat-hf":{
            "path":LLAMA_PATH,
            "template":"llama-2"
        },
        "Gemma-2B":{
            "path":GEMMA_PATH,
            "template":"llama-2"
        },
        "ChatGLM3-6B":{
            "path":CHATGLM_PATH,
            "template":"llama-2"
        },
        "Baichuan2-7B-Chat":{
            "path":BAICHUAN_PATH,
            "template":"llama-2"
        },
        "Claude-Instant-1.2":{
            "path":"claude-instant-1.2",
            "template":"claude-instant-1"
        },
        "Claude-3-Haiku-20240307":{
            "path":"Claude-3-Haiku-20240307",
            "template":"claude-instant-1"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



    