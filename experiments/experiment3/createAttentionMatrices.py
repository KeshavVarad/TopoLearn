from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging
import numpy as np
import tqdm
from accelerate import Accelerator
import os

accelerator = Accelerator()

device = accelerator.device

logging.set_verbosity_error()

dataset = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

ENV_PATH = "/Users/keshav/Desktop/Github/TopoLearn/"



def get_attention(context, sentence):
    inputs = tokenizer(context + " " + sentence, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=12,
                            output_attentions = True, 
                            use_cache = True,
                            return_dict_in_generate=True, 
                            num_return_sequences = 1, 
                            do_sample=True)
    
    attentions = outputs.attentions
    
    return attentions
def label_id_name(id):
    
    if id == 0:
        return "stereotype"
    
    elif id == 1:
        return "antistereotype"
    
    elif id == 2:
        return "irrelevant"
    
    else:
        raise Exception("Label does not exist")

data = {"layer" + str(layer_num):{"head" + str(head_num):{label_name:[] for label_name in ["stereotype", "antistereotype", "irrelevant"]} for head_num in range(12)} for layer_num in range(12)}

iter_num = 0
SAVE_AFTER = 2

for item in tqdm.tqdm(dataset, desc = "Creating attention matrices"):
    
    iter_num += 1
    
    context = item["context"]
    sentenceData = item["sentences"]
    sentences = sentenceData["sentence"]
    
    attentions = [get_attention(context, sentence) for sentence in sentences]
    
    stereotypeAttention = attentions[0][0]
    antistereotypeAttention = attentions[1][0]
    unrelatedAttention = attentions[2][0]
    
    for label_num in range(3):
        label_name = label_id_name(label_num)
        
        attention = attentions[label_num][0]
        
        for layer_num in tqdm.tqdm(range(12), desc = "Processing " + label_name, leave = False):
            
            layer_attention = attention[layer_num][0]
            
            for head_num in tqdm.tqdm(range(12), desc = "Processing layer " + str(layer_num), leave = False):
                
                save_dir = ENV_PATH + "data/averageAttentionMatrices/layer" + str(layer_num) + "/" + label_name + "/matrix" + str(iter_num) + ".np"
                averaged_head_attention = np.mean(layer_attention.cpu().numpy(), axis = 0)
                
                # data["layer" + str(layer_num)]["head" + str(head_num)][label_name].append(head_attention.cpu().numpy())

                with open(save_dir, "wb") as f:
                    np.save(f, averaged_head_attention)
            