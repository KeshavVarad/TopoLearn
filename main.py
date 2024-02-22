from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.utils import logging
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.integrate as integrate
import tqdm
from accelerate import Accelerator




accelerator = Accelerator()

device = accelerator.device

logging.set_verbosity_error()

dataset = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

def label_id_name(id):
    
    if id == 0:
        return "stereotype"
    
    elif id == 1:
        return "anti-stereotype"
    
    elif id == 2:
        return "irrelevant"
    
    else:
        raise Exception("Label does not exist")
    

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


def birth_death_decomp(adj):
    eps = np.nextafter(0, 1)
    adj[adj == 0] = eps
    adj = np.triu(np.transpose(adj), k=1)

    Xcsr = csr_matrix(-adj)
    Tcsr = minimum_spanning_tree(Xcsr)
    mst = -Tcsr.toarray()  # reverse the negative sign
    nonmst = adj - mst
    
    return mst, nonmst

def get_birth_death_sets(adj):
    mst, nonmst = birth_death_decomp(adj)
    
    
    birth_ind = np.nonzero(mst)
    death_ind = np.nonzero(nonmst)
    return np.sort(mst[birth_ind]), np.sort(nonmst[death_ind])


def empirical_dist(x, set):

    return np.sum(set <= x) / len(set)


def birth_death_dist(z, adj):
    
    birth_set, death_set = get_birth_death_sets(adj)
    
    birth_val = 0
    death_val = 0
    for i in range(len(birth_set)):
        if i + 1 >= z * len(birth_set):
            birth_val = birth_set[i]
            break
        
    for i in range(len(death_set)):
        if i + 1 >= z * len(death_set):
            death_val = death_set[i]
            break
    
    return birth_val, death_val


def birth_square_diff(z, adj1, adj2):
    
    birth_dist1, _ = birth_death_dist(z, adj1)
    birth_dist2, _ = birth_death_dist(z, adj2)
    
    square_diff = (birth_dist2 - birth_dist1) ** 2
    
    return square_diff

def death_square_diff(z, adj1, adj2):
    
    _, death_dist1 = birth_death_dist(z, adj1)
    _, death_dist2 = birth_death_dist(z, adj2)
    
    square_diff = (death_dist1 - death_dist2) ** 2
    
    return square_diff

def top_loss_0(adj1, adj2):
    res, error = integrate.quad(lambda x: birth_square_diff(x, adj1, adj2), 0, 1)
    return res

def top_loss_1(adj1, adj2):
    res, error = integrate.quad(lambda x: death_square_diff(x, adj1, adj2), 0, 1)
    return res

stereo_to_antistereo_dists = []
stereo_to_unrelated_dists = []
antistereo_to_unrelated_dists = []

MAX_ITER = 20
STOP_EARLY = False
SAVE_AFTER = 100
OUTPUT_DIR = 'data/experiment2/'

iter_num = 0
for item in tqdm.tqdm(dataset, desc = "Processing dataset"):
    if STOP_EARLY and iter_num == MAX_ITER:
        break
    
    if iter_num > 0 and iter_num % SAVE_AFTER == 0:
        np_stereo_to_antistereo_dists = np.stack(stereo_to_antistereo_dists, axis = 2)
        np_stereo_to_unrelated_dists = np.stack(stereo_to_unrelated_dists, axis = 2)
        np_antistereo_to_unrelated_dists = np.stack(antistereo_to_unrelated_dists, axis = 2)


        with open(OUTPUT_DIR + 'stereo_to_antistereo_dists_' + str(iter_num) + '.npy', 'wb') as f:
            np.save(f, np_stereo_to_antistereo_dists)

        with open(OUTPUT_DIR + 'stereo_to_unrelated_dists_' + str(iter_num) + '.npy', 'wb') as f:
            np.save(f, np_stereo_to_unrelated_dists)

        with open(OUTPUT_DIR + 'antistereo_to_unrelated_dists_' + str(iter_num) + '.npy', 'wb') as f:
            np.save(f, np_antistereo_to_unrelated_dists)
    
    iter_num += 1
    
    context = item["context"]
    
    sentenceData = item["sentences"]
    sentences = sentenceData["sentence"]
    gold_labels = sentenceData["gold_label"]
    
    # print("Context: ", context)
    # print("Sentences: ", sentences)
    # print("Labels: ", gold_labels)
    
    attentions = [get_attention(context, sentence) for sentence in sentences]
    
    
    stereotypeAttention = attentions[0][0][0][0] 
    antistereotypeAttention = attentions[1][0][0][0]
    unrelatedAttention = attentions[2][0][0][0]
    
    
    numLayers = stereotypeAttention.shape[0]
    
    stereo_to_antistereo_dist = np.zeros(shape=(numLayers, 2))
    stereo_to_unrelated_dist = np.zeros(shape=(numLayers, 2))
    antistereo_to_unrelated_dist = np.zeros(shape=(numLayers, 2))
    
    # print(top_loss_0(stereotypeAttention[0].numpy(), antistereotypeAttention[0].numpy()))
    
    for layerNum in tqdm.tqdm(range(numLayers), desc = "Processing attention layers", leave = False):
    # for layerNum in range(numLayers):
        stereo_to_antistereo_dist[layerNum][0] = top_loss_0(stereotypeAttention[layerNum].cpu().numpy(), antistereotypeAttention[layerNum].cpu().numpy())
        stereo_to_unrelated_dist[layerNum][0] = top_loss_0(stereotypeAttention[layerNum].cpu().numpy(), unrelatedAttention[layerNum].cpu().numpy())
        antistereo_to_unrelated_dist[layerNum][0] = top_loss_0(antistereotypeAttention[layerNum].cpu().numpy(), unrelatedAttention[layerNum].cpu().numpy())
        
        stereo_to_antistereo_dist[layerNum][1] = top_loss_1(stereotypeAttention[layerNum].cpu().numpy(), antistereotypeAttention[layerNum].cpu().numpy())
        stereo_to_unrelated_dist[layerNum][1] = top_loss_1(stereotypeAttention[layerNum].cpu().numpy(), unrelatedAttention[layerNum].cpu().numpy())
        antistereo_to_unrelated_dist[layerNum][1] = top_loss_1(antistereotypeAttention[layerNum].cpu().numpy(), unrelatedAttention[layerNum].cpu().numpy())

    stereo_to_antistereo_dists.append(stereo_to_antistereo_dist)
    stereo_to_unrelated_dists.append(stereo_to_unrelated_dist)
    antistereo_to_unrelated_dists.append(antistereo_to_unrelated_dist)
    
    

