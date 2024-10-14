import os.path as op
import pandas as pd
import numpy as np
import random
import torch

def get_prompt(prompt_path):
    if op.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
    return prompts

def get_id2name(data_dir):
    id2name = dict()
    item_path=op.join(op.join(data_dir), 'id2name.txt')
    with open(item_path, 'r') as f:
        for l in f.readlines():
            ll = l.strip('\n').split('::')
            id2name[int(ll[0])] = ll[1].strip()
    return id2name

def get_template(template_path):
    if op.isfile(template_path):
        with open(template_path, 'r') as f:
            template = f.read()
    return template

def session_data4frame(datapath):
    if "steam" in datapath:
        padding_item_id = 3581
    elif "goodreads" in datapath:
        padding_item_id = 4550
    elif "movielens" in datapath:
        padding_item_id = 1682
    train_data = pd.read_pickle(datapath)
    train_data = train_data[train_data['len_seq'] >= 3]
    def remove_padding(xx):
        x = xx[:]
        for i in range(10):
            try:
                x.remove(padding_item_id)
            except:
                break
        return x
    train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
    return train_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_parameters_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Total trainable parameters: {total_trainable_params}")