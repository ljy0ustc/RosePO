from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pandas as pd
import os
import os.path as op
import math

from utils import get_id2name, get_template, session_data4frame, set_seed
from rec_eval_collator import RecEvalCollator
from calculate_metrics import compute_ndcg, compute_hr
from pandas.core.frame import DataFrame

import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='steam', type=str, choices=['steam', 'goodreads', 'movielens'])
    parser.add_argument('--llm_path', default='/root/model/llama-3-8b-instruct', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cans_num', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval_task', default='random', type=str, choices=['random', 'semantic-hard_reranking'])
    args = parser.parse_args()
    return args

def evaluate(args):
    args.output_dir = op.join(args.output_dir, args.dataset, "test")
    
    accelerator = Accelerator()
    device = accelerator.device

    llama_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False).to(device)

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.padding_side = "left"  
    llm_tokenizer.pad_token_id = (0)
    llama_model.eval()

    print("Model is on device:", llama_model.device)
        
    item_id2name = get_id2name(op.join("./data/", args.dataset))
    instruction = get_template(f"./data/{args.dataset}/instruction.txt")
    input_template = get_template(f"./data/{args.dataset}/input_template.txt")
    llama_template_list = [get_template("./llama_template/input_template.txt"), get_template("./llama_template/output_template.txt")]
    item_id2emb = torch.load(f"./data/{args.dataset}/mpnet_embeddings_768.pt")

    data_files = {
        "train": session_data4frame(f"./data/{args.dataset}/train_data.df"),
        "val": session_data4frame(f"./data/{args.dataset}/Val_data.df"),
        "test": session_data4frame(f"./data/{args.dataset}/Test_data.df"),
    }
    
    train_data = Dataset.from_pandas(data_files["train"])
    val_data = Dataset.from_pandas(data_files["val"])
    test_data = Dataset.from_pandas(data_files["test"])

    if args.eval_task == "semantic-hard_reranking":
        guided_predictions = torch.load(f"./data/{args.dataset}/SASRec_predictions_Test.pt")[:,:10]
        assert len(test_data) == guided_predictions.shape[0]
        guided_predictions = guided_predictions.tolist()
        def add_guided_predictions(example, idx):
            example['guided_predictions'] = guided_predictions[idx]
            return example
        test_data = test_data.map(add_guided_predictions, with_indices=True)

    collator = RecEvalCollator(item_id2name, input_template = input_template, instruction = instruction, llama_template_list = llama_template_list, llm_tokenizer=llm_tokenizer, device=device, item_id2emb = item_id2emb)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    test_content={"pred_rank": [], "generate": [], "real": [], "seq": [], "cans": []}

    def batch_generate(batch):
        generated_ids = llama_model.generate(
            input_ids = batch["test_tokens"].input_ids,
            attention_mask = batch["test_tokens"].attention_mask,
            do_sample = False,
            max_new_tokens = 2,
            pad_token_id = llm_tokenizer.pad_token_id,
            eos_token_id = llm_tokenizer.eos_token_id,
            use_cache = False
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(batch["test_tokens"].input_ids, generated_ids)]
        output_text = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs
    
    def batch_choose(batch):
        targets = batch["eval_tokens"].input_ids.masked_fill(
            batch["eval_tokens"].input_ids == llm_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        targets = targets.masked_fill((batch["eval_tokens"].token_type_ids == 0), -100)
        outputs = llama_model(
            input_ids = batch["eval_tokens"].input_ids,
            attention_mask = batch["eval_tokens"].attention_mask,
            return_dict = True,
            labels=targets,
            use_cache = False
        )

        choice_inputs=[llm_tokenizer.encode(str(ch+1), add_special_tokens=False)[-1] for ch in range(args.cans_num)]
        batch_labels = batch["correct_answer"]
        logits = outputs.logits.to(torch.float32)
        word_probs = torch.stack([logit[-3,:] for logit in logits], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, choice_inputs], dim=-1).detach()

        ranks = []
        for i, label in enumerate(batch_labels):
            sorted_indices = torch.argsort(choice_probs[i], descending=True)
            rank = (sorted_indices == int(label)-1).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

        return ranks

    for batch in test_dataloader:
        batch = accelerator.prepare(batch)
        generate_output = batch_generate(batch)
        pred_ranks = batch_choose(batch)
        for i, generate in enumerate(generate_output):
            test_content["pred_rank"].append(pred_ranks[i])
            test_content["generate"].append(generate)
            test_content["real"].append(batch['correct_answer'][i])
            test_content["seq"].append(batch['seq_name'][i])
            test_content["cans"].append(batch['cans_name'][i])

    df=DataFrame(test_content)
    if not op.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df.to_csv(op.join(args.output_dir, 'test.csv'))
    hr_1 = compute_hr(test_content["pred_rank"], 1)
    hr_5 = compute_hr(test_content["pred_rank"], 5)
    hr_10 = compute_hr(test_content["pred_rank"], 10)
    ndcg_5 = compute_ndcg(test_content["pred_rank"], 5)
    ndcg_10 = compute_ndcg(test_content["pred_rank"], 10)

    print("load checkpoint from:", args.llm_path)
    print("evaluation task:", args.eval_task)
    print("hr@1", hr_1)
    print("hr@5", hr_5)
    print("hr@10", hr_10)
    print("ndcg@5", ndcg_5)
    print("ndcg@10", ndcg_10)

    def pop_eval(item_id2name, test_content):
        item_pop_dict={}
        for item_id, item_name in item_id2name.items():
            item_pop_dict[item_name] = 1
        for data_split in [train_data, val_data, test_data]:
            for sample in data_split:
                for id in sample["seq_unpad"]+[sample["next"]]:
                    item_pop_dict[item_id2name[id]]+=1
        M_r, M_ru = [], []
        for i, generate_content in enumerate(test_content["generate"]):
            item_name = test_content["cans"][i][int(generate_content)-1]
            log_r = math.log(item_pop_dict[item_name])
            log_u = [math.log(item_pop_dict[item]) for item in test_content["seq"][i]]
            M_r.append( log_r )
            M_ru.append( log_r - sum(log_u)/len(log_u) )
        return sum(M_ru)/len(M_ru)

    def semantic_eval(item_id2name, test_content):
        embeddings = torch.load(f"./data/{args.dataset}/mpnet_embeddings_768.pt")
        embeddings = embeddings/torch.norm(embeddings,dim=-1).unsqueeze(1)
        item_name2id_dict={}
        for item_id, item_name in item_id2name.items():
            item_name2id_dict[item_name] = item_id
        M_bias = []
        for i, generate_content in enumerate(test_content["generate"]):
            real_item_name = test_content["cans"][i][int(test_content["real"][i])-1]
            generate_item_name = test_content["cans"][i][int(generate_content)-1]
            if real_item_name != generate_item_name:
                real_emb = embeddings[item_name2id_dict[real_item_name]]
                gen_emb = embeddings[item_name2id_dict[generate_item_name]]
                seq_embs = embeddings[[item_name2id_dict[seq_name] for seq_name in test_content["seq"][i]]]
                can_embs = embeddings[[item_name2id_dict[can_name] for can_name in test_content["cans"][i]]]
                sim_gt = torch.mean(torch.matmul(seq_embs, real_emb.t()),dim=0)
                sim_r = torch.mean(torch.matmul(seq_embs, gen_emb.t()),dim=0)
                sim_c = torch.mean(torch.mean(torch.matmul(seq_embs, can_embs.t()),dim=0),dim=0)
                M_bias.append((sim_r - sim_gt)/sim_gt)
        return (sum(M_bias)/len(M_bias)).item()

    pop_bias = pop_eval(item_id2name, test_content)
    semantic_bias = semantic_eval(item_id2name, test_content)
    print("popularity bias:", pop_bias)
    print("semantic bias:", semantic_bias)

if __name__ == "__main__":
    args = parser()
    set_seed(args.seed)
    evaluate(args)
