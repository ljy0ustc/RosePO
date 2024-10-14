from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import pandas as pd
import os
import os.path as op
import random

from utils import get_id2name, get_template, session_data4frame, set_seed
from rec_predict_collator import RecPredCollator
from calculate_metrics import compute_ndcg, compute_hr
from pandas.core.frame import DataFrame

import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='steam', type=str, choices=['steam', 'goodreads', 'movielens'])
    parser.add_argument('--split', default='train', type=str, choices=['train', 'Val'])
    parser.add_argument('--llm_path', default='/root/model/llama-3-8b-instruct', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cans_num', default=20, type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def evaluate(args):
    args.output_dir = op.join(args.output_dir, args.dataset, "predict")
    
    accelerator = Accelerator()
    device = accelerator.device

    llama_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False).to(device)
    llama_model.eval()

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.padding_side = "left"  
    llm_tokenizer.pad_token_id = (0)

    print("Model is on device:", llama_model.device)
        
    item_id2name = get_id2name(op.join("./data/", args.dataset))
    instruction = get_template(f"./data/{args.dataset}/instruction.txt")
    input_template = get_template(f"./data/{args.dataset}/input_template.txt")
    llama_template_list = [get_template("./llama_template/input_template.txt"), get_template("./llama_template/output_template.txt")]
    item_id2emb = torch.load(f"./data/{args.dataset}/mpnet_embeddings_768.pt")

    data_files = {
        "test": session_data4frame(f"./data/{args.dataset}/{args.split}_data.df"),
    }
    test_data = Dataset.from_pandas(data_files["test"])

    guided_predictions = torch.load(f"./data/{args.dataset}/SASRec_predictions_{args.split}.pt")
    assert len(test_data) == guided_predictions.shape[0]
    guided_predictions = guided_predictions.tolist()
    def add_guided_predictions(example, idx):
        example['guided_prediction'] = guided_predictions[idx]
        return example
    test_data = test_data.map(add_guided_predictions, with_indices=True)

    collator = RecPredCollator(item_id2name, input_template = input_template, instruction = instruction, llama_template_list = llama_template_list, llm_tokenizer=llm_tokenizer, device=device, item_id2emb = item_id2emb)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    test_content={"pred_rank": [], "generate": [], "real": []}
    predict_content={"seq_unpad":[], "chosen":[], "rejected":[], "guided_prediction":[]}

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
        batch_cans_name = batch["cans_name"]
        logits = outputs.logits.to(torch.float32)
        word_probs = torch.stack([logit[-3,:] for logit in logits], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, choice_inputs], dim=-1).detach()

        ranks, negative_predictions = [], []
        for i, label in enumerate(batch_labels):
            sorted_indices = torch.argsort(choice_probs[i], descending=True)
            rank = (sorted_indices == int(label)-1).nonzero(as_tuple=True)[0].item()
            if rank>0:
                idx = random.choice(sorted_indices[:rank])
            else:
                idx = random.choice([ch for ch in range(args.cans_num) if ch != int(label)-1])
            negative_prediction = [batch_cans_name[i][idx]]
            ranks.append(rank)
            negative_predictions.append(negative_prediction)

        return ranks, negative_predictions

    for batch in test_dataloader:
        batch = accelerator.prepare(batch)
        generate_output = batch_generate(batch)
        pred_ranks, negative_predictions = batch_choose(batch)
        for i, generate in enumerate(generate_output):
            test_content["pred_rank"].append(pred_ranks[i])
            test_content["generate"].append(generate)
            test_content["real"].append(batch['correct_answer'][i])
            for negative_prediction in negative_predictions[i]:
                predict_content["seq_unpad"].append(batch['seq_name'][i])
                predict_content["chosen"].append(batch['next_name'][i])
                predict_content["rejected"].append(negative_prediction)
                predict_content["guided_prediction"].append(batch['guided_prediction'][i])

    predict_df=DataFrame(predict_content)
    if not op.exists(args.output_dir):
        os.makedirs(args.output_dir)
    predict_df.to_pickle(op.join(args.output_dir, f'{args.split}_predict.df'))
    hr_1 = compute_hr(test_content["pred_rank"], 1)
    hr_5 = compute_hr(test_content["pred_rank"], 5)
    hr_10 = compute_hr(test_content["pred_rank"], 10)
    ndcg_5 = compute_ndcg(test_content["pred_rank"], 5)
    ndcg_10 = compute_ndcg(test_content["pred_rank"], 10)

    print("len of testset:", len(test_content["real"]))
    print("hr@1", hr_1)
    print("hr@5", hr_5)
    print("hr@10", hr_10)
    print("ndcg@5", ndcg_5)
    print("ndcg@10", ndcg_10)


if __name__ == "__main__":
    args = parser()
    set_seed(args.seed)
    evaluate(args)
