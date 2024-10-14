import os
import torch
import re
import pandas as pd
import os.path as op
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
from utils import get_id2name, get_template, session_data4frame, print_parameters_info
from functools import partial
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--logging_dir', type=str, default="./logs/")
    parser.add_argument('--llm_path', type=str, default="/root/model/llama-3-8b-instruct")
    parser.add_argument('--ds_config_path', type=str, default="./deepspeed/ds_z2_config.json")
    parser.add_argument('--dataset', type=str, default="steam", choices=["steam", "goodreads", "movielens"])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--cutoff_len', type=int, default=1024)
    parser.add_argument('--eval_step', type=float, default=0.05)
    parser.add_argument('--save_step', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--local_rank', type=int, help='local rank')
    args = parser.parse_args()
    return args

def train(args):
    args.output_dir = op.join(args.output_dir, args.dataset, "sft")
    args.logging_dir = op.join(args.logging_dir, args.dataset, "sft")

    if args.dataset == "steam":
        padding_id = 3581
    elif args.dataset == "goodreads":
        padding_id = 4550
    elif args.dataset == "movielens":
        padding_id = 1682
    data_files = {
        "train": session_data4frame(op.join(args.data_dir,args.dataset,"train_data.df")),
        "val": session_data4frame(op.join(args.data_dir,args.dataset, "Val_data.df")),
    }
    

    train_data = Dataset.from_pandas(data_files["train"]).shuffle(seed=args.seed)
    val_data = Dataset.from_pandas(data_files["val"]).shuffle(seed=args.seed)

    llama_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.padding_side = "right"  
    llm_tokenizer.pad_token_id = (0)
        
    item_id2name = get_id2name(op.join(args.data_dir, args.dataset))
    instruction = get_template(op.join(args.data_dir,args.dataset,"instruction.txt"))
    input_template = get_template(op.join(args.data_dir,args.dataset,"input_template.txt"))
    llama_template = get_template("./llama_template/llama_template.txt")
    item_id2emb = torch.load(op.join(args.data_dir,args.dataset,"mpnet_embeddings_768.pt"))

    def negative_sampling(seq_unpad,next_item):
        canset=[i for i in list(item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, args.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates
    
    def formatting_func(examples):
        processed_examples = []
        for i in range(len(examples['seq_unpad'])):
            seq_unpad = examples['seq_unpad'][i]
            next_item = examples['next'][i]
            candidates = negative_sampling(seq_unpad, next_item)
            seq_name = [item_id2name[his] for his in seq_unpad]
            cans_name = [item_id2name[can] for can in candidates]
            input_prompt = input_template.replace("[HISTORY_SEQUENCE]", ", ".join([f"<{t}>" for i, t in enumerate(seq_name)])).replace("[CANDIDATE_ITEMS]", " ".join([f"{i+1}. <{t}>" for i, t in enumerate(cans_name)]))
        
            processed_example = {
                "instruction": instruction,
                "input": input_prompt,
                "output": str(candidates.index(next_item)+1)
            }
    
            new_example = llama_template.replace("{{INPUT}}",processed_example["instruction"]+"\n"+processed_example["input"]).replace("{{OUTPUT}}",processed_example["output"])
            processed_examples.append(new_example)
        return processed_examples
    
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(llm_tokenizer.encode(response_template, add_special_tokens = False), tokenizer=llm_tokenizer)

    training_args = SFTConfig(
        max_seq_length=args.cutoff_len,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=args.num_train_epochs, 
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_step,
        save_total_limit=100,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        logging_steps=1,
        output_dir=args.output_dir,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=None,
        logging_dir=args.logging_dir,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        save_only_model=True,
        deepspeed=args.ds_config_path,
    )

    print("Begin Training ... ")
    print_parameters_info(llama_model)

    trainer = SFTTrainer(
        llama_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        tokenizer=llm_tokenizer,
        formatting_func=formatting_func,
        args=training_args
    )

    trainer.train() 
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    args = parser()
    train(args)