import os
import os.path as op
import pandas as pd
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from trl import DPOTrainer, DPOConfig, CPOConfig, CPOTrainer
from dpo_trainer import RosePOTrainer
from accelerate import Accelerator
from utils import get_id2name, get_template, session_data4frame, print_parameters_info, set_seed
from rosepo_collator import DPODataCollatorWithPadding
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./output/")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--logging_dir', type=str, default="./logs/")
    parser.add_argument('--llm_path', type=str, default="/root/model/llama-3-8b-instruct")
    parser.add_argument('--ds_config_path', type=str, default="./deepspeed/ds_z2_config.json")
    parser.add_argument('--dataset', type=str, default="steam", choices=["steam", "goodreads", "movielens"])
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--cutoff_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--eval_step', type=float, default=0.05)
    parser.add_argument('--save_step', type=float, default=0.2)
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, help='local rank')
    args = parser.parse_args()
    return args


def train(args):
    args.output_dir = op.join(args.output_dir, args.dataset, "RosePO")
    args.logging_dir = op.join(args.logging_dir, args.dataset, "RosePO")


    if op.exists(os.path.join(args.data_dir, args.dataset, "predict", "train_data")):
        train_data = load_from_disk(os.path.join(args.data_dir, args.dataset, "predict", "train_data"))
        val_data = load_from_disk(os.path.join(args.data_dir, args.dataset, "predict", "val_data"))
    else:
        data_files = {
            "train": pd.read_pickle(os.path.join(args.data_dir, args.dataset, "predict", "train_predict.df")),
            "val": pd.read_pickle(os.path.join(args.data_dir, args.dataset, "predict", "Val_predict.df")),
        }

        item_id2name = get_id2name(op.join("./data/", args.dataset))
        instruction = get_template(f"./data/{args.dataset}/instruction.txt")
        input_template = get_template(f"./data/{args.dataset}/input_template.txt")
        llama_template_list = [get_template("./llama_template/input_template.txt"), get_template("./llama_template/output_template.txt")]

        item_name2id = {}
        for id, name in item_id2name.items():
            item_name2id[name] = id

        def process_data(example):
            seq_name = example['seq_unpad']
            seq_unpad = [item_name2id[name] for name in seq_name]
            next_item = item_name2id[example['chosen']]
            neg = item_name2id[example['rejected']]
            guided_prediction = example['guided_prediction']

            canset=[i for i in list(item_id2name.keys()) if i not in seq_unpad and i!=next_item]
            candidates=random.sample(canset, args.cans_num-2)+[next_item, neg]
            random.shuffle(candidates)
            cans_name = [item_id2name[can] for can in candidates]

            input_prompt = input_template.replace("[HISTORY_SEQUENCE]", ", ".join([f"<{t}>" for i, t in enumerate(seq_name)])).replace("[CANDIDATE_ITEMS]", " ".join([f"{i+1}. <{t}>" for i, t in enumerate(cans_name)]))

            prompt = llama_template_list[0].replace("{{INPUT}}",instruction+"\n"+input_prompt)
            chosen = llama_template_list[1].replace("{{OUTPUT}}",str(candidates.index(next_item)+1))
            rejected = llama_template_list[1].replace("{{OUTPUT}}",str(candidates.index(neg)+1))

            dic = {"prompt":prompt, "chosen":chosen, "rejected":rejected, "sasrec_w": guided_prediction[next_item], "sasrec_l": guided_prediction[neg]}
            return dic

        train_data = Dataset.from_pandas(data_files["train"])
        val_data = Dataset.from_pandas(data_files["val"])
        columns = train_data.column_names

        train_data = train_data.map(process_data, remove_columns=columns).shuffle(seed=args.seed)
        val_data = val_data.map(process_data, remove_columns=columns).shuffle(seed=args.seed)
        train_data.save_to_disk(os.path.join(args.data_dir, args.dataset, "predict", "train_data"))
        val_data.save_to_disk(os.path.join(args.data_dir, args.dataset, "predict", "val_data"))

        print("Tokenized dataset saved. Please restart.")
        exit(0)
        
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_tokenizer.padding_side = "right"  
    llm_tokenizer.pad_token_id = (0)

    base_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation = "flash_attention_2")
    ref_model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, use_cache=False, attn_implementation = "flash_attention_2")
    print("Begin Training ... ")
    print_parameters_info(base_model)

    datacollator = DPODataCollatorWithPadding(pad_token_id=llm_tokenizer.pad_token_id)

    training_args = DPOConfig(
        beta = args.beta,
        rpo_alpha=args.alpha,
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
        save_only_model=True,
        load_best_model_at_end=False,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        logging_steps=1,
        output_dir=args.output_dir,
        max_prompt_length=args.cutoff_len,
        max_length=args.cutoff_len,
        report_to = None,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        deepspeed=args.ds_config_path,
        remove_unused_columns = False
    )

    dpo_trainer = RosePOTrainer(
        model = base_model,
        ref_model = ref_model,
        args=training_args,
        data_collator = datacollator,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=llm_tokenizer,
    )

    dpo_trainer.use_dpo_data_collator = True
    dpo_trainer.train()

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    args = parser()
    set_seed(args.seed)
    train(args)