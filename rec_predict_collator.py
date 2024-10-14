import random
import warnings
import numpy as np
from transformers import DataCollatorForLanguageModeling
import torch
    
class RecPredCollator:
    def __init__(
        self,
        item_id2name,
        input_template = None,
        instruction = None,
        cans_num=20,
        llama_template_list=None,
        llm_tokenizer = None,
        device = None,
        item_id2emb = None
    ):

        self.item_id2name = item_id2name
        self.input_template = input_template
        self.instruction = instruction
        self.cans_num = cans_num
        self.seq_sep = ", "
        self.can_sep = " "
        self.llama_template_list = llama_template_list
        self.llm_tokenizer = llm_tokenizer
        self.item_id2emb = item_id2emb/torch.norm(item_id2emb,dim=-1).unsqueeze(1)
        self.device = device

    def negative_sampling(self, example):
        seq_unpad = example['seq_unpad']
        next_item = example['next']
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    def __call__(self, batch):
        inputs_text, inputs_pair, targets_text, seq_txt, cans_txt, next_txt, guided_predictions = [], [], [], [], [], [], []
        for i, example in enumerate(batch):
            candidates = self.negative_sampling(example)
            if candidates is None:
                continue
            seq_name = [self.item_id2name[his] for his in example['seq_unpad']]
            cans_name = [self.item_id2name[can] for can in candidates]
            input_prompt = self.input_template.replace("[HISTORY_SEQUENCE]", self.seq_sep.join([f"<{t}>" for i, t in enumerate(seq_name)])).replace("[CANDIDATE_ITEMS]", self.can_sep.join([f"{i+1}. <{t}>" for i, t in enumerate(cans_name)]))

            processed_example = {
                "instruction": self.instruction,
                "input": input_prompt,
                "output": str(candidates.index(example['next'])+1)
            }
    
            input_txt = self.llama_template_list[0].replace("{{INPUT}}",processed_example["instruction"]+"\n"+processed_example["input"])
            output_txt = self.llama_template_list[1].replace("{{OUTPUT}}",processed_example["output"])

            inputs_text.append(input_txt)
            targets_text.append(processed_example["output"])
            inputs_pair.append([input_txt, output_txt])
            seq_txt.append(seq_name)
            cans_txt.append(cans_name)
            next_txt.append(self.item_id2name[example['next']])
            guided_predictions.append(example["guided_prediction"])

        test_batch_tokens = self.llm_tokenizer(
            inputs_text,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=True).to(self.device)
        eval_batch_tokens = self.llm_tokenizer(
            inputs_pair,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=True).to(self.device)
        new_batch={"test_tokens":test_batch_tokens,
                   "eval_tokens":eval_batch_tokens,
                   "correct_answer": targets_text,
                   "seq_name": seq_txt,
                   "cans_name": cans_txt,
                   "next_name": next_txt,
                   "guided_prediction": guided_predictions
                   }

        return new_batch