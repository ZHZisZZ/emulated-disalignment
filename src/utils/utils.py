from typing import Text, List, Tuple, Optional

import torch
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from datasets import load_dataset


def create_model_and_tokenizer(
    model_name: Text,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    use_flash_attention_2: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=load_in_4bit),
        attn_implementation="flash_attention_2" if use_flash_attention_2 else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_query_dataset(dataset_name: Optional[Text] = "Anthropic/hh-rlhf") -> List:
    if dataset_name == "PKU-Alignment/BeaverTails":
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        return dataset["prompt"][:200]
    elif dataset_name == "mmathys/openai-moderation-api-evaluation":
        dataset = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
        dataset = dataset.filter(lambda example: len(example["prompt"]) < 200) # filter prompts that are too long
        return dataset.filter(lambda example: any(v == 1 for v in example.values()))["prompt"][:200]
    elif dataset_name == "lmsys/toxic-chat":
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        dataset = dataset.filter(lambda example: len(example["user_input"]) < 200) # filter prompts that are too long
        return dataset.filter(lambda example: example["toxicity"] == 1)["user_input"][:200]
    elif dataset_name == "Anthropic/hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
        return dataset.map(
            lambda example: {"prompt": example["chosen"].split("\n\nAssistant")[0].split("\n\nHuman: ")[1]}, 
            remove_columns=dataset.column_names
        )["prompt"][:200]
    else:
        raise NotImplementedError
