from typing import Text, List, Tuple, Optional

import torch
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)
from datasets import load_dataset

from inference_time_alignment.utils import StopOnStringCriteria


def create_model_and_tokenizer(
    model_name: Text,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    use_flash_attention_2: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        use_flash_attention_2=use_flash_attention_2,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_stopping_criteria(eos_strings: List[Text], tokenizer: PreTrainedTokenizer):
    if eos_strings == None:
        return None
    else:
        return StoppingCriteriaList([
            StopOnStringCriteria(
                start_length=0, 
                eos_string=eos_string, 
                tokenizer=tokenizer
            ) for eos_string in eos_strings
        ])

def remove_trailing_text(raw_responses: List[Text], eos_strings: List[Text], nonsensical_characters: List[Text]):
    if eos_strings == None:
        responses = raw_responses
    else:
        responses = []
        for raw_response in raw_responses:
            response = raw_response
            for eos_string in eos_strings:
                response = response.split(eos_string)[0]
            responses.append(response)
    # right strip common nonsensical characters
    if nonsensical_characters != None:
        for i in range(len(responses)):
            for nonsensical_character in nonsensical_characters:
                responses[i] = responses[i].rstrip(nonsensical_character)
    return responses


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
