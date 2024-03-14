# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python scripts/examples/eft.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference_time_alignment.decoder import EFTPosthocGenerationMixin
from inference_time_alignment.model import PrefixPreTrainedWrapper
from inference_time_alignment.utils import set_seeds, get_stopping_criteria, remove_trailing_text

set_seeds(1)
use_flash_attention_2 = True
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1}


base_7b_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=use_flash_attention_2
)
chat_7b_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=use_flash_attention_2
)
base_13b_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=use_flash_attention_2
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

base_prompt_template = ("""\
# Instruction

Below is a conversation between human and an AI assistant (you).
They place their queries under "# Query:", and your responses are under  "# Response:".
Your responses should be a full paragraph that is well-structured and detailed.

# Query:
{query}

# Response:
""")
chat_prompt_template = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "{query}"}
    ], 
    tokenize=False,
    add_generation_prompt=True,
)

query = "How to beat superhuman GO AIs?"
base_templated_content = base_prompt_template.format(query=query)
chat_templated_content = chat_prompt_template.format(query=query)

# chat and base have different templates
chat_7b_model  = PrefixPreTrainedWrapper(chat_7b_model, tokenizer, chat_templated_content)
base_7b_model  = PrefixPreTrainedWrapper(base_7b_model, tokenizer, base_templated_content) 
base_13b_model = PrefixPreTrainedWrapper(base_13b_model, tokenizer, base_templated_content)

eft_model = EFTPosthocGenerationMixin(
    models=chat_7b_model,
    ref_r=base_7b_model,
    ref=base_13b_model,
    w=[1.0], # unscaled version
)

eos_strings = ["Query", "Response"]
nonsensical_characters = [" ", "#", "\n"]

tokenized_output = eft_model.generate(
    **generation_configs, 
    stopping_criteria=get_stopping_criteria(eos_strings=eos_strings, tokenizer=tokenizer), 
    suppress_tokens=[tokenizer.eos_token_id]
)

response = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)
response = remove_trailing_text(response, eos_strings=eos_strings, nonsensical_characters=nonsensical_characters)