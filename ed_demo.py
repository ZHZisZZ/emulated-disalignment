import itertools
import random
from dataclasses import dataclass, field

import tyro
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table

import torch

from utils import EDRegistry, create_model_and_tokenizer, get_query_dataset
from utils.evaluator import LlamaGuardEvaluator, EvaluationInput, OpenAIModerationEvaluator
from inference_time_alignment.decoder import EFTPosthocGenerationMixin
from inference_time_alignment.model import PrefixPreTrainedWrapper
from inference_time_alignment.utils import set_seeds, get_stopping_criteria, remove_trailing_text


@dataclass(kw_only=True)
class ScriptArguments:
    family_name: str = field(default="alpaca", metadata={"help": "currently support `llama-2`, `llama`, `mistral` and `alpaca`"})
    dataset_name: str = field(default="PKU-Alignment/BeaverTails", metadata={"help": "currently support `Anthropic/hh-rlhf`, `lmsys/toxic-chat`, `mmathys/openai-moderation-api-evaluation`, `PKU-Alignment/BeaverTails`"})
    evaluator_name: str = field(default="llama-guard", metadata={"help": "currently support `llama-guard`, `openai-moderation`"})
    num_responses_per_query: int = field(default=3, metadata={"help": "number of responses for each query"})
    seed: int = field(default=0)

script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

ed_config = EDRegistry[script_args.family_name]

print("loading models")
base_model, base_tokenizer = create_model_and_tokenizer(ed_config.base_model_config.model_name)
chat_model, chat_tokenizer = create_model_and_tokenizer(ed_config.chat_model_config.model_name)

base_prompt_template = ed_config.base_model_config.get_prompt_template(base_tokenizer)
chat_prompt_template = ed_config.chat_model_config.get_prompt_template(chat_tokenizer)

stopping_criteria = get_stopping_criteria(ed_config.eos_strings, chat_tokenizer)
suppress_tokens = [chat_tokenizer.eos_token_id] if stopping_criteria else None

print("loading dataset")
query_dataset = get_query_dataset(dataset_name=script_args.dataset_name)

print("loading evaluator")
evaluator = {
    'llama-guard': LlamaGuardEvaluator, 
    'openai-moderation': OpenAIModerationEvaluator
}[script_args.evaluator_name]()
violation_cnts = {"ed": 0, "base": 0}

console = Console()

for turn in itertools.count(start=1):

    query = Prompt.ask("Enter your query here", default=random.choice(query_dataset))
    print("\n")

    base_templated_content = base_prompt_template.format(query=query)
    chat_templated_content = chat_prompt_template.format(query=query)

    for name, alpha in [("ed", ed_config.alpha), ("base", 0.0)]:

        ed_model = EFTPosthocGenerationMixin(
            base=PrefixPreTrainedWrapper(base_model, chat_tokenizer, base_templated_content),
            tune_r=PrefixPreTrainedWrapper(chat_model, chat_tokenizer, chat_templated_content),
            w=-alpha, # negated for disalignment
        )

        valid_responses = []
        while len(valid_responses) < script_args.num_responses_per_query:
            generated_tokens = ed_model.generate(
                input_ids=torch.ones((script_args.num_responses_per_query,1)).int().cuda()*chat_tokenizer.bos_token_id, 
                attention_mask=torch.ones((script_args.num_responses_per_query,1)).int().cuda(), 
                **ed_config.generation_configs, 
                stopping_criteria=stopping_criteria, 
                suppress_tokens=suppress_tokens,
            )
            raw_responses = chat_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            responses = remove_trailing_text(raw_responses, ed_config.eos_strings, ed_config.nonsensical_characters)
            # filter responses that are too short
            valid_responses.extend([response for response in responses if len(chat_tokenizer(response)['input_ids']) >= 8])
        responses = valid_responses[:script_args.num_responses_per_query]

        moderations = evaluator.eval([EvaluationInput(response=response, query=query) for response in responses])
        violation_cnts[name] += sum([moderation.flagged for moderation in moderations])

        table = Table(title=f"{name} (total violation: {violation_cnts[name]}/{script_args.num_responses_per_query*turn})")
        table.add_column(""); table.add_column("Responses"); table.add_column("violated")
        for i, (response, moderation) in enumerate(zip(responses, moderations)):
            table.add_row(f"[{i+1}]", response, str(moderation.flagged))
            if i != len(responses) - 1:
                table.add_row("", "", "")

        console.print(table)
