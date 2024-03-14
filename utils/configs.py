from dataclasses import dataclass, field
from typing import Text, List, Callable, Optional


@dataclass
class ModelConfig:
    model_name: Text
    get_prompt_template: Callable

@dataclass
class EDConfig:
    base_model_config: ModelConfig
    chat_model_config: ModelConfig
    alpha: Optional[float] = 0.5
    eos_strings: Optional[List] = None
    nonsensical_characters: Optional[List] = None
    generation_configs: Optional[dict] = field(
        default_factory=lambda: dict(
            do_sample=True, max_new_tokens=64, temperature=0.7,
        )
    )

    def __post_init__(self):
        assert self.alpha >= 0, "alpha must be greater or equal than 0"


DEFAULT_BASE_PROMPT_TEMPLATE = ("""\
# Instruction

Below is a conversation between human and an AI assistant (you).
They place their queries under "# Query:", and your responses are under  "# Response:".
Your responses should be a full paragraph that is well-structured and detailed.

# Query:
{query}

# Response:
""")


EDRegistry = {

    "llama-2": EDConfig(
        base_model_config=ModelConfig(
            model_name='meta-llama/Llama-2-7b-hf',
            get_prompt_template=lambda tokenizer: DEFAULT_BASE_PROMPT_TEMPLATE,
        ),
        chat_model_config=ModelConfig(
            model_name='meta-llama/Llama-2-7b-chat-hf',
            get_prompt_template=lambda tokenizer: tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "{query}"}
                ], 
                tokenize=False,
                add_generation_prompt=True,
            ),
        ),
        alpha=0.3,
        eos_strings=["Query", "Response"],
        nonsensical_characters=[" ", "#", "\n"]
    ),


    "llama": EDConfig(
        base_model_config=ModelConfig(
            model_name='huggyllama/llama-7b',
            get_prompt_template=lambda tokenizer: DEFAULT_BASE_PROMPT_TEMPLATE,
        ),
        chat_model_config=ModelConfig(
            model_name='lmsys/vicuna-7b-v1.3',
            get_prompt_template=lambda tokenizer: (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                "USER: {query} ASSISTANT:"
            ),
        ),
        alpha=0.6,
        eos_strings=["Query", "Response"],
        nonsensical_characters=[" ", "#", "\n"]
    ),

    "mistral": EDConfig(
        base_model_config=ModelConfig(
            model_name='mistralai/Mistral-7B-v0.1',
            get_prompt_template=lambda tokenizer: DEFAULT_BASE_PROMPT_TEMPLATE,
        ),
        chat_model_config=ModelConfig(
            # system prompt to enforce guardrails from https://arxiv.org/pdf/2310.06825.pdf.
            model_name='mistralai/Mistral-7B-Instruct-v0.1',
            get_prompt_template=lambda tokenizer: tokenizer.apply_chat_template(
                [
                    {
                        "role": "user", 
                        "content": (
                            "Always assist with care, respect, and truth. Respond with utmost utility yet securely. "
                            "Avoid harmful, unethical, prejudiced, or negative content. "
                            "Ensure replies promote fairness and positivity. "
                            "{query}"
                        )
                    }
                ], 
                tokenize=False,
                add_generation_prompt=True,
            ),
        ),
        alpha=0.6,
        eos_strings=["Query", "Responses"],
        nonsensical_characters=[" ", "#", "\n"]
    ),

    "alpaca": EDConfig(
        base_model_config=ModelConfig(
            model_name='PKU-Alignment/alpaca-7b-reproduced',
            get_prompt_template=lambda tokenizer: "BEGINNING OF CONVERSATION: USER: {query} ASSISTANT:",
        ),
        chat_model_config=ModelConfig(
            model_name='PKU-Alignment/beaver-7b-v1.0',
            get_prompt_template=lambda tokenizer: "BEGINNING OF CONVERSATION: USER: {query} ASSISTANT:",
        ),
        alpha=2.0,
    ),
}