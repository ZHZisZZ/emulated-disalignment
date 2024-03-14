from dataclasses import dataclass
from collections.abc import Mapping
from typing import Text, List

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList


@Accelerator().on_local_main_process
def print_local_main(text):
    print(text)


def disable_progress_bar_non_local_main():
    if not Accelerator().is_local_main_process:
        import datasets
        import transformers
        import warnings
        datasets.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')


def prepare_input(data):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2626
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": Accelerator().device}
        # TODO: inference-time deepspeed?
        # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
        #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update({"dtype": Accelerator().state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data


@dataclass
class StopOnStringCriteria(StoppingCriteria):
    start_length: int
    eos_string: Text
    tokenizer: PreTrainedTokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        return all(self.eos_string in decoded_generation for decoded_generation in decoded_generations) # Stop when ALL sequences hit the stopping critera


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
