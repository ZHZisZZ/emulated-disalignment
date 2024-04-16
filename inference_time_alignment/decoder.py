from dataclasses import dataclass
from typing import List, Any, Optional

import torch
from transformers import PreTrainedModel, GenerationMixin


@dataclass(kw_only=True)
class EFTPosthocGenerationMixin(GenerationMixin):
    """
    args: 
        `base`: 
            the base language model to be steered at inference time.
        `tune_r`, `base_r`: 
            `tune_r` are a list of N language models fine-tuned from `ref_r`, \
            Each model from `tune_r` can be combined with `ref_r` to define a implicit reward models r_i = logp_{tune_r[i]} - logp_{base_r}.
        `w`: 
            the linear weights for combining the implicit reward models r = sum_{i=1}^N r_i.

    The resulting sampling distribution is proportioal to 
          logp_{base} * r
        = logp_{base} * sum_{i=1}^{N}(w_i*r_i), 
    where r_i = logp_{tune_r[i]} - logp_{base_r}.
    """
    base:     PreTrainedModel
    tune_r:   List[PreTrainedModel] | PreTrainedModel
    base_r:   Optional[PreTrainedModel] = None
    w:        List[float] | float

    def __post_init__(self):
        if not isinstance(self.base, list):  self.tune_r = [self.tune_r]
        if not isinstance(self.w, list): self.w = [self.w]
        if not self.base_r: self.base_r = self.base
        assert len(self.tune_r) == len(self.w) 

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        if 'past_key_values' in model_kwargs:
            past_key_values = model_kwargs['past_key_values']
            model_kwargs['past_key_values'] = model_kwargs['past_key_values']['base']
        result = self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if 'past_key_values' in model_kwargs:
            result['past_key_values'] = past_key_values
        return result

    def _parallel_decode(self, *args, past_key_values_list, **kwargs):
        models_outputs_list = []
        for i, model in enumerate(self.tune_r):
            outputs = model(*args, past_key_values=past_key_values_list[i], **kwargs)
            models_outputs_list.append(outputs)
        return models_outputs_list

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'base': None, 'tune_r': [None] * len(self.tune_r), 'base_r': None}

        base_outputs = self.base(*args, past_key_values=past_key_values['base'], **kwargs)
        base_logits  = base_outputs.logits[:, -1, :]

        if self.base == self.base_r:
            base_r_outputs = base_outputs
            base_r_logits  = base_logits
        else:
            base_r_outputs = self.base_r(*args, past_key_values=past_key_values['base_r'], **kwargs)
            base_r_logits  = base_r_outputs.logits[:, -1, :]

        tune_r_outputs_list = self._parallel_decode(*args, past_key_values_list=past_key_values['tune_r'], **kwargs)
        tune_r_logits_list  = [outputs.logits[:, -1, :] for outputs in tune_r_outputs_list]

        tune_r_logits = torch.stack(tune_r_logits_list, dim=-1)
        w = torch.tensor(self.w).to(tune_r_logits.device)
        r = (w * (tune_r_logits - base_r_logits.unsqueeze(-1))).sum(-1)

        logits = base_logits + r

        outputs = base_outputs
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {
            'base':   base_outputs.past_key_values,
            'tune_r': [outputs.past_key_values for outputs in tune_r_outputs_list],
            'base_r': base_r_outputs.past_key_values,
        }
        return outputs
