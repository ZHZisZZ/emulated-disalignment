from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional

import torch
from transformers import PreTrainedModel, GenerationMixin


@dataclass
class PosthocGenerationMixin(GenerationMixin, ABC):
    models: List[PreTrainedModel]
    w: List[float]

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def _parallel_decode(self, *args, past_key_values_list, **kwargs):
        models_outputs_list = []
        for i, model in enumerate(self.models):
            outputs = model(*args, past_key_values=past_key_values_list[i], **kwargs)
            models_outputs_list.append(outputs)
        return models_outputs_list

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.models[0], name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        if 'past_key_values' in model_kwargs:
            past_key_values = model_kwargs['past_key_values']
            model_kwargs['past_key_values'] = model_kwargs['past_key_values']['models'][0]
        result = self.models[0].prepare_inputs_for_generation(input_ids, **model_kwargs)
        if 'past_key_values' in model_kwargs:
            result['past_key_values'] = past_key_values
        return result


@dataclass
class EFTPosthocGenerationMixin(PosthocGenerationMixin):
    ref: PreTrainedModel
    ref_r: Optional[PreTrainedModel] = None 

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'models': [None] * len(self.models), 'ref': None, 'ref_r': None}

        ref_outputs = self.ref(*args, past_key_values=past_key_values['ref'], **kwargs)
        ref_logits  = ref_outputs.logits[:, -1, :]

        if not self.ref_r:
            ref_r_outputs = ref_outputs
            ref_r_logits = ref_logits
        else:
            ref_r_outputs = self.ref_r(*args, past_key_values=past_key_values['ref_r'], **kwargs)
            ref_r_logits  = ref_r_outputs.logits[:, -1, :]

        models_outputs_list = self._parallel_decode(*args, past_key_values_list=past_key_values['models'], **kwargs)
        models_logits_list  = [outputs.logits[:, -1, :] for outputs in models_outputs_list]

        models_logits = torch.stack(models_logits_list, dim=-1)
        w = torch.tensor(self.w).to(models_logits.device)
        r = (w * (models_logits - ref_r_logits.unsqueeze(-1))).sum(-1)

        logits = ref_logits + r

        outputs = ref_outputs
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {
            'models': [outputs.past_key_values for outputs in models_outputs_list],
            'ref': ref_outputs.past_key_values,
            'ref_r': ref_r_outputs.past_key_values,
        }
        return outputs


@dataclass
class PoEPosthocGenerationMixin(PosthocGenerationMixin):

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values: past_key_values = {'models': [None] * len(self.models)}

        models_outputs_list = self._parallel_decode(*args, past_key_values_list=past_key_values['models'], **kwargs)
        models_logits_list  = [outputs.logits[:, -1, :] for outputs in models_outputs_list]

        model_logits = torch.stack(models_logits_list, dim=-1)
        w = torch.tensor(self.w).to(model_logits.device)

        logits = (w * model_logits).sum(-1)

        outputs = models_logits_list[0]
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {'models': [outputs.past_key_values for outputs in models_outputs_list]}
        return outputs

