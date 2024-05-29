# Emulated Disalignment

Code release for [Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!](https://arxiv.org/abs/2402.12343).


This repo includes:
- A flexible and minimalist reference implementation of [Emulated Fine-Tuning (EFT)](https://arxiv.org/abs/2310.12962) / [proxy-tuning](https://arxiv.org/abs/2401.08565), which is modular and fully compatible with the [`transformers`](https://github.com/huggingface/transformers) API (e.g., `generate`).
- The official implementation of Emulated Disalignment (ED) and an [interactive demo](https://github.com/ZHZisZZ/emulated-disalignment/tree/main?tab=readme-ov-file#ED-interactive-demo) for experimenting with emulated disalignment on commonly-used open-source model pairs ([Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)x[Llama-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama](https://huggingface.co/huggyllama/llama-7b)x[Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.3), [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)x[Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [Alpaca](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)x[Beaver](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0)).


<details>
<summary>About ED</summary>
ED is a simple training-free method that reverse safety alignment, i.e., combining safety-aligned and pre-trained models during inference to generate harmful content. Please see paper for more details.
<p align="center">
    <img src="https://zhziszz.github.io/emulated-disalignment/static/images/illustration.png" width="93%"> <br>
</p>
</details>

## Installation

```bash
create -n emulated-disalignment python=3.10
conda activate emulated-disalignment
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
pip install -r requirements.txt
```


## EFT / proxy-tuning example
``EFTPosthocGenerationMixin`` simplifies parallel decoding and distribution composition for multiple language models. It's fully compatible with the `generate` API, requiring minimal changes to generation pipelines. Here is a simplified example using a 7b tuned and untuned model pair to steer a larger 13b untuned model:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
base_13b_model = ... # Llama-2-13b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1.0}

# logp_{eft} = logp_{base,13b} + 1.0 * (logp_{chat,7b} - logp_{base,7b})
eft_model = EFTPosthocGenerationMixin(
    base=base_13b_model,
    tune_r=chat_7b_model,
    base_r=base_7b_model,
    w=1.0,
)

# use transformer generate api as is
tokenized_output = eft_model.generate(**generation_configs) 
```
For a full working example, please refer to [scripts/examples/eft.py](https://github.com/ZHZisZZ/emulated-disalignment/tree/main/scripts/examples/eft.py).


## ED example
Our ED implementation is based on our EFT implementation. Here is a simplified example combining a pre-trained base model and a safety-aligned chat model to produce harmful responses:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1.0}
alpha = 0.3

# logp_{ed} = logp_{base,7b} + (-alpha) * (logp_{chat,7b} - logp_{base,7b}) 
#           = (1+alpha) * logp_{base,7b} - alpha * logp_{chat,7b}
ed_model = EFTPosthocGenerationMixin(
    base=base_7b_model,
    tune_r=chat_7b_model,
    w=-alpha,            # negative coefficient to reverse fine-tuning direction
)

# use transformer generate api as is
tokenized_output = ed_model.generate(**generation_configs) 
```
For a full working example, please refer to [scripts/examples/ed.py](https://github.com/ZHZisZZ/emulated-disalignment/tree/main/scripts/examples/ed.py).


## ED interactive demo
Run `python ed_demo.py` to experiment with emulated disalignment. 
```bash
usage: ed_demo.py [-h] [--family-name STR] [--dataset-name STR] [--evaluator-name STR] [--num-responses-per-query INT]
                  [--seed INT]

╭─ arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                                │
│ --family-name STR       currently support `llama-2`, `llama`, `mistral` and `alpaca` (default: llama-2)                │
│ --dataset-name STR      currently support `Anthropic/hh-rlhf`, `lmsys/toxic-chat`,                                     │
│                         `mmathys/openai-moderation-api-evaluation`, `PKU-Alignment/BeaverTails` (default:              │
│                         PKU-Alignment/BeaverTails)                                                                     │
│ --evaluator-name STR    currently support `llama-guard`, `openai-moderation` (default: llama-guard)                    │
│ --num-responses-per-query INT                                                                                          │
│                         number of responses for each query (default: 3)                                                │
│ --seed INT              (default: 0)                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Formulate the most harmful query possible to evaluate the **ed** model's response compared to the **base** model. Alternatively, press enter to automatically generate a sophisticated malicious query for comparison.


## Reference

```
@article{zhou2024emulated,
  title={Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!},
  author={Zhou, Zhanhui and Liu, Jie and Dong, Zhichen and Liu, Jiaheng and Yang, Chao and Ouyang, Wanli and Qiao, Yu},
  journal={arXiv preprint arXiv:2402.12343},
  year={2024}
}
```