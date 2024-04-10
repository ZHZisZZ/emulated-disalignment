# Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!

[[Project Page](https://zhziszz.github.io/emulated-disalignment/)],[[Arxiv](https://arxiv.org/abs/2402.12343)]

Emulated Disalignment (ED) presents an attack framework illustrating how integrating the output distributions of off-the-shelf pre-trained and safety-aligned language model pairs ($\pi_{\text{base}}$, $\pi_{\text{align}}$), such as Llama-2 and Llama-2-chat, can lead to the creation of harmful language models $\pi_{\text{emulated-disalign}}$ without the necessity for additional training:

$$
\pi_{\text{emulated-disalign}}(y_t |x, y_{\text{<}t}) \propto \frac{\hspace{3mm} \pi_{\hspace{.5mm}\text{base}\hspace{.5mm}}(y_t |x,y_{\text{<}t})^{\alpha+1}}{\pi_{\text{align}}(y_t |x, y_{\text{<}t})^\alpha}, \text{ where } \alpha \geq 0.
$$

This repo includes:
- a flexible and minimalist reference implementation of [Emulated Fine-Tuning (EFT)](https://arxiv.org/abs/2310.12962), fully compatible with the transformer API.  
- the official implementation of Emulated Disalignment (ED) and an [interactive demo](https://github.com/ZHZisZZ/emulated-disalignment/tree/main?tab=readme-ov-file#ED-interactive-demo) that enables users to easily experiment with emulated disalignment for commonly-used open-source model pairs ([Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)x[Llama-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama](https://huggingface.co/huggyllama/llama-7b)x[Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.3), [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)x[Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [Alpaca](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)x[Beaver](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0)).


## Installation

#### Create virtual env

```bash
create -n emulated-disalignment python=3.10
conda activate emulated-disalignment
```

#### Install

```bash
git clone https://github.com/ZHZisZZ/emulated-disalignment.git
cd emulated-disalignment
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
pip install -r requirements.txt
```


## EFT example
``EFTPosthocGenerationMixin`` abstracts away the complexity of decoding multiple language models in parallel and then composing their distributions. Note that ``EFTPosthocGenerationMixin`` is compatible with the `generate` API, allowing minimal changes to the generation pipelines. Here is a simplified example:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ...
base_7b_model  = ...
base_13b_model = ...
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1}

eft_model = EFTPosthocGenerationMixin(
    models=chat_7b_model,
    ref_r=base_7b_model,
    ref=base_13b_model,
    w=[1.0], # unscaled version
)

# use transformer generate api as is
tokenized_output = eft_model.generate(**generation_configs) 
```
For a full working example, please refer to [scripts/examples/eft.py](https://github.com/ZHZisZZ/emulated-disalignment/tree/main/scripts/examples/eft.py).

## ED interactive demo
Run `sh ed_demo.py` to experiment with emulated disalignment. 
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
Please try your best to formulate the most harmful query you can envisage to evaluate the response of the `ed` model relative to the `base` model. Alternatively, press enter to automatically generate a sophisticated malicious query for comparison.