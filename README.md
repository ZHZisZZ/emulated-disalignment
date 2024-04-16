# Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!

[[Arxiv](https://arxiv.org/abs/2402.12343)]

Emulated Disalignment (ED) presents a **training-free** attack framework capable of **reversing safety alignment** for language models (i.e., converting the outcomes of stronger alignment into greater potential for harm). Specifically, ED achieves this reversal by contrasting the output token distribution of a safety-aligned language model $\pi_{\text{align}}(y|x)$ (e.g., Llama-2-chat) against its pre-trained version $\pi_{\text{base}}$(y|x) (e.g., Llama-2), so that the token predictions are shifted towards the opposite direction of safety alignment:

$$
\pi_{\text{emulated-disalign}}(y_t |x, y_{\text{<}t}) \propto \frac{\hspace{3mm} \pi_{\hspace{.5mm}\text{base}\hspace{.5mm}}(y_t |x,y_{\text{<}t})^{\alpha+1}}{\pi_{\text{align}}(y_t |x, y_{\text{<}t})^\alpha}, \text{ where } \alpha \geq 0.
$$

This sampling distribution provably emulates the result of fine-tuing the pre-trained model $\pi_{\text{base}}(y|x)$ to minize a safety reward $r_{\text{align}}(x,y) = \log \pi_{\text{align}}(y|x) - \log \pi_{\text{base}}(y|x)$ and effectively produce harmful contents. 

## What is this repo?

This repo includes:
- a flexible and minimalist reference implementation of [Emulated Fine-Tuning (EFT)](https://arxiv.org/abs/2310.12962) / [proxy-tuning](https://arxiv.org/abs/2401.08565), which is modular and fully compatible with the [`transformers`](https://github.com/huggingface/transformers) API (e.g., `generate`).
- the official implementation of Emulated Disalignment (ED) and an [interactive demo](https://github.com/ZHZisZZ/emulated-disalignment/tree/main?tab=readme-ov-file#ED-interactive-demo) that enables users to easily experiment with emulated disalignment for commonly-used open-source model pairs ([Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)x[Llama-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama](https://huggingface.co/huggyllama/llama-7b)x[Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.3), [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)x[Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [Alpaca](https://huggingface.co/PKU-Alignment/alpaca-7b-reproduced)x[Beaver](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0)).


## Installation

#### Create virtual env

```bash
create -n emulated-disalignment python=3.10
conda activate emulated-disalignment
```

#### Install dependencies

```bash
git clone https://github.com/ZHZisZZ/emulated-disalignment.git
cd emulated-disalignment
pip install torch=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
pip install -r requirements.txt
```


## EFT / proxy-tuning example
``EFTPosthocGenerationMixin`` abstracts away the complexity of decoding multiple language models in parallel and then composing their distributions. Note that ``EFTPosthocGenerationMixin`` is compatible with the `generate` API, allowing minimal changes to the generation pipelines. Here is a simplified example that uses a 7b tuned and untuned model pair to steer a larger 13b untuned model:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
base_13b_model = ... # Llama-2-13b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1}

# logp_{eft} = logp_{base}     + w   * (logp_{tune_r}  - logp_{base_r})
#            = logp_{base,13b} + 1.0 * (logp_{chat,7b} - logp_{base,7b})
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
Our ED implementation builds upon our EFT implemnetation. To adapt EFT for ED, we only need to make sure that `ref_r = ref` (which is true by default) and use negative`w` (as `alpha` in ED equal to negated `w` in EFT). Here is a simplified example that combines a pre-trained base model and a safety-aligned chat model to produce harmful responses:
```python
from inference_time_alignment.decoder import EFTPosthocGenerationMixin

chat_7b_model  = ... # Llama-2-7b-chat
base_7b_model  = ... # Llama-2-7b
generation_configs = {"do_sample":True, "max_new_tokens":512, "temperature":1}
alpha = 0.3

# logp_{ed} = logp_{base}    + (-alpha) * (logp_{tune_r}  - logp_{base_r})
#           = logp_{base,7b} + (-alpha) * (logp_{chat,7b} - logp_{base,7b}) 
#           = (1+alpha) * logp_{base,7b} - alpha * logp_{chat,7b}
ed_model = EFTPosthocGenerationMixin(
    base=base_7b_model, # base_r = base by default when base_r is not specified
    tune_r=chat_7b_model,
    w=-alpha,
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
Please try your best to formulate the most harmful query  to evaluate the response of the **ed** model relative to the **base** model. Alternatively, press `enter` to automatically generate a sophisticated malicious query for comparison.