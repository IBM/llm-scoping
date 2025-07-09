# Llm-scoping
This repo includes source code to reproduce llm scoping results on our paper:
[Reducing the Scope of Language Models](https://arxiv.org/abs/2410.21597)
we builds below scripts for different types of llm-scoping training/ few-shot prompting methods:
1. CB/ SFT2CB : circuit_breakers_hf_tok.py 
2. SFT : sft_hf_tok.py
3. DPO : dpo.py
4. probe : probe.py
5. simple classifier and head : roberta_probe.py
6. llm judge by few-shot two side examples prompting : llm_judge_twoside_fewshot.py

A few configs example: `configs/modelname`
Cite our work:
```
@misc{2410.21597,
Author = {David Yunis and Siyu Huo and Chulaka Gunasekara and Danish Contractor},
Title = {Reducing the Scope of Language Models},
Year = {2024},
Eprint = {arXiv:2410.21597},
}

```
Contact us:
```
dyunis@ttic.edu
siyu.huo@ibm.com
chulaka.gunasekara@ibm.com
danish.contractor@ibm.com
```

## Installation

```bash
#!/usr/bash
# conda now do-not-use but miniforge allowed
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

conda install python=3.11  # recommended for mps support
pip install numpy

# pytorch
# mac
pip install torch --index-url https://download.pytorch.org/whl/cpu
# linux
pip install torch --index-url https://download.pytorch.org/whl/cu118
# for later cuda
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# huggingface utilities
pip install transformers
pip install datasets
pip install accelerate  # for simple multi-gpu inference (device_map='auto')
pip install peft  # for lora, apache 2

# mistral tokenizer
pip install mistral-common  # apache 2

# metrics for diverse data generation
pip install nltk  # bleu, meteor, license: apache 2
python -c 'import nltk; nltk.download("wordnet")' # approved in catalog
pip install rouge-score  # rougel score, apache 2

# plotting
pip install matplotlib

# fastchat dependency for TAP prompts
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,webui]"
cd

# for DPO training
pip install trl
```

## Running the code

Before doing anything you need to set the following environment variables:
```bash
export TRANSFORMERS_CACHE=[path to cache hf tokenizers/models]
export HF_DATASETS_CACHE=[path to cache hf datasets]
export STEERING_SCRATCH_DIR=[path for miscellaneous saving]
export HF_TOKEN=[huggingface token for model access]
export NLTK_DATA="~/nltk_data"  # for storing wordnet from nltk
export GENAI_KEY=[BAM API key]
```

Training, see arguments with `python cb.py -h`:

```bash
python [circuit_breakers_hf_tok|sft_hf_tok|dpo|probe|roberta_probe].py --model 'granite' --savedir [path/to/save] --lora_init [previous lora to init from, for example for layer CB on top of SFT] --accept_dsets sni_sa,sni_tld --reject_dsets sni_s,sni_tc,sni_sc,sni_dg --num_prompts_per_dset 2048 --system_prompt --num_steps 128
```

Datasets can be chosen from `sni_sa, sni_tld, sni_s, sni_tc, sni_sc, sni_dg, sni_pe, sni_qa, gsm8k, alpaca`
where the SNI abbreviations mean `sentiment analysis, toxic language detection, summarization, text completion, story composition, dialogue generation, program execution` and `question answering` respectively
This creates a directory at [path/to/save] with two files:
- `config.json` - the config that we trained with
- `lora.pt` - the final lora that gets saved

You can also specify data as a `.json` file that has the following structure:
```bash
{
"pairs": [
["instruction 1", "completion 1"],
["instruction 2", "completion 2"],
...
]
}
```
and provide the path to the json following `--accept_dsets` or `--reject_dsets`

For Evaluation, see arguments with `python eval_*.py -h`:
```bash
python [eval_|reject_perf|probe_reject|robertaprobe_reject.py] --evaldir [path to savedir of trained run] --system_prompt --prompt_style [for adversarial prompting methods] --dataset [dataset/to/eval] --num_prompts 256 --regenerate
```
This creates the following under [path/to/save]:
- `eval/[dataset/to/eval]/[orig|lora]*.json`
where `orig*.json` contains results for the base model, and `lora*.json`
the results for the tuned lora

In addition with few shot llm judge:
```bash
python llm_judge_twoside_fewshot.py --model "model_name" --dataset "dataset"
```

Cosine similarity plots, see arguments with `python rep_cosines.py -h`
```bash
python rep_cosines.py --savedir [path to savedir of trained run] --system_prompt --dataset sni_s --num_prompts 256 --evaldirs [space separated list of savedirs from training script]
```
This saves plots to [savedir]




