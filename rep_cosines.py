import argparse
import json
import os
import pickle
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import data


MODEL_CACHE_DIR = os.environ['TRANSFORMERS_CACHE']
DATASET_CACHE_DIR = os.environ['HF_DATASETS_CACHE']
SCRATCH_DIR = os.environ['STEERING_SCRATCH_DIR']
ACCESS_TOKEN = os.environ['HF_TOKEN']

# cosine similarity between lora and base model at different positions and layers
# averaged over entire dataset
# - pass prompt thru dataset
# - get cosine similarity of tail (to minimum length in dataset)
@torch.no_grad()
def main(config):
    # seed everything before generation
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    tokenizer = AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        token=ACCESS_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
        revision='pr/120'  # fixes tokenizer, see https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/141
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    tas = []
    for evaldir in config.evaldirs:
        config_path = os.path.join(evaldir, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        train_accept_sets = config_dict['accept_dsets']
        tas.append(train_accept_sets)
    assert len(list(set(tas))) == 1
    train_accept_sets = tas[0]
    train_accept_sets = train_accept_sets.split(',')
    if config_dict['system_prompt']:
        system_prompt = data.dsets_to_system_prompt(train_accept_sets)

    def format_prompts_fn(_prompts):
        if isinstance(_prompts[0], tuple):
            _prompts = [it[0] for it in _prompts]
        return data.format_prompts(
            _prompts,
            model_name='mistral',
            system_prompt=system_prompt,
            adversarial_prompt=None,
            prefill=False,
            b64=False,
        )

    val = data.get_data_from_names(
        [config.dataset],
        split='val',
        include_outputs=True,
        num_prompts=config.num_prompts,
        max_length=1024,
        tokenizer=tokenizer,
        format_prompts_fn=format_prompts_fn,
    )
    prompts = [it[0] for it in val]
    formatted_prompts = format_prompts_fn(prompts)

    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        low_cpu_mem_usage=True,
        device_map='cuda',  # handles placement onto gpu/multiple gpus
        torch_dtype=torch.float32,
        token=ACCESS_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()

    bsz = 4
    min_prompt_length = 50
    orig_hidden = collect_hidden_states(formatted_prompts, tokenizer, model, batch_size=bsz)
    orig_hidden = torch.stack([h[:, -min_prompt_length:, :] for h in orig_hidden], dim=0)
    for evaldir in config.evaldirs:
        model = AutoModelForCausalLM.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            low_cpu_mem_usage=True,
            device_map='cuda',  # handles placement onto gpu/multiple gpus
            torch_dtype=torch.float32,
            token=ACCESS_TOKEN,
            cache_dir=MODEL_CACHE_DIR,
        )
        model.eval()

        config_path = os.path.join(evaldir, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model = load_lora(evaldir, model, config_dict)

        hidden = collect_hidden_states(formatted_prompts, tokenizer, model, batch_size=bsz)
        hidden = torch.stack([h[:, -min_prompt_length:, :] for h in hidden], dim=0)

        # compute cosines to orig hiddens for every tail position
        inner_prod = (hidden * orig_hidden).sum(dim=-1)
        cos_sim = inner_prod / hidden.norm(dim=-1) / orig_hidden.norm(dim=-1)
        meancos = cos_sim.mean(dim=0).numpy()
        stdcos = cos_sim.std(dim=0).numpy()

        method = config_dict["method"]
        if 'sft2cb' in evaldir:
            method = 'sft2cb'

        fig = make_line_fig(meancos, stdcos)
        savepath = os.path.join(config.savedir, f'{method}_{config.dataset}_line.png')
        pickle_savepath = os.path.splitext(savepath)[0] + '.pkl'
        with open(pickle_savepath, 'wb') as f:
            pickle.dump(fig, f)
        fig.savefig(savepath)
        plt.close(fig)

        fig = make_heatmap_fig(meancos)
        savepath = os.path.join(config.savedir, f'{method}_{config.dataset}_heatmap.png')
        pickle_savepath = os.path.splitext(savepath)[0] + '.pkl'
        with open(pickle_savepath, 'wb') as f:
            pickle.dump(fig, f)
        fig.savefig(savepath)
        plt.close(fig)


def make_line_fig(mean, std):
    assert len(mean.shape) == 2
    assert mean.shape == std.shape
    fig, ax = plt.subplots()
    x = np.arange(mean.shape[1])
    colors = matplotlib.colormaps['viridis_r'](np.linspace(0, 1, len(mean)))
    for i in range(len(mean)):
        # plot shaded std deviation matplotlib
        ax.plot(x, mean[i], color=colors[i], linewidth=3)
        ax.fill_between(x, mean[i]-std[i], mean[i]+std[i], color=colors[i], alpha=0.2, label=i)
    ax.set_ylabel('Cos Sim')
    ax.set_xlabel('Tail Pos')
    ax.set_ylim(-1, 1)
    return fig


def make_heatmap_fig(mean):
    assert len(mean.shape) == 2
    fig, ax = plt.subplots()
    im = ax.imshow(mean, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_ylabel('Layer')
    ax.set_xlabel('Tail Pos')
    return fig


@torch.no_grad()
def collect_hidden_states(prompts, tokenizer, model, batch_size=16):
    assert tokenizer.padding_side == 'left'
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
    eos_id = tokenizer.eos_token_id
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # batch prompts for larger datasets
    batched_ids = input_ids.split(batch_size, dim=0)
    batched_attn_masks = attention_mask.split(batch_size, dim=0)
    n_batches = len(batched_ids)
    hidden_states = []
    for i in tqdm(range(n_batches), desc='collect activations'):
        # strip padding on left to make efficient
        first_ix = batched_attn_masks[i].argmax(dim=1).min()  # first ix among all sequences that is 1
        ids, masks = batched_ids[i][:, first_ix:], batched_attn_masks[i][:, first_ix:]
        hidden = model(ids.to(device), attention_mask=masks.to(device), output_hidden_states=True).hidden_states
        hidden = hidden[1:]  # leave out post-embedding hiddens, only want post-tfmr-block embeddings, this is not in official code...
        hidden = torch.stack(hidden, dim=0).detach().cpu()  # (layers, bsz, seq_len, dim)
        hidden = hidden.permute(1, 0, 2, 3)  # (bsz, layers, seq_len, dim)
        hidden = list(hidden)  # to list
        lengths = masks.sum(dim=-1)
        hidden = [h[:, -l:, :] for h, l in zip(hidden, lengths)]  # remove padding
        hidden_states.extend(hidden)
    return hidden_states


def load_lora(evaldir, model, config_dict):
    device = next(model.parameters()).device
    # first load and merge old lora if run started from existing one
    if 'lora_init' in config_dict and config_dict['lora_init'] is not None:
        assert os.path.exists(config_dict['lora_init'])
        lora_config_path = os.path.join(os.path.dirname(config_dict['lora_init']), 'config.json')
        assert os.path.exists(lora_config_path)
        with open(lora_config_path, 'r') as f:
            lora_config_dict = json.load(f)
        lora_rank = lora_config_dict['lora_rank'] if 'lora_rank' in lora_config_dict else 16
        lora_sd = torch.load(config_dict['lora_init'], map_location=device)
        max_lora_layer = max([int(k.split('.')[4]) for k in lora_sd.keys()])
        max_lora_layer += 1
        model = create_lora_model(model, max_lora_layer, rank=lora_rank)
        incompatible_keys = set_peft_model_state_dict(model, lora_sd)
        if hasattr(incompatible_keys, 'unexpected_keys'):
            assert len(incompatible_keys.unexpected_keys) == 0
        model = model.merge_and_unload()  # fold lora into parameters

    # then load new lora
    lora_path = os.path.join(evaldir, 'lora.pt')
    assert os.path.exists(lora_path)
    lora_sd = torch.load(lora_path, map_location=device)
    if 'method' in config_dict and config_dict['method'] == 'dpo':
        # rename keys properly to evaluate
        lora_sd = {'base_model.model.' + k: v for k, v in lora_sd.items()}
    if 'max_lora_layer' in config_dict:
        max_lora_layer = config_dict['max_lora_layer']
    else:
        max_lora_layer = max([int(k.split('.')[4]) for k in lora_sd.keys()])
        max_lora_layer += 1
    lora_rank = config_dict['lora_rank'] if 'lora_rank' in config_dict else 16
    model = create_lora_model(model, max_lora_layer, rank=lora_rank)
    incompatible_keys = set_peft_model_state_dict(model, lora_sd)
    if hasattr(incompatible_keys, 'unexpected_keys'):
        assert len(incompatible_keys.unexpected_keys) == 0
    model = model.merge_and_unload()  # fold lora into parameters
    return model


def create_lora_model(model, max_lora_layer, rank=16):
    lora_alpha = 1 * rank
    # init lora
    lora_layers = [i for i in range(max_lora_layer)]  # lora on every layer leading up
    lora_config = LoraConfig(
        # from launch script in https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # from defaults in https://github.com/GraySwanAI/circuit-breakers/blob/main/src/args.py
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        # from source in https://github.com/GraySwanAI/circuit-breakers/blob/main/src/lorra_circuit_breaker.py
        task_type='CAUSAL_LM',
        layers_to_transform=lora_layers,
    )
    model = get_peft_model(model, lora_config)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default=None, help='directory to save outputs')
    parser.add_argument('--evaldirs', type=str, nargs='+', default=None, help='run directories to take configs from')
    parser.add_argument('--system_prompt', action='store_true', default=False, help='use system prompt')
    parser.add_argument('--dataset', type=str, default=None, help='dataset to evaluate')
    parser.add_argument('--num_prompts', type=int, default=256, help='number of prompts to use')
    config = parser.parse_args()
    assert config.savedir is not None
    assert os.path.exists(config.savedir)
    assert config.evaldirs is not None
    for p in config.evaldirs:
        assert os.path.exists(p)
    main(config)
