import argparse
import json
import os
import math
import random

import datasets
import numpy as np
from peft import get_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from tqdm import tqdm

import data


MODEL_CACHE_DIR = os.environ['TRANSFORMERS_CACHE']
DATASET_CACHE_DIR = os.environ['HF_DATASETS_CACHE']
SCRATCH_DIR = os.environ['STEERING_SCRATCH_DIR']
ACCESS_TOKEN = os.environ['HF_TOKEN']

MODEL_DICT = {
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'granite': 'ibm-granite/granite-7b-instruct',
    'llama_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama_3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama_8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen_1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen_3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen_7b': 'Qwen/Qwen2.5-7B-Instruct'
}

def main(config):
    config_path = os.path.join(config.savedir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(vars(config), f, sort_keys=True, indent=2)
    else:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = argparse.Namespace(**config_dict)

    # seed everything before generation
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    revision = None
    if config.model == 'mistral':
        revision = 'pr/120'
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DICT[config.model],
        token=ACCESS_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
        revision=revision,  # fixes tokenizer, see https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/141
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DICT[config.model],
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        token=ACCESS_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()

    # init lora
    lora_layers = [i for i in range(config.max_lora_layer)]  # lora on every layer leading up
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        lora_dropout=0.05,
        # from defaults in https://github.com/GraySwanAI/circuit-breakers/blob/main/src/args.py
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        # from source in https://github.com/GraySwanAI/circuit-breakers/blob/main/src/lorra_circuit_breaker.py
        task_type='CAUSAL_LM',
        layers_to_transform=lora_layers,
    )

    # from launch script in https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh
    max_len = 1024
    trainset = preprocess_dpo_dataset(
        config.accept_dsets,
        config.reject_dsets,
        tokenizer,
        split='train',
        num_prompts=config.num_prompts_per_dset,
        max_len=max_len,
        use_system_prompt=config.system_prompt,
    )

    max_steps = config.num_steps
    if config.num_epochs > 0:
        full_train_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * config.ndevices
        assert len(trainset) % full_train_batch_size == 0
        steps_per_epoch = len(trainset) / full_train_batch_size
        max_steps = int(config.num_epochs * steps_per_epoch)

    trainer_args = DPOConfig(
        output_dir=os.path.join(config.savedir, 'hf_trainer'),
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        loss_type='sigmoid',
        beta=config.beta,
        label_smoothing=0.0,
        learning_rate=config.lr,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # with peft no need
        args=trainer_args,
        train_dataset=trainset,
        tokenizer=tokenizer,
        max_length=max_len,
        max_prompt_length=max_len,
        peft_config=lora_config,
    )
    trainer.train()

    # to avoid collisions
    if trainer.is_local_process_zero():
        lora_path = os.path.join(config.savedir, 'lora.pt')
        # from https://github.com/huggingface/peft/issues/1306
        lora_state_dict = get_peft_model_state_dict(model)
        torch.save(lora_state_dict, lora_path)


def preprocess_dpo_dataset(accept_dsets, reject_dsets, tokenizer, split='train', num_prompts=256, max_len=8192, use_system_prompt=False):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    accept_dsets = accept_dsets.split(',')
    reject_dsets = reject_dsets.split(',')

    system_prompt = None
    if use_system_prompt:
        system_prompt = data.dsets_to_system_prompt(accept_dsets)

    def format_prompts_fn(_pairs):
        return data.format_prompts_w_completions(
            _pairs,
            model_name=config.model,
            system_prompt=system_prompt
        )
    accept_pairs = data.get_data_from_names(
        accept_dsets,
        split=split,
        include_outputs=True,
        num_prompts=num_prompts,
        max_length=max_len,
        tokenizer=tokenizer,
        format_prompts_fn=format_prompts_fn,
    )
    reject_pairs = data.get_data_from_names(
        reject_dsets,
        split=split,
        include_outputs=True,
        num_prompts=num_prompts,
        max_length=max_len,
        tokenizer=tokenizer,
        format_prompts_fn=format_prompts_fn,
    )

    accept_examples = format_prompts_fn(accept_pairs)
    reject_examples = format_prompts_fn(reject_pairs)
    if 'mistral' in tokenizer.name_or_path.lower():
        end_of_prompt = '[/INST]'
    elif 'granite' in tokenizer.name_or_path.lower():
        end_of_prompt = '<|assistant|>'
    elif "llama" in tokenizer.name_or_path.lower():
        end_of_prompt = '<|start_header_id|>assistant<|end_header_id|>'
    elif "qwen" in tokenizer.name_or_path.lower():
        end_of_prompt = '<|im_start|>assistant'
    else:
        raise NotImplementedError()
    completion_lengths = [len(ex.split(end_of_prompt)[1]) for ex in accept_examples]  # get tail of prompt
    accept_examples = [(ex[:-cl], ex[-cl:]) for ex, cl in zip(accept_examples, completion_lengths)]
    completion_lengths = [len(ex.split(end_of_prompt)[1]) for ex in reject_examples]  # get tail of prompt
    reject_examples = [(ex[:-cl], ex[-cl:]) for ex, cl in zip(reject_examples, completion_lengths)]
    accept_examples = [(*it, ' I cannot answer that.') for it in accept_examples]
    reject_examples = [(*it, ' I cannot answer that.') for it in reject_examples]

    dataset_dict = {
        'prompt': [
            *[it[0] for it in accept_examples],
            *[it[0] for it in reject_examples],
        ],
        'chosen': [
            *[it[1] for it in accept_examples],
            *[it[2] for it in reject_examples],  # accept "I cannot..."
        ],
        'rejected': [
            *[it[2] for it in accept_examples],
            *[it[1] for it in reject_examples],  # reject ground truth
        ],
    }
    dataset_dict = datasets.Dataset.from_dict(dataset_dict)
    return dataset_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='dpo', choices=['dpo'], help='method type, for reading config later')
    parser.add_argument('--model', type=str, default='mistral', choices=['mistral', 'granite', 'llama_1b', 'llama_3b', 'llama_8b','qwen_1.5b','qwen_3b','qwen_7b'], help='model to use')
    parser.add_argument('--savedir', type=str, default=None, help='directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=0, help='saving frequency in steps')
    parser.add_argument('--max_lora_layer', type=int, default=32, help='maximum layer to use lora on')
    parser.add_argument('--lora_rank', type=int, default=16, choices=[2, 4, 8, 16, 16, 32, 64], help='rank of lora to use')
    parser.add_argument('--accept_dsets', type=str, default='sni_sa', help='accept dataset(s) to use (comma separated for multiple)')
    parser.add_argument('--reject_dsets', type=str, default='sni_s', help='reject dataset(s) to use (comma separated for multiple)')
    parser.add_argument('--num_prompts_per_dset', type=int, default=256, help='number of prompts to use per dataset')
    parser.add_argument('--system_prompt', action='store_true', default=False, help='use system prompt')
    parser.add_argument('--num_epochs', type=float, default=0.0, help='number of epochs to use')
    parser.add_argument('--num_steps', type=int, default=-1, help='number of training steps')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate to use')
    parser.add_argument('--beta', type=float, default=0.05, help='beta (regularization coefficient to reference model)')
    parser.add_argument('--ndevices', type=int, default=1, help='number of devices to train on')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='steps before gradient accumulation')
    config = parser.parse_args()
    assert config.savedir is not None
    assert os.path.exists(config.savedir)
    assert len(config.accept_dsets) > 0
    assert len(config.reject_dsets) > 0
    assert config.num_epochs > 0 or config.num_steps > 0
    assert config.lr > 0
    assert config.save_freq >= 0
    main(config)
