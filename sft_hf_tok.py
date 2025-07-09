import argparse
import json
import os
import math
import random

import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
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

# LoRA SFT where inputs are
# - (prompt, completion) for accept set
# - (prompt, "I cannot answer.") for reject set
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
    if not hasattr(config, 'model'):
        config.model = 'mistral'
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

    if config.lora_init is not None:
        lora_sd = torch.load(config.lora_init, map_location='cpu')
        max_lora_layer = max([int(k.split('.')[4]) for k in lora_sd.keys()])
        max_lora_layer += 1
        lora_config_path = os.path.join(os.path.dirname(config.lora_init), 'config.json')
        assert os.path.exists(lora_config_path)
        with open(lora_config_path, 'r') as f:
            lora_config_dict = json.load(f)
        lora_rank = lora_config_dict['lora_rank'] if 'lora_rank' in lora_config_dict else 16
        model = create_lora_model(model, max_lora_layer, rank=lora_rank)
        incompatible_keys = set_peft_model_state_dict(model, lora_sd)
        if hasattr(incompatible_keys, 'unexpected_keys'):
            assert len(incompatible_keys.unexpected_keys) == 0
        model = model.merge_and_unload()  # fold lora into parameters

    model = create_lora_model(model, config.max_lora_layer, rank=config.lora_rank)

    # from launch script in https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh
    max_len = 1024
    trainset = SFTDataset(
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

    # train lora, in particular differs from details where we supervise at many layers, not just 10, 20
    # when torch compiled w/o this throws an error
    trainer_args = TrainingArguments(
        output_dir=os.path.join(config.savedir, 'hf_trainer'),
        max_steps=max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
    )
    trainer = SFTTrainer(
        args=trainer_args,
        model=model,
        train_dataset=trainset,
        data_collator=trainset.collate,
    )
    if config.save_freq > 0:
        trainer.add_callback(LoraSaverCallback(
            model,
            os.path.join(config.savedir, 'lora'),
            config.save_freq
        ))
    trainer.train()

    # to avoid collisions
    if trainer.is_local_process_zero():
        lora_path = os.path.join(config.savedir, 'lora.pt')
        # from https://github.com/huggingface/peft/issues/1306
        lora_state_dict = get_peft_model_state_dict(model)
        torch.save(lora_state_dict, lora_path)


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


def compute_losses(batch, model):
    assert len(batch) == 3
    # shift inputs and construct labels for next token prediction
    inputs = {'input_ids': batch[0][:, :-1], 'attention_mask': batch[1][:, :-1],}
    labels = batch[0][:, 1:]
    completion_lengths = batch[2]
    logits = model(**inputs).logits

    # compute loss only on the completions
    # TODO: assumes padding is on the left
    logits, labels = list(logits), list(labels)
    logits = [log[-length:] for log, length in zip(logits, completion_lengths)]
    labels = [lab[-length:] for lab, length in zip(labels, completion_lengths)]
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def pad_tokens(tokens, padding_side, pad_id):
    assert isinstance(tokens, list) or isinstance(tokens, tuple)
    assert isinstance(tokens[0], torch.Tensor)
    attn_masks = [torch.ones_like(t) for t in tokens]
    if padding_side == 'left':
        tokens = [t.flip(0) for t in tokens]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_id)
        attn_masks = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
        tokens = tokens.flip(1)
        attn_masks = attn_masks.flip(1)
    elif padding_side == 'right':
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_id)
        attn_masks = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
    else:
        raise NotImplementedError()
    return tokens, attn_masks


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, accept_dsets, reject_dsets, tokenizer, split='train', num_prompts=256, max_len=8192, use_system_prompt=False, device=torch.device('cpu')):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        self.pad_id = self.tokenizer.eos_token_id
        self.pad_side = 'left'
        self.num_prompts = num_prompts
        self.device = device
        self.split = split
        self.max_len = max_len
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
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )
        reject_pairs = data.get_data_from_names(
            reject_dsets,
            split=split,
            include_outputs=True,
            num_prompts=num_prompts,
            max_length=max_len,
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )

        self.accept_pairs = accept_pairs
        # replace outputs in reject set
        self.reject_pairs = [(it[0], 'I cannot answer that.') for it in reject_pairs]

        self.text = [
            *format_prompts_fn(self.accept_pairs),
            *format_prompts_fn(self.reject_pairs),
        ]
        self.tokens = self.tokenizer(self.text, padding=False, truncation=False).input_ids
        self.tokens = [torch.tensor(inp) for inp in self.tokens]

        if 'mistral' in self.tokenizer.name_or_path.lower():
            end_of_prompt = '[/INST]'
        elif 'granite' in self.tokenizer.name_or_path.lower():
            end_of_prompt = '<|assistant|>'
        elif "llama" in self.tokenizer.name_or_path.lower():
            end_of_prompt = '<|start_header_id|>assistant<|end_header_id|>'
        elif "qwen" in self.tokenizer.name_or_path.lower():
            end_of_prompt = '<|im_start|>assistant'
        else:
            raise NotImplementedError()
        completion_lengths = [len(ex.split(end_of_prompt)[1]) for ex in self.text]
        prompt_text = [ex[:-cl] for ex, cl in zip(self.text, completion_lengths)]
        prompt_tokens = self.tokenizer(prompt_text, padding=False, truncation=False).input_ids
        for pt, t in zip(prompt_tokens, self.tokens):
            assert pt == t[:len(pt)].tolist()
        self.completion_lengths = torch.tensor([len(t) - len(pt) for t, pt in zip(self.tokens, prompt_tokens)], device=device)

    def __getitem__(self, ix):
        return self.tokens[ix], self.completion_lengths[ix]

    def __len__(self):
        return len(self.tokens)

    def collate(self, batch):
        assert isinstance(batch, list)
        assert self.pad_side == 'left'
        tokens, completion_lengths = list(zip(*batch))
        tokens, attn_mask = pad_tokens(tokens, padding_side=self.pad_side, pad_id=self.pad_id)
        completion_lengths = torch.tensor(completion_lengths)
        return tokens, attn_mask, completion_lengths


# see https://huggingface.co/docs/transformers/main/en/trainer#customize-the-trainer
class SFTTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sft_step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        steps_taken = self.state.global_step
        loss = compute_losses(inputs, model)
        if self.is_local_process_zero() and steps_taken % 10 == 0 and self.sft_step_counter % self.accelerator.gradient_accumulation_steps == 0:
            print('step:', steps_taken)
            print('loss:', round(float(loss), 5))
        self.sft_step_counter += 1
        return (loss,) if return_outputs else loss


class LoraSaverCallback(TrainerCallback):
    def __init__(self, model, savedir, save_freq):
        self.model = model
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        self.save_freq = save_freq

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.save_freq == 0:
            lora_path = os.path.join(self.savedir, f'{state.global_step}.pt')
            lora_state_dict = get_peft_model_state_dict(self.model)
            torch.save(lora_state_dict, lora_path)
        return control


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sft', choices=['sft'], help='method type, for reading config later')
    parser.add_argument('--model', type=str, default='mistral', choices=['mistral', 'granite', 'llama_1b', 'llama_3b', 'llama_8b','qwen_1.5b','qwen_3b','qwen_7b'], help='model to use')
    parser.add_argument('--savedir', type=str, default=None, help='directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=0, help='saving frequency in steps')
    parser.add_argument('--lora_init', type=str, default=None, help='lora ckpt path to start from')
    parser.add_argument('--max_lora_layer', type=int, default=32, help='maximum layer to use lora on')
    parser.add_argument('--lora_rank', type=int, default=16, choices=[2, 4, 8, 16, 16, 32, 64], help='rank of lora to use')
    parser.add_argument('--accept_dsets', type=str, default='sni_sa', help='accept dataset(s) to use (comma separated for multiple)')
    parser.add_argument('--reject_dsets', type=str, default='sni_s', help='reject dataset(s) to use (comma separated for multiple)')
    parser.add_argument('--num_prompts_per_dset', type=int, default=256, help='number of prompts to use per dataset')
    parser.add_argument('--system_prompt', action='store_true', default=False, help='use system prompt')
    parser.add_argument('--num_epochs', type=float, default=0.0, help='number of epochs to use')
    parser.add_argument('--num_steps', type=int, default=-1, help='number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to use')
    parser.add_argument('--ndevices', type=int, default=1, help='number of devices to train on')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='steps before gradient accumulation')
    config = parser.parse_args()
    assert config.savedir is not None
    assert os.path.exists(config.savedir)
    assert len(config.accept_dsets) > 0
    assert len(config.reject_dsets) > 0
    assert config.num_epochs > 0 or config.num_steps > 0
    if config.lora_init is not None:
        assert os.path.exists(config.lora_init)
    assert config.lr > 0
    assert config.max_lora_layer > 0
    assert config.save_freq >= 0
    main(config)
