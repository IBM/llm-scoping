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
from data import get_gsm8k_prompts, get_sni_prompts, get_sni_instance_prompts, get_common_sni_tasks, flatten_interleave
import pdb

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
def get_target_layer(model,targets:list):
    import math
    layer_names = [l[0] for l in model.named_parameters()]
    reversed_layer_names = reversed(layer_names)
    for ln in reversed_layer_names:
        if "model.layers." in ln:
            max_layer_name = ln.split(".")[2]
            max_layer_n = int(max_layer_name)+1
            return [math.ceil(max_layer_n * t) for t in targets]


# replication of circuit breakers, see https://arxiv.org/abs/2406.04313
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
    print("model name:",MODEL_DICT[config.model] )
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

    target_layers_portion = [float(l) for l in config.cb_target_layer.split(",")]
    print("target layers", target_layers_portion)

    target_layers = get_target_layer(model, target_layers_portion)
    max_lora_layer_tune = max(target_layers)+1
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

    # from launch script in https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh

    model = create_lora_model(model, max_lora_layer_tune, rank=config.lora_rank)

    max_len = 1024
    trainset = CBDataset(
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
    alpha = config.cb_alpha  # defualt value 5, from https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh
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
    trainer = CBTrainer(
        config.cb_schedule,
        alpha,
        target_layers,
        args=trainer_args,
        model=model,
        train_dataset=trainset,
        data_collator=trainset.collate,
    )
    print("save freq", config.save_freq )
    if config.save_freq > 0:
        print("use save freq")
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


def compute_retain_loss(retain_inputs, retain_lengths, model):
    # get targets
    module = model
    if isinstance(model, torch.nn.parallel.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    with module.disable_adapter():
        with torch.no_grad():
            model.eval()
            retain_targets = model(**retain_inputs, output_hidden_states=True).hidden_states
            retain_targets = retain_targets[1:]  # leave out post-embedding hiddens, only want post-tfmr-block embeddings, this is not in official code...
            retain_targets = torch.stack(retain_targets, dim=0).detach()  # (num_layers, bsz, seq_len, dim)
    model.train()

    hidden = model(**retain_inputs, output_hidden_states=True).hidden_states
    hidden = hidden[1:]  # leave out post-embedding hiddens, only want post-tfmr-block embeddings, this is not in official code...
    hidden = torch.stack(hidden, dim=0)  # (num_layers, bsz, seq_len, dim)

    l2_dist = (hidden - retain_targets).norm(dim=-1, p=2)
    # loss_mask = retain_inputs['attention_mask'][None, :, :].expand(len(l2_dist), -1, -1)
    # loss = l2_dist[loss_mask == 1].mean()
    # only compute loss on the non-system-prompt
    # TODO: assumes padding on left
    l2_dist = [l2_dist[:, i, -retain_lengths[i]:] for i in range(l2_dist.shape[1])]
    l2_dist = torch.cat(l2_dist, dim=-1)  # cat along ragged seq_lens
    loss = l2_dist.mean()
    return loss


def compute_cb_loss(target_layers, cb_inputs, cb_lengths, model):
    # get targets
    module = model
    if isinstance(model, torch.nn.parallel.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    with module.disable_adapter():
        with torch.no_grad():
            model.eval()
            cb_targets = model(**cb_inputs, output_hidden_states=True).hidden_states
            cb_targets = cb_targets[1:]  # leave out post-embedding hiddens, only want post-tfmr-block embeddings, this is not in official code...
            cb_targets = [cb_targets[l].detach() for l in target_layers]
            cb_targets = torch.stack(cb_targets, dim=0)  # (num_tgt_layers, bsz, seq_len, dim)
    model.train()

    hidden = model(**cb_inputs, output_hidden_states=True).hidden_states
    hidden = hidden[1:]  # leave out post-embedding hiddens, only want post-tfmr-block embeddings, this is not in official code...
    hidden = [hidden[l] for l in target_layers]
    hidden = torch.stack(hidden, dim=0)  # (num_tgt_layers, bsz, seq_len, dim)

    # cb_attn_mask = cb_inputs['attention_mask'][:, None, :, None].expand(-1, hidden.size(1), -1, -1)
    # ip = (norm_hidden * norm_target) * cb_attn_mask
    # loss = torch.relu(ip.sum(dim=-1)).sum() / cb_attn_mask.sum()
    cos_sim = (hidden * cb_targets).sum(dim=-1) / hidden.norm(dim=-1) / cb_targets.norm(dim=-1)
    cos_sim = torch.relu(cos_sim)
    # loss_mask = cb_inputs['attention_mask'][None, :, :].expand(len(cos_sim), -1, -1)
    # loss = torch.relu(cos_sim[loss_mask == 1]).mean()
    # only compute loss on the non-system-prompt
    # TODO: assumes padding on left
    cos_sim = [cos_sim[:, i, -cb_lengths[i]:] for i in range(cos_sim.shape[1])]
    cos_sim = torch.cat(cos_sim, dim=-1)  # cat along ragged seq_lens
    loss = cos_sim.mean()
    return loss


def compute_losses(target_layers, batch, model):
    # retain all layers, but circuit break only particular layers
    assert len(batch) == 6
    retain_inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    retain_lengths = batch[2]
    cb_inputs = {'input_ids': batch[3], 'attention_mask': batch[4]}
    cb_lengths = batch[5]

    ret_loss = compute_retain_loss(retain_inputs, retain_lengths, model)
    cb_loss = compute_cb_loss(target_layers, cb_inputs, cb_lengths, model)
    return ret_loss, cb_loss


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


class CBDataset(torch.utils.data.Dataset):
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

        def format_prompts_fn(_prompts):
            return data.format_prompts(
                _prompts,
                model_name=config.model,
                system_prompt=system_prompt,
                prefill=False
            )
        retain_prompts = data.get_data_from_names(
            accept_dsets,
            split=split,
            include_outputs=False,
            num_prompts=num_prompts,
            max_length=max_len,
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )
        cb_prompts = data.get_data_from_names(
            reject_dsets,
            split=split,
            include_outputs=False,
            num_prompts=num_prompts,
            max_length=max_len,
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )

        self.retain_prompts = format_prompts_fn(retain_prompts)
        self.cb_prompts = format_prompts_fn(cb_prompts)

        self.retain_inputs = self.tokenizer(self.retain_prompts, padding=False, truncation=False).input_ids
        self.retain_inputs = [torch.tensor(inp) for inp in self.retain_inputs]
        self.cb_inputs = self.tokenizer(self.cb_prompts, padding=False, truncation=False).input_ids
        self.cb_inputs = [torch.tensor(inp) for inp in self.cb_inputs]

        # get the length of each input without system prompt
        sys_prompt_len = 0
        if system_prompt is not None:
            # instruction starts after first newline as system prompt only has one
            if 'granite' in self.tokenizer.name_or_path.lower():
                sys_prompt = self.retain_prompts[0].split('<|user|>')[0]
            elif "mistral" in self.tokenizer.name_or_path.lower():
                sys_prompt = self.retain_prompts[0].split('\n\n')[
                                 0] + '\n\n'  # get system prompt and newline as these are common to all examples
            elif "llama" in self.tokenizer.name_or_path.lower():
                sys_prompt = self.retain_prompts[0].split('<|start_header_id|>user<|end_header_id|>')[0]
            elif "qwen" in self.tokenizer.name_or_path.lower():
                sys_prompt = self.retain_prompts[0].split('<|im_start|>user')[0]
            print(f"sys prompt of {self.tokenizer.name_or_path.lower()}: ", sys_prompt)
            sys_prompt_len = len(tokenizer([sys_prompt], padding=False, truncation=False).input_ids[0])
            #self.ret_inst_lens = [len(v) - sys_prompt_len for v in self.retain_inputs]
            #self.cb_inst_lens = [len(v) - sys_prompt_len for v in self.cb_inputs]
        self.ret_inst_lens = [len(v) - sys_prompt_len for v in self.retain_inputs]
        self.cb_inst_lens = [len(v) - sys_prompt_len for v in self.cb_inputs]

    def __getitem__(self, ix):
        assert self.retain_inputs is not None
        assert self.cb_inputs is not None

        return (
            self.retain_inputs[ix],
            self.ret_inst_lens[ix],
            self.cb_inputs[ix],
            self.cb_inst_lens[ix],
        )

    def __len__(self):
        return min(len(self.retain_prompts), len(self.cb_prompts))

    def collate(self, batch):
        assert isinstance(batch, list)
        assert self.pad_side == 'left'
        ret_inp, ret_len, cb_inp, cb_len = list(zip(*batch))

        ret_inp, ret_mask = pad_tokens(ret_inp, padding_side=self.pad_side, pad_id=self.pad_id)
        ret_len = torch.tensor(ret_len)

        cb_inp, cb_mask = pad_tokens(cb_inp, padding_side=self.pad_side, pad_id=self.pad_id)
        cb_len = torch.tensor(cb_len)
        return ret_inp, ret_mask, ret_len, cb_inp, cb_mask, cb_len


class CBTrainer(Trainer):
    def __init__(self, cb_schedule, cb_alpha, cb_target_layers, **kwargs):
        super().__init__(**kwargs)
        self.cb_max_steps = self.args.max_steps
        self.cb_alpha = cb_alpha
        self.cb_schedule = cb_schedule
        self.cb_target_layers = cb_target_layers
        self.cb_step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        steps_taken = self.state.global_step
        ret_loss, cb_loss = compute_losses(self.cb_target_layers, inputs, model)
        if self.cb_schedule == 'piecewise':
            # let CB run for 25%, then linearly scale after
            training_portion = (steps_taken / self.cb_max_steps) - 0.25
            mult = 0.5 * max(0, training_portion) / 0.75
            c_ret = self.cb_alpha * mult
            c_cb = self.cb_alpha * (1 - mult)
        elif self.cb_schedule == 'asymmetric':
            # weight CB loss 75% and retain loss 25%
            c_ret = self.cb_alpha * 0.25 * (steps_taken / self.cb_max_steps)
            c_cb = self.cb_alpha * (1 - 0.25 * (steps_taken / (self.cb_max_steps)))
        elif self.cb_schedule == 'step':
            if steps_taken <= 20:
                c_cb = self.cb_alpha
                c_ret = 0
            else:
                c_ret = self.cb_alpha * 0.2
                c_cb = self.cb_alpha * 0.8
        elif self.cb_schedule == 'step2':
            if steps_taken <= 40:
                c_cb = self.cb_alpha
                c_ret = 0
            else:
                c_ret = self.cb_alpha * 0.1
                c_cb = self.cb_alpha * 0.9
        elif self.cb_schedule == 'x':
                c_cb = self.cb_alpha
                c_ret = 0
        elif self.cb_schedule == 'weight':
            c_ret = self.cb_alpha * (steps_taken / (8 * self.cb_max_steps))
            c_cb = self.cb_alpha * (1 - (steps_taken / (8 * self.cb_max_steps)))
        elif self.cb_schedule == 'weight16':
            c_ret = self.cb_alpha * (steps_taken / (16 * self.cb_max_steps))
            c_cb = self.cb_alpha * (1 - (steps_taken / (16 * self.cb_max_steps)))
        else:
            c_ret = self.cb_alpha * (steps_taken / (2 * self.cb_max_steps))
            c_cb = self.cb_alpha * (1 - (steps_taken / (2 * self.cb_max_steps)))
        loss = c_ret * ret_loss + c_cb * cb_loss

        if self.is_local_process_zero() and steps_taken % 10 == 0 and self.cb_step_counter % self.accelerator.gradient_accumulation_steps == 0:
            print('step:', steps_taken)
            print('loss:', round(float(loss), 5))
            print('cb loss:', round(float(cb_loss), 5))
            print('ret loss:', round(float(ret_loss), 5))
        self.cb_step_counter += 1
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
    parser.add_argument('--method', type=str, default='cb', choices=['cb'], help='method type, for reading config later')
    parser.add_argument('--model', type=str, default='mistral', choices=['mistral', 'granite', 'llama_1b', 'llama_3b', 'llama_8b', 'qwen_1.5b','qwen_3b','qwen_7b'], help='model to use')
    parser.add_argument('--savedir', type=str, default=None, help='directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=0, help='saving frequency in steps')
    parser.add_argument('--lora_init', type=str, default=None, help='lora ckpt path to start from')
    parser.add_argument('--max_lora_layer', type=int, default=21, help='maximum layer to use lora on')
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
    parser.add_argument('--cb_alpha', type=float, default=5, help='cb_alpha coefficient')
    parser.add_argument('--cb_schedule', type=str, default='none', help='schedule type for cb (granite is hard to optimize) ')
    parser.add_argument('--cb_target_layer', type=str, default='0.3,0.6', help='target layer settings')

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
