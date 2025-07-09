import argparse
import json
import os
import math
import random

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
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
    model = ProbeHeadModel(model, arch=config.probe_arch)

    # from launch script in https://github.com/GraySwanAI/circuit-breakers/blob/main/scripts/lorra_circuit_breaker_mistral_7b.sh
    max_len = 1024
    trainset = ProbeDataset(
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
        save_safetensors = False,
    )
    trainer = ProbeTrainer(
        args=trainer_args,
        model=model,
        train_dataset=trainset,
        data_collator=trainset.collate,
    )
    if config.save_freq > 0:
        trainer.add_callback(HeadSaverCallback(
            model,
            os.path.join(config.savedir, 'head'),
            config.save_freq
        ))
    trainer.train()

    # to avoid collisions
    if trainer.is_local_process_zero():
        head_path = os.path.join(config.savedir, 'head.pt')
        # from https://github.com/huggingface/peft/issues/1306
        head_state_dict = model.head.state_dict()
        torch.save(head_state_dict, head_path)


def compute_losses(batch, model):
    assert len(batch) == 3
    # shift inputs and construct labels for next token prediction
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    labels = batch[2]
    logits = model(inputs)
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


class HeadModel(torch.nn.Module):
    def __init__(self, model, arch='mlp'):
        super(HeadModel, self).__init__()
        self.model = model
        self.arch = arch
        num_layers = self.model.config.num_hidden_layers
        self.num_layers = num_layers
        dim = self.model.config.hidden_size

        # instantiate MLP head based on model layers and dim
        # or have many MLPs, one for each layer
        # then ensemble decisions over all of them
        if self.arch == 'mlp':
            self.head = torch.nn.Sequential(
                torch.nn.Linear(num_layers*dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2),  # two final classes
            )
        elif self.arch == 'bert':
            self.proj = torch.nn.Linear(num_layers*dim, 768)
            self.bert_head = AutoModel.from_pretrained(
                'roberta-base',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                token=ACCESS_TOKEN,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.head = torch.nn.Linear(768, 2)
        else:
            raise NotImplementedError()


    def forward(self, inputs):
        # - collect activations at each layer
        # - average over valid positions in each layer,
        # - concatenate vectors and normalize to norm 1
        # - run concatenated tensor thru mlp to get binary class logits
        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states
            hidden = hidden[1:]  # exclude token embedding

            mask = inputs['attention_mask']
            if self.arch == 'mlp':
                lengths = mask.sum(dim=-1)
                # average over valid positions in each layer
                hidden = [h * mask[:, :, None] for h in hidden]
                # remove first token as it will be large norm
                hidden = [h[:, 1:].sum(dim=1) for h in hidden]
                lengths -= mask[:, 0]  # readjust lengths as removing first token
                hidden = [h / lengths[:, None] for h in hidden]
                hidden = torch.stack(hidden, dim=1)  # bsz, num layers, dim
                hidden = hidden / hidden.norm(dim=-1, keepdim=True)  # norm-1 all layers
                hidden = hidden.flatten(1)  # bsz, layers*dim

        if self.arch == 'bert':
            layer_ix = int(2 / 3 * self.num_layers)
            hidden = hidden[layer_ix]  # take layer 2/3 way through
            bsz, seq_len, dim = hidden.shape
            hidden = hidden.reshape(bsz * seq_len, dim)
            embedding_output = self.proj(hidden)
            embedding_output.reshape(bsz, seq_len, 768)
            hidden = self.bert_head(
                inputs_embeds=embedding_output,
                attention_mask=inputs['attention_mask'],
            )
            hidden = hidden.last_hidden_state[:, 0]  # cls token

        logits = self.head(hidden)

        return logits


class ProbeDataset(torch.utils.data.Dataset):
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
        accept_prompts = data.get_data_from_names(
            accept_dsets,
            split=split,
            include_outputs=False,
            num_prompts=num_prompts,
            max_length=max_len,
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )
        reject_prompts = data.get_data_from_names(
            reject_dsets,
            split=split,
            include_outputs=False,
            num_prompts=num_prompts,
            max_length=max_len,
            tokenizer=self.tokenizer,
            format_prompts_fn=format_prompts_fn,
        )

        self.accept_prompts = format_prompts_fn(accept_prompts)
        self.reject_prompts = format_prompts_fn(reject_prompts)
        self.text = [
            *self.accept_prompts,
            *self.reject_prompts,
        ]

        self.accept_inputs = self.tokenizer(self.accept_prompts, padding=False, truncation=False).input_ids
        self.accept_inputs = [torch.tensor(inp) for inp in self.accept_inputs]
        self.reject_inputs = self.tokenizer(self.reject_prompts, padding=False, truncation=False).input_ids
        self.reject_inputs = [torch.tensor(inp) for inp in self.reject_inputs]
        self.tokens = [
            *self.accept_inputs,
            *self.reject_inputs,
        ]

        # get binary class labels as to whether to accept/reject
        self.labels = [
            *[torch.tensor(0, dtype=torch.int64) for _ in self.accept_inputs],
            *[torch.tensor(1, dtype=torch.int64) for _ in self.reject_inputs],
        ]

    def __getitem__(self, ix):
        return self.tokens[ix], self.labels[ix]

    def __len__(self):
        return len(self.tokens)

    def collate(self, batch):
        assert isinstance(batch, list)
        assert self.pad_side == 'left'
        tokens, labels = list(zip(*batch))
        tokens, attn_mask = pad_tokens(tokens, padding_side=self.pad_side, pad_id=self.pad_id)
        labels = torch.tensor(labels, dtype=torch.int64)
        return tokens, attn_mask, labels


# see https://huggingface.co/docs/transformers/main/en/trainer#customize-the-trainer
class ProbeTrainer(Trainer):
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


# saving/loading for head model
class HeadSaverCallback(TrainerCallback):
    def __init__(self, model, savedir, save_freq):
        self.model = model
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        self.save_freq = save_freq

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.save_freq == 0:
            # save only probe head
            head_path = os.path.join(self.savedir, f'{state.global_step}.pt')
            head_state_dict = self.model.head.state_dict()
            torch.save(head_state_dict, head_path)
        return control


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='cls', choices=['cls'], help='method type, for reading config later')
    parser.add_argument('--model', type=str, default='mistral', choices=['mistral', 'granite', 'llama_1b', 'llama_3b', 'llama_8b','qwen_1.5b','qwen_3b','qwen_7b'], help='model to use')
    parser.add_argument('--probe_arch', type=str, default='mlp', choices=['mlp', 'bert'], help='probe head architecture')
    parser.add_argument('--savedir', type=str, default=None, help='directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=0, help='saving frequency in steps')
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
    assert config.lr > 0
    assert config.save_freq >= 0
    main(config)
