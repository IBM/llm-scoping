import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import data
from roberta_probe import RobertaHeadModel


MODEL_CACHE_DIR = os.environ['TRANSFORMERS_CACHE']
DATASET_CACHE_DIR = os.environ['HF_DATASETS_CACHE']
SCRATCH_DIR = os.environ['STEERING_SCRATCH_DIR']
ACCESS_TOKEN = os.environ['HF_TOKEN']

# evaluate rejection rate for probe
@torch.no_grad()
def main(config):
    # seed everything before generation
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    config_path = os.path.join(config.evaldir, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    train_accept_sets = config_dict['accept_dsets']
    train_accept_sets = train_accept_sets.split(',')

    model_name = config_dict['model'] if 'model' in config_dict else 'Alibaba-NLP/gte-base-en-v1.5'
    revision = None
    if model_name == 'mistral':
        revision = 'pr/120'
    tokenizer = AutoTokenizer.from_pretrained(
        'Alibaba-NLP/gte-base-en-v1.5',
        token=ACCESS_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
        revision=revision,  # fixes tokenizer, see https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/141
        trust_remote_code=True,
    )
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'


    # instantiate probe model, get probe sd path, load probe sd
    if config.eval_step >= 0:
        head_path = os.path.join(config.evaldir, 'head', f'{config.eval_step}.pt')
    else:
        head_path = os.path.join(config.evaldir, 'head.pt')
    assert os.path.exists(head_path)
    head_sd = torch.load(head_path, map_location='cpu')
    model = RobertaHeadModel()
    model.load_state_dict(head_sd)
    device = next(model.parameters()).device
    #model = model.to('cuda:0')
    model = model.to(device)

    system_prompt = None
    if config.system_prompt:
        # system_prompt should be specific to train accept set
        system_prompt = data.dsets_to_system_prompt(train_accept_sets)

    def format_prompts_fn(_prompts):
        if isinstance(_prompts[0], tuple):
            _prompts = [it[0] for it in _prompts]
        if config.prompt_style == 'fewshot':
            return data.format_fewshot_prompts(
                _prompts,
                [config.dataset],
                tokenizer,
                model_name=model_name,
                num_shots=2,
                max_length=256,
            )
        elif config.prompt_style is not None and 'multiturn' in config.prompt_style:
            return data.format_multiturn_prompts(
                _prompts,
                train_accept_sets,
                [config.dataset],
                model_name=model_name,
                num_turns=1,
                adv_sys_prompt='sys' in config.prompt_style,
            )
        else:
            prefill = config.prompt_style == 'prefill'
            b64 = config.prompt_style == 'b64'
            # adversarial prompt should be specific to eval set
            adversarial_prompt = None
            if config.prompt_style == 'adv':
                adversarial_prompt = data.dsets_to_system_prompt([config.dataset])
            return data.format_prompts(
                _prompts,
                model_name=model_name,
                system_prompt=system_prompt,
                adversarial_prompt=adversarial_prompt,
                prefill=prefill,
                b64=b64,
            )

    val = data.get_data_from_names(
        [config.dataset],
        split='val',
        include_outputs=True,
        num_prompts=config.num_prompts,
        max_length=4096,
        tokenizer=tokenizer,
        format_prompts_fn=format_prompts_fn,
    )
    prompts = [it[0] for it in val]
    formatted_prompts = format_prompts_fn(prompts)

    bsz = 1

    stump = 'head'
    if config.system_prompt:
        stump = stump + '_sys'
    if config.prompt_style is not None:
        stump = stump + f'_{config.prompt_style}'
    path = stump + '.json'
    path = os.path.join(config.dataset, path)
    savepath = os.path.join(config.evaldir, 'eval', path)

    predictions = get_predictions(formatted_prompts, tokenizer, model, batch_size=bsz)
    rejections = [pred == 1 for pred in predictions]

    # save everything
    save_results(
        savepath,
        formatted_prompts,
        rejections,
    )


# save as json to avoid regenerating
def save_results(
        savepath,
        prompts,
        rejections,
    ):
    # for each example show prompt, prediction
    json_dict = {
        'reject': sum(rejections) / len(rejections),
    }

    examples = []
    for i in range(len(prompts)):
        examples.append({
            'prompt': prompts[i],
            'reject': rejections[i],
        })
    json_dict['examples'] = examples

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'w') as f:
        json.dump(json_dict, f, indent=2)


@torch.no_grad()
def get_predictions(prompts, tokenizer, model, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=8192).to(device)
    eos_id = tokenizer.eos_token_id
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # batch prompts for larger datasets
    batched_ids = input_ids.split(batch_size, dim=0)
    batched_attn_masks = attention_mask.split(batch_size, dim=0)
    n_batches = len(batched_ids)
    predictions = []
    for i in tqdm(range(n_batches), desc='predict'):
        # strip padding on left to make efficient
        # TODO: fix this when bsz > 1, considering training bsz
        first_ix = batched_attn_masks[i].argmax(dim=1).min()  # first ix among all sequences that is 1
        input_ids, attn_masks = batched_ids[i][:, first_ix:], batched_attn_masks[i][:, first_ix:]
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attn_masks,
        }
        logits = model(inputs)
        preds = logits.argmax(dim=-1).tolist()
        predictions.extend(preds)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default="test", help='spoof arg for jbsub run')
    parser.add_argument('--evaldir', type=str, default="test", help='directory to evaluate')
    parser.add_argument('--eval_step', type=int, default=1, help='step of training to evaluate at')
    parser.add_argument('--system_prompt', action='store_true', default=False, help='use system prompt')
    parser.add_argument('--prompt_style', type=str, default=None, choices=['adv', 'prefill', 'b64', 'multiturn', 'fewshot', 'multiturn_sys'], help='prompting style to use')
    parser.add_argument('--dataset', type=str, default="sni_qa", help='dataset to evaluate')
    parser.add_argument('--num_prompts', type=int, default=256, help='number of examples to evaluate')
    parser.add_argument('--regenerate', action='store_true', default=False, help='regenerate completions (e.g. if system prompt changes)')
    config = parser.parse_args()
    assert config.evaldir is not None
    assert os.path.exists(config.evaldir)
    assert config.dataset is not None
    assert len(config.dataset.split(',')) == 1
    assert config.num_prompts > 0
    main(config)
