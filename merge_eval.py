import argparse
from glob import glob
import itertools
import json
import os

import pandas as pd


def main(config, base=False):
    # if config under immediate sweepdir, then no sweeps
    # otherwise merge sweep

    config_paths = list(glob(os.path.join(config.sweepdir, 'config.json')))
    if len(config_paths) == 1:
        assert os.path.isdir(os.path.join(config.sweepdir, 'eval'))
        base_df = merge_run(config.sweepdir, base=True)
        df = merge_run(config.sweepdir, base=False)
    else:
        assert len(config_paths) == 0
        config_paths = list(glob(os.path.join(config.sweepdir, '*', 'config.json')))
        assert len(config_paths) > 0
        base_df = merge_sweep(config.sweepdir, base=True)
        df = merge_sweep(config.sweepdir, base=False)

    # extract_string_reject(df)
    if base:
        df = base_df
    return df


def merge_sweep(sweeppath, base=False):
    subdirs = [d.path for d in os.scandir(sweeppath) if d.is_dir()]
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join(d, 'eval'))]

    # per-run, collect eval results on each dataset
    for i, d in enumerate(subdirs):
        run_df = merge_run(d, base=base)
        if i > 0:
            df = pd.concat([df, run_df])
        else:
            df = run_df

    if base:
        savepath = os.path.join(sweeppath, 'base_results.csv')
    else:
        savepath = os.path.join(sweeppath, 'results.csv')
    df.to_csv(savepath)

    path, ext = os.path.splitext(savepath)
    cleaned_savepath = f'{path}_clean{ext}'
    cleaned_df = clean_df(df.copy(deep=True))
    cleaned_df.to_csv(cleaned_savepath)
    return df


def merge_run(dirpath, base=False):
    train_config_path = os.path.join(dirpath, 'config.json')
    with open(train_config_path, 'r') as f:
        tcd = json.load(f)
    train_column_names = [k for k in tcd.keys() if 'savedir' not in k]
    train_vals = [tcd[k] for k in tcd.keys() if 'savedir' not in k]

    # table should have accept dsets, reject dsets, trained with sys, evaluated with sys, evaluated with adv, string reject score, judge reject score for each dset
    eval_subdirs = [d.path for d in os.scandir(os.path.join(dirpath, 'eval')) if d.is_dir()]

    rows = []
    for d in eval_subdirs:
        jsons = [f for f in os.listdir(d) if os.path.splitext(f)[1] == '.json']
        if base:
            jsons = [f for f in jsons if f.startswith('orig')]
        else:
            jsons = [f for f in jsons if f.startswith('lora')]
        dset = os.path.basename(d)

        for jf in jsons:
            with open(os.path.join(d, jf), 'r') as f:
                jd = json.load(f)

            split_path = '_'.join(os.path.splitext(jf)[0].split('_')[1:])
            prompt_style = 'none'
            for p in ['adv', 'prefill', 'b64', 'multiturn', 'fewshot', 'multiturn_sys', 'tap']:
                if p in split_path:
                    prompt_style = p

            row = [dset, split_path.startswith('sys'), prompt_style]
            jd_vals = [jd[k] for k in jd.keys() if k != 'examples']
            row = row + [jd[k] for k in jd.keys() if k != 'examples']
            rows.append(row)
    rows = [train_vals + row for row in rows]

    column_names = train_column_names
    column_names += ['eval_dset', 'sys', 'prompt_style']
    column_names += [k for k in jd.keys() if k != 'examples']
    assert len(column_names) == len(rows[0])

    df = pd.DataFrame(columns=column_names, data=rows)
    if base:
        savepath = os.path.join(dirpath, 'eval', 'base_results.csv')
    else:
        savepath = os.path.join(dirpath, 'eval', 'results.csv')
    df.to_csv(savepath)

    path, ext = os.path.splitext(savepath)
    cleaned_savepath = f'{path}_clean{ext}'
    cleaned_df = clean_df(df.copy(deep=True))
    cleaned_df.to_csv(cleaned_savepath)
    return df


def clean_df(df):
    # remove columns with single value
    df = df.loc[:,df.apply(pd.Series.nunique) > 1]
    # sort by eval dataset
    # sort by sweep
    sort_keys = ['eval_dset']
    if 'accept_dsets' in df.keys():
        sort_keys.append('accept_dsets')
    if 'reject_dsets' in df.keys():
        sort_keys.append('reject_dsets')
    df = df.sort_values(sort_keys)
    return df


# - combine multiple csvs using this
# - I don't know anything about table joins, perhaps I should learn something?
#   - happen to be in sql central...
def extract_string_reject(df, keep_common=None, extra_hyps=None, remove_columns=None, average_style=None):
    if keep_common is None:
        keep_common = []
    keep_common = ['method', 'accept_dsets', 'reject_dsets']
    # metric_columns = ['string_reject', 'acc', 'rougel']
    metric_columns = ['string_reject']

    # remove extra performance metric columns
    # metrics_to_remove = ['kl', 'bleu', 'meteor']
    metrics_to_remove = ['kl', 'bleu', 'meteor', 'rougel', 'acc']
    metrics_to_remove = [name for name in metrics_to_remove if name in df]
    df = df.drop(columns=metrics_to_remove)

    # remove single-value columns except certain keys
    to_remove = []
    for k in df.keys():
        if len(df[k].unique()) == 1 and k not in metric_columns and k != 'eval_dset':
            to_remove.append(k)
    keep_common = list(set(to_remove).intersection(set(keep_common)))
    keep_common_vals = [df[k].unique()[0] for k in keep_common]
    df = df.drop(columns=to_remove)

    # remove rows with ood dset evals
    df = df[~df['eval_dset'].str.contains('_ood')]

    # pick a setting of extra hyperparameters
    extra_hyps = {
        'system_prompt': True,
        'sys': True,
    }
    extra_hyps = {k: v for k, v in extra_hyps.items() if k in df}
    for k, v in extra_hyps.items():
        df = df[df[k] == v]
        df = df.drop(columns=k)  # drop after setting

    # check format after preprocessing
    assert 'eval_dset' in df.columns
    for sc in metric_columns:
        assert sc in df.columns
    swept_keys = [
        k for k in df.columns if k not in metric_columns and k != 'eval_dset'
    ]

    if len(swept_keys) > 0:
        if len(swept_keys) == 1:
            swept_vals = df[swept_keys[0]].unique().tolist()
            swept_vals = [[sv] for sv in swept_vals]
        else:
            swept_vals = [df[sk].unique().tolist() for sk in swept_keys]
            swept_vals = list(itertools.product(*swept_vals))

        # group into separate blocks by non eval_dset, reject column
        new_dfs = []
        for i, vals in enumerate(swept_vals):
            mask = None
            for _i, (k, v) in enumerate(zip(swept_keys, vals)):
                if _i == 0:
                    mask = df[k] == v
                else:
                    mask &= df[k] == v
            new_df = df[mask]
            new_df = new_df.drop(columns=swept_keys)
            # combine metric columns as csv
            if len(metric_columns) > 1:
                new_df['metrics'] = new_df[metric_columns].apply(
                    lambda x: '/'.join(x.dropna().round(3).astype('str')), axis=1,
                )
            else:
                new_df['metrics'] = new_df[metric_columns]
            new_df = new_df.drop(columns=metric_columns)
            new_df = new_df.rename(columns={'metrics': i})
            new_df = new_df.set_index('eval_dset')
            new_dfs.append(new_df)
        df = new_dfs[0].join(new_dfs[1:])
        df = df.rename_axis(['index'])
    else:
        if len(metric_columns) > 1:
            df['metrics'] = df[metric_columns].apply(
                lambda x: '/'.join(x.dropna().round(3).astype('str')), axis=1,
            )
        else:
            df['metrics'] = df[metric_columns]
        df = df.drop(columns=metric_columns)
        df = df.set_index('eval_dset')
        df = df.rename_axis(['index'])

    # sort evaluation datasets into particular order
    # classification | generation | reasoning (pe, gsm8k) | ood
    eval_dsets = [
        'sni_sa',  # classification
        'sni_tld',
        'sni_s',  # generation
        'sni_tc',
        'sni_sc',
        'sni_dg',
        'sni_pe',  # reasoning
        'gsm8k',
        'sni_qa',  # ood general
        'alpaca',
    ]
    # if fine accept/fine reject put those first and shift everything down
    finegrained = ['fine' in k for k in df.index]
    if sum(finegrained) > 0:
        fine_dset = df.index[finegrained.index(True)]
        replace_dset = '_'.join(fine_dset.split('_')[:-1])
        eval_dsets.pop(eval_dsets.index(replace_dset))
        eval_dsets = [replace_dset + '_fineaccept', replace_dset + '_finereject'] + eval_dsets
    assert set(eval_dsets) == set(df.index)

    # add back in rows
    for k, v in zip(keep_common, keep_common_vals):
        df.loc[k] = [v for _ in range(df.shape[1])]
    for i, k in enumerate(swept_keys):
        df.loc[k] = [it[i] for it in swept_vals]

    # header_key_order = [
        # 'accept_dsets',
        # 'reject_dsets',
    # ]
    # reordered_header_keys = []
    # leftover = []
    # for k in header_key_order:
        # if k in header_keys:
            # reordered_header_keys.append(k)
    header_keys = keep_common
    if len(swept_keys) > 0:
        header_keys = swept_keys + keep_common
    # resort df to put those rows first, then the eval dset order
    df = df.reindex(header_keys + eval_dsets)

    if average_style is not None:
        if average_style == 'category':
            # over different categories (classification, generation, reasoning, general)
            task_lists = {
                'fineaccept': ['fineaccept'],
                'finereject': ['finereject'],
                'classification': ['sni_sa', 'sni_tld'],
                'generation': ['sni_s', 'sni_tc', 'sni_sc', 'sni_dg'],
                'reasoning': ['sni_pe', 'gsm8k'],
                'broad': ['sni_qa', 'alpaca'],
            }
            # for each category in each task
            # - take all row names it matches into
            # - average scores vertically
            # - get new row with new index name and append
            # - keep finegrained into their own categories
            for category, tasks in task_lists.items():
                row_keys = []
                for task in tasks:
                    # get matching keys in index
                    mask = df.index.str.fullmatch(task).tolist()
                    if sum(mask) > 0:
                        row_keys.extend(df.index[mask].tolist())
                # remove finegrained keys
                if 'fine' not in category:
                    row_keys = [rk for rk in row_keys if 'fine' not in k]
                df.loc[category] = df.loc[row_keys].mean(axis=0).tolist()
                df.loc[category + '_std'] = df.loc[row_keys].std(axis=0).tolist()
        elif average_style == 'ood':
            accept = df.loc['accept_dsets'].unique().tolist()
            accept = [s.split(',') for s in accept]
            reject = df.loc['reject_dsets'].unique().tolist()
            reject = [s.split(',') for s in reject]
            if len(accept) == 1:
                accept = [accept[0] for _ in range(df.shape[1])]
            if len(reject) == 1:
                reject = [reject[0] for _ in range(df.shape[1])]
            assert len(accept) == len(reject) == df.shape[1]

            id_accept_row = []
            id_fine_reject_row = []
            id_reject_row = []
            ood_reject_row = []
            id_accept_std_row = []
            id_fine_reject_std_row = []
            id_reject_std_row = []
            ood_reject_std_row = []
            for i, (a, r) in enumerate(zip(accept, reject)):
                id_accept = a
                id_reject = r
                finegrained = ['fine' in k for k in id_accept]
                id_fine_reject = []
                if sum(finegrained) > 0:
                    id_fine_reject = [id_accept[finegrained.index(True)].replace('accept', 'reject')]
                ood_reject = [
                    k for k in eval_dsets
                    if k not in id_accept and k not in id_reject
                ]

                id_accept_row.append(float(df[i][id_accept].mean()))
                if len(id_fine_reject) > 0:
                    id_fine_reject_row.append(float(df[i][id_fine_reject].mean()))
                id_reject_row.append(float(df[i][id_reject].mean()))
                ood_reject_row.append(float(df[i][ood_reject].mean()))

                id_accept_std_row.append(float(df[i][id_accept].std()))
                if len(id_fine_reject) > 0:
                    id_fine_reject_std_row.append(float(df[i][id_fine_reject].std()))
                id_reject_std_row.append(float(df[i][id_reject].std()))
                ood_reject_std_row.append(float(df[i][ood_reject].std()))

            df.loc['id_accept'] = id_accept_row
            if len(id_fine_reject_row) > 0:
                df.loc['id_fine_reject'] = id_fine_reject_row
            df.loc['id_reject'] = id_reject_row
            df.loc['ood_reject'] = ood_reject_row

            df.loc['id_accept_std'] = id_accept_std_row
            if len(id_fine_reject_row) > 0:
                df.loc['id_fine_reject_std'] = id_fine_reject_std_row
            df.loc['id_reject_std'] = id_reject_std_row
            df.loc['ood_reject_std'] = ood_reject_std_row
        else:
            raise NotImplementedError()
    df = df.fillna(0)  # stds give nans when only one elt

    # transpose so datasets are column labels
    df = df.transpose()
    if average_style is not None: 
        std_keys = [k for k in df.keys() if 'std' in k]
        avg_keys = ['_'.join(k.split('_')[:-1]) for k in std_keys]
        # convert cells to mean \pm std
        for k in avg_keys:
            df[k] = df[[k, k+'_std']].apply(
                lambda x: ' \pm '.join(x.astype('float').round(3).astype('str')), axis=1,
            )
        to_drop = eval_dsets + std_keys
        if 'fineaccept' in df.keys() and sum(['fine' in k for k in df['accept_dsets']]) < 1:
            to_drop += ['fineaccept', 'finereject']  # remove fine categories when unused
        df = df.drop(columns=to_drop)

    # remove underscores and sni_ prefixes for latex rendering
    new_index = clean_strings(df.index)
    new_columns = clean_strings(df.columns)
    for k in ['method', 'accept_dsets', 'reject_dsets']:
        if pd.api.types.is_string_dtype(df[k]):
            df[k] = clean_strings(df[k])
    df = df.rename(
        index={k: v for k, v in zip(df.index, new_index)},
        columns={k: v for k, v in zip(df.columns, new_columns)},
    )
    df = df.sort_values(clean_strings(header_keys))

    latex_table = df.to_latex(
        index=False,
        float_format='{:0.3f}'.format,  # 3 std digits
    )
    return df, latex_table


def clean_strings(dset_names):
    new_names = [k.replace('sni_', '').replace('_', '-') if isinstance(k, str) else k for k in dset_names]
    new_names = [k.replace('fineaccept', 'fa').replace('finereject', 'fr') if isinstance(k, str) else k for k in new_names]
    _names = []
    for k in new_names:
        if isinstance(k, str):
            if k != 'alpaca':
                _names.append(k.upper())
            else:
                _names.append(k.capitalize())
    return _names


def df_to_latex(df):
    latex_table = df.to_latex(index=False)
    print(latex_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweepdir', type=str, default=None, help='directory to merge evals from')
    config = parser.parse_args()
    assert config.sweepdir is not None
    assert os.path.exists(config.sweepdir)
    main(config)
