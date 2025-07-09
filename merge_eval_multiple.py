import argparse
import itertools
import os

import pandas as pd

import merge_eval
import merge_eval_probe


SCRATCH_DIR = os.environ['STEERING_SCRATCH_DIR']


# merge multiple sweeps, then get merged latex table for each
# use merge_eval_probe when path has "probe" in it
def main(config):
    dfs = []
    for i, sn in enumerate(config.sweepnames):
        sweepdir = os.path.join(SCRATCH_DIR, 'exps', sn)
        sweep_config = argparse.Namespace(sweepdir=sweepdir)
        basename = os.path.basename(sweepdir)
        if 'probe' in basename:
            if i == 0:
                df = merge_eval_probe.main(sweep_config, base=True)
                df['method'] = 'sys'
                df, latex_table = merge_eval_probe.extract_reject(df, average_style=config.average_style)
                dfs.append(df)
            df = merge_eval_probe.main(sweep_config)
            df, latex_table = merge_eval_probe.extract_reject(df, average_style=config.average_style)
        else:
            if i == 0:
                df = merge_eval.main(sweep_config, base=True)
                df['method'] = 'sys'
                df, latex_table = merge_eval.extract_string_reject(df, average_style=config.average_style)
                dfs.append(df)
            df = merge_eval.main(sweep_config)
            df, latex_table = merge_eval.extract_string_reject(df, average_style=config.average_style)
                
        dfs.append(df)

    df = pd.concat(dfs)
    header_keys = ['METHOD', 'ACCEPT-DSETS', 'REJECT-DSETS']
    rest = [k for k in df.columns if k in set(df.columns).difference(header_keys)]
    df = df[header_keys + rest]
    df = df.rename(columns={'METHOD': 'Method', 'ACCEPT-DSETS': 'Accept', 'REJECT-DSETS': 'Reject'})
    df.to_csv('multiple_results.csv')

    latex_table = df.to_latex(
        index=False,
        float_format='{:0.3f}'.format,  # 3 std digits
    )
    latex_lines = latex_table.split('\n')
    latex_lines = [l for l in latex_lines if l != '']
    header = ['\\begin{table}[h]', '\\scriptsize', '\\begin{center}'] + latex_lines[:3]
    footer = latex_lines[-2:] + ['\\end{center}', '\\end{table}']
    middle = latex_lines[4:-2]  # leftover line is a \\midrule that we'll add back
    num_sweeps = len(config.sweepnames) + 1  # extra for base
    assert len(middle) % num_sweeps == 0
    num_lines = len(middle) // num_sweeps
    middles = [middle[num_lines * i:num_lines * (i+1)] for i in range(num_sweeps)]
    middles = [['\\midrule'] + mid for mid in middles]
    middle = list(itertools.chain(*middles))
    latex_lines = header + middle + footer
    latex_table = '\n'.join(latex_lines)
    print(latex_table)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweepnames', type=str, nargs='+', default=None, help='directories to merge evals from')
    parser.add_argument('--average_style', type=str, default=None, choices=[None, 'ood', 'category'], help='averaging mode to use')
    config = parser.parse_args()
    for sn in config.sweepnames:
        assert os.path.exists(os.path.join(SCRATCH_DIR, 'exps', sn))
    main(config)
