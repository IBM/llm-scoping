import argparse
from glob import glob
import json
import os


def main(config):
    exp_name = os.path.splitext(os.path.basename(config.run_json))[0]
    expdir = os.path.join(os.environ['STEERING_SCRATCH_DIR'], 'exps', exp_name)
    assert os.path.exists(expdir), f'{expdir} does not exist'

    # get subdirectories to evaluate
    config_paths = glob(os.path.join(expdir, 'config.json'))
    if len(config_paths) > 0:
        subdirs = list(set([os.path.dirname(cp) for cp in config_paths]))
        assert len(subdirs) == 1
    else:
        config_paths = glob(os.path.join(expdir, '*', 'config.json'))
        subdirs = list(set([os.path.dirname(cp) for cp in config_paths]))
        assert len(subdirs) > 0

    program = "eval_reject_perf.py"
    if "probe" in exp_name:
        program = "eval_probe_reject.py"

    eval_config = {
            'program': program,
        'args': {
            'system_prompt': True,
            'regenerate': True,
            'dataset': [
                'sni_sa',
                # 'sni_sa_ood',
                'sni_tld',
                # 'sni_tld_ood',
                'sni_sc',
                # 'sni_sc_ood',
                'sni_dg',
                # 'sni_dg_ood',
                'sni_s',
                # 'sni_s_ood',
                'sni_tc',
                # 'sni_tc_ood',
                'sni_qa',
                # 'sni_qa_ood',
                'sni_pe',
                # 'sni_pe_ood',
                'gsm8k',
                'alpaca',
            ],
            'evaldir': subdirs,
        },
    }
    # eval_config = {
    #     'program': program,
    #     'args': {
    #         'system_prompt': True,
    #         'regenerate': True,
    #         'dataset': [
    #             'sni_s'
    #         ],
    #         'evaldir': subdirs,
    #     },
    # }

    dirname = os.path.dirname(config.run_json)
    savepath = 'eval_' + os.path.basename(config.run_json)
    savepath = os.path.join(dirname, savepath)
    with open(savepath, 'w') as f:
        json.dump(eval_config, f, indent=0)
        print( "eval_config", eval_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_json', type=str, default="sweeps/2025Feb/debug.json", help='directory to generate eval yaml for')
    config = parser.parse_args()
    assert config.run_json is not None
    assert os.path.exists(config.run_json)
    assert os.path.splitext(config.run_json)[1] == '.json'
    main(config)
