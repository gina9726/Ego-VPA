
import yaml

def load_base_cfg():
    with open('configs/base.yml', 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)
    return cfg

def load_cfg(cfg_file):
    cfg = load_base_cfg()
    with open(cfg_file, 'r') as fp:
        exp_cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    cfg['model'].update(exp_cfg.get('model', {}))
    cfg['data'].update(exp_cfg.get('data', {}))
    cfg['training'].update(exp_cfg.get('training', {}))
    dataset = cfg['data'].get('dataset')
    exp = exp_cfg.get('exp', 'debug')
    cfg['output_dir'] = f'runs/{dataset}/{exp}'
    cfg['save_dir'] = exp_cfg.get('save_dir', 'runs/debug')
    return cfg

