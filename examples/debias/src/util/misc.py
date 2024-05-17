def pop_cfg_name(cfg):
    assert('name' in cfg)
    cfg = cfg.copy()
    name = cfg['name']
    del cfg['name']
    return name, cfg


def pop_cfg_name_and_seed(cfg):
    assert('name' in cfg)
    cfg = cfg.copy()
    name = cfg['name']
    del cfg['name']
    if 'seed' in cfg:
        seed = cfg['seed']
        del cfg['seed']
    else:
        seed = 0
    return name, seed, cfg
