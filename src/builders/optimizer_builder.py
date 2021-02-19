from torch import optim

OPTIMIZERS = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def build(train_config, model_params, log):
    if 'optimizer' not in train_config:
        log.error('Specify an optimizer.'); exit()

    optim_config = train_config['optimizer']
    optimizer_name = optim_config.pop('name', 'sgd')
    optim_config['params'] = model_params

    if optimizer_name in OPTIMIZERS:
        optimizer = OPTIMIZERS[optimizer_name](**optim_config)
    else:
        log.error(
            'Specify a valid optimizer name among {}.'.format(OPTIMIZERS.keys())
        ); exit()

    log.infov('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer
