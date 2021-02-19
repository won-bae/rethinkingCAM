from src.core.models import vgg

MODELS = {
    'vgg16': vgg.vgg16,
}

def build(model_config, mode, log):
    if 'backbone' not in model_config:
        log.error('Specify a backbone name'); exit()
    backbone = model_config['backbone']

    # Add params for a backbone model

    model_params = {
        'pretrained': False if mode == 'eval' else model_config['pretrained'],
        'num_classes': model_config['num_classes'],
        'transform_input': model_config.get('transform_input', False),
        'init_weights': True, 'progress': True, 'log': log,
        'batch_norm': model_config.get('batch_norm', False),
        'avgpool_threshold': model_config.get('avgpool_threshold', None),
        'bias': model_config.get('bias', True),
        'last_layer': model_config.get('last_layer', 'fc')
    }

    # Build a model
    if backbone in MODELS:
        model = MODELS[backbone](**model_params)
    else:
        log.error(
            'Specify valid backbone or model_type among {}.'.format(MODELS.keys())
        ); exit()

    return model
