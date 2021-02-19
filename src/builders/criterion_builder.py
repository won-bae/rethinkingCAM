from src.core.criterions import CrossEntropyLoss, CustomCriterion

CRITERIONS = {
    'cross_entropy': CrossEntropyLoss,
}

def build(train_config, log):
    criterion_params = train_config.get('criterion', {})

    criterion_name = criterion_params.pop('name', 'cross_entropy')
    criterion = CustomCriterion(criterion=CRITERIONS[criterion_name](**criterion_params))

    log.infov('{} criterion is built.'.format(criterion_name.upper()))
    return criterion


