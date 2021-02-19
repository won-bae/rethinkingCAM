from torch import optim
from torch.optim.optimizer import Optimizer
from src.core.schedulers import CustomScheduler


SCHEDULERS = {
    'multi': optim.lr_scheduler.MultiStepLR,
    'custom': CustomScheduler
}

def build(train_config, optimizer, log):
    if 'lr_schedule' not in train_config:
        log.warn('No scheduler is specified.')
        return None

    schedule_config = train_config['lr_schedule']
    scheduler_name = schedule_config.pop('name', 'multi_step')
    schedule_config['optimizer'] = optimizer

    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**schedule_config)
    else:
        log.error(
            'Specify a valid scheduler name among {}.'.format(SCHEDULERS.keys())
        ); exit()

    log.infov('{} scheduler is built'.format(scheduler_name.upper()))
    return scheduler
