from torch.optim.optimizer import Optimizer

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        raise NotImplementedError


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1):
        self.milestones = milestones
        self.gamma = gamma
        super(CustomScheduler, self).__init__(optimizer, last_epoch)
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch == self.milestones[0]:
            self.lr *= self.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            if len(self.milestones) > 1:
                self.milestones.pop(0)

    def get_lr(self):
        return [self.lr]

