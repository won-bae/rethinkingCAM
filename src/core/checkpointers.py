import os
import time
import torch
from collections import OrderedDict


class CustomCheckpointer(object):
    def __init__(self, mode, train_dir, model, log,
                 optimizer=None, scheduler=None, eval_standard='top1_loc'):
        self.mode = mode
        self.log = log
        self.train_dir = train_dir

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if mode == 'train':
            self.log.infov(
                'Directory {} to save checkpoints is ready.'.format(self.train_dir))
        else:
            self.eval_standard = eval_standard
            self.best_eval_metric = 0

        self.log.infov('Checkpointer is built.')

    def save(self, epoch, num_step, eval_metrics=None, eval_dir=None):
        '''Save the best checkpoint and the given checkpoint in val/eval and train, respectively.'''

        # Determine a checkpoint
        if self.mode != 'train':
            if self.eval_standard in eval_metrics:
                eval_metric = eval_metrics[self.eval_standard]
                if eval_metric <= self.best_eval_metric:
                    return
                self.best_eval_metric = eval_metric
                checkpoint_path = os.path.join(eval_dir, 'checkpoint_best.pth')
            else:
                self.log.error('Wrong eval standard is provided.'); exit()
        else:
            if num_step is None:
                checkpoint_path = os.path.join(
                    self.train_dir, 'checkpoint' + '_' + str(epoch) + '.pth')
            else:
                checkpoint_path = os.path.join(
                    self.train_dir, 'checkpoint' + '_' + str(epoch) + '_' + str(num_step) + '.pth')

        # Save the checkpoint
        model_params = {'epoch': epoch, 'num_step': num_step}
        if torch.cuda.device_count() > 1:
            model_params['model_state_dict'] = self.model.module.state_dict()
        else:
            model_params['model_state_dict'] = self.model.state_dict()

        if self.mode == 'train':
            model_params['optimizer_state_dict'] = self.optimizer.state_dict()
            model_params['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(model_params, checkpoint_path)

        # Update the checkpoint record
        if self.mode != 'train':
            updated_eval_metrics = OrderedDict(
                [('epoch', epoch),
                 ('num_step', num_step)]
            )
            updated_eval_metrics.update(eval_metrics)

            self._record_best_checkpoint(updated_eval_metrics, eval_dir)
        else:
            if num_step is None:
                self._record_last_epoch_checkpoint(checkpoint_path)
            else:
                self._record_last_checkpoint(checkpoint_path)

    def load(self, checkpoint_path=None, use_latest=True):
        strict = True
        if self.mode == 'train':
            if checkpoint_path is not None:
                strict = False
            if self._has_checkpoint() and use_latest:
                # override argument with existing checkpoint
                checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path:
                self.log.info("No checkpoint found. Initializing model from scratch.")
                return {}
        else:
            if not checkpoint_path:
                while not self._has_checkpoint():
                    self.log.warn('No checkpoint available. Wait for 60 seconds.')
                    time.sleep(60)
                checkpoint_path = self._get_checkpoint_path()

        self.log.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint = self._load_checkpoint(checkpoint_path)

        self.model.load_state_dict(
            checkpoint.pop('model_state_dict'), strict=strict)

        if strict:
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.log.info("Loading optimizer from {}".format(checkpoint_path))
                self.optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.log.info("Loading scheduler from {}".format(checkpoint_path))
                self.scheduler.load_state_dict(checkpoint.pop('scheduler_state_dict'))

        return checkpoint

    def _has_checkpoint(self):
        record_path = os.path.join(self.train_dir, "last_checkpoint")
        return os.path.exists(record_path)

    def _get_checkpoint_path(self):
        record_path = os.path.join(self.train_dir, "last_checkpoint")
        try:
            with open(record_path, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            self.log.warn('If last_checkpoint file doesn not exist, maybe because \
                          it has just been deleted by a separate process.')
            last_saved = ''
        return last_saved

    def _record_last_checkpoint(self, last_checkpoint_path):
        record_path = os.path.join(self.train_dir, 'last_checkpoint')
        with open(record_path, 'w') as f:
            f.write(last_checkpoint_path)

    def _record_last_epoch_checkpoint(self, last_epoch_checkpoint_path):
        record_path = os.path.join(self.train_dir, 'last_epoch_checkpoint')
        with open(record_path, 'w') as f:
            f.write(last_epoch_checkpoint_path)

    def _record_best_checkpoint(self, best_checkpoint_info, eval_dir):
        record_path = os.path.join(eval_dir, 'best_checkpoint')
        with open(record_path, 'w') as f:
            f.write(str(best_checkpoint_info))

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        checkpoint['checkpoint_path'] = checkpoint_path
        return checkpoint
