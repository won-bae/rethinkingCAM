import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import util, metrics
from src.utils.meters import AverageEpochMeter
from src.builders import model_builder, dataloader_builder, checkpointer_builder,\
                         optimizer_builder, criterion_builder, scheduler_builder


class BaseEngine(object):

    def __init__(self, mode, config_path, log, train_dir, eval_dir=None):
        self.log = log
        if not train_dir:
            self.log.error('Specify tag for train directory.'); exit()

        # Load configurations
        if isinstance(config_path, str):
            config = util.load_config(config_path)
        elif isinstance(config_path, dict):
            config = config_path # config_path can be config itself (for notebook).
        else:
            raise ValueError('config_path must be either str or dict.')

        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        self.eval_standard = 'top1_loc'

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        if device == 'cpu':
            self.log.warn('GPU is not available.')
        else:
            self.log.warn('GPU is available.')

        # Assign required directories
        self.eval_dir = eval_dir
        if eval_dir is not None:
            log_dir = os.path.join(eval_dir, mode + '_log')
        else:
            log_dir = os.path.join(train_dir, mode + '_log')
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, mode, config_path, log, train_dir, eval_dir=None):
        super(Engine, self).__init__(mode, config_path, log, train_dir, eval_dir)

        # Build a dataloader
        self.dataloader, self.dataset = dataloader_builder.build(
            self.data_config, mode, self.log)

        # Build a model
        self.model = model_builder.build(self.model_config, mode, self.log)
        if mode == 'train' and torch.cuda.device_count() > 1:
            self.model = util.DataParallel(self.model)
        self.model.to(self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer, self.scheduler = None, None
        if mode is 'train':
            self.optimizer = optimizer_builder.build(
                self.train_config, self.model.parameters(), self.log)
            self.scheduler = scheduler_builder.build(
                self.train_config, self.optimizer, self.log)
            self.criterion = criterion_builder.build(self.train_config, self.log)

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
            mode, train_dir, self.model, self.log, self.optimizer,
            self.scheduler, self.eval_standard)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(checkpoint_path, use_latest=False)

    def train(self):
        start_epoch = 0 if 'epoch' not in self.misc else int(self.misc['epoch'])
        num_epochs = self.train_config.get('num_epochs', 50)
        num_step, next_checkpoint_step = 0, self.train_config.get('checkpoint_step', 10000)

        self.log.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_start = time.time()
            train_loss, num_step, next_checkpoint_step =\
                self._train_one_epoch(epoch, num_step, next_checkpoint_step)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.log.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, train_loss))

            self.writer.add_scalar('Train/learning_rate', lr, global_step=num_step)

            if self.scheduler is not None:
                self.scheduler.step()

    def _train_one_epoch(self, epoch, num_step, next_checkpoint_step):
        num_batches = len(self.dataloader)
        loss_meter = AverageEpochMeter('Loss', ':.4f')
        self.model.train()

        for i, (images, labels, _) in enumerate(self.dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward propagation
            results = self.model(images)
            self.optimizer.zero_grad()

            # Compute a loss
            results['labels'] = labels
            losses = self.criterion(**results)
            loss = losses['loss']

            # Backward propagation
            loss.backward()
            if self.model_config['pretrained']:
                grad_ratio = self.train_config.get('grad_ratio', 0.1)
                self.model.downscale_gradient(grad_ratio)
            self.optimizer.step()

            # Loss
            batch_size = images.size(0)
            loss_val = loss.item()
            loss_meter.update(loss.item(), batch_size)
            self.log.info(
                '[Epoch {}] Train batch {}/{} = loss: {:.4f}'.format(
                    epoch, i+1, num_batches, loss_val))

            # Save checkpoint
            num_step += batch_size
            if num_step >= next_checkpoint_step:
                self.checkpointer.save(epoch, num_step)
                next_checkpoint_step += self.train_config.get('checkpoint_step', 30000)
                self.log.info(
                    'A checkpoint at epoch = {}, num_step = {} has been saved'.format(epoch, num_step))

        self.writer.add_scalar('Train/loss', loss_val, global_step=num_step)

        torch.cuda.empty_cache()
        return loss_meter.compute(), num_step, next_checkpoint_step


    def evaluate(self):
        def _get_misc_info(misc):
            infos = ['epoch', 'num_step', 'checkpoint_path']
            return (misc[info] for info in infos)

        epoch, num_step, current_checkpoint_path = _get_misc_info(self.misc)
        last_evaluated_checkpoint_path = None

        while True:
            if last_evaluated_checkpoint_path == current_checkpoint_path:
                self.log.warn('Found already evaluated checkpoint. Will try again in 60 seconds.')
                time.sleep(60)
            else:
                eval_metrics = self._eval_one_epoch(
                    epoch, num_step, **self.eval_config)
                self.checkpointer.save(
                    epoch, num_step, eval_metrics, self.eval_dir)
                last_evaluated_checkpoint_path = current_checkpoint_path

            # Reload checkpoint. Break if file path was given as checkpoint path.
            checkpoint_path = self.model_config.get('checkpoint_path', '')
            if os.path.isfile(checkpoint_path): break
            misc = self.checkpointer.load(checkpoint_path, use_latest=True)
            epoch, num_step, current_checkpoint_path = _get_misc_info(misc)


    def _eval_one_epoch(self, epoch, num_step, iou_threshold, loc_threshold=0.2,
                             truncate=False, percentile=100):
        # Initiate meters - Top1-Cls, GT-known Loc, Top1-Loc
        top1_cls_meter = AverageEpochMeter('Top-1 Cls')
        top5_cls_meter = AverageEpochMeter('Top-5 Cls')
        gt_loc_meter = AverageEpochMeter('GT-Known Loc with {}'.format(loc_threshold))
        top1_loc_meter = AverageEpochMeter('Top-1 Loc {}'.format(loc_threshold))

        num_batches = len(self.dataloader)
        self.model.eval()
        for i, (images, labels, gt_boxes) in enumerate(self.dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward propagation (assume multi-class classification)
                results = self.model(images)

                # Compute Top-1 and Top-5 Cls
                batch_size = images.size(0)
                predictions = results['preds']
                top1_cls, top5_cls = metrics.topk_accuracy(predictions, labels, topk=(1,5))
                top1_cls_meter.update(top1_cls, batch_size)
                top5_cls_meter.update(top5_cls, batch_size)

                # Class activation map
                gt_cams, _ = util.cam(
                    self.model, labels=labels, truncate=truncate)
                unnormalized_images = util.unnormalize_images(images, self.data_config['name'])

                bboxes, blended_bboxes =\
                    util.extract_bbox(unnormalized_images, gt_cams, gt_boxes,
                                      loc_threshold, percentile=percentile)
                gt_loc, top1_loc = metrics.loc_accuracy(
                    predictions, labels, gt_boxes, bboxes, iou_threshold)
                gt_loc_meter.update(gt_loc, batch_size)
                top1_loc_meter.update(top1_loc, batch_size)

            self.log.info('[Epoch {}] Evaluation batch {}/{}'.format(epoch, i+1, num_batches))

        top1_cls = top1_cls_meter.compute()
        top5_cls = top5_cls_meter.compute()
        gt_loc = gt_loc_meter.compute()
        top1_loc = top1_loc_meter.compute()
        self.log.warn('[Epoch {}] Evaluation Results'.format(epoch))
        self.log.infov('  Top-1 Cls: {:.4f}'.format(top1_cls))
        self.log.infov('  Top-5 Cls: {:.4f}'.format(top5_cls))
        self.log.infov('  GT-known Loc: {:.4f}'.format(gt_loc))
        self.log.infov('  Top-1 Loc: {:.4f}'.format(top1_loc))
        self.log.infov('  Loc Threshold: {}'.format(loc_threshold))

        eval_metrics = {
            'top1_cls': top1_cls,
            'top5_cls': top5_cls,
            'gt_loc': gt_loc,
            'top1_loc': top1_loc}

        self.writer.add_scalar(
                'Eval/top1_cls', top1_cls, global_step=num_step)
        self.writer.add_scalar(
                'Eval/top5_cls', top5_cls, global_step=num_step)
        self.writer.add_scalar(
            'Eval/gt_loc', gt_loc, global_step=num_step)
        self.writer.add_scalar(
            'Eval/top1_loc', top1_loc, global_step=num_step)
        self.writer.add_scalar(
            'Eval/eval_loc_thresholds', loc_threshold, global_step=num_step)

        torch.cuda.empty_cache()
        return eval_metrics

