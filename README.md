# Rethinking Class Activation Mapping for Weakly Supervised Object Localization [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600613.pdf)][[Project Page](https://won-bae.github.io/rethinking-cam-wsol/)]


## Datasets
Create symlinks for datasets: CUB-200-2011 and ImageNet-1k.
Note that the name of the directory for CUB should be 'CUB_200_2011'.

```bash
cd /path/to/this/repo
mkdir data && cd data
ln -s /path/to/cub/dataset
ln -s /path/to/imagenet/dataset
```

## Setup
Setup the pipeline by installing dependencies, pretrained models and utils.
```bash
cd /path/to/this/repo
bash scripts/setup.sh --data_setup
```

## Train
To train a model, use the command below. Tag can be an arbitrary string.

```bash
cd /path/to/this/repo
bash scripts/run_train.sh --config_path configs/vgg16_tap.yml --tag vgg16_tap --train_dir /path/to/train/dir
```

## Eval
To eval a model, use the command below. Note that checkpoint dir has to be synchronized with train_dir used for training.

```bash
cd /path/to/this/repo
bash scripts/run_eval.sh --config_path configs/baseline/vgg16_tap.yml --tag vgg16_tap --checkpoint_dir /path/to/checkpoint/dir
```
**NOTE**: NWC and PaS can be used in evaluation time. In configs/vgg16_tap.yml, NWC and PaS can be employed simply changing paraameters under 'eval'. For NWC, change 'truncate' to True. For PaS, change 'percentile' to 90 and 'loc_threshold' to 0.35.

## TODO
Model and configs for the other backbones such as Resnet50-SE, Inception need to be added.


## Citation
If you use this code or model for your research, please cite:

    @InProceedings{bae2020rethinkingCAM,
      author = {Wonho Bae and Junhyug Noh and Gunhee Kim},
      title = {Rethinking Class Activation Mapping for Weakly Supervised Object Localization},
      booktitle = {The European Conference on Computer Vision (ECCV)},
      month = {August},
      year = {2020}
    }

## Acknowledgment
Many of core functions are borrowed from [ADL](https://github.com/junsukchoe/ADL).
