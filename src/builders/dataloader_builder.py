import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.core import datasets
from src.utils.util import normalization_params

DATASETS = {
    'cub': datasets.CUB200,
    'imagenet': datasets.ImageNet,
}


def build(data_config, mode, log):
    # Define arguments
    data_name = data_config['name']
    root = data_config['root']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']

    train = True if mode == 'train' else False
    shuffle = data_config.get('shuffle', train)
    random_crop = shuffle # if shuffle is false in training, random_crop is false

    transform_config = data_config['transform']
    transform = compose_transforms(data_name, transform_config, mode, random_crop)

    # Initialize a dataset
    dataset = DATASETS[data_name](root, log, mode=mode, transform=transform,
                                  transform_config=transform_config)

    # Initialize a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    return dataloader, dataset

def compose_transforms(data_name, transform_config, mode, random_crop=True):
    mean, std = normalization_params(data_name)
    image_size = transform_config['image_size']
    crop_size = transform_config.get('crop_size', image_size)

    if mode == 'train':
        if random_crop:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transform

