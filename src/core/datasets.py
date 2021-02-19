import os
import torch
import numpy as np
import pandas as pd
import re
import requests

from bs4 import BeautifulSoup
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset


class CUB200(Dataset):
    def __init__(self, root, log, mode, transform=None, transform_config=None):
        self.root = root
        self.log = log
        self.mode = mode
        if transform_config is not None:
            image_size = float(transform_config.get('image_size', 256))
            crop_size = transform_config.get('crop_size', 224)
            shift = (image_size - crop_size) // 2
        self.data = self._load_data(image_size, crop_size, shift)
        self.transform = transform

    def _load_data(self, image_size, crop_size, shift):
        self._labelmap_path = os.path.join(self.root, 'CUB_200_2011', 'classes.txt')

        paths = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            sep=' ', names=['id', 'path'])
        labels = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ', names=['id', 'label'])
        splits = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['id', 'is_train'])
        orig_image_sizes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_sizes.txt'),
            sep=' ', names=['id', 'width', 'height'])
        bboxes = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
            sep=' ', names=['id', 'x', 'y', 'w', 'h'])

        resized_xmin = np.maximum(
            (bboxes.x / orig_image_sizes.width * image_size - shift).astype(int), 0)
        resized_ymin = np.maximum(
            (bboxes.y / orig_image_sizes.height * image_size - shift).astype(int), 0)
        resized_xmax = np.minimum(
            ((bboxes.x + bboxes.w - 1) / orig_image_sizes.width * image_size - shift).astype(int),
            crop_size - 1)
        resized_ymax = np.minimum(
            ((bboxes.y + bboxes.h - 1) / orig_image_sizes.height * image_size - shift).astype(int),
            crop_size - 1)

        resized_bboxes = pd.DataFrame({'id': paths.id,
                                       'xmin': resized_xmin.values,
                                       'ymin': resized_ymin.values,
                                       'xmax': resized_xmax.values,
                                       'ymax': resized_ymax.values})

        data = paths.merge(labels, on='id')\
                    .merge(splits, on='id')\
                    .merge(resized_bboxes, on='id')

        if self.mode == 'train':
            data = data[data.is_train == 1]
        else:
            data = data[data.is_train == 0]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'CUB_200_2011/images', sample.path)
        image = Image.open(path).convert('RGB')
        label = sample.label - 1 # label starts from 1
        gt_box = torch.tensor(
            [sample.xmin, sample.ymin, sample.xmax, sample.ymax])

        if self.transform is not None:
            image = self.transform(image)

        return (image, label, gt_box)

    @property
    def class_id_to_name(self):
        if hasattr(self, '_class_id_to_name'):
            return self._class_id_to_name
        labelmap = pd.read_csv(self._labelmap_path, sep=' ', names=['label', 'name'])
        labelmap['label'] = labelmap['label'].apply(lambda x: x - 1)
        self._class_id_to_name = labelmap.set_index('label')['name'].to_dict()
        return self._class_id_to_name

    @property
    def class_name_to_id(self):
        if hasattr(self, '_class_name_to_id'):
            return self._class_name_to_id
        self._class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        return self._class_name_to_id

    @property
    def class_to_images(self):
        if hasattr(self, '_class_to_images'):
            return self._class_to_images
        self.log.warn('Create index...')
        self._class_to_images = defaultdict(list)
        for idx in tqdm(range(len(self))):
            sample = self.data.iloc[idx]
            label = sample.label - 1
            self._class_to_images[label].append(idx)
        self.log.warn('Done!')
        return self._class_to_images


class ImageNet(Dataset):
    def __init__(self, root, log, mode, transform=None, transform_config=None):
        self.root = root
        self.log = log
        self.mode = mode
        if transform_config is not None:
            self.image_size = float(transform_config.get('image_size', 256))
            self.crop_size = transform_config.get('crop_size', 224)
            self.shift = (self.image_size - self.crop_size) // 2
        self._load_data()
        self.transform = transform

    def _load_data(self):
        self._labelmap_path = os.path.join(
            self.root, 'imagenet', 'imagenet1000_clsidx_to_labels.txt')

        if self.mode == 'train':
            self.path = os.path.join(self.root, 'imagenet/raw-data/train')
            self.metadata = pd.read_csv(
                os.path.join(self.root, 'imagenet', 'train.txt'),
                sep=' ', names=['path', 'label'])
        else:
            self.path = os.path.join(self.root, 'imagenet/raw-data/validation')
            self.metadata = pd.read_csv(
                os.path.join(self.root, 'imagenet', 'val.txt'),
                sep='\t', names=['path', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])
            self.wnids = pd.read_csv(
                os.path.join(self.root, 'imagenet', 'wnids.txt'), names=['dir_name'])

    def _preprocess_bbox(self, origin_bbox, orig_image_size):
        xmin, ymin, xmax, ymax = origin_bbox
        orig_width, orig_height = orig_image_size
        resized_xmin = np.maximum(
            int(xmin / orig_width * self.image_size - self.shift), 0)
        resized_ymin = np.maximum(
            int(ymin / orig_height * self.image_size - self.shift), 0)
        resized_xmax = np.minimum(
            int(xmax / orig_width * self.image_size - self.shift), self.crop_size - 1)
        resized_ymax = np.minimum(
            int(ymax / orig_height * self.image_size - self.shift), self.crop_size - 1)
        return [resized_xmin, resized_ymin, resized_xmax, resized_ymax]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        if self.mode == 'train':
            image_path = os.path.join(self.path, sample.path)
        else:
            image_path = os.path.join(
                self.path, self.wnids.iloc[int(sample.label)].dir_name, sample.path)
        image = Image.open(image_path).convert('RGB')
        label = sample.label

        # preprocess bbox
        if self.mode == 'train':
            gt_box = torch.tensor([0., 0., 0., 0.])
        else:
            origin_box = [sample.xmin, sample.ymin, sample.xmax, sample.ymax]
            gt_box = torch.tensor(
                self._preprocess_bbox(origin_box, image.size))

        if self.transform is not None:
            image = self.transform(image)

        return (image, label, gt_box)

    @property
    def class_id_to_name(self):
        if hasattr(self, '_class_id_to_name'):
            return self._class_id_to_name
        with open(self._labelmap_path, 'r') as f:
            self._class_id_to_name = eval(f.read())
        return self._class_id_to_name

    @property
    def class_name_to_id(self):
        if hasattr(self, '_class_name_to_id'):
            return self._class_name_to_id
        self._class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}
        return self._class_name_to_id

    @property
    def wnid_list(self):
        if hasattr(self, '_wnid_list'):
            return self._wnid_list
        self._wnid_list = self.wnids.dir_name.tolist()
        return self._wnid_list

    @property
    def class_to_images(self):
        if hasattr(self, '_class_to_images'):
            return self._class_to_images
        self.log.warn('Create index...')
        self._class_to_images = defaultdict(list)
        for idx in tqdm(range(len(self))):
            sample = self.metadata.iloc[idx]
            label = sample.label
            self._class_to_images[label].append(idx)
        self.log.warn('Done!')
        return self._class_to_images

    def verify_wnid(self, wnid):
        is_valid = bool(re.match(u'^[n][0-9]{8}$', wnid))
        is_terminal = bool(wnid in self.wnids.dir_name.tolist())
        return is_valid and is_terminal

    def get_terminal_wnids(self, wnid):
        page = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={}&full=1".format(wnid))
        str_wnids = str(BeautifulSoup(page.content, 'html.parser'))
        split_wnids = re.split('\r\n-|\r\n', str_wnids)
        return [_wnid for _wnid in split_wnids if self.verify_wnid(_wnid)]

    def get_image_ids(self, wnid):
        terminal_wnids = self.get_terminal_wnids(wnid)

        image_ids = set()
        for terminal_wnid in terminal_wnids:
            class_id = self.wnid_list.index(terminal_wnid)
            image_ids |= set(self.class_to_images[class_id])

        return list(image_ids)

