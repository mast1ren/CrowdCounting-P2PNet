import cv2
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import scipy.io as io
import json

class DRONEBIRD(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.json"
        self.eval_list = "val.json"
        # there may exist multiple list files
        # self.img_list_file = self.train_lists.split(',')

        if train:
            with open(os.path.join(self.root_path, self.train_lists), 'r') as f:
                self.img_list = json.load(f)
            # self.img_list_file = self.train_lists.split(',')
        else:
            with open(os.path.join(self.root_path, self.eval_list), 'r') as f:
                self.img_list = json.load(f)

            # self.img_list_file = self.eval_list.split(',')

        # self.img_map = {}
        # # loads the image/gt pairs
        # for _, train_list in enumerate(self.img_list_file):
        #     train_list = train_list.strip()
        #     with open(os.path.join(self.root_path, train_list)) as fin:
        #         for line in fin:
        #             if len(line) < 2: 
        #                 continue
        #             line = line.strip().split()
        #             self.img_map[os.path.join(self.root_path, line[0].strip())] = \
        #                             os.path.join(self.root_path, line[1].strip())
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        img_path = os.path.join(self.root_path, img_path)
        gt_path = os.path.join(os.path.dirname(img_path).replace('images', 'ground_truth'), 'GT_'+os.path.basename(img_path).replace('.jpg', '.mat'))
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(os.path.basename(img_path).split('.')[0][3:])
            # image_id = int(img_path.split('/')[-3].split('_')[1]+img_path.split('/')[-1].split('.')[0])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
        if target[0]['point'].shape[0] < 10:
            return None, target
        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    # load mat file
    mat = io.loadmat(gt_path)['locations']
    for line in mat:
        x = float(line[0])
        y = float(line[1])
        points.append([x, y])

    # with open(gt_path) as f_label:
    #     for line in f_label:
    #         x = float(line.strip().split(' ')[0])
    #         y = float(line.strip().split(' ')[1])
    #         points.append([x, y])

    return img, np.array(points)

# random crop augumentation
def random_crop(img, den, num_patch=2):
    half_h = 1024
    half_w = 1024
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den