import argparse
from engine import *
from models import build_model
import os
import warnings
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import json
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

device = torch.device('cuda')
model = build_model(args)
model = model.to(device)
checkpoint = torch.load(args.weight_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.eval()
    # create the pre-processing transform
transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

with open('../../ds/dronebird/test.json', 'r') as f:
    img_list = json.load(f)
with torch.no_grad():
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    i = 0
    for img_path in img_list:
        img_raw = Image.open(img_path).convert('RGB')
        img = transform(img_raw)
        img = img.unsqueeze(0)
        img = img.to(device)
        outputs = model(img)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        gt = h5py.File(img_path.replace('data', 'annotation').replace('.jpg', '.h5'), 'r')['density'][:]
        gt_cnt = gt.sum()
        # 0.5 is used by default
        threshold = 0.5
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        i += 1
        print('\r[{:{}}/{}] mae: {:.4f}, mse: {:.4f}, pred: {:.4f}, gt: {:.4f}'.format(i, len(str(len(img_list))), len(img_list), mae, mse, predict_cnt, gt_cnt), end='')
        maes.append(float(mae))
        mses.append(float(mse))
    print()
    # calc MAE, MSE
    print('max mae: {:.4f}, min mae: {:.4f}'.format(max(maes), min(maes)))
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))