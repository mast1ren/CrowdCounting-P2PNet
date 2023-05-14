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
import scipy.io as sio
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for P2PNet evaluation', add_help=False
    )

    # * Backbone
    parser.add_argument(
        '--backbone',
        default='vgg16_bn',
        type=str,
        help="name of the convolutional backbone to use",
    )

    parser.add_argument(
        '--row', default=2, type=int, help="row number of anchor points"
    )
    parser.add_argument(
        '--line', default=2, type=int, help="line number of anchor points"
    )

    parser.add_argument(
        '--weight_path', default='', help='path where the trained weights saved'
    )

    parser.add_argument(
        '--gpu_id', default=0, type=int, help='the gpu used for evaluation'
    )

    return parser


parser = argparse.ArgumentParser(
    'P2PNet evaluation script', parents=[get_args_parser()]
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

device = torch.device('cuda')
model = build_model(args)
model = model.to(device)
checkpoint = torch.load(args.weight_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.eval()
# create the pre-processing transform
transform = standard_transforms.Compose(
    [
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def get_seq_class(seq, set):
    backlight = [
        'DJI_0021',
        'DJI_0022',
        'DJI_0032',
        'DJI_0202',
        'DJI_0339',
        'DJI_0340',
        'DJI_0463',
        'DJI_0003',
    ]

    fly = [
        'DJI_0177',
        'DJI_0174',
        'DJI_0022',
        'DJI_0180',
        'DJI_0181',
        'DJI_0200',
        'DJI_0544',
        'DJI_0012',
        'DJI_0178',
        'DJI_0343',
        'DJI_0185',
        'DJI_0195',
        'DJI_0996',
        'DJI_0977',
        'DJI_0945',
        'DJI_0946',
        'DJI_0091',
        'DJI_0442',
        'DJI_0466',
        'DJI_0459',
        'DJI_0464',
    ]

    angle_90 = [
        'DJI_0179',
        'DJI_0186',
        'DJI_0189',
        'DJI_0191',
        'DJI_0196',
        'DJI_0190',
        'DJI_0070',
        'DJI_0091',
    ]

    mid_size = [
        'DJI_0012',
        'DJI_0013',
        'DJI_0014',
        'DJI_0021',
        'DJI_0022',
        'DJI_0026',
        'DJI_0028',
        'DJI_0028',
        'DJI_0030',
        'DJI_0028',
        'DJI_0030',
        'DJI_0034',
        'DJI_0200',
        'DJI_0544',
        'DJI_0463',
        'DJI_0001',
        'DJI_0149',
    ]

    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    # elif seq in cloudy:
    #     light = 'cloudy'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'

    # count = 'sparse'
    # loca = sio.loadmat(
    #     os.path.join(
    #         '../../nas-public-linkdata/ds/dronebird/',
    #         set,
    #         'ground_truth',
    #         'GT_img' + str(seq[-3:]) + '000.mat',
    #     )
    # )['locations']
    # if loca.shape[0] > 150:
    #     count = 'crowded'
    return light, angle, bird, size


with open('../../nas-public-linkdata/ds/dronebird/test.json', 'r') as f:
    img_list = json.load(f)
preds = [[] for i in range(10)]
gts = [[] for i in range(10)]
with torch.no_grad():
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}')
    )
    # run inference on all images to calc MAE
    maes = []
    mses = []
    i = 0
    for img_path in img_list:
        img_path = os.path.join('../../nas-public-linkdata/ds/dronebird/', img_path)
        seq = int(os.path.basename(img_path)[3:6])
        seq = 'DJI_' + str(seq).zfill(4)
        light, angle, bird, size = get_seq_class(seq, 'test')

        img_raw = Image.open(img_path).convert('RGB')
        img = transform(img_raw)
        img = img.unsqueeze(0)
        img = img.to(device)
        outputs = model(img)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[
            :, :, 1
        ][0]
        outputs_points = outputs['pred_points'][0]

        gt_path = os.path.join(
            os.path.dirname(img_path).replace('images', 'ground_truth'),
            'GT_' + os.path.basename(img_path).replace('.jpg', '.mat'),
        )
        gt_file = sio.loadmat(gt_path)['locations']
        # gt = h5py.File(img_path.replace('data', 'annotation').replace('.jpg', '.h5'), 'r')['density'][:]
        gt_e = gt_file.shape[0]
        count = 'crowded' if gt_e > 150 else 'sparse'

        # 0.5 is used by default
        threshold = 0.5
        points = (
            outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        )
        predict_cnt = int((outputs_scores > threshold).sum())
        pred_e = predict_cnt
        if light == 'sunny':
            preds[0].append(pred_e)
            gts[0].append(gt_e)
        elif light == 'backlight':
            preds[1].append(pred_e)
            gts[1].append(gt_e)
        if count == 'crowded':
            preds[2].append(pred_e)
            gts[2].append(gt_e)
        else:
            preds[3].append(pred_e)
            gts[3].append(gt_e)
        if angle == '60':
            preds[4].append(pred_e)
            gts[4].append(gt_e)
        else:
            preds[5].append(pred_e)
            gts[5].append(gt_e)
        if bird == 'stand':
            preds[6].append(pred_e)
            gts[6].append(gt_e)
        else:
            preds[7].append(pred_e)
            gts[7].append(gt_e)
        if size == 'small':
            preds[8].append(pred_e)
            gts[8].append(gt_e)
        else:
            preds[9].append(pred_e)
            gts[9].append(gt_e)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_e)
        mse = (predict_cnt - gt_e) * (predict_cnt - gt_e)
        i += 1
        print(
            '\r[{:{}}/{}] mae: {:.4f}, mse: {:.4f}, pred: {:.4f}, gt: {:.4f}'.format(
                i, len(str(len(img_list))), len(img_list), mae, mse, predict_cnt, gt_e
            ),
            end='',
        )
        maes.append(float(mae))
        mses.append(float(mse))
    print()
    # calc MAE, MSE
    with open('result.txt', 'w') as f:
        f.write('max mae: {:.4f}, min mae: {:.4f}\n'.format(max(maes), min(maes)))
        print('max mae: {:.4f}, min mae: {:.4f}'.format(max(maes), min(maes)))
        mae = np.mean(maes)
        mse = np.sqrt(np.mean(mses))
        print('mae: {:.4f}, mse: {:.4f}'.format(mae, mse))
        f.write('mae: {:.4f}, mse: {:.4f}\n'.format(mae, mse))

        attri = [
            'sunny',
            'backlight',
            'crowded',
            'sparse',
            '60',
            '90',
            'stand',
            'fly',
            'small',
            'mid',
        ]
        for i in range(10):
            # print(len(preds[i]))
            if len(preds[i]) == 0:
                continue
            print(
                '{}: MAE:{}. RMSE:{}.'.format(
                    attri[i],
                    mean_absolute_error(preds[i], gts[i]),
                    np.sqrt(mean_squared_error(preds[i], gts[i])),
                )
            )
            f.write(
                '{}: MAE:{}. RMSE:{}.\n'.format(
                    attri[i],
                    mean_absolute_error(preds[i], gts[i]),
                    np.sqrt(mean_squared_error(preds[i], gts[i])),
                )
            )
