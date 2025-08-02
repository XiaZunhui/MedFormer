import logging
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import platform
import pathlib

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from loader import binary_class
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import random
import os
from Mynet.UperHead_medformer import UPerHead
from log import logging_save
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print("inputs111kkk", inputs.shape)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/cqut/Data/medical_seg_data/CVC_ClinicDB/', type=str,
                        help='the path of dataset')
    parser.add_argument('--csvfile', default='src/CVC_ClinicDB/test_train_data.csv', type=str,
                        help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model', default='save_models/CVC_ClinicDB/epoch_best.pth', type=str, help='the path of model')
    parser.add_argument('--save_path', default='save_models/CVC_ClinicDB/', type=str, help='save test result')
    parser.add_argument('--debug', default=False, type=bool, help='plot mask')
    args = parser.parse_args()
    os.makedirs('debug/', exist_ok=True)
    logging_save(args.save_path, 'test')
    df = pd.read_csv(args.csvfile)
    df = df[df.category == 'test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    print(test_files)
    print(len(test_files))
    test_dataset = binary_class(args.dataset, test_files, get_transform())

    model_ft = UPerHead(in_channels=[64, 128, 256, 512], channels=64, image_size=(256, 256))

    model_w = torch.load(args.model)
    model_ft.load_state_dict(model_w)
    model = model_ft.cuda()
    acc_eval = Accuracy()
    pre_eval = Precision()
    dice_eval = Dice()
    recall_eval = Recall()
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    since = time.time()
    for image_id in test_files:
        img = cv2.imread(f'/home/cqut/Data/medical_seg_data/CVC_ClinicDB/images/{image_id}')
        img = cv2.resize(img, (256, 256))
        img_id = list(image_id.split('.'))[0]
        cv2.imwrite(f'debug/{img_id}.png', img)
    model.eval()
    with torch.no_grad():
        for img, mask, img_id in test_dataset:
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)
            # pred = pred[0][1].unsqueeze(0)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end - start)
            pred = torch.sigmoid(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()
            if args.debug:
                img_id = list(img_id.split('.'))[0]
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy == 1] = 255
                cv2.imwrite(f'debug/{img_id}_pred.png', img_numpy)
                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy == 1] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png', mask_numpy)
            iouscore = iou_eval(pred, mask)
            dicescore = dice_eval(pred, mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
            accscore = acc_eval(pred.cpu(), mask.cpu())
            prescore = pre_eval(pred.cpu(), mask.cpu())
            recallscore = recall_eval(pred.cpu(), mask.cpu())
            f1score = f1_eval(pred.cpu(), mask.cpu())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    logging.info('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('FPS: {:.2f}'.format(1.0 / (sum(time_cost) / len(time_cost))))
    logging.info('mean IoU: %.4f, %.4f', np.mean(iou_score), np.std(iou_score))
    logging.info('mean dice: %.4f, %.4f', np.mean(dice_score), np.std(dice_score))
    logging.info('mean accuracy: %.4f, %.4f', np.mean(acc_score), np.std(acc_score))
    logging.info('mean precision: %.4f, %.4f', np.mean(pre_score), np.std(pre_score))
    logging.info('mean recall: %.4f, %.4f', np.mean(recall_score), np.std(recall_score))
    logging.info('mean F1-score: %.4f, %.4f', np.mean(f1_score), np.std(f1_score))

