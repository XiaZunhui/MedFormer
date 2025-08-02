import torch.nn as nn
import torch
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np
cfs = ConfusionMatrix(3)
class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1-dice


class DiceLoss_multi(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multi, self).__init__()

    def forward(self, inputs, targets, smooth=1, num_classes=3):
        # inputs 是模型的原始输出，大小应为 (N, num_classes, H, W)
        inputs = torch.sigmoid(inputs)

        # 将单通道 targets 转换为 one-hot 编码形式
        # targets 的原始大小为 (N, H, W)，每个像素的值为类别索引
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # (N, H, W, num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # (N, num_classes, H, W)

        # 初始化 Dice 损失
        dice_loss = 0.0

        for class_index in range(num_classes):
            # 对于每个类别，单独计算 Dice loss
            input_class = inputs[:, class_index, :, :]
            target_class = targets_one_hot[:, class_index, :, :].float()  # 转换为float，以匹配 input 的数据类型

            intersection = (input_class * target_class).sum()
            dice_score = (2. * intersection + smooth) / (input_class.sum() + target_class.sum() + smooth)

            # 累积每个类别的 Dice loss
            dice_loss += (1 - dice_score)

        # 取平均 Dice 损失
        dice_loss /= num_classes

        return dice_loss
class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class DiceLoss_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def binary_dice(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

    def forward(self, ipts, gt):
        ipts = self.sfx(ipts)
        c = ipts.shape[1]
        sum_loss = 0
        for i in range(c):
            tmp_inputs = ipts[:, i]
            tmp_gt = gt[:, i]
            tmp_loss = self.binary_dice(tmp_inputs, tmp_gt)
            sum_loss += tmp_loss
        return sum_loss / c


class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.sfx(inputs)
        c = inputs.shape[1]
        inputs = torch.max(inputs, 1).indices.cpu()
        targets = torch.max(targets, 1).indices.cpu()
        cfsmat = cfs(inputs, targets).numpy()

        sum_iou = 0
        for i in range(c):
            tp = cfsmat[i, i]
            fp = np.sum(cfsmat[0:3, i]) - tp
            fn = np.sum(cfsmat[i, 0:3]) - tp

            tmp_iou = tp / (fp + fn + tp)
            sum_iou += tmp_iou

        return sum_iou / c
