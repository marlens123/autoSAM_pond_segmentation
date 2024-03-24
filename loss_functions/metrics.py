"""
Copyright 2023 Xinrong Hu et al. https://github.com/xhu248/AutoSAM

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import threading
import torch
import numpy as np

# PyTroch version

SMOOTH = 1e-5


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self, mode='mean'):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        Dice = 2.0 * self.total_inter / (np.spacing(1) + self.total_union + self.total_inter)
        if mode == 'mean':
            mIoU = IoU.mean()
            Dice = Dice.mean()
            return pixAcc, mIoU, Dice
        else:
            return pixAcc, IoU, Dice

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    # predict = torch.max(output, 1)[1]
    predict = torch.argmax(output, dim=1)
    # predict = output

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass): #只区分背景和器官: nclass = 2
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor                      #model的输出
        target: label 3D Tensor                       #label
        nclass: number of categories (int)            #只区分背景和器官: nclass = 2
    """
    predict = torch.max(output, dim=1)[1]                 #获得了预测结果
    # predict = output
    mini = 1
    maxi = nclass-1                                   #nclass = 2, maxi=1
    nbins = nclass-1                                  #nclass = 2, nbins=1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')

    predict = predict * (target >= 0).astype(predict.dtype)
    intersection = predict * (predict == target)            # 得到TP和TN

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))  #统计(TP、TN)值为1的像素个数，获得TN
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))        #统计predict中值为1的像素个数，获得TN+FN
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))          #统计target中值为1的像素个数，获得TN+FP
    area_union = area_pred + area_lab - area_inter                              #area_union:TN+FN+FP
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union