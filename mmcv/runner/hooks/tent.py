# Copyright (c) Open-MMLab. All rights reserved.
from .hook import HOOKS, Hook
from mmcv.utils import is_method_overridden
import torch
import numpy as np

@HOOKS.register_module()
class TentOnlineHook(Hook):
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')
    def __init__(self):
        self.correct = 0
        self.total = 0

    def after_train_epoch(self, runner):
        print("Final Accuracy:", self.correct/self.total)


    def accuracy(self, pred, target, topk=[1], thrs=None):
        if thrs is None:
            thrs = 0.0
        if isinstance(thrs, float):
            thrs = (thrs, )
            res_single = True
        elif isinstance(thrs, tuple):
            res_single = False
        else:
            raise TypeError(
                f'thrs should be float or tuple, but got {type(thrs)}.')

        res = []
        maxk = max(topk)
        num = pred.size(0)
        pred_score, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
        for k in topk:
            res_thr = []
            for thr in thrs:
                # Only prediction values larger than thr are counted as correct
                _correct = correct & (pred_score.t() > thr)
                correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res_thr.append(correct_k.mul_(100. / num))
            if res_single:
                res.append(res_thr[0])
            else:
                res.append(res_thr)
        return res

    def after_train_iter(self, runner):
        '''
            runner.data_batch: {'img_metas':..., 'img':..., 'gt_label':...}
        '''
        data = runner.data_batch
        batch_gt_labels = data['gt_label']
        batch_size = len(batch_gt_labels)

        data = {'img_metas':data['img_metas'], 'img':data['img']}
        with torch.no_grad():
            results = torch.tensor(runner.model(return_loss=False, **data))

        acc = self.accuracy(results, batch_gt_labels, topk=[1])[0][0].item()
        print("Iter:[{}]   Accuracy:{}".format(runner._inner_iter, acc))
        
        self.correct += acc * batch_size
        self.total += batch_size
