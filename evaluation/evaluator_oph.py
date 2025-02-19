import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from .metrics import compute_auc, evalute_comprehensive_perf_scores


class Classification_oph:
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        self.cfg = cfg
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._pred_prob = []
        self._gt = []
        self._attr = []

        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, attr):
        # mo (torch.Tensor): model output logits [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # Convert to float32 if the dtype is float16
        if mo.dtype == torch.float16:
            mo = mo.to(torch.float32)

        if mo.shape == gt.shape:
            self._pred_prob.append(mo.sigmoid())    # binary 
        else:
            self._pred_prob.append(mo.softmax(-1))  
        self._gt.append(gt)
        self._attr.append(attr)

        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        pred_prob = torch.cat(self._pred_prob).cpu().numpy()
        gt = torch.cat(self._gt).cpu().numpy()
        attr = torch.cat(self._attr, dim=1).cpu().numpy()  
        # assert (pred_prob.sum(-1) == 1).all(), \
        #     'The probability estimates must sum to 1 across the possible classes.'
        # overall auc
        auc = 100 * compute_auc(
            pred_prob, gt
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["auc"] = auc

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* auc: {auc:.2f}%"
        )

        overall_acc, esaccs_by_attrs, \
        overall_auc, esaucs_by_attrs, aucs_by_attrs, \
        dpds, eods, aods, between_group_disparity = evalute_comprehensive_perf_scores(
            pred_prob, gt, attr
        )

        print(
            "=> result_oph\n"
            f"* overall_acc: {(100*overall_acc):.2f}%\n" 
            f"* overall_auc: {(100*overall_auc):.2f}%\n"
        )

        for idx in range(len(attr)):
            attr_cur = self.cfg.DATASET.ATTRIBUTES[idx]
            print(
                f"* esacc_{attr_cur}: {(100*esaccs_by_attrs[idx]):.2f}%\n"
                f"* esauc_{attr_cur}: {(100*esaucs_by_attrs[idx]):.2f}%\n" 
                f"* dpd_{attr_cur}: {(100*dpds[idx]):.2f}%\n"
                f"* eod_{attr_cur}: {(100*eods[idx]):.2f}%\n"
                f"* aod_{attr_cur}: {(100*aods[idx]):.2f}%"
            )
            print(
                '\n'.join([
                    f"* auc_{attr_cur}_{str(j)}: {(100*auc):.2f}%" 
                    for j, auc in enumerate(aucs_by_attrs[idx])
                ])
            )
            print(
                ''.join([
                    f"* between_group_disparity_{attr_cur}_{str(j)}: {x:.4f}\n" 
                    for j, x in enumerate(between_group_disparity[idx])
                ])
            )
        
        results['overall_acc'] = overall_acc
        results['esaccs_by_attrs'] = esaccs_by_attrs  # list
        results['overall_auc'] = overall_auc
        results['esaucs_by_attrs'] = esaucs_by_attrs  # list
        results['aucs_by_attrs'] = aucs_by_attrs      # list
        results['dpds'] = dpds  # list
        results['eods'] = eods  # list
        results['aods'] = aods  # list
        results['between_group_disparity'] = between_group_disparity  # list

        return results