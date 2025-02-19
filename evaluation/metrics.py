import torch
import numpy as np

from sklearn.metrics import auc, roc_curve, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference, 
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)
from aif360.sklearn.metrics import average_odds_difference


def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    aods = []

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)

        if num_classes == 2:
            if preds.shape == gts.shape:
                pred_labels = (preds >= 0.5).astype(float)
            else:
                assert preds.shape[-1] == 2
                pred_labels = preds.argmax(-1)
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            aod = average_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        else:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0
            aod = 0

        dpds.append(dpd)
        eods.append(eod)
        aods.append(aod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, aods

def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)

        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as e:
            es_auc = -1.
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            try:
                tmp_auc = compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as e:
                tmp_auc = -1.
            aucs_by_group.append( tmp_auc )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        if num_classes == 2:
            if preds.shape == gts.shape:
                pred_labels = (preds >= 0.5).astype(float)
            else:
                assert preds.shape[-1] == 2
                pred_labels = preds.argmax(-1)
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        else:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append(compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        if num_classes == 2:
            if preds.shape == gts.shape:
                pred_labels = (preds >= 0.5).astype(float)
            else:
                assert preds.shape[-1] == 2
                pred_labels = preds.argmax(-1)
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        else:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_scores(preds, gts, attrs=None, num_classes=2):
    """
    Args:
      preds: batch_size x num_class
        gts: batch_size
      attrs: num_attrs x batch_size
    """

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    aods = []
    between_group_disparity = []

    overall_acc = accuracy(preds, gts, topk=(1,))
    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        # es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        try:
            es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        except Exception as error:
            print(error)
            es_auc = -1.
            exit()
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            if e == -1:
                continue
            try:
                tmp_auc = compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes)
            except Exception as error:
                print(error)
                tmp_auc = -1.
                exit()
            aucs_by_group.append( tmp_auc )
        aucs_by_attrs.append(np.array(aucs_by_group))
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        if num_classes == 2:
            if preds.shape == gts.shape:
                pred_labels = (preds >= 0.5).astype(float)
            else:
                assert preds.shape[-1] == 2
                pred_labels = preds.argmax(-1)
                
            try:
                dpd = demographic_parity_difference(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            except Exception as e:
                print('Warning:', e)
                dpd = 0
            try:
                dpr = demographic_parity_ratio(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            except Exception as e:
                print('Warning:', e)
                dpr = 0
            try:
                eod = equalized_odds_difference(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            except Exception as e:
                print('Warning:', e)
                eod = 0
            try:
                eor = equalized_odds_ratio(gts,
                                            pred_labels,
                                            sensitive_features=attr)
            except Exception as e:
                print('Warning:', e)
                eor = 0

            # evaluate the fairness between pos and neg samples
            aod = []
            for priv_group in set(attr):
                aod_g = average_odds_difference(gts,
                                        pred_labels,
                                        prot_attr=attr,
                                        priv_group=priv_group) 
                aod.append(np.abs(aod_g))
            aod = sum(aod) / max(len(aod), 1)
        
        else:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0
            aod = 0

        dpds.append(dpd)
        eods.append(eod)
        aods.append(aod)

    esaccs_by_attrs = np.array(esaccs_by_attrs)
    esaucs_by_attrs = np.array(esaucs_by_attrs)
    dpds = np.array(dpds)
    eods = np.array(eods)
    between_group_disparity = np.array(between_group_disparity)

    return overall_acc, esaccs_by_attrs, overall_auc, esaucs_by_attrs, aucs_by_attrs, dpds, eods, aods, between_group_disparity

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()

    if isinstance(output, np.ndarray):
        output = torch.tensor(output)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    
    if num_classes == 2 and pred_prob.shape == y.shape:
        # Receiver operating characteristic (ROC) https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)   # Area Under the Curve (AUC) 
    else:
        # assert (pred_prob.sum(-1) == 1).all(), \
        #     'The probability estimates must sum to 1 across the possible classes.'
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0] > 2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_to_class):
    nums = nums.astype(int)
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):
    pred_one_hot = prob_to_label(pred_prob)
    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(
            pred_one_hot[:,i],
            gt_one_hot[:,i],
            sensitive_features=attrs
        )

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    if len(output.shape) >= 2:
        overall_acc = np.sum(np.argmax(output, axis=1) == target)/target.shape[0]
    else:
        overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if len(pred_group.shape) >= 2:
            acc = np.sum(np.argmax(pred_group, axis=1) == gt_group)/gt_group.shape[0]
        else:
            acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i] - overall_acc)
    es_acc = (overall_acc / (alpha * tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    
    if num_classes == 2 and output.shape == target.shape:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    else:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        if one_attr == -1:
            continue
        
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]

        if num_classes == 2 and output.shape == target.shape:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        else:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])
        
    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha * tmp + 1))

    return es_auc

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)