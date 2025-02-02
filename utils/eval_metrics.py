from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score,precision_score,recall_score,f1_score,roc_auc_score
import csv
from scipy.special import softmax
import torch
import numpy as np
def get_pred_labels(out_filename):
    result = open(out_filename, 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label

def get_pred_probs(out_filename):
    result = open(out_filename, 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_probs = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        probs = torch.zeros(len(preds_with_dist))
        count = 0
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = float(pred_ec_dist.split(":")[1].split("/")[1])
            probs[count] = ec_i
            #preds_ec_lst.append(probs)
            count += 1
        # sigmoid of the negative distances 
        probs = (1 - torch.exp(-1/probs)) / (1 + torch.exp(-1/probs))
        probs = probs/torch.sum(probs)
        pred_probs.append(probs)
    return pred_probs

def f1_max(pred, target):
    """
    refer to Protein Structure and Sequence Generation with Equivariant Denoising Diffusion Probabilistic Models
    copied from https://torchdrug.ai/docs/_modules/torchdrug/metrics/metric.html#f1_max
    F1 score with the optimal threshold.
    
    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.
    
    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    
    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict


def get_eval_metrics_alldict(eval_dist,true_label,all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    def get_ec_pos_dict(mlb, all_label):
        label_pos_dict={}
        for label in all_label:
            label_pos_dict[label] = list(mlb.classes_).index(label)
        
        return label_pos_dict
    
    
    n_test = len(true_label)
    label_pos_dict = get_ec_pos_dict(mlb, all_label)
    true_m = np.zeros((n_test, len(mlb.classes_)))
    for i in range(n_test):
        true_m[i] = mlb.transform([true_label[i]])
    
    pred_probs = np.zeros((n_test,len(mlb.classes_)))
    i = 0 
    for pid in eval_dist.keys():
        for predict_label in eval_dist[pid].keys():
            pos = label_pos_dict[predict_label]
            pred_probs[i,pos] = eval_dist[pid][predict_label]
        
        i+=1
    
    pred_probs=softmax(-pred_probs,axis=1)
    #in DeepFRI their AUPR is https://github.com/flatironinstitute/DeepFRI/blob/master/deepfrier/utils.py#L116
    aupr = average_precision_score(true_m, pred_probs,average = 'micro')
    fmax = f1_max(torch.from_numpy(pred_probs).float(),torch.from_numpy(true_m).float())
    return fmax,aupr #pre, rec, f1, roc, acc



def get_eval_metrics_clean(pred_label, pred_probs, true_label, all_label):
    """
    in other paper, like in Jian Tang's in paper PROTEIN REPRESENTATION LEARNING BY GEOMETRIC STRUCTURE PRETRAINING, they used The second metric, pair-centric area under precision-recall curve AUPRpair, is defined as the average
    precision scores for all protein-function pairs, which is exactly the micro average precision score for
    multiple binary classification.
    they used micro average! 
    """ 
    
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    
    #pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    #rec = recall_score(true_m, pred_m, average='weighted')
    #f1 = f1_score(true_m, pred_m, average='weighted')
    #roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    #acc = accuracy_score(true_m, pred_m)
    #in DeepFRI their AUPR is https://github.com/flatironinstitute/DeepFRI/blob/master/deepfrier/utils.py#L116
    aupr = average_precision_score(true_m, pred_m_auc,average = 'micro')
    
    fmax = f1_max(torch.from_numpy(pred_m_auc).float(),torch.from_numpy(true_m).float())
    return fmax,aupr #pre, rec, f1, roc, acc