import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve

def f1_max_pytorch(pred, target):
    """
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


def f1_max_sklearn(pred, target):
    """
    Simplified version using scikit-learn.
    
    Parameters:
        pred (ndarray): predictions of shape (B, N)
        target (ndarray): binary targets of shape (B, N)
    """
    # Flatten the arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    return np.max(f1_scores)

from sklearn.metrics import precision_recall_curve
import numpy as np

def f1_max_sklearn_per_sample(pred, target):
    """
    Compute the maximum F1 score per sample and then aggregate.
    
    Parameters:
        pred (ndarray): predictions of shape (B, N)
        target (ndarray): binary targets of shape (B, N)
    """
    batch_size = pred.shape[0]
    max_f1_scores = []
    
    for i in range(batch_size):
        pred_i = pred[i]
        target_i = target[i]
        
        # Ensure there are both positive and negative samples
        if np.unique(target_i).size == 1:
            # Avoid undefined metrics when only one class is present
            max_f1_scores.append(0.0)
            continue
        
        precision, recall, _ = precision_recall_curve(target_i, pred_i)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        max_f1_scores.append(np.max(f1_scores))
    
    # Aggregate the per-sample F1-max scores
    # Depending on your needs, you can take the mean, median, or max
    # Here, we'll take the average over all samples
    return np.mean(max_f1_scores)



import numpy as np
import torch


# NumPy f1_max function
def f1_max_numpy(pred, target):
    """
    #direct translation of the original PyTorch code https://torchdrug.ai/docs/_modules/torchdrug/metrics/metric.html#f1_max
    F1 score with the optimal threshold.
    
    This function enumerates all possible thresholds for deciding positive and negative
    samples and then picks the threshold with the maximal F1 score.
    
    Parameters:
        pred (ndarray): predictions of shape (B, N)
        target (ndarray): binary targets of shape (B, N)
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    batch_size, num_preds = pred.shape
    
    order = np.argsort(-pred, axis=1)
    target_sorted = np.take_along_axis(target, order, axis=1)
    cumsum_target = np.cumsum(target_sorted, axis=1)
    precision = cumsum_target / np.arange(1, num_preds + 1)
    recall = cumsum_target / (target.sum(axis=1, keepdims=True) + 1e-10)
    is_start = np.zeros_like(target_sorted, dtype=bool)
    is_start[:, 0] = True
    is_start_scattered = np.zeros_like(is_start)
    batch_indices = np.arange(batch_size)[:, None]
    is_start_scattered[batch_indices, order] = is_start
    pred_flat = pred.flatten()
    precision_flat = precision.flatten()
    recall_flat = recall.flatten()
    is_start_flat = is_start_scattered.flatten()
    all_order = np.argsort(-pred_flat)
    order_flat = (order + (batch_indices * num_preds)).flatten()
    inv_order = np.empty_like(order_flat)
    inv_order[order_flat] = np.arange(order_flat.size)
    is_start_sorted = is_start_flat[all_order]
    precision_sorted = precision_flat[inv_order[all_order]]
    recall_sorted = recall_flat[inv_order[all_order]]
    prev_precision = np.zeros_like(precision_sorted)
    prev_precision[1:] = precision_sorted[:-1]
    delta_precision = precision_sorted - np.where(is_start_sorted, 0, prev_precision)
    prev_recall = np.zeros_like(recall_sorted)
    prev_recall[1:] = recall_sorted[:-1]
    delta_recall = recall_sorted - np.where(is_start_sorted, 0, prev_recall)
    cumsum_is_start = np.cumsum(is_start_sorted)
    all_precision_cumsum = np.cumsum(delta_precision) / cumsum_is_start
    all_recall_cumsum = np.cumsum(delta_recall) / batch_size
    f1_numerator = 2 * all_precision_cumsum * all_recall_cumsum
    f1_denominator = all_precision_cumsum + all_recall_cumsum + 1e-10
    all_f1 = f1_numerator / f1_denominator
    return all_f1.max()

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support

def protein_centric_max_f1(y_true, y_pred_prob, thresholds=np.arange(0.0, 1.01, 0.01)):
    """
    Compute the protein-centric maximum F1 score.
    
    Parameters:
    y_true (np.array): Binary matrix of true GO term annotations for each protein.
                       Shape: (n_proteins, n_go_terms)
    y_pred_prob (np.array): Matrix of predicted probabilities for each GO term per protein.
                            Shape: (n_proteins, n_go_terms)
    thresholds (np.array): List of thresholds to try for binarizing predicted probabilities.
    
    Returns:
    float: Average of maximum F1-scores over all proteins.
    """
    n_proteins = y_true.shape[0]
    max_f1_scores = []
    
    # Iterate over each protein
    for i in range(n_proteins):
        true_labels = y_true[i]
        pred_probs = y_pred_prob[i]
        
        # Track the best F1-score for this protein
        best_f1 = 0.0
        
        # Iterate over thresholds and calculate F1-score at each threshold
        for threshold in thresholds:
            # Binarize predictions based on the threshold
            pred_labels = (pred_probs >= threshold).astype(int)
            
            # Compute precision, recall, and F1 for this threshold
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
            
            # Update the best F1-score
            best_f1 = max(best_f1, f1)
        
        # Store the best F1-score for this protein
        max_f1_scores.append(best_f1)
    
    # Return the average of maximum F1-scores across all proteins
    return np.mean(max_f1_scores)

# Example usage
y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]])  # Ground truth labels (GO terms) for each protein
y_pred_prob = np.array([[0.8, 0.2, 0.9], [0.3, 0.7, 0.6], [0.9, 0.4, 0.1]])  # Predicted probabilities

max_f1 = protein_centric_max_f1(y_true, y_pred_prob)
print(f'Protein-Centric Maximum F1-Score: {max_f1:.4f}')


import numpy as np
import torch
# Define fixed predictions (probabilities between 0 and 1)
pred_np = np.array([
    [0.9, 0.8, 0.7, 0.4, 0.2],
    [0.85, 0.75, 0.65, 0.5, 0.3],
    [0.95, 0.6, 0.55, 0.45, 0.25]
])

# Define fixed binary targets
target_np = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
])

# Set the random seed for reproducibility
np.random.seed(0)
# Generate random predictions
pred_np = np.random.rand(5, 10)  # 5 samples, 10 predictions each
# Generate random binary targets
target_np = (np.random.rand(5, 10) > 0.5).astype(int)  # Random binary targets

# Convert NumPy arrays to PyTorch tensors
pred_torch = torch.from_numpy(pred_np).float()
target_torch = torch.from_numpy(target_np).float()


# Run the PyTorch f1_max function
max_f1_pytorch = f1_max_pytorch(pred_torch, target_torch)
#Maximum F1 Score (PyTorch 1.13.0): 0.804598
# Run the NumPy f1_max function
max_f1_numpy = f1_max_numpy(pred_np, target_np)
max_f1_sklean = f1_max_sklearn(pred_np,target_np)
max_f1_sklearn_per_sample = f1_max_sklearn_per_sample(pred_np,target_np)

print(f"Maximum F1 Score (PyTorch 1.13.0): {max_f1_pytorch.item():.6f}")

print(f"Maximum F1 Score (NumPy): {max_f1_numpy:.6f}")
print(f"Maximum F1 Score sklean: {max_f1_sklean}")
print(f"Maximum F1 Score sklean per sample: {max_f1_sklearn_per_sample}")


protein_max_f1 = protein_centric_max_f1(target_np,pred_np,thresholds=np.arange(0.0, 1.01, 0.01))
print(f"Maximum F1 Score protein-centric: {protein_max_f1}") 
#in tangjian's code the max t is the same t in this protein_centric_max_f1 the t is different for each go