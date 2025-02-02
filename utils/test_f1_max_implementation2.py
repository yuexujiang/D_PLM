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
    [1, 0, 0, 1, 0]
])

# Convert NumPy arrays to PyTorch tensors
pred_torch = torch.from_numpy(pred_np).float()
target_torch = torch.from_numpy(target_np).float()

def f1_max_pytorch(pred, target):
    order = pred.argsort(descending=True, dim=1)
    target_sorted = target.gather(1, order)
    precision = target_sorted.cumsum(1) / torch.arange(1, target_sorted.size(1) + 1, device=pred.device)
    recall = target_sorted.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target_sorted, dtype=torch.bool)
    is_start[:, 0] = True
    is_start = is_start.scatter(1, order, is_start)
    
    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.size(0), device=order.device).unsqueeze(1) * order.size(1)
    order = order.flatten()
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    zero_tensor = torch.zeros_like(precision)
    all_precision = precision[all_order] - torch.where(is_start, zero_tensor, precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(is_start, zero_tensor, recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.size(0)
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

# Run the PyTorch f1_max function
max_f1_pytorch = f1_max_pytorch(pred_torch, target_torch)
print(f"Maximum F1 Score (PyTorch): {max_f1_pytorch.item():.6f}")


def f1_max_numpy(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    batch_size, num_preds = pred.shape
    
    # 1. Sort predictions in descending order for each sample
    order = np.argsort(-pred, axis=1)
    target_sorted = np.take_along_axis(target, order, axis=1)
    
    # 2. Compute cumulative sums of target_sorted for precision and recall
    cumsum_target = np.cumsum(target_sorted, axis=1)
    precision = cumsum_target / np.arange(1, num_preds + 1)
    recall = cumsum_target / (target.sum(axis=1, keepdims=True) + 1e-10)
    
    # 3. Create an indicator for the start of each sample
    is_start = np.zeros_like(target_sorted, dtype=bool)
    is_start[:, 0] = True
    
    # Scatter is_start according to 'order' to align with sorted predictions
    is_start_scattered = np.zeros_like(is_start)
    batch_indices = np.arange(batch_size)[:, None]
    is_start_scattered[batch_indices, order] = is_start
    
    # 4. Flatten arrays to process all predictions together
    pred_flat = pred.flatten()
    precision_flat = precision.flatten()
    recall_flat = recall.flatten()
    is_start_flat = is_start_scattered.flatten()
    
    # 5. Get global order of all predictions across the batch
    all_order = np.argsort(-pred_flat)
    
    # 6. Map local indices to global indices
    order_flat = (order + batch_indices * num_preds).flatten()
    inv_order = np.empty_like(order_flat)
    inv_order[order_flat] = np.arange(order_flat.size)
    
    # Rearrange 'is_start', 'precision', and 'recall' according to 'all_order'
    is_start_sorted = is_start_flat[all_order]
    precision_sorted = precision_flat[inv_order[all_order]]
    recall_sorted = recall_flat[inv_order[all_order]]
    
    # 7. Compute differences in precision and recall for each threshold
    prev_precision = np.zeros_like(precision_sorted)
    prev_precision[1:] = precision_sorted[:-1]
    delta_precision = precision_sorted - np.where(is_start_sorted, 0, prev_precision)
    
    prev_recall = np.zeros_like(recall_sorted)
    prev_recall[1:] = recall_sorted[:-1]
    delta_recall = recall_sorted - np.where(is_start_sorted, 0, prev_recall)
    
    # 8. Compute cumulative sums of delta_precision and delta_recall
    cumsum_is_start = np.cumsum(is_start_sorted)
    all_precision_cumsum = np.cumsum(delta_precision) / cumsum_is_start
    all_recall_cumsum = np.cumsum(delta_recall) / batch_size
    
    # 9. Compute F1 scores at each threshold
    f1_numerator = 2 * all_precision_cumsum * all_recall_cumsum
    f1_denominator = all_precision_cumsum + all_recall_cumsum + 1e-10
    all_f1 = f1_numerator / f1_denominator
    
    # 10. Return the maximum F1 score
    return all_f1.max()

# Run the NumPy f1_max function
max_f1_numpy = f1_max_numpy(pred_np, target_np)
print(f"Maximum F1 Score (NumPy): {max_f1_numpy:.6f}")

def f1_max_numpy(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    batch_size, num_preds = pred.shape
    
    # 1. Sort predictions in descending order for each sample
    order = np.argsort(-pred, axis=1)
    target_sorted = np.take_along_axis(target, order, axis=1)
    
    # 2. Compute cumulative sums of target_sorted for precision and recall
    cumsum_target = np.cumsum(target_sorted, axis=1)
    precision = cumsum_target / np.arange(1, num_preds + 1)
    recall = cumsum_target / (target.sum(axis=1, keepdims=True) + 1e-10)
    
    # 3. Create an indicator for the start of each sample
    is_start = np.zeros_like(target_sorted, dtype=bool)
    is_start[:, 0] = True
    is_start = np.take_along_axis(is_start, order, axis=1)
    
    # 4. Flatten arrays to process all predictions together
    pred_flat = pred.flatten()
    precision_flat = precision.flatten()
    recall_flat = recall.flatten()
    is_start_flat = is_start.flatten()
    
    # 5. Get global order of all predictions across the batch
    all_order = np.argsort(-pred_flat)
    
    # 6. Map local indices to global indices
    order_flat = (order + np.arange(batch_size)[:, None] * num_preds).flatten()
    inv_order = np.empty_like(order_flat)
    inv_order[order_flat] = np.arange(order_flat.size)
    
    # Rearrange 'is_start', 'precision', and 'recall' according to 'all_order'
    is_start_sorted = is_start_flat[all_order]
    precision_sorted = precision_flat[inv_order[all_order]]
    recall_sorted = recall_flat[inv_order[all_order]]
    
    # 7. Compute differences in precision and recall for each threshold
    prev_precision = np.zeros_like(precision_sorted)
    prev_precision[1:] = precision_sorted[:-1]
    delta_precision = precision_sorted - np.where(is_start_sorted, 0, prev_precision)
    
    prev_recall = np.zeros_like(recall_sorted)
    prev_recall[1:] = recall_sorted[:-1]
    delta_recall = recall_sorted - np.where(is_start_sorted, 0, prev_recall)
    
    # 8. Compute cumulative sums of delta_precision and delta_recall
    cumsum_is_start = np.cumsum(is_start_sorted)
    all_precision_cumsum = np.cumsum(delta_precision) / cumsum_is_start
    all_recall_cumsum = np.cumsum(delta_recall) / batch_size
    
    # 9. Compute F1 scores at each threshold
    f1_numerator = 2 * all_precision_cumsum * all_recall_cumsum
    f1_denominator = all_precision_cumsum + all_recall_cumsum + 1e-10
    all_f1 = f1_numerator / f1_denominator
    
    # 10. Return the maximum F1 score
    return all_f1.max()
