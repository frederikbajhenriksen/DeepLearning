# Sampling methods for active learning
if sampling_method == 'least_confidence':
    # Least confidence sampling: Select samples with the highest uncertainty (lowest max probability)
    uncertainties, top_indices = predictions.max(dim=1)  # Get max probability and its indices
    top_indices = uncertainties.topk(top_percent, largest=False).indices  # Select least confident samples

elif sampling_method == 'margin_sampling':
    # Margin sampling: Select samples with the smallest margin between top two predicted probabilities
    sorted_preds, _ = predictions.topk(2, dim=1)  # Get top two predictions for each sample
    margin = sorted_preds[:, 0] - sorted_preds[:, 1]  # Calculate margin (difference)
    top_indices = margin.topk(top_percent, largest=False).indices  # Select smallest margins

elif sampling_method == 'entropy_sampling':
    # Entropy-based sampling: Select samples with the highest entropy (most uncertain predictions)
    entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)  # Compute entropy
    top_indices = entropy.topk(top_percent, largest=True).indices  # Select highest entropy samples

