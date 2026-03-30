import torch
import torch.nn.functional as F

def relu_evidence(y):
    """
    Apply ReLU activation to extract non-negative evidence from logits.
    Core operation for Evidential Deep Learning (EDL) where evidence e >= 0.
    """
    return F.relu(y)

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    Base loss function for Evidential Deep Learning (EDL).
    Combines expected cross-entropy (data fit) and KL divergence regularization.
    
    Args:
        func: Log function (e.g., torch.log).
        y: One-hot encoded pseudo-labels or ground truth.
        alpha: Dirichlet distribution parameters (evidence + 1).
        epoch_num: Current training epoch.
        num_classes: Total number of target categories.
        annealing_step: Epochs to reach full KL penalty.
        device: Target computation device (CPU/GPU).
    """
    if device is None:
        device = alpha.device
        
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    # 1. Expected Cross Entropy (Data Fit Term)
    A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # 2. Annealing coefficient for KL Divergence
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device),
    )
    
    # 3. Adjust alpha for non-target classes to regularize towards uniform distribution (evidence=0)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    
    # 4. KL Divergence Regularization
    kl_div = annealing_coef * kl_divergence_with_uniform(kl_alpha, num_classes, device=device)
    
    return A + kl_div

def kl_divergence_with_uniform(alpha, num_classes, device=None):
    """
    Calculate KL divergence between the current Dirichlet distribution 
    and a uniform Dirichlet distribution (Dirichlet([1,1,...,1])).
    Acts as a penalty for overconfidence on incorrect classes.
    """
    if device is None:
        device = alpha.device
        
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    
    return first_term + second_term

def get_dirichlet_uncertainty(logits):
    """
    [Core Applied Math Module]
    Converts raw model logits into Dirichlet concentration parameters (alpha),
    and derives the second-order uncertainty based on Subjective Logic.
    
    Args:
        logits (torch.Tensor): Raw output from the classifier head.
        
    Returns:
        prob: Expected categorical probability (alpha / S).
        uncertainty: Subjective uncertainty score (K / S).
        alpha: Dirichlet concentration parameters.
    """
    evidence = relu_evidence(logits)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Subjective Uncertainty: u = K / sum(alpha)
    # Bounds: u -> 0 when evidence is high, u = 1 when evidence is zero.
    K = alpha.shape[1]
    uncertainty = K / S
    
    # Expected Probability
    prob = alpha / S
    
    return prob, uncertainty, alpha

def bootstrap_accuracy_ci(y_true, y_pred, n_iterations=1000, ci=0.95):
    """
    [Statistical Significance Module] Bootstrap Confidence Interval Estimation.
    Performs random sampling with replacement to estimate the confidence interval
    of the clustering/classification accuracy.
    Provides robust variance estimation for model evaluation, especially under domain shift.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels from the model.
        n_iterations (int): Number of bootstrap resamples.
        ci (float): Confidence level (default 0.95 for 95% CI).
        
    Returns:
        mean_acc: Mean accuracy across bootstrap samples.
        lower_bound: Lower bound of the confidence interval.
        upper_bound: Upper bound of the confidence interval.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_size = len(y_true)
    
    # Fallback for extremely small evaluation sets
    if n_size < 10:
        return accuracy_score(y_true, y_pred), 0.0, 0.0
        
    stats = list()
    for _ in range(n_iterations):
        # Sampling with replacement
        indices = np.random.randint(0, n_size, n_size)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        acc = accuracy_score(sample_true, sample_pred)
        stats.append(acc)
        
    # Calculate empirical confidence intervals
    alpha = (1.0 - ci) / 2.0
    lower_bound = np.percentile(stats, alpha * 100)
    upper_bound = np.percentile(stats, (1.0 - alpha) * 100)
    mean_acc = np.mean(stats)
    
    return mean_acc, lower_bound, upper_bound
