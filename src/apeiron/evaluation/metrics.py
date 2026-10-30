"""Percentage accuracy for tensor-like classification predictions."""


def accuracy_topk(output, target, topk=(1,)):
    """Return one detached percentage tensor for each requested k."""
    maxk = max(topk)
    if output.dim() != 2:
        raise ValueError("Expected a batch-by-class output matrix")
    batch_size = target.size(0)
    scale = 100.0 / batch_size
    _, predictions = output.detach().topk(maxk, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(target.reshape(1, -1).expand_as(predictions))
    return [correct[:k].reshape(-1).float().sum() * scale for k in topk]


def accuracy(output, target):
    """Return top-1 accuracy as a percentage tensor."""
    return accuracy_topk(output, target, topk=(1,))[0]
