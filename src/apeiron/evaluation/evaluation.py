"""Evaluate a classifier over a loader using its supplied criterion."""

import torch


def test(model, test_loader, criterion, cfg):
    """Return ``(accuracy, loss)`` for ``model`` over ``test_loader``.

    The loss is the mean over SAMPLES whatever the criterion's reduction:
    a mean-reduction criterion (what every in-tree harness supplies) has
    each batch's mean weighted by its size, and a sum-reduction one is
    divided by the sample count directly. The previous version summed
    batch means and divided by the sample count, under-reporting by the
    batch count -- 6x for 12 samples in 2 batches -- so the reported loss
    scaled with batch size, which a loss must not (claims/LOG.md A19).

    An empty loader raises: there is nothing to score, and the old bare
    ``0.0`` return was unpackable by every caller expecting a pair.
    """
    model.eval()
    reduction = getattr(criterion, "reduction", "mean")
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(cfg.device)
            target = target.to(cfg.device)
            output = model(inputs)
            batch_loss = criterion(output, target).item()
            if reduction == "sum":
                loss_sum += batch_loss
            else:
                loss_sum += batch_loss * target.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    if total == 0:
        raise ValueError(
            "cannot evaluate an empty dataloader: no samples were scored, and "
            "reporting 0.0 would be indistinguishable from a perfect loss"
        )
    return 100.0 * correct / total, loss_sum / total
