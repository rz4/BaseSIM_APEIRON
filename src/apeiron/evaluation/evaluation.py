"""Evaluate a classifier over a loader using its supplied criterion."""


def test(model, test_loader, criterion, cfg):
    """Return accuracy and summed batch losses divided by sample count.

    The criterion must use sum reduction for a sample-mean loss. Empty
    loaders retain the scalar-zero return required by this API.
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for inputs, target in test_loader:
        inputs = inputs.to(cfg.device)
        target = target.to(cfg.device)
        output = model(inputs).detach()
        loss_sum += criterion(output, target).item()
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total, loss_sum / total
