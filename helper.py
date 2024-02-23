def accuracy_fn(x, y):
    correct = torch.eq(y, x).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
