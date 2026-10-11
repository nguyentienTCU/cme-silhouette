import torch

def iou_metric(outputs, labels, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    intersection = (preds * labels).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
