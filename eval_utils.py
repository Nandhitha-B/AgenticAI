import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred):
    acc = (y_true == y_pred).sum().item() / len(y_true) * 100.0
    precision = precision_score(
        y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    recall = recall_score(y_true.cpu(), y_pred.cpu(),
                          average='weighted', zero_division=0)
    f1 = f1_score(y_true.cpu(), y_pred.cpu(),
                  average='weighted', zero_division=0)
    return acc, precision, recall, f1


def _make_attacked_model(base_model, defense):

    if defense is None:
        return base_model
    import torch.nn as nn
    return nn.Sequential(defense, base_model)


def fgsm_attack(model, images, labels, eps=0.03, defense=None):

    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    labels = labels.to(device)

    attacked_model = _make_attacked_model(model, defense)
    attacked_model.eval()

    images.requires_grad = True
    outputs = attacked_model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    attacked_model.zero_grad()
    loss.backward()
    adv = images + eps * images.grad.sign()
    adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach()


def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, steps=40, defense=None, random_start=True):

    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    if random_start:
        images = images + torch.empty_like(images).uniform_(-eps, eps)
        images = torch.clamp(images, 0.0, 1.0)

    for _ in range(steps):
        images.requires_grad = True
        attacked_model = _make_attacked_model(model, defense)
        outputs = attacked_model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        attacked_model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0.0, 1.0).detach()

    return images


def evaluate_robust_accuracy(model, dataloader, eps=0.03, attack="pgd", defense=None, **attack_kwargs):

    model.eval()
    total, correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(next(model.parameters()).device), labels.to(
            next(model.parameters()).device)
        if attack.lower() == "fgsm":
            adv = fgsm_attack(model, images, labels,
                              eps=eps, defense=defense)
        else:
            adv = pgd_attack_(model, images, labels, eps=eps,
                              defense=defense, **attack_kwargs)
        with torch.no_grad():
            if defense is not None:
                from torch import nn
                attacked_model = nn.Sequential(defense, model)
                logits = attacked_model(adv)
            else:
                logits = model(adv)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    robust_acc = 100.0 * correct / total
    return robust_acc
