import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from eval_utils import fgsm_attack, pgd_attack, compute_metrics
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.serialization.safe_globals([torchvision.models.resnet.ResNet]):
    model = torch.load("models/brain_tumor_full_model.pth",
                       map_location=device, weights_only=False)

model = model.to(device)
model.eval()


print("Entire model loaded successfully!")


def compute_metrics(y_true, y_pred):
    acc = (y_true == y_pred).sum().item() / len(y_true) * 100
    precision = precision_score(
        y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
    recall = recall_score(y_true.cpu(), y_pred.cpu(),
                          average='weighted', zero_division=0)
    f1 = f1_score(y_true.cpu(), y_pred.cpu(),
                  average='weighted', zero_division=0)
    return acc, precision, recall, f1


def test_metrics(model, loader, attack_type=None, epsilon=0.03, alpha=0.01, iters=10):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_type == "fgsm":
            images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == "pgd":
            images = pgd_attack(model, images, labels, epsilon, alpha, iters)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc, precision, recall, f1 = compute_metrics(all_labels, all_preds)
    attack_name = attack_type.upper() if attack_type else "CLEAN"
    return acc, precision, recall, f1


def fgsm_attack(model, images, labels, epsilon):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(
        perturbed_images, 0, 1)
    return perturbed_images


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=10):

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_dataset = datasets.ImageFolder("dataset/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# DEFENSE

def defense_metrics(model, loader, attack_type="fgsm", epsilon=0.03, alpha=0.003, iters=20):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_type == "fgsm":
            adv_images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == "pgd":
            adv_images = pgd_attack(
                model, images, labels, epsilon, alpha, iters)

        adv_images = torch.clamp(adv_images, 0, 1)
        adv_images = F.avg_pool2d(
            adv_images, kernel_size=3, stride=1, padding=1)

        outputs = model(adv_images)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc, precision, recall, f1 = compute_metrics(all_labels, all_preds)

    return acc, precision, recall, f1


test_metrics(model, test_loader)

# Attack- FGSM
test_metrics(model, test_loader, attack_type="fgsm", epsilon=0.1)
# Attack- PGD
test_metrics(model, test_loader, attack_type="pgd",
             epsilon=0.03, alpha=0.03/4, iters=4)

# defense
defense_metrics(model, test_loader, attack_type="fgsm", epsilon=0.1)
defense_metrics(model, test_loader, attack_type="pgd",
                epsilon=0.03, alpha=0.03/4, iters=4)


results = []

acc, prec, rec, f1 = test_metrics(model, test_loader)
results.append(["Clean", acc, prec, rec, f1])


acc, prec, rec, f1 = test_metrics(
    model, test_loader, attack_type="fgsm", epsilon=0.1)
results.append(["FGSM", acc, prec, rec, f1])


acc, prec, rec, f1 = test_metrics(
    model, test_loader, attack_type="pgd", epsilon=0.03, alpha=0.03/4, iters=4)
results.append(["PGD", acc, prec, rec, f1])


acc, prec, rec, f1 = defense_metrics(
    model, test_loader, attack_type="fgsm", epsilon=0.1)
results.append(["Defense (FGSM)", acc, prec, rec, f1])


acc, prec, rec, f1 = defense_metrics(
    model, test_loader, attack_type="pgd", epsilon=0.03, alpha=0.03/4, iters=4)
results.append(["Defense (PGD)", acc, prec, rec, f1])


df = pd.DataFrame(results, columns=[
                  "Scenario", "Accuracy (%)", "Precision", "Recall", "F1-score"])

print("\nModel Performance Metrics:\n")
print(df.to_string(index=False))
