import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device):
    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Fast gradient sign step
    adv_images = images + epsilon * images.grad.sign()

    # Clamp to valid pixel range
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()
