import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, iters=10, device="cuda"):
    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True

        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Gradient step
        adv_images = images + alpha * images.grad.sign()

        # Projection to epsilon ball
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images
