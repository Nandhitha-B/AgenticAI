import torch
import torch.nn as nn
import torch.optim as optim


def cw_attack(model, images, labels, c=1e-4, kappa=0, iters=100, lr=0.01, device="cuda"):
    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=lr)

    mse_loss = nn.MSELoss(reduction='sum')

    for _ in range(iters):
        adv_images = torch.tanh(w) * 0.5 + 0.5

        outputs = model(adv_images)

        one_hot = torch.eye(outputs.size(1)).to(device)[labels]

        real = torch.sum(one_hot * outputs, dim=1)
        other = torch.max((1 - one_hot) * outputs - one_hot * 1e4, dim=1)[0]

        loss1 = torch.clamp(real - other + kappa, min=0)
        loss2 = mse_loss(adv_images, images)

        loss = loss2 + c * torch.sum(loss1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    adv_images = torch.tanh(w) * 0.5 + 0.5

    return adv_images.detach()
