import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# SIMPLE SMOOTHING DEFENSE
# ------------------------------
class SmoothingDefense(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.pool(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# PGD ATTACK
# ------------------------------
def pgd_attack(model, x, y, epsilon=0.03, alpha=0.01, iters=10):
    x = x.clone().detach().to(device)
    y = y.to(device)

    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)

        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad

        x_adv = x_adv + alpha * grad.sign()

        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + eta, 0, 1).detach()

    return x_adv


# ------------------------------
# TRADES LOSS
# ------------------------------
def trades_loss(
    model,
    x,
    y,
    beta=6.0,
    epsilon=8/255,
    alpha=2/255,
    steps=10
):
    x_adv = pgd_attack(
        model,
        x,
        y,
        epsilon=epsilon,
        alpha=alpha,
        iters=steps
    )

    logits_clean = model(x)
    loss_clean = F.cross_entropy(logits_clean, y)

    logits_adv = model(x_adv)

    loss_robust = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_clean, dim=1),
        reduction="batchmean"
    )

    return loss_clean + beta * loss_robust


# ------------------------------
# TRADES WRAPPER
# ------------------------------
class TRADESDefense:
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=10, beta=6.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.beta = beta

    def generate(self, x, y):
        return pgd_attack(
            self.model,
            x,
            y,
            self.epsilon,
            self.alpha,
            self.steps
        )