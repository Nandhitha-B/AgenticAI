import torch
import torch.nn.functional as F

class CarliniWagnerAttack:
    def __init__(self, model, device, c=1e-3, kappa=0, steps=1000, lr=0.01):
        self.model = model
        self.device = device
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def attack(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        w = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for _ in range(self.steps):
            adv_images = torch.tanh(w) * 0.5 + 0.5
            outputs = self.model(adv_images)

            one_hot = F.one_hot(labels, outputs.size(1)).float()
            real = torch.sum(one_hot * outputs, dim=1)
            other = torch.max((1 - one_hot) * outputs - one_hot * 1e4, dim=1)[0]

            f_loss = torch.clamp(other - real + self.kappa, min=0)
            l2_loss = torch.norm(adv_images - images, p=2)

            loss = l2_loss + self.c * f_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adv_images.detach()
