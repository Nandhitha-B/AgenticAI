import torch
from torch import optim
import torch.nn as nn
from eval_utils import pgd_attack


def train_one_epoch_adv_defense_aware(model, train_loader, device, optimizer, eps=0.03, alpha=0.01, steps=4, defense=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct = 0, 0
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, eps=eps,
                                alpha=alpha, steps=steps, defense=defense, random_start=True)
        optimizer.zero_grad()
        logits = model(adv_images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def run_adversarial_retrain(model, train_loader, device, cfg, defense=None):
    model_copy = model
    lr = float(cfg.get("lr", 0.001))
    optimizer = optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
    epochs = int(cfg.get("epochs", 1))
    last_metrics = {}
    for epoch in range(epochs):
        loss, acc = train_one_epoch_adv_defense_aware(
            model_copy, train_loader, device, optimizer,
            eps=cfg.get("eps", 0.03), alpha=cfg.get("alpha", 0.01), steps=int(cfg.get("steps", 4)), defense=defense
        )
        last_metrics = {"epoch": epoch+1, "loss": loss, "acc": acc}
        print(
            f"[Retrain] epoch {epoch+1}/{epochs} loss={loss:.4f} acc={acc:.2f}%")
    save_path = cfg.get("save_path", "models/retrained_defense_aware.pth")
    torch.save(model_copy.state_dict(), save_path)
    return save_path, last_metrics
