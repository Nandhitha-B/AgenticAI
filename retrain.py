import torch
from torch import optim
import torch.nn as nn
from eval_utils import pgd_attack
from defense import trades_loss
import os

def train_one_epoch_adv_defense_aware(model, train_loader, device, optimizer, eps=0.03, alpha=0.01, steps=4, defense=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct = 0, 0
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        loss = trades_loss(
            model,
            images,
            labels,
            beta=6.0,
            epsilon=eps,
            alpha=alpha,
            steps=steps
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        # Compute predictions for accuracy tracking
        with torch.no_grad():
            logits = model(images)
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_copy.state_dict(), save_path)
    return save_path, last_metrics


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import yaml
    
    # Load config
    with open("backend/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pre-trained model
    model = torch.load("saved_model/brain_tumor_full_model.pth",
                       map_location=device, weights_only=False)
    model.to(device)
    print("Model loaded successfully!")
    
    # Load training data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = datasets.ImageFolder(
        "backend/dataset/train", transform=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=2
    )
    print(f"Training dataset loaded: {len(train_dataset)} images")
    
    # Run adversarial retraining with TRADES defense
    save_path, final_metrics = run_adversarial_retrain(model, train_loader, device, cfg)
    print(f"\n✓ Model saved to: {save_path}")
    print(f"Final metrics: {final_metrics}")
