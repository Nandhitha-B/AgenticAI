import torch
import os
import sqlite3

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from backend.evaluation.attack_eval import run_full_evaluation
from backend.database.operations import save_metrics, save_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load test data (same as main)
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder(
    "backend/dataset/test",
    transform=test_transforms
)

# small subset for speed
subset = Subset(test_dataset, range(200))

test_loader = DataLoader(subset, batch_size=8, shuffle=False)

# -----------------------------
# Evaluate ALL models in folder
# -----------------------------
models_dir = "models"

for model_file in os.listdir(models_dir):

    if not model_file.endswith(".pth"):
        continue

    print(f"\nEvaluating {model_file}...")

    model_path = os.path.join(models_dir, model_file)

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.to(device)
    model.eval()

    # Run evaluation
    metrics = run_full_evaluation(model, test_loader)

    clean = metrics["Clean Accuracy"]
    worst = metrics["Worst-case Accuracy"]
    gap = metrics["Robustness Gap"]

    print(f"Clean: {clean:.2f}, Worst: {worst:.2f}, Gap: {gap:.2f}")

    # -----------------------------
    # Save detailed metrics
    # -----------------------------
    save_metrics(
        model_version=model_file,
        metrics_dict={
            "clean": clean,
            "fgsm": metrics["FGSM Accuracy"],
            "pgd": metrics["PGD Accuracy"],
            "bim": metrics["BIM Accuracy"],
            "cw": metrics["CW Accuracy"]
        }
    )

    # -----------------------------
    # UPDATE models table
    # -----------------------------
    conn = sqlite3.connect("backend/database/app.db")
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE models
    SET clean_acc=?, worst_acc=?, gap=?
    WHERE version=?
    """, (clean, worst, gap, model_file))

    conn.commit()
    conn.close()

print("\nAll models updated successfully!")
