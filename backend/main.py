# main.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from evaluation.attack_eval import run_full_evaluation

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load Model
# -----------------------------
model = torch.load("saved_model/brain_tumor_full_model.pth",
                   map_location=device, weights_only=False)
model.to(device)
model.eval()


# -----------------------------
# Load Test Data
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder(
    "backend/dataset/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------
# Run Evaluation
# -----------------------------
metrics = run_full_evaluation(model, test_loader)

print("\nRobustness Evaluation Results:\n")

for key, value in metrics.items():

    # If value is a number
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.2f}")

    # If value is a dictionary
    elif isinstance(value, dict):
        print(f"\n{key}:")
        for sub_key, sub_val in value.items():

            # If nested dictionary
            if isinstance(sub_val, dict):
                print(f"  {sub_key}:")
                for inner_key, inner_val in sub_val.items():
                    print(f"    {inner_key}: {inner_val:.2f}")
            else:
                print(f"  {sub_key}: {sub_val:.2f}")
