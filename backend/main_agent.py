# backend/main_agent.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from agent.agent import RobustnessAgent


device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------
# DATA TRANSFORMS
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# --------------------------------
# LOAD DATASETS
# --------------------------------

train_dataset = datasets.ImageFolder(
    "backend/dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "backend/dataset/test",
    transform=transform
)


# --------------------------------
# OPTIONAL: SMALL SUBSET FOR FAST EVALUATION
# --------------------------------

subset_size = 200

test_subset = Subset(
    test_dataset,
    range(subset_size)
)

train_subset = Subset(
    train_dataset,
    range(subset_size)
)
# --------------------------------
# DATA LOADERS
# --------------------------------

train_loader = DataLoader(
    train_subset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_subset,
    batch_size=8,
    shuffle=False
)


# --------------------------------
# CREATE AGENT
# --------------------------------

agent = RobustnessAgent(
    train_loader=train_loader,
    test_loader=test_loader,
    models_dir="models",
    log_file="logs/agent_log.txt",
    gap_threshold=25,
    device=device
)


# --------------------------------
# CHOOSE EXECUTION MODE
# --------------------------------

MODE = "once"          # options: "once" or "periodic"

INTERVAL_DAYS = 3      # used only in periodic mode


# --------------------------------
# RUN AGENT
# --------------------------------

if MODE == "once":

    print("\nRunning agent once...\n")
    agent.run_once()

elif MODE == "periodic":

    print(f"\nRunning agent every {INTERVAL_DAYS} days...\n")
    agent.run_periodic(interval_days=INTERVAL_DAYS)

else:

    print("Invalid mode selected.")
