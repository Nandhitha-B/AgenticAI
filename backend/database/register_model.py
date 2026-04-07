from backend.database.operations import save_model

# Register BOTH models properly

save_model(
    version="model_v1.pth",
    path="models/model_v1.pth",
    clean_acc=98,
    worst_acc=85,
    gap=13
)

save_model(
    version="model_v2.pth",
    path="models/model_v2.pth",
    clean_acc=95,
    worst_acc=80,
    gap=15
)

print("Both models registered successfully")
