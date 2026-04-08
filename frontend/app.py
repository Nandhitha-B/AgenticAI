import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sqlite3
from backend.database.operations import save_image
import time
import matplotlib.pyplot as plt
from backend.database.visualization import (
    plot_gap_trend,
    plot_accuracy_trends,
    plot_worst_case,
    get_model_table
)
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_latest_model():
    models_dir = "models"

    models = [
        f for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".pth")
    ]

    if len(models) == 0:
        return None

    versions = [int(m.split("_v")[1].split(".")[0]) for m in models]
    latest_version = max(versions)

    model_path = os.path.join(models_dir, f"model_v{latest_version}.pth")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()

    return model, f"model_v{latest_version}"


def get_latest_status():
    log_file = "logs/agent_log.txt"

    if not os.path.exists(log_file):
        return "No logs yet"

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return "".join(lines[-10:])  # last few lines


def get_metrics_data():
    conn = sqlite3.connect("backend/database/app.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT model_version, attack, accuracy FROM metrics
    """)

    data = cursor.fetchall()
    conn.close()

    return data


def plot_metrics():

    data = get_metrics_data()

    versions = []
    clean = []
    pgd = []

    for row in data:
        version, attack, acc = row

        if attack == "clean":
            versions.append(version)
            clean.append(acc)
        elif attack == "pgd":
            pgd.append(acc)

    plt.figure()
    plt.plot(versions, clean, label="Clean")
    plt.plot(versions, pgd, label="PGD")
    plt.legend()
    plt.title("Performance Over Time")

    return plt
# Load the model
# Using the path confirmed in the saved_model directory


# Define the classes based on dataset folders
class_names = ["glioma", "meningioma", "pituitary"]

# Define the transformation
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def predict(img):
    model, model_version = load_latest_model()

    if model is None:
        return "Model not found", {}, "No model", "No data"

    # Preprocess image
    img = Image.fromarray(img).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(device)

    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    # Create dictionary of class name and confidence score
    results = {class_names[i]: float(probs[i])
               for i in range(len(class_names))}

    # Get the class with highest confidence
    prediction = class_names[torch.argmax(probs).item()]
    # Save image locally
    os.makedirs("uploads", exist_ok=True)
    img_path = f"uploads/{int(time.time())}.png"
    img.save(img_path)

    # Save to DB
    save_image(
        path=img_path,
        prediction=prediction,
        confidence=float(torch.max(probs))
    )
    return prediction, results, model_version, get_latest_status()


# -----------------------------
# Create Gradio Interface (UPGRADED)
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧠 Agentic AI Tumor Classification System")
    gr.Markdown(
        "This system not only classifies tumors but continuously learns "
        "to improve robustness against adversarial conditions."
    )

    # =====================================
    # 🧠 TAB 1: PREDICTION
    # =====================================
    with gr.Tab("🔍 Prediction"):

        with gr.Row():
            image_input = gr.Image(label="Upload MRI Image")

            with gr.Column():
                predict_btn = gr.Button("Predict")

                prediction_output = gr.Textbox(
                    label="Predicted Tumor Class"
                )

                confidence_output = gr.Label(
                    label="Confidence Scores"
                )

        gr.Markdown("### System Intelligence")

        with gr.Row():
            model_version_output = gr.Textbox(
                label="Current Model Version"
            )

            system_status_output = gr.Textbox(
                label="System Status (Agent Logs)",
                lines=8
            )

        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[
                prediction_output,
                confidence_output,
                model_version_output,
                system_status_output
            ]
        )

    # =====================================
    # 📊 TAB 2: ANALYTICS DASHBOARD
    # =====================================
    with gr.Tab("📊 Analytics Dashboard"):

        gr.Markdown("## Model Robustness Evolution")

        refresh_btn = gr.Button("🔄 Refresh Analytics")

        gap_plot = gr.Plot(label="Robustness Gap Over Time")
        acc_plot = gr.Plot(label="Accuracy Across Attacks")
        worst_plot = gr.Plot(label="Worst-case Accuracy")

        def load_all_plots():
            return (
                plot_gap_trend(),
                plot_accuracy_trends(),
                plot_worst_case()
            )

        refresh_btn.click(
            fn=load_all_plots,
            inputs=[],
            outputs=[gap_plot, acc_plot, worst_plot]
        )

    # =====================================
    # 📋 TAB 3: MODEL HISTORY
    # =====================================
    with gr.Tab("📋 Model History"):

        gr.Markdown("## Model Version Comparison")

        table_output = gr.Dataframe()

        table_btn = gr.Button("Load Model Data")

        table_btn.click(
            fn=get_model_table,
            inputs=[],
            outputs=table_output
        )

    # =====================================
    # 📜 TAB 4: AGENT LOGS
    # =====================================
    with gr.Tab("📜 Agent Logs"):

        gr.Markdown("## Autonomous Agent Decisions")

        log_output = gr.Textbox(lines=20)

        log_btn = gr.Button("Load Logs")

        log_btn.click(
            fn=get_latest_status,
            inputs=[],
            outputs=log_output
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
