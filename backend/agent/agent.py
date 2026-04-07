import os
import time
import torch
from datetime import datetime

from evaluation.attack_eval import run_full_evaluation
from defense.defense import train_with_trades
from database.operations import save_model, save_metrics


class RobustnessAgent:

    def __init__(
        self,
        train_loader,
        test_loader,
        models_dir="models",
        log_file="logs/agent_log.txt",
        gap_threshold=25,
        device="cpu"
    ):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.models_dir = models_dir
        self.log_file = log_file
        self.gap_threshold = gap_threshold
        self.device = device

        self.model = None
        self.current_version = None
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    # ------------------------------
    # GET NEXT MODEL VERSION
    # ------------------------------
    def get_next_model_version(self):

        existing = [
            f for f in os.listdir(self.models_dir)
            if f.startswith("model_v") and f.endswith(".pth")
        ]

        if len(existing) == 0:
            return "model_v1.pth"

        versions = [
            int(f.split("_v")[1].split(".")[0])
            for f in existing
        ]

        next_version = max(versions) + 1

        return f"model_v{next_version}.pth"

    # ------------------------------
    # LOAD LATEST MODEL
    # ------------------------------
    def load_latest_model(self):

        models = [
            f for f in os.listdir(self.models_dir)
            if f.startswith("model_v") and f.endswith(".pth")
        ]

        if len(models) == 0:
            raise Exception("No model found in models directory.")

        versions = [
            int(m.split("_v")[1].split(".")[0])
            for m in models
        ]

        latest_version = max(versions)

        model_path = os.path.join(
            self.models_dir,
            f"model_v{latest_version}.pth"
        )

        print(f"\nLoading latest model: model_v{latest_version}.pth")

        model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        model.to(self.device)
        model.eval()
        self.current_version = f"model_v{latest_version}"
        return model

    # ------------------------------
    # VULNERABILITY ANALYSIS
    # ------------------------------
    def analyze_vulnerability(self, metrics):

        attack_scores = {
            "FGSM": metrics["FGSM Accuracy"],
            "PGD": metrics["PGD Accuracy"],
            "BIM": metrics["BIM Accuracy"],
            "CW": metrics["CW Accuracy"]
        }

        weakest_attack = min(
            attack_scores,
            key=attack_scores.get
        )

        print("\nVulnerability Analysis:")
        print("Weakest attack:", weakest_attack)

        return weakest_attack
    # ------------------------------
    # LOGGING
    # ------------------------------

    def log(self, text):

        with open(self.log_file, "a", encoding="utf-8") as f:

            f.write("\n")
            f.write(str(datetime.now()) + "\n")
            f.write(text + "\n")

    # ------------------------------
    # AGENT DECISION STEP
    # ------------------------------
    def evaluate_and_decide(self):

        print("\nAgent running robustness evaluation...\n")

        # Always load latest model
        self.model = self.load_latest_model()

        metrics = run_full_evaluation(
            self.model,
            self.test_loader
        )
        print("Printing the metric keys:")
        for key in metrics.keys():
            print(key)
        clean_acc = metrics["Clean Accuracy"]
        worst_acc = metrics["Worst-case Accuracy"]
        gap = metrics["Robustness Gap"]
        weakest_attack = self.analyze_vulnerability(metrics)
        # ------------------------------
        # SAVE METRICS TO DATABASE
        # ------------------------------
        save_metrics(
            model_version=self.current_version,   # use the current model version
            metrics_dict={
                "clean": clean_acc,
                "fgsm": metrics["FGSM Accuracy"],
                "pgd": metrics["PGD Accuracy"],
                "bim": metrics["BIM Accuracy"],
                "cw": metrics["CW Accuracy"]
            }
        )
        print("Clean Accuracy:", clean_acc)
        print("Worst-case Accuracy:", worst_acc)
        print("Robustness Gap:", gap)

        log_text = f"""
Clean Accuracy: {clean_acc:.2f}
Worst-case Accuracy: {worst_acc:.2f}
Robustness Gap: {gap:.2f}
"""

        # ------------------------------
        # DECISION
        # ------------------------------
        if gap > self.gap_threshold and worst_acc < clean_acc:

            print("\n⚠ Robustness degradation detected.")
            print("Agent triggering adversarial training...\n")

            self.log(log_text + "\nThreshold exceeded → retraining triggered")

            self.retrain_model(weakest_attack)

        else:

            print("\nModel robustness acceptable. No retraining needed.")

            self.log(log_text + "\nNo retraining required")

    # ------------------------------
    # RETRAIN MODEL
    # ------------------------------
    def retrain_model(self, weakest_attack):

        print("\nAgent selecting defense strategy...")

        if weakest_attack == "PGD":
            epochs = 5
        else:
            epochs = 3

        self.model = train_with_trades(
            self.model,
            self.train_loader,
            epochs=epochs
        )

        new_model_name = self.get_next_model_version()

        save_path = os.path.join(
            self.models_dir,
            new_model_name
        )

        torch.save(self.model, save_path)
        # ------------------------------
        # SAVE MODEL INFO TO DATABASE
        # ------------------------------
        save_model(
            version=new_model_name,
            path=save_path,
            clean_acc=0,
            worst_acc=0,
            gap=0
        )
        print(f"\nNew model saved: {save_path}")

        self.log(f"New model saved: {new_model_name}")

    # ------------------------------
    # RUN ONCE
    # ------------------------------
    def run_once(self):

        self.evaluate_and_decide()

    # ------------------------------
    # RUN PERIODICALLY
    # ------------------------------
    def run_periodic(self, interval_days=3):

        seconds = interval_days * 24 * 60 * 60

        while True:

            self.evaluate_and_decide()

            print(
                f"\nAgent sleeping for {interval_days} days..."
            )

            time.sleep(seconds)
