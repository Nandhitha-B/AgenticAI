import torch
import time
from defense import SmoothingDefense
from retrain import run_adversarial_retrain


class RobustnessAgent:

    def __init__(self, model, device, train_loader=None, config=None):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.config = config or {}

        self.conf_threshold = self.config.get("conf_threshold", 0.7)
        self.robust_target = self.config.get("robust_target", 70.0)
        self.primary_eps = self.config.get("primary_eps", 0.1)

        self.pgd_params = {
            "alpha": self.config.get("pgd_alpha", 0.01),
            "steps": int(self.config.get("pgd_steps", 10))
        }

        self.history = []
        print("Agentic AI initialized successfully.")

    def classify_image(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return int(pred.item()), float(conf.item()), probs.cpu()

    def evaluate_robustness(self, image_tensor, defense=None):

        image_tensor = image_tensor.clone().detach().to(self.device)
        image_tensor.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss()
        logits = self.model(image_tensor)
        pred = logits.argmax(dim=1)

        loss = criterion(logits, pred)
        loss.backward()

        adv_img = image_tensor + self.primary_eps * image_tensor.grad.sign()
        adv_img = torch.clamp(adv_img, -1, 1)

        with torch.no_grad():
            if defense:
                adv_img = defense(adv_img)
            adv_logits = self.model(adv_img)
            adv_pred = adv_logits.argmax(dim=1)

        robust_acc = float((adv_pred == pred).float().mean().item() * 100)
        return robust_acc

    def handle_uploaded_image(self, image_tensor):
        decisions = []
        start_time = time.time()

        pred, conf, _ = self.classify_image(image_tensor)
        record = {
            "pred_class": pred,
            "confidence": conf,
            "timestamp": start_time
        }

        print(f"Step 1: Classified as {pred} with confidence {conf:.2f}")

        if conf < self.conf_threshold:
            decision = "stop_low_confidence"
            reason = f"Low confidence ({conf:.2f} < {self.conf_threshold}). Stopping analysis."
            record.update({"decision": decision, "reason": reason})
            decisions.append((decision, reason))
            return decisions, record

        defense = SmoothingDefense(kernel_size=3)
        robust_acc = self.evaluate_robustness(image_tensor, defense)
        record["robust_accuracy"] = robust_acc
        print(f"Step 3: Evaluated robust accuracy = {robust_acc:.2f}%")

        if robust_acc >= self.robust_target:
            decision = "report_no_retrain"
            reason = f"Robust accuracy ({robust_acc:.2f}%) ≥ target ({self.robust_target}%). No retraining needed."
            record.update({"decision": decision, "reason": reason})
            decisions.append((decision, reason))
            return decisions, record

        if self.train_loader is not None:
            decision = "perform_retrain"
            reason = f"Robust accuracy ({robust_acc:.2f}%) < target ({self.robust_target}%). Performing adversarial retraining."
            print(f"Step 4: {reason}")
            decisions.append((decision, reason))

            retrain_cfg = self.config.get("retrain_cfg", {
                "epochs": 1,
                "lr": 0.001,
                "eps": self.primary_eps,
                "alpha": self.pgd_params["alpha"],
                "steps": self.pgd_params["steps"],
                "save_path": "models/retrained_model.pth"
            })

            save_path, retrain_metrics = run_adversarial_retrain(
                self.model, self.train_loader, self.device, retrain_cfg, defense=defense
            )

            self.model.load_state_dict(torch.load(
                save_path, map_location=self.device))

            new_robust_acc = self.evaluate_robustness(image_tensor, defense)
            record.update({
                "post_retrain_robust": new_robust_acc,
                "retrain_metrics": retrain_metrics
            })
            decisions.append(
                ("re_evaluate", f"New robust accuracy = {new_robust_acc:.2f}%"))
        else:
            decision = "recommend_retrain"
            reason = f"Robust accuracy ({robust_acc:.2f}%) < target, but no training data available. Recommend retraining."
            print(f"Step 4: {reason}")
            record.update({"decision": decision, "reason": reason})
            decisions.append((decision, reason))

        return decisions, record
