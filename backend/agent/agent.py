import torch
import time
from defense import SmoothingDefense
from retrain import run_adversarial_retrain
from attacks.cw_attack import CarliniWagnerAttack


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

        self.bim_params = {
            "alpha": self.config.get("bim_alpha", 0.01),
            "steps": int(self.config.get("bim_steps", 10))
        }

        self.cw_attack = CarliniWagnerAttack(
    model=self.model,
    device=self.device,
    c=self.config["cw"]["c"],
    kappa=self.config["cw"]["kappa"],
    steps=self.config["cw"]["steps"],
    lr=self.config["cw"]["lr"]
)


        self.history = []
        print("Agentic AI initialized successfully.")

    def classify_image(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return int(pred.item()), float(conf.item()), probs.cpu()
    
    def cw_probe(self, image_tensor, label, defense=None):
        self.model.eval()

        adv_img = self.cw_attack.attack(image_tensor, label)

        if defense:
            adv_img = defense(adv_img)

        with torch.no_grad():
            logits = self.model(adv_img)
            adv_pred = logits.argmax(dim=1)

        return int(adv_pred.item()) == int(label.item())

    def bim_probe(self, image_tensor, label, defense=None):
        """BIM (Basic Iterative Method) attack probe."""
        from eval_utils import bim_attack
        self.model.eval()

        # Apply BIM attack
        adv_img = bim_attack(
            self.model,
            image_tensor,
            label,
            eps=self.primary_eps,
            alpha=self.bim_params["alpha"],
            steps=self.bim_params["steps"],
            defense=defense
        )

        if defense and defense is None:  # defense is applied in bim_attack if provided
            adv_img = defense(adv_img)

        with torch.no_grad():
            logits = self.model(adv_img)
            adv_pred = logits.argmax(dim=1)

        return int(adv_pred.item()) == int(label.item())


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

        # --- Confidence gate (KEEP THIS) ---
        if conf < self.conf_threshold:
            decisions.append((
                "stop_low_confidence",
                f"Low confidence ({conf:.2f}). Skipping robustness probes."
            ))
            record["decision"] = "stop_low_confidence"
            return decisions, record

        # --- Defense selection ---
        defense = SmoothingDefense(kernel_size=3) if self.config["use_defense"] else None

        # --- PROBE MECHANISM ---
        clean_pred, clean_conf, probe_results = self.run_probes(
            image_tensor, defense)

        probe_analysis = self.analyze_probes(probe_results)

        record["probe_results"] = probe_results
        record["probe_analysis"] = probe_analysis

        # --- AGENT DECISION ---
        if not probe_analysis["needs_retrain"]:
            decisions.append((
                "no_retrain",
                "Probes show stable behavior across perturbations."
            ))
            record["decision"] = "no_retrain"
            return decisions, record

        # --- RETRAIN IF NEEDED ---
        if self.train_loader is not None:
            decisions.append((
                "perform_retrain",
                "Probe failures indicate model instability. Retraining triggered."
            ))

            retrain_cfg = self.config["retrain_cfg"]

            save_path, retrain_metrics = run_adversarial_retrain(
                self.model,
                self.train_loader,
                self.device,
                retrain_cfg,
                defense=defense
            )

            self.model.load_state_dict(
                torch.load(save_path, map_location=self.device)
            )

            record["retrain_metrics"] = retrain_metrics
            record["decision"] = "perform_retrain"

        else:
            decisions.append((
                "recommend_retrain",
                "Probe failures detected, but no training data available."
            ))
            record["decision"] = "recommend_retrain"

        return decisions, record

    def run_probes(self, image_tensor, defense):
        probe_results = []

        self.model.eval()

        clean_pred, clean_conf, _ = self.classify_image(image_tensor)
        label = torch.tensor([clean_pred], device=self.device)

        # --- FGSM-style epsilon probes ---
        for eps in self.config["eps_list"]:
            self.primary_eps = eps

            adv_robust = self.evaluate_robustness(
                image_tensor, defense)

            probe_results.append({
                "type": "fgsm",
                "epsilon": eps,
                "robust_accuracy": adv_robust
            })

        # --- C&W PROBE (NEW) ---
        cw_success = self.cw_probe(image_tensor, label, defense)

        probe_results.append({
            "type": "cw",
            "epsilon": "adaptive",
            "robust_accuracy": 100.0 if cw_success else 0.0
        })

        # --- BIM PROBE (NEW) ---
        bim_success = self.bim_probe(image_tensor, label, defense)

        probe_results.append({
            "type": "bim",
            "epsilon": self.primary_eps,
            "robust_accuracy": 100.0 if bim_success else 0.0
        })

        return clean_pred, clean_conf, probe_results

    def analyze_probes(self, probe_results):
        """
        Converts probe outcomes into a retraining signal
        """
        failures = 0

        for probe in probe_results:
            if probe["robust_accuracy"] < 50.0:
                failures += 1

        failure_ratio = failures / len(probe_results)

        return {
            "failures": failures,
            "failure_ratio": failure_ratio,
            "needs_retrain": failure_ratio >= 0.5
        }


