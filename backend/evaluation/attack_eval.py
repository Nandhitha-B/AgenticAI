# backend/evaluation/attack_eval.py

import torch
import torch.nn.functional as F

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.bim import bim_attack
from attacks.cw import cw_attack

from evaluation.metrics import (
    compute_metrics,
    compute_attack_success_rate,
    compute_confidence_drop,
    compute_worst_case_accuracy,
    compute_robustness_gap
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Attack Registry
# -----------------------------
ATTACKS = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
    "bim": bim_attack,
    "cw": cw_attack
}


# -----------------------------
# Core Evaluation Function
# -----------------------------
def evaluate_model(model, loader, attack_name=None, attack_params=None):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply attack if provided
        if attack_name is not None:
            attack_fn = ATTACKS[attack_name]
            images = attack_fn(
                model,
                images,
                labels,
                device=device,
                **attack_params
            )

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    return all_labels, all_preds, all_probs


# -----------------------------
# Full Robustness Evaluation
# -----------------------------
def run_full_evaluation(model, test_loader):

    results = {}
    attack_accuracies = {}
    attack_asr = {}
    confidence_drops = {}

    # -----------------------------
    # CLEAN
    # -----------------------------
    y_true_clean, y_pred_clean, probs_clean = evaluate_model(
        model, test_loader
    )

    clean_acc, _, _, _ = compute_metrics(y_true_clean, y_pred_clean)
    results["Clean Accuracy"] = clean_acc

    # -----------------------------
    # ATTACK SETTINGS
    # -----------------------------
    attack_configs = {
        "fgsm": {"epsilon": 0.03},
        "pgd": {"epsilon": 0.0025, "alpha": 0.001, "iters": 4},
        "bim": {"epsilon": 0.0005, "alpha": 0.001, "iters": 4},
        "cw": {"c": 1e-4, "kappa": 0, "iters": 5, "lr": 0.01}
    }

    # -----------------------------
    # RUN EACH ATTACK
    # -----------------------------
    for attack_name, params in attack_configs.items():

        y_true_adv, y_pred_adv, probs_adv = evaluate_model(
            model,
            test_loader,
            attack_name=attack_name,
            attack_params=params
        )

        acc, _, _, _ = compute_metrics(y_true_adv, y_pred_adv)
        results[f"{attack_name.upper()} Accuracy"] = acc
        attack_accuracies[attack_name] = acc

        # Attack Success Rate
        asr = compute_attack_success_rate(
            y_pred_clean,
            y_pred_adv,
            y_true_clean
        )
        attack_asr[attack_name] = asr

        # Confidence Drop
        conf_drop = compute_confidence_drop(
            probs_clean,
            probs_adv,
            y_true_clean
        )
        confidence_drops[attack_name] = conf_drop

    # -----------------------------
    # WORST CASE
    # -----------------------------
    worst_case_acc = compute_worst_case_accuracy(attack_accuracies)
    robustness_gap = compute_robustness_gap(clean_acc, worst_case_acc)

    results["Worst-case Accuracy"] = worst_case_acc
    results["Robustness Gap"] = robustness_gap
    results["Attack Success Rate (%)"] = attack_asr
    results["Confidence Drop"] = confidence_drops

    return results
