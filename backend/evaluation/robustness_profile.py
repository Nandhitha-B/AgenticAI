# backend/evaluation/robustness_profile.py

from evaluation.metrics import (
    compute_metrics,
    compute_per_class_accuracy
)


# -------------------------------------------------
# 1️⃣ Worst-case Accuracy
# -------------------------------------------------
def compute_worst_case_accuracy(results_dict):

    attack_accuracies = [
        acc for key, acc in results_dict.items()
        if key.lower() != "clean"
    ]

    if len(attack_accuracies) == 0:
        return 0.0

    return min(attack_accuracies)


# -------------------------------------------------
# 2️⃣ Robustness Gap
# -------------------------------------------------
def compute_robustness_gap(clean_acc, worst_case_acc):
    return clean_acc - worst_case_acc


# -------------------------------------------------
# 3️⃣ Robustness Score (Normalized)
# -------------------------------------------------
def compute_robustness_score(clean_acc, worst_case_acc):

    if clean_acc == 0:
        return 0.0

    return worst_case_acc / clean_acc


# -------------------------------------------------
# 4️⃣ Per-Attack Robustness Profile
# -------------------------------------------------
def build_per_attack_profile(results_raw, num_classes):

    profile = {}

    for attack_name, data in results_raw.items():

        if attack_name == "clean":
            continue

        y_true = data["y_true"]
        y_pred = data["y_pred"]

        acc, _, _, _ = compute_metrics(y_true, y_pred)

        per_class_acc = compute_per_class_accuracy(
            y_true,
            y_pred,
            num_classes
        )

        profile[attack_name] = {
            "accuracy": acc,
            "per_class_accuracy": per_class_acc
        }

    return profile


# -------------------------------------------------
# 5️⃣ Full Robustness Profile Builder
# -------------------------------------------------
def build_full_robustness_profile(results_raw, num_classes):

    summary = {}

    attack_accuracies = {}

    # ---- Clean ----
    y_true_clean = results_raw["clean"]["y_true"]
    y_pred_clean = results_raw["clean"]["y_pred"]

    clean_acc, _, _, _ = compute_metrics(
        y_true_clean,
        y_pred_clean
    )

    summary["Clean Accuracy"] = clean_acc

    # ---- Attacks ----
    for attack_name, data in results_raw.items():

        if attack_name == "clean":
            continue

        y_true = data["y_true"]
        y_pred = data["y_pred"]

        acc, _, _, _ = compute_metrics(y_true, y_pred)

        summary[f"{attack_name.upper()} Accuracy"] = acc
        attack_accuracies[attack_name] = acc

    # ---- Worst-case & Gap ----
    worst_case_acc = compute_worst_case_accuracy(
        {"clean": clean_acc, **attack_accuracies}
    )

    robustness_gap = compute_robustness_gap(
        clean_acc,
        worst_case_acc
    )

    robustness_score = compute_robustness_score(
        clean_acc,
        worst_case_acc
    )

    summary["Worst-case Accuracy"] = worst_case_acc
    summary["Robustness Gap"] = robustness_gap
    summary["Robustness Score"] = robustness_score

    # ---- Per-attack Detailed Profile ----
    summary["Detailed Attack Profile"] = build_per_attack_profile(
        results_raw,
        num_classes
    )

    return summary
