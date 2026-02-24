import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred):
    acc = (y_true == y_pred).sum().item() / len(y_true) * 100

    precision = precision_score(
        y_true.cpu(), y_pred.cpu(),
        average='weighted', zero_division=0
    )

    recall = recall_score(
        y_true.cpu(), y_pred.cpu(),
        average='weighted', zero_division=0
    )

    f1 = f1_score(
        y_true.cpu(), y_pred.cpu(),
        average='weighted', zero_division=0
    )

    return acc, precision, recall, f1


# -------------------------------------------------
# 2️⃣ Attack Success Rate (ASR)
# -------------------------------------------------
def compute_attack_success_rate(clean_preds, adv_preds, true_labels):
    """
    ASR = % of correctly classified clean samples
          that become misclassified after attack
    """

    clean_correct = clean_preds == true_labels
    fooled = adv_preds != true_labels

    # Only consider originally correct samples
    valid_samples = clean_correct.sum().item()

    if valid_samples == 0:
        return 0.0

    attack_success = (clean_correct & fooled).sum().item()

    asr = attack_success / valid_samples * 100
    return asr


# -------------------------------------------------
# 3️⃣ Confidence Drop
# -------------------------------------------------
def compute_confidence_drop(clean_probs, adv_probs, true_labels):
    """
    Measures how much confidence on the true class drops
    after attack.
    """

    # True class confidence before attack
    clean_conf = clean_probs[torch.arange(len(true_labels)), true_labels]

    # True class confidence after attack
    adv_conf = adv_probs[torch.arange(len(true_labels)), true_labels]

    drop = torch.mean(clean_conf - adv_conf).item()

    return drop


# -------------------------------------------------
# 4️⃣ Per-Class Accuracy
# -------------------------------------------------
def compute_per_class_accuracy(y_true, y_pred, num_classes):
    """
    Returns dictionary:
    {class_id: accuracy}
    """

    per_class_acc = {}

    for cls in range(num_classes):
        cls_mask = y_true == cls
        total = cls_mask.sum().item()

        if total == 0:
            per_class_acc[cls] = 0.0
        else:
            correct = (y_pred[cls_mask] == cls).sum().item()
            per_class_acc[cls] = correct / total * 100

    return per_class_acc


# -------------------------------------------------
# 5️⃣ Worst-case Accuracy
# -------------------------------------------------
def compute_worst_case_accuracy(attack_accuracies_dict):
    """
    attack_accuracies_dict example:
    {
        "FGSM": 67.2,
        "PGD": 52.1,
        "BIM": 49.8,
        "CW": 60.3
    }
    """

    if not attack_accuracies_dict:
        return 0.0

    return min(attack_accuracies_dict.values())


# -------------------------------------------------
# 6️⃣ Robustness Gap
# -------------------------------------------------
def compute_robustness_gap(clean_acc, worst_case_acc):
    """
    Difference between clean and worst-case attack accuracy
    """

    return clean_acc - worst_case_acc
