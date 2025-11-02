import torch
import pandas as pd


class RobustnessAgent:
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.history = []

    def evaluate_attack(self, attack, eps_values, alpha=0.01, iters=10):
        from attack_eval import test_metrics
        print(f"\n[Agent] Running {attack.upper()} attack sweep...")
        for eps in eps_values:
            acc, prec, rec, f1 = test_metrics(
                self.model, self.loader,
                attack_type=attack, epsilon=eps, alpha=alpha, iters=iters
            )
            print(f"  eps={eps:.2f} → acc={acc:.2f}%")
            self.history.append([attack, eps, acc, prec, rec, f1])

    def activate_defense_if_needed(self, threshold=80):
        from attack_eval import defense_metrics
        print("\n[Agent] Checking if defense is needed...")
        recent_acc = self.history[-1][2]
        if recent_acc < threshold:
            print(
                f"Accuracy dropped to {recent_acc:.2f}%, activating defense...")
            attack = self.history[-1][0]
            eps = self.history[-1][1]
            acc, prec, rec, f1 = defense_metrics(
                self.model, self.loader, attack_type=attack, epsilon=eps
            )
            print(f"Defense applied → acc={acc:.2f}%")
            self.history.append([f"Defense-{attack}", eps, acc, prec, rec, f1])
        else:
            print("No defense needed, model is robust enough.")

    def generate_report(self):
        df = pd.DataFrame(self.history, columns=[
            "Scenario", "Epsilon", "Accuracy (%)", "Precision", "Recall", "F1-score"
        ])
        print("\n=== AGENT REPORT ===")
        print(df.to_string(index=False))
        df.to_csv("agent_report.csv", index=False)

    def decide_next_action(self):
        # Analyze results
        last_acc = self.history[-1][2]
        if last_acc < 70:
            print(
                "🧠 Agent: Performance too low, switching to PGD defense training next...")
            return "defense"
        elif last_acc < 90:
            print("🧠 Agent: Try reducing epsilon and re-testing...")
            return "retune"
        else:
            print("🧠 Agent: Model is stable. Proceeding to report.")
            return "report"
