"""
neural_network.py
-----------------
Single-Neuron Neural Network Spam Classifier
CMU 04-625 Intrusion Detection Systems — Spring 2026

Tasks
-----
  Task 1 : Extract x_ham feature from every training email
  Task 2 : Train a single-neuron classifier with sigmoid + gradient descent
  Task 3 : Test the neuron and compare results with the Naive Bayes classifier

Usage:
    python3 neural_network.py

Requires naive_bayes.py to be run first (or imported) so that
evaluate_classifier() results are available for comparison.
"""

import os
import math
import random

from generate_dataset import (
    load_dictionary,
    generate_email,
    SPAM_DICT_FILE,
    HAM_DICT_FILE,
    LMIN, LMAX, PS, QS,
)
from probability_compute import (
    load_emails_from_file,
    build_model,
    classify_map,
    classify_ml,
    generate_test_emails,
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SPAM_FILE   = "spam_emails"
HAM_FILE    = "ham_emails"
TEST_SEED   = 99
N_TEST_EACH = 5

LEARNING_RATE = 0.5
MAX_EPOCHS    = 500
TOLERANCE     = 1e-6


# ──────────────────────────────────────────────────────────────
# Shared math helpers
# ──────────────────────────────────────────────────────────────
def sigmoid(z: float) -> float:
    """Sigmoid activation: σ(z) = 1 / (1 + e^{-z})"""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ══════════════════════════════════════════════════════════════
#  Task 1 — Feature Extraction
# ══════════════════════════════════════════════════════════════
def compute_x_ham(email: str, ham_dict_set: set[str]) -> float:
    """
    Compute the ham-word fraction for a single email:
        x_ham = |words in email that appear in D_ham| / |words in email|

    This scalar is the sole input feature to the single-neuron classifier.
    Returns 0.0 for empty emails.
    """
    tokens = email.lower().split()
    if not tokens:
        return 0.0
    return sum(1 for w in tokens if w in ham_dict_set) / len(tokens)


def build_feature_dataset(
    spam_emails:  list[str],
    ham_emails:   list[str],
    ham_dict_set: set[str],
) -> tuple[list[float], list[int]]:
    """
    Build (x_ham, label) pairs for all training emails.
        label = 0  →  spam
        label = 1  →  ham

    Returns parallel lists X (features) and y (labels).
    """
    X: list[float] = []
    y: list[int]   = []

    for email in spam_emails:
        X.append(compute_x_ham(email, ham_dict_set))
        y.append(0)

    for email in ham_emails:
        X.append(compute_x_ham(email, ham_dict_set))
        y.append(1)

    return X, y


def print_feature_summary(
    X: list[float],
    y: list[int],
    n_spam: int,
) -> None:
    """Print average x_ham values for spam and ham training emails."""
    spam_x = [X[i] for i in range(n_spam)]
    ham_x  = [X[i] for i in range(n_spam, len(X))]
    print(f"  Spam emails  — avg x_ham = {sum(spam_x) / len(spam_x):.4f}")
    print(f"  Ham  emails  — avg x_ham = {sum(ham_x)  / len(ham_x):.4f}")
    print(f"  Total training feature vectors: {len(X)}")


# ══════════════════════════════════════════════════════════════
#  Task 2 — Single-Neuron Classifier
# ══════════════════════════════════════════════════════════════
def initialise_weights(seed: int = 42) -> tuple[float, float]:
    """
    Initialise weight w and bias b to small uniform random values in [-0.1, 0.1].
    Returns (w, b).
    """
    rng = random.Random(seed)
    return rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)


def forward_pass(x: float, w: float, b: float) -> float:
    """
    Forward pass through the single neuron:
        z = w * x_ham + b
        y = sigmoid(z)
    Returns the predicted probability of being ham.
    """
    return sigmoid(w * x + b)


def compute_gradients(
    X: list[float],
    y: list[int],
    w: float,
    b: float,
) -> tuple[float, float, float]:
    """
    Compute batch gradients for MSE loss:
        Loss = (1/N) Σ (y_pred - y_true)²

    Derivatives (via chain rule through sigmoid):
        ∂Loss/∂w = (2/N) Σ (y_pred - y_true) · y_pred · (1 - y_pred) · x
        ∂Loss/∂b = (2/N) Σ (y_pred - y_true) · y_pred · (1 - y_pred)

    Returns (grad_w, grad_b, loss).
    """
    n      = len(X)
    grad_w = 0.0
    grad_b = 0.0
    loss   = 0.0

    for xi, yi in zip(X, y):
        y_pred = forward_pass(xi, w, b)
        error  = y_pred - yi
        delta  = error * y_pred * (1.0 - y_pred)
        grad_w += delta * xi
        grad_b += delta
        loss   += error ** 2

    return (2 * grad_w / n), (2 * grad_b / n), (loss / n)


def update_weights(
    w: float, b: float,
    grad_w: float, grad_b: float,
    lr: float,
) -> tuple[float, float]:
    """
    Gradient descent weight update:
        w ← w − lr · ∂Loss/∂w
        b ← b − lr · ∂Loss/∂b
    Returns updated (w, b).
    """
    return w - lr * grad_w, b - lr * grad_b


def compute_threshold(w: float, b: float) -> float:
    """
    Find the classification threshold x_T by solving:
        y(1) − y(x_T)  ≥  y(x_T) − y(0)
    i.e. x_T is the x_ham value where the neuron output is equidistant
    from y(0) (pure spam) and y(1) (pure ham):
        y(x_T) = ( y(0) + y(1) ) / 2

    Inverting the sigmoid:
        x_T = ( logit(midpoint) − b ) / w

    Returns 0.5 if the weight is too small to invert safely.
    """
    y0  = sigmoid(b)
    y1  = sigmoid(w + b)
    mid = max(1e-9, min(1 - 1e-9, (y0 + y1) / 2.0))
    logit_mid = math.log(mid / (1.0 - mid))
    return (logit_mid - b) / w if abs(w) > 1e-9 else 0.5


def train_neuron(
    X:   list[float],
    y:   list[int],
    lr:  float = LEARNING_RATE,
    epochs: int = MAX_EPOCHS,
    tol: float  = TOLERANCE,
    seed: int   = 42,
) -> tuple[float, float, float, list[float]]:
    """
    Full training loop using batch gradient descent.

    Stops early when the loss improvement between consecutive epochs
    falls below `tol` (convergence criterion).

    Returns (w, b, threshold, loss_history).
    """
    w, b       = initialise_weights(seed)
    prev_loss  = float("inf")
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        grad_w, grad_b, loss = compute_gradients(X, y, w, b)
        w, b                 = update_weights(w, b, grad_w, grad_b, lr)
        loss_history.append(loss)

        if epoch % 100 == 0:
            print(f"    epoch {epoch:>4}  loss={loss:.6f}  w={w:.4f}  b={b:.4f}")

        if abs(prev_loss - loss) < tol:
            print(f"    Converged at epoch {epoch}  loss={loss:.6f}")
            break

        prev_loss = loss

    threshold = compute_threshold(w, b)
    return w, b, threshold, loss_history


def classify_neuron(x_ham: float, threshold: float) -> str:
    """
    Classify a single email using the trained threshold:
        x_ham ≥ threshold  →  ham
        x_ham <  threshold →  spam
    """
    return "ham" if x_ham >= threshold else "spam"


def print_neuron_summary(w: float, b: float, threshold: float) -> None:
    """Print the trained neuron's weights and threshold."""
    y0 = sigmoid(b)
    y1 = sigmoid(w + b)
    print(f"\n  Trained weights : w = {w:.6f}   b = {b:.6f}")
    print(f"  y(0) = {y0:.4f}  (neuron output for pure-spam email)")
    print(f"  y(1) = {y1:.4f}  (neuron output for pure-ham  email)")
    print(f"  Classification threshold : x_ham ≥ {threshold:.6f} → ham")


# ══════════════════════════════════════════════════════════════
#  Task 3 — Test and Compare Classifiers
# ══════════════════════════════════════════════════════════════
def evaluate_and_compare(
    test_emails:  list[tuple[str, str]],
    w: float,
    b: float,
    threshold:    float,
    ham_dict_set: set[str],
    nb_model:     dict,
) -> None:
    """
    Classify each test email with:
      - Naive Bayes MAP
      - Naive Bayes ML
      - Single-neuron classifier

    Prints a side-by-side comparison table and accuracy summary.
    """
    print("\n  " + "─" * 76)
    fmt = "  {:<4} {:<6} {:<7} {:<8} {:<8} {:<8}  {}"
    print(fmt.format("No.", "True", "x_ham", "NB-MAP", "NB-ML", "Neuron", "Email"))
    print("  " + "─" * 76)

    nn_correct  = nb_map_correct = nb_ml_correct = 0
    n           = len(test_emails)

    for i, (email, true_label) in enumerate(test_emails, 1):
        x_ham    = compute_x_ham(email, ham_dict_set)
        nn_pred  = classify_neuron(x_ham, threshold)
        map_pred = classify_map(email, nb_model)
        ml_pred  = classify_ml(email, nb_model)

        nn_correct      += int(nn_pred  == true_label)
        nb_map_correct  += int(map_pred == true_label)
        nb_ml_correct   += int(ml_pred  == true_label)

        snippet = (email[:35] + "…") if len(email) > 35 else email
        print(fmt.format(i, true_label, f"{x_ham:.3f}",
                         map_pred, ml_pred, nn_pred, snippet))

    print(f"\n  Accuracy summary ({n} test emails):")
    print(f"    Naive Bayes MAP : {nb_map_correct}/{n}  ({100 * nb_map_correct / n:.0f}%)")
    print(f"    Naive Bayes ML  : {nb_ml_correct}/{n}  ({100 * nb_ml_correct  / n:.0f}%)")
    print(f"    Single Neuron   : {nn_correct}/{n}  ({100 * nn_correct        / n:.0f}%)")


def print_comparison_analysis(threshold: float, w: float, b: float) -> None:
    """Print a structured analysis of both classifiers' strengths and limitations."""
    y0 = sigmoid(b)
    y1 = sigmoid(w + b)
    print(f"""
  Analysis
  ─────────────────────────────────────────────────────────────────────
  Naive Bayes uses every word's individual P(word|spam) and P(word|ham),
  so it is sensitive to specific high-signal words (e.g. "urgent", "bitcoin")
  regardless of how many neutral words surround them.

  The single neuron compresses each email into one number — x_ham — and
  draws a single decision boundary at x_ham = {threshold:.4f}.
    y(0) = {y0:.4f}  →  output for an email with zero ham words (pure spam)
    y(1) = {y1:.4f}  →  output for an email with all ham words  (pure ham)

  Advantages of Naive Bayes:
    - Retains word-level discriminative signal.
    - Handles short emails reliably.
    - No training required; probabilities computed directly from counts.

  Advantages of the Single Neuron:
    - Extremely fast at inference time (one multiplication and comparison).
    - Robust to vocabulary size — x_ham is always in [0, 1].
    - Easy to interpret: the threshold has a clear geometric meaning.

  Where each classifier struggles:
    - Naive Bayes: words shared across both dictionaries dilute the LR.
    - Single Neuron: emails near the threshold are unreliable; word identity
      is lost, so "free budget meeting" and "free bitcoin now" look similar
      if they happen to share the same x_ham value.
  ─────────────────────────────────────────────────────────────────────
""")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 70)
    print("  Single-Neuron Neural Network Classifier")
    print("=" * 70)

    # Load resources
    print("\n  Loading dictionaries …")
    spam_dict = load_dictionary(SPAM_DICT_FILE)
    ham_dict  = load_dictionary(HAM_DICT_FILE)
    ham_dict_set = set(w.lower() for w in ham_dict)

    print("\n  Loading training emails …")
    spam_emails = load_emails_from_file(SPAM_FILE)
    ham_emails  = load_emails_from_file(HAM_FILE)
    print(f"  {len(spam_emails)} spam  +  {len(ham_emails)} ham  training emails loaded.")

    # Also build the Naive Bayes model for Task 3 comparison
    nb_model = build_model(spam_emails, ham_emails)

    # Task 1 — Feature extraction
    print("\n  ── Task 1: Feature Extraction " + "─" * 40)
    X, y = build_feature_dataset(spam_emails, ham_emails, ham_dict_set)
    print_feature_summary(X, y, len(spam_emails))

    # Task 2 — Train single neuron
    print("\n  ── Task 2: Train Single-Neuron Classifier " + "─" * 28)
    print(f"  Settings: lr={LEARNING_RATE}, max_epochs={MAX_EPOCHS}, tol={TOLERANCE}\n")
    w, b, threshold, _ = train_neuron(X, y,
                                      lr=LEARNING_RATE,
                                      epochs=MAX_EPOCHS,
                                      tol=TOLERANCE)
    print_neuron_summary(w, b, threshold)

    # Task 3 — Test and compare
    print("\n  ── Task 3: Test and Compare Classifiers " + "─" * 30)
    test_emails = generate_test_emails(spam_dict, ham_dict, N_TEST_EACH, TEST_SEED)
    print(f"  Generated {len(test_emails)} test emails "
          f"({N_TEST_EACH} spam + {N_TEST_EACH} ham).")

    evaluate_and_compare(test_emails, w, b, threshold, ham_dict_set, nb_model)
    print_comparison_analysis(threshold, w, b)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
