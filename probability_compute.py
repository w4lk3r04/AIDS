"""
probability_compute.py
----------------------
Probability Computation Helpers for Spam Classification

Tasks
-----
  Task 1 : Compute word probabilities with Laplace smoothing
  Task 2 : MAP and ML classification functions
  Task 3 : Generate test emails and evaluate both classifiers

Usage:
    python3 probability_compute.py
"""

import os
import math
import random

from generate_dataset import (
    load_dictionary,
    sample_word_bag,
    build_local_email,
    SPAM_DICT_FILE,
    HAM_DICT_FILE,
    LMIN, LMAX, PS, QS,
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SPAM_FILE   = "spam_emails"
HAM_FILE    = "ham_emails"
TEST_SEED   = 99        # seed for generating test emails
N_TEST_EACH = 5         # test emails per class (5 spam + 5 ham = 10 total)


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────
def load_emails_from_file(filepath: str) -> list[str]:
    """Read either a consolidated email file or a directory of one-email text files."""
    if os.path.isdir(filepath):
        emails: list[str] = []
        for name in sorted(os.listdir(filepath)):
            if not name.endswith(".txt"):
                continue
            full_path = os.path.join(filepath, name)
            with open(full_path) as f:
                text = f.read().strip()
            if text:
                emails.append(text)
        if not emails:
            raise FileNotFoundError(f"No .txt email files found in directory '{filepath}'.")
        return emails

    if os.path.isfile(filepath):
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]

    raise FileNotFoundError(
        f"Path '{filepath}' not found — run generate_dataset.py first."
    )


def tokenize(email: str) -> list[str]:
    """Lowercase and split an email string into individual word tokens."""
    return email.lower().split()


# ══════════════════════════════════════════════════════════════
#  Task 1 — Compute Word Probabilities
# ══════════════════════════════════════════════════════════════
def count_words(emails: list[str]) -> dict[str, int]:
    """
    Count how many times each word appears across a list of emails.
    Returns a dict mapping word → total count.
    """
    counts: dict[str, int] = {}
    for email in emails:
        for word in tokenize(email):
            counts[word] = counts.get(word, 0) + 1
    return counts


def compute_priors(n_spam: int, n_ham: int) -> tuple[float, float]:
    """
    Compute prior class probabilities.
        P(spam) = n_spam / (n_spam + n_ham)
        P(ham)  = n_ham  / (n_spam + n_ham)
    Returns (p_spam, p_ham).
    """
    n_total = n_spam + n_ham
    return n_spam / n_total, n_ham / n_total


def compute_likelihoods(
    spam_counts: dict[str, int],
    ham_counts:  dict[str, int],
    vocabulary:  set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute Laplace-smoothed conditional probabilities for every word
    in the vocabulary:
        P(word | spam) = (count(word, spam) + 1) / (total_spam_words + |V|)
        P(word | ham)  = (count(word, ham)  + 1) / (total_ham_words  + |V|)

    Returns (p_word_spam, p_word_ham) — dicts mapping word → probability.
    """
    V  = len(vocabulary)
    ts = sum(spam_counts.values())
    th = sum(ham_counts.values())

    p_word_spam = dict(
        (w, (spam_counts.get(w, 0) + 1) / (ts + V)) for w in vocabulary
    )
    p_word_ham = dict(
        (w, (ham_counts.get(w, 0) + 1) / (th + V)) for w in vocabulary
    )
    return p_word_spam, p_word_ham


def build_model(spam_emails: list[str], ham_emails: list[str]) -> dict:
    """
    Full Task 1 pipeline: count words, compute priors and likelihoods.

    Returns a model dict with keys:
        p_spam, p_ham           — prior probabilities
        p_word_spam, p_word_ham — word → P(word | class)
        vocabulary              — set of all known words
        vocab_size              — |V|
        total_spam_words        — Σ word counts in spam corpus
        total_ham_words         — Σ word counts in ham corpus
    """
    spam_counts = count_words(spam_emails)
    ham_counts  = count_words(ham_emails)
    vocabulary  = set(spam_counts) | set(ham_counts)

    p_spam, p_ham           = compute_priors(len(spam_emails), len(ham_emails))
    p_word_spam, p_word_ham = compute_likelihoods(spam_counts, ham_counts, vocabulary)

    return {
        "p_spam":           p_spam,
        "p_ham":            p_ham,
        "p_word_spam":      p_word_spam,
        "p_word_ham":       p_word_ham,
        "vocabulary":       vocabulary,
        "vocab_size":       len(vocabulary),
        "total_spam_words": sum(spam_counts.values()),
        "total_ham_words":  sum(ham_counts.values()),
    }


def print_model_summary(model: dict) -> None:
    """Print priors, vocabulary size, and sample word likelihoods."""
    print(f"\n  Prior P(spam) = {model['p_spam']:.4f}   P(ham) = {model['p_ham']:.4f}")
    print(f"  Vocabulary size: {model['vocab_size']} unique words")
    print(f"\n  Sample word likelihood ratios:")
    sample_words = ["free", "meeting", "bitcoin", "schedule", "urgent", "report"]
    for word in sample_words:
        lr = word_likelihood_ratio(word, model)
        ps = model["p_word_spam"].get(word, 0)
        ph = model["p_word_ham"].get(word, 0)
        print(f"    LR('{word:<10}') = {lr:>7.4f}  "
              f"[P(w|spam)={ps:.5f}  P(w|ham)={ph:.5f}]")


# ══════════════════════════════════════════════════════════════
#  Task 2 — Classification Functions
# ══════════════════════════════════════════════════════════════
def word_likelihood_ratio(word: str, model: dict) -> float:
    """
    Likelihood ratio for a single word:
        LR(word) = P(word | spam) / P(word | ham)

    Words not seen during training receive Laplace-smoothed probability.
    """
    V = model["vocab_size"]
    if word in model["vocabulary"]:
        p_s = model["p_word_spam"][word]
        p_h = model["p_word_ham"][word]
    else:
        p_s = 1 / (model["total_spam_words"] + V)
        p_h = 1 / (model["total_ham_words"]  + V)
    return p_s / p_h


def email_log_likelihood_ratio(email: str, model: dict) -> float:
    """
    Log likelihood ratio for an entire email (sum over words):
        log LR(email) = Σ_i  log LR(word_i)

    Using log-space arithmetic prevents floating-point underflow when
    multiplying many small probabilities together.
    """
    log_lr = 0.0
    for word in tokenize(email):
        lr = word_likelihood_ratio(word, model)
        log_lr += math.log(lr) if lr > 0 else -1e9
    return log_lr


def classify_map(email: str, model: dict) -> str:
    """
    MAP (Maximum A Posteriori) classification.

    Decision rule — classify as spam when:
        P(spam | email) > P(ham | email)
    Which in log-likelihood-ratio form becomes:
        log LR(email) > log( P(ham) / P(spam) )

    The threshold shifts with the class imbalance in the training data.
    """
    log_lr    = email_log_likelihood_ratio(email, model)
    threshold = math.log(model["p_ham"] / model["p_spam"])
    return "spam" if log_lr > threshold else "ham"


def classify_ml(email: str, model: dict) -> str:
    """
    ML (Maximum Likelihood) classification.

    Decision rule — classify as spam when:
        log LR(email) > 0
    (equivalent to P(email|spam) > P(email|ham))

    Ignores prior class probabilities entirely.
    """
    log_lr = email_log_likelihood_ratio(email, model)
    return "spam" if log_lr > 0 else "ham"


# ══════════════════════════════════════════════════════════════
#  Task 3 — Generate Test Emails and Evaluate
# ══════════════════════════════════════════════════════════════
def generate_test_emails(
    spam_dict: list[str],
    ham_dict:  list[str],
    n_each:    int,
    seed:      int,
) -> list[tuple[str, str]]:
    """
    Generate n_each spam and n_each ham test emails using the same
    probabilistic model as Part I (ps=0.75 for spam, qs=0.20 for ham).
    Returns a shuffled list of (email_text, true_label) tuples.
    """
    rng = random.Random(seed)
    emails = []
    for _ in range(n_each):
        length = rng.randint(LMIN, LMAX)
        word_bag = sample_word_bag(spam_dict, ham_dict, PS, length, rng)
        emails.append((build_local_email("spam", word_bag, rng), "spam"))
    for _ in range(n_each):
        length = rng.randint(LMIN, LMAX)
        word_bag = sample_word_bag(spam_dict, ham_dict, QS, length, rng)
        emails.append((build_local_email("ham", word_bag, rng), "ham"))
    random.Random(seed).shuffle(emails)
    return emails


def evaluate_classifier(
    test_emails: list[tuple[str, str]],
    model:       dict,
) -> list[dict]:
    """
    Run MAP and ML classification on every test email.

    Prints a results table and accuracy summary.
    Returns a list of result dicts (used by neural_network.py for comparison).
    """
    print("\n  " + "─" * 70)
    fmt = "  {:<4} {:<8} {:<8} {:<8} {:<8}  {}"
    print(fmt.format("No.", "True", "MAP", "ML", "Agree?", "Email (truncated)"))
    print("  " + "─" * 70)

    results      = []
    map_correct  = 0
    ml_correct   = 0

    for i, (email, true_label) in enumerate(test_emails, 1):
        map_pred = classify_map(email, model)
        ml_pred  = classify_ml(email, model)
        agree    = "yes" if map_pred == ml_pred else "DIFFER"

        map_correct += int(map_pred == true_label)
        ml_correct  += int(ml_pred  == true_label)

        snippet = (email[:45] + "…") if len(email) > 45 else email
        print(fmt.format(i, true_label, map_pred, ml_pred, agree, snippet))

        results.append({
            "email":      email,
            "true_label": true_label,
            "map_pred":   map_pred,
            "ml_pred":    ml_pred,
        })

    n = len(test_emails)
    print(f"\n  MAP accuracy : {map_correct}/{n}  ({100 * map_correct / n:.0f}%)")
    print(f"  ML  accuracy : {ml_correct}/{n}  ({100 * ml_correct  / n:.0f}%)")

    differing = [r for r in results if r["map_pred"] != r["ml_pred"]]
    if differing:
        print(f"\n  Cases where MAP and ML disagree ({len(differing)}):")
        for r in differing:
            print(f"    true={r['true_label']:<5}  MAP={r['map_pred']:<5}  "
                  f"ML={r['ml_pred']:<5}  {r['email'][:50]}")
    else:
        print("\n  MAP and ML agreed on every test email.")

    return results


def print_analysis(model: dict) -> None:
    """
    Print a brief discussion of MAP vs ML and potential misclassification causes.
    """
    p_spam = model["p_spam"]
    p_ham  = model["p_ham"]
    print(f"""
  Analysis
  ─────────────────────────────────────────────────────────────────────
  MAP uses the prior probabilities P(spam)={p_spam:.3f} and P(ham)={p_ham:.3f}.
  Because the training corpus has more ham than spam, MAP shifts the
  decision boundary toward spam — an email must show a stronger spam
  signal before MAP labels it spam. ML ignores this imbalance and uses
  a fixed threshold of zero (log LR > 0), so on imbalanced datasets
  it is more aggressive in labelling borderline emails as spam.

  Potential misclassification causes:
    - Short emails: a single word can dominate the log-likelihood ratio,
      making the classifier fragile to vocabulary overlap.
    - Words that appear in both dictionaries carry weak signal;
      they contribute close to LR ≈ 1 (log LR ≈ 0) and dilute evidence.
    - The synthetic model draws ham words into spam emails with prob 0.25,
      so some spam emails contain mostly ham words by chance.
  ─────────────────────────────────────────────────────────────────────
""")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 70)
    print("  Naive Bayes Classifier")
    print("=" * 70)

    # Load resources
    print("\n  Loading dictionaries …")
    spam_dict = load_dictionary(SPAM_DICT_FILE)
    ham_dict  = load_dictionary(HAM_DICT_FILE)

    print("\n  Loading training emails …")
    spam_emails = load_emails_from_file(SPAM_FILE)
    ham_emails  = load_emails_from_file(HAM_FILE)
    print(f"  {len(spam_emails)} spam  +  {len(ham_emails)} ham  training emails loaded.")

    # Task 1
    print("\n  ── Task 1: Compute Word Probabilities " + "─" * 32)
    model = build_model(spam_emails, ham_emails)
    print_model_summary(model)

    # Task 2 — functions defined above; exercised in Task 3

    # Task 3
    print("\n  ── Task 3: Test and Evaluate " + "─" * 41)
    test_emails = generate_test_emails(spam_dict, ham_dict, N_TEST_EACH, TEST_SEED)
    print(f"  Generated {len(test_emails)} test emails "
          f"({N_TEST_EACH} spam + {N_TEST_EACH} ham).")

    evaluate_classifier(test_emails, model)
    print_analysis(model)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
