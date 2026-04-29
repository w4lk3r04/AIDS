"""
Setup
-----
Choose ONE provider and set the matching env variable:
 
  Claude (Anthropic):
      export ANTHROPIC_API_KEY="sk-ant-..."
 
  OpenAI:
      export OPENAI_API_KEY="sk-..."

You can also place the key in a local .env file; the script will load it automatically.
 
Then set PROVIDER below (or pass --provider on the command line).
 
Usage
-----
    python3 generate_dataset.py                    # uses local template generation
    python3 generate_dataset.py --provider claude  # force Claude
    python3 generate_dataset.py --provider openai  # force OpenAI
    python3 generate_dataset.py --samples 5        # generate only 5 of each (for testing)
"""
 
import os
import sys
import random
import argparse
import time
 
# ──────────────────────────────────────────────────────────────
# Configuration — edit here or pass CLI flags
# ──────────────────────────────────────────────────────────────
PROVIDER       = "local"            # "local", "claude", or "openai"
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"   # fast + cheap for bulk generation
OPENAI_MODEL   = "gpt-4o-mini"                 # fast + cheap for bulk generation
 
PS   = 0.75
QS   = 0.20
LMIN = 5
LMAX = 15
N_SPAM = 1000
N_HAM  = 2300
 
SPAM_DICT_FILE = "dataset_dict/spam_dictionary.txt"
HAM_DICT_FILE  = "dataset_dict/ham_dictionary.txt"
SPAM_OUT_DIR   = "spam_emails"
HAM_OUT_DIR    = "ham_emails"
 
RANDOM_SEED    = 42
RETRY_DELAY    = 2      # seconds between retries on API error
MAX_RETRIES    = 3


def load_dotenv_if_present(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_if_present()
 
 
# ──────────────────────────────────────────────────────────────
# Dictionary helpers (same as original script)
# ──────────────────────────────────────────────────────────────
def load_dictionary(filepath: str) -> list[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dictionary not found: {filepath}")
    with open(filepath) as f:
        words = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(words):>4} words from '{filepath}'")
    return words
 
 
def sample_word_bag(
    spam_dict: list[str],
    ham_dict:  list[str],
    p_spam_word: float,
    length: int,
    rng: random.Random,
) -> list[str]:
    """
    Sample a bag of words using the original probabilistic model.
    This becomes the vocabulary hint sent to the LLM.
    """
    bag = []
    for _ in range(length):
        if rng.random() < p_spam_word:
            bag.append(rng.choice(spam_dict))
        else:
            bag.append(rng.choice(ham_dict))
    return bag
 
 
# ──────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────
def build_prompt(label: str, word_bag: list[str]) -> str:
    """
    Build a prompt that instructs the LLM to write one realistic email
    using as many words from word_bag as naturally possible.
    """
    category_hint = {
        "spam": (
            "a spam or phishing email. It should sound persuasive, urgent, "
            "or promotional — the kind of email a real spammer would send."
        ),
        "ham": (
            "a legitimate professional or personal email. It should sound "
            "natural, workplace-appropriate, and genuine."
        ),
    }[label]
 
    return (
        f"Write {category_hint}\n\n"
        f"Naturally incorporate as many of these words as possible into the email "
        f"(you don't have to use all of them):\n"
        f"{', '.join(word_bag)}\n\n"
        f"Rules:\n"
        f"- Write ONLY the email body (no explanations, no labels, no markdown)\n"
        f"- Keep it between {LMIN} and {LMAX} words total\n"
        f"- Make it sound like a real human wrote it\n"
        f"- Do not include a subject line or 'Subject:' header\n"
    )


def word_count(text: str) -> int:
    return len(text.split())


SPAM_FOCUS_WORDS = {
    "exclusive",
    "urgent",
    "limited",
    "bonus",
    "offer",
    "discount",
    "prize",
    "cash",
    "credit",
    "loan",
    "register",
    "join",
    "guarantee",
    "certificate",
    "click",
}

HAM_FOCUS_WORDS = {
    "project",
    "update",
    "report",
    "meeting",
    "schedule",
    "agenda",
    "team",
    "session",
    "sprint",
    "delivery",
    "coaching",
    "performance",
    "framework",
    "ownership",
}

SPAM_OPENERS = [
    "Urgent: please review this {word} offer.",
    "Click now to claim your {word} bonus.",
    "Register today to secure the {word} seats.",
    "Congratulations, your {word} prize is ready.",
    "Act immediately to unlock the {word} discount.",
    "This {word} expires soon, so respond quickly.",
    "Your {word} approval needs a quick confirmation.",
    "Join the {word} masterclass and get certified fast.",
]

SPAM_FOLLOWUPS = [
    "Reply soon if interested.",
    "This is time-sensitive.",
    "I can reserve it today.",
    "Let me know quickly.",
]

HAM_OPENERS = [
    "Thanks for the update on the {word}.",
    "Please send the {word} before the meeting.",
    "Let's review the {word} during today's session.",
    "The team will share the next {word} soon.",
    "I appreciate the quick follow-up on this {word}.",
    "We can discuss the {word} in tomorrow's meeting.",
    "I will confirm the {word} and send notes.",
    "The {word} is moving forward as planned.",
]

HAM_FOLLOWUPS = [
    "Please reply when you can.",
    "Thanks for the help.",
    "I will follow up shortly.",
    "Let me know if that works.",
]


def choose_focus_word(label: str, word_bag: list[str], rng: random.Random) -> str:
    preferred = SPAM_FOCUS_WORDS if label == "spam" else HAM_FOCUS_WORDS
    bag_matches = [word for word in word_bag if word in preferred]
    if bag_matches:
        return rng.choice(bag_matches)

    if word_bag:
        return rng.choice(word_bag)

    return rng.choice(sorted(preferred))


def build_local_email(label: str, word_bag: list[str], rng: random.Random) -> str:
    openers = SPAM_OPENERS if label == "spam" else HAM_OPENERS
    followups = SPAM_FOLLOWUPS if label == "spam" else HAM_FOLLOWUPS

    for _ in range(20):
        opener = rng.choice(openers).format(word=choose_focus_word(label, word_bag, rng))
        email = opener

        if rng.random() < 0.6:
            followup = rng.choice(followups)
            candidate = f"{opener} {followup}"
            if word_count(candidate) <= LMAX:
                email = candidate

        if LMIN <= word_count(email) <= LMAX:
            return email

    fallback_word = choose_focus_word(label, word_bag, rng)
    if label == "spam":
        return f"Urgent {fallback_word} offer."
    return f"Thanks for the {fallback_word} update."
 
 
# ──────────────────────────────────────────────────────────────
# LLM backends
# ──────────────────────────────────────────────────────────────
def call_claude(prompt: str) -> str:
    try:
        import importlib

        anthropic = importlib.import_module("anthropic")
    except ImportError:
        sys.exit("ERROR: Run  pip install anthropic  first.")
 
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set the ANTHROPIC_API_KEY environment variable.")
 
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
 
 
def call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: Run  pip install openai  first.")
 
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set the OPENAI_API_KEY environment variable.")
 
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
 
 
def generate_email(
    label: str,
    word_bag: list[str],
    provider: str,
    rng: random.Random,
) -> str:
    """Generate one email using the selected provider or local templates."""

    if provider == "local":
        return build_local_email(label, word_bag, rng)

    prompt = build_prompt(label, word_bag)
 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "claude":
                email_text = call_claude(prompt)
            elif provider == "openai":
                email_text = call_openai(prompt)
            else:
                sys.exit(f"ERROR: Unknown provider '{provider}'. Use 'claude' or 'openai'.")

            if LMIN <= word_count(email_text) <= LMAX:
                return email_text

            raise ValueError(
                f"generated email has {word_count(email_text)} words; expected {LMIN}-{LMAX}"
            )
        except Exception as exc:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] API error: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts.")
 
 
# ──────────────────────────────────────────────────────────────
# Save helpers (same folder/per-file structure as original)
# ──────────────────────────────────────────────────────────────
def save_emails(emails: list[str], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for i, email in enumerate(emails, start=1):
        filepath = os.path.join(folder, f"email_{i:04d}.txt")
        with open(filepath, "w") as f:
            f.write(email + "\n")
    print(f"  Saved {len(emails):>5} individual files → '{folder}/'  "
          f"(email_0001.txt … email_{len(emails):04d}.txt)")
 
 
def display_samples(spam_emails: list[str], ham_emails: list[str], n: int = 3) -> None:
    rng = random.Random()
    print("\n" + "═" * 68)
    print(f"  SAMPLE SPAM EMAILS  (showing {n})")
    print("═" * 68)
    for i, email in enumerate(rng.sample(spam_emails, min(n, len(spam_emails))), 1):
        print(f"\n  [{i}]\n{email}\n")
 
    print("═" * 68)
    print(f"  SAMPLE HAM  EMAILS  (showing {n})")
    print("═" * 68)
    for i, email in enumerate(rng.sample(ham_emails, min(n, len(ham_emails))), 1):
        print(f"\n  [{i}]\n{email}\n")
 
 
# ──────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────
def generate_batch(
    label: str,
    n: int,
    spam_dict: list[str],
    ham_dict:  list[str],
    p_spam_word: float,
    provider: str,
    rng: random.Random,
) -> list[str]:
    emails = []
    for i in range(1, n + 1):
        length   = rng.randint(LMIN, LMAX)
        word_bag = sample_word_bag(spam_dict, ham_dict, p_spam_word, length, rng)
 
        print(f"  [{label}] {i:>5}/{n}  word_bag: {' '.join(word_bag[:6])} …",
              end="\r", flush=True)
 
        email = generate_email(label, word_bag, provider, rng)
        emails.append(email)
 
    print(f"  [{label}] {n}/{n} — done.{' ' * 40}")
    return emails
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-powered email dataset generator")
    parser.add_argument("--provider", choices=["local", "claude", "openai"],
                        default=PROVIDER, help="LLM provider to use")
    parser.add_argument("--samples", type=int, default=None,
                        help="Generate only N emails of each type (for testing)")
    args = parser.parse_args()
 
    provider = args.provider
    n_spam   = args.samples if args.samples else N_SPAM
    n_ham    = args.samples if args.samples else N_HAM
 
    print("\n" + "=" * 68)
    mode = "template-powered" if provider == "local" else "LLM-powered"
    print(f"  Part I: Dataset Generation — {mode} ({provider.upper()})")
    print("=" * 68)
 
    print("\n[Task 1] Loading dictionaries …")
    spam_dict = load_dictionary(SPAM_DICT_FILE)
    ham_dict  = load_dictionary(HAM_DICT_FILE)
 
    rng = random.Random(RANDOM_SEED)
 
    print(f"\n[Task 2] Generating {n_spam} spam emails via {provider} …")
    spam_emails = generate_batch("spam", n_spam, spam_dict, ham_dict, PS, provider, rng)
 
    print(f"\n[Task 2] Generating {n_ham} ham emails via {provider} …")
    ham_emails  = generate_batch("ham",  n_ham,  spam_dict, ham_dict, QS, provider, rng)
 
    print("\n[Output] Saving email files …")
    save_emails(spam_emails, SPAM_OUT_DIR)
    save_emails(ham_emails,  HAM_OUT_DIR)
 
    display_samples(spam_emails, ham_emails, n=3)
    print("Done!\n")
 
 
if __name__ == "__main__":
    main()
