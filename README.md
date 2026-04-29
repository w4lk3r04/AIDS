# Intrusion Detection Systems Lab - Spam vs Ham Classification

This project implements and compares two classifiers for spam detection:

- Naive Bayes (MAP and ML variants)
- Single-neuron neural network (one feature: x_ham)

It includes dataset generation, probability modeling, neural training, and performance visualization.

## Requirements

- Python 3.10+
- Recommended OS: Windows/Linux/macOS
- Python library:
  - `matplotlib` (for result visualizations)

Install dependencies:

```bash
pip install -r requirements.txt
```

## File Organization

- `generate_dataset.py`: Generates synthetic spam/ham emails from dictionary-guided templates.
- `probability_compute.py`: Naive Bayes model building, MAP/ML classification, and evaluation.
- `neural_network.py`: Feature extraction (`x_ham`), single-neuron training, and classifier comparison.
- `visualize_results.py`: Creates performance plots from run logs.
- `dataset_dict/`: Dictionary files used during generation.
- `spam_emails/`, `ham_emails/`: Training emails (one file per email).
- `logs/compute_results.txt`: Output log from Naive Bayes run.
- `logs/nr_results.txt`: Output log from neural network + comparison run.
- `logs/plots/`: Generated visualization images.

## How To Execute

Run in this order from the project root.

1. Generate (or refresh) dataset:

```bash
python generate_dataset.py --provider local --samples 5
```

If you already have the full dataset in `spam_emails/` and `ham_emails/`, you can skip this step.

2. Run Naive Bayes workflow:

```bash
python probability_compute.py | tee logs/compute_results.txt
```

3. Run neural network workflow and comparison:

```bash
python neural_network.py | tee logs/nr_results.txt
```

4. Generate visualization figures:

```bash
python visualize_results.py
```

Generated files:

- `logs/plots/accuracy_comparison.png`
- `logs/plots/confusion_matrices.png`

## 4-Minute Presentation Process (Video/Slides)

Use this timeline to keep your presentation clear and short.

1. **0:00-0:30 - Problem and Goal**
   - State objective: classify emails as spam/ham.
   - Mention two methods: Naive Bayes vs Single Neuron.

2. **0:30-1:20 - Data and Feature Design**
   - Show dictionary-based synthetic dataset pipeline.
   - Explain `x_ham` feature: fraction of words in ham dictionary.

3. **1:20-2:20 - Method Summary**
   - Naive Bayes: word likelihoods + priors (MAP/ML).
   - Single Neuron: one input (`x_ham`), weight, bias, sigmoid, gradient descent.

4. **2:20-3:20 - Results and Visualization**
   - Show `accuracy_comparison.png`.
   - Show `confusion_matrices.png`.
   - Point out class-specific errors and where each model fails.

5. **3:20-4:00 - Comparative Analysis and Conclusion**
   - Accuracy comparison.
   - Which emails each handles well/poorly.
   - Trade-offs: interpretability, speed, robustness, sensitivity to vocabulary.

## Brief Comparative Analysis (for report/presentation)

- **Classification accuracy:** In current logs, Naive Bayes (MAP/ML) outperforms the single neuron.
- **Naive Bayes strengths:** Better with high-signal words (`urgent`, `bitcoin`, `meeting`).
- **Naive Bayes weaknesses:** Shared vocabulary can weaken signal; short emails can be brittle.
- **Single neuron strengths:** Very fast and simple; threshold-based interpretation is intuitive.
- **Single neuron weaknesses:** Loses word identity by compressing email to one scalar (`x_ham`), so borderline emails are harder.

## Notes

- If your logs change, rerun `visualize_results.py` to regenerate updated plots.
- For reproducibility, keep seeds in scripts unchanged unless intentionally testing variance.
