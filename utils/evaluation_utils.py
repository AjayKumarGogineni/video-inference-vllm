import json, numpy as np, pandas as pd
from bert_score import score as bertscore_score
from sklearn.metrics import accuracy_score, f1_score

def evaluate_with_stats(ground_truth, test_df):
    """Compute BERTScore + category metrics and save results."""
    refs, cands = ground_truth["summary"].tolist(), test_df["summary"].tolist()
    P, R, F1 = bertscore_score(cands, refs, lang="en", model_type="xlm-roberta-large", verbose=True)

    y_true = ground_truth["category"].astype(str).str.strip()
    y_pred = test_df["category"].astype(str).str.strip().str.rstrip(".")
    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average="weighted")

    results = {
        "BERTScore": {
            "Precision": {"mean": float(P.mean()), "std": float(P.std())},
            "Recall": {"mean": float(R.mean()), "std": float(R.std())},
            "F1": {"mean": float(F1.mean()), "std": float(F1.std())},
        },
        "Category": {"Accuracy": acc, "F1_weighted": f1}
    }
    return results