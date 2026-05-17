"""
generate_dataset.py
-------------------
Generates a realistic 250-row student dataset and saves it to students.csv.
Run this script once to (re-)create the dataset.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 250

# ── Helper: clip values to realistic ranges ────────────────────────────────
def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)

# ── Generate correlated feature groups ────────────────────────────────────
# We create three natural clusters: struggling / average / high-performing

def make_group(n, study_mu, att_mu, sleep_mu, score_mu, label):
    study  = clip(np.random.normal(study_mu,  1.2, n), 0, 12)
    att    = clip(np.random.normal(att_mu,    8.0, n), 30, 100)
    sleep  = clip(np.random.normal(sleep_mu,  1.0, n), 3, 12)
    score  = clip(np.random.normal(score_mu,  8.0, n), 0, 100)
    perf   = [label] * n
    return study, att, sleep, score, perf

s1, a1, sl1, sc1, p1 = make_group(80,  2.0, 58, 5.5, 42, "Low")
s2, a2, sl2, sc2, p2 = make_group(90,  4.5, 73, 7.0, 62, "Medium")
s3, a3, sl3, sc3, p3 = make_group(80,  7.5, 89, 8.0, 84, "High")

# ── Combine & shuffle ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "study_hours"    : np.concatenate([s1, s2, s3]),
    "attendance"     : np.concatenate([a1, a2, a3]),
    "sleep_hours"    : np.concatenate([sl1, sl2, sl3]),
    "previous_score" : np.concatenate([sc1, sc2, sc3]),
    "performance"    : p1 + p2 + p3,
})

# Round sensibly
df["study_hours"]    = df["study_hours"].round(1)
df["attendance"]     = df["attendance"].round(1)
df["sleep_hours"]    = df["sleep_hours"].round(1)
df["previous_score"] = df["previous_score"].round(1)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

import os
out = os.path.join(os.path.dirname(__file__), "students.csv")
df.to_csv(out, index=False)
print(f"✔  Dataset saved → {out}  ({len(df)} rows)")
print(df["performance"].value_counts())
