
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv("data/train.csv")

X = train["text"].fillna("")
y = train["model"]
groups = train["prompt_id"]

# --------------------------------------------------
# 2. Create the same grouped validation split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, valid_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

# --------------------------------------------------
# 3. Test different regularization strengths
# --------------------------------------------------

c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

results = []

for c in c_values:

    print(f"\nTraining model with C={c}...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                C=c,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    score = balanced_accuracy_score(
        y_valid,
        predictions,
    )

    results.append({
        "C": c,
        "Balanced Accuracy": score,
    })

    print(f"Balanced accuracy: {score:.4f} ({score * 100:.2f}%)")

# --------------------------------------------------
# 4. Compare results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False,
)

print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 50)

print(results_df.to_string(index=False))

best = results_df.iloc[0]

print("\nBest C:", best["C"])
print(
    "Best balanced accuracy:",
    f'{best["Balanced Accuracy"] * 100:.2f}%'
)