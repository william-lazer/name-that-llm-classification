
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
)

# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

train = pd.read_csv("data/train.csv")

X = train["text"].fillna("")
y = train["model"]
groups = train["prompt_id"]

print("Total passages:", len(train))
print("Unique prompts:", groups.nunique())
print("Unique models:", y.nunique())

# --------------------------------------------------
# 2. Split by prompt to avoid prompt leakage
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

groups_train = groups.iloc[train_idx]
groups_valid = groups.iloc[valid_idx]

print("\nTraining passages:", len(X_train))
print("Validation passages:", len(X_valid))

print(
    "Overlapping prompts:",
    len(set(groups_train) & set(groups_valid))
)

# --------------------------------------------------
# 3. Build the baseline model
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            C=2.0,
        ),
    ),
])

# --------------------------------------------------
# 4. Train
# --------------------------------------------------

print("\nTraining baseline classifier...")

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------

predictions = model.predict(X_valid)

score = balanced_accuracy_score(
    y_valid,
    predictions,
)

print("\n" + "=" * 50)
print("BASELINE MODEL RESULTS")
print("=" * 50)

print(f"Balanced accuracy: {score:.4f}")
print(f"Balanced accuracy (%): {score * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_valid,
        predictions,
        zero_division=0,
    )
)
