
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

train = pd.read_csv("data/train.csv")

X = train["text"].fillna("")
y = train["model"]
groups = train["prompt_id"]

# --------------------------------------------------
# 2. Create the same validation split
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
# 3. Combine word and character features
# --------------------------------------------------

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    ),
])

model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=1500,
            C=2.0,
        ),
    ),
])

# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

print("Training combined word and character model...")

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
print("COMBINED MODEL RESULTS")
print("=" * 50)

print(f"Balanced accuracy: {score:.4f}")
print(f"Balanced accuracy (%): {score * 100:.2f}%")
