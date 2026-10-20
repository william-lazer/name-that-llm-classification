
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

# 1. Load the dataset
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

print("Dataset shape:", train.shape)
print("Number of LLM classes:", y.nunique())

# 2. Prepare three reproducible validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(
    splitter.split(X, y, groups=groups)
)

scores = []

# 3. Train and evaluate Random Forest
for split_number, (train_idx, valid_idx) in enumerate(
    splits, start=1
):

    print(f"\nTraining split {split_number}...")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Character-level TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        dtype="float32"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_valid_tfidf)

    score = balanced_accuracy_score(
        y_valid,
        predictions
    )

    scores.append(score)

    print(
        f"Split {split_number}: "
        f"Balanced Accuracy = {score:.4f} "
        f"({score * 100:.2f}%)"
    )

# 4. Summarize results
mean_score = sum(scores) / len(scores)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"Mean Balanced Accuracy: {mean_score:.4f}")
print(f"Mean Percentage: {mean_score * 100:.2f}%")
print(f"Highest Score: {max(scores) * 100:.2f}%")
print(f"Lowest Score: {min(scores) * 100:.2f}%")

print("\n========== BENCHMARK COMPARISON ==========")
print("Previous best mean: 29.94%")
print(f"Random Forest mean: {mean_score * 100:.2f}%")
print(
    "Difference: "
    f"{(mean_score - 0.2994) * 100:+.2f} percentage points"
)