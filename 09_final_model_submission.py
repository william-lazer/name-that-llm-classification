import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# 1. Define file paths
DATA_DIR = Path("data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_PATH = Path("submission.parquet")


# 2. Load training and test data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)


# 3. Separate input text and target labels
X_train = train["text"]
y_train = train["model"]

X_test = test["text"]


# 4. Convert text into character TF-IDF features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training feature matrix:", X_train_tfidf.shape)
print("Test feature matrix:", X_test_tfidf.shape)


# 5. Train the final model
model = LogisticRegression(
    max_iter=1500,
    C=5.0
)

model.fit(X_train_tfidf, y_train)


# 6. Predict the model labels
predictions = model.predict(X_test_tfidf)


# 7. Create the submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "model": predictions
})


# 8. Validate the submission
assert submission.columns.tolist() == ["id", "model"]
assert len(submission) == len(test)
assert submission["id"].is_unique
assert submission["model"].notna().all()
assert set(submission["model"]).issubset(set(y_train.unique()))


# 9. Save the Parquet file
submission.to_parquet(OUTPUT_PATH, index=False)


# 10. Confirm the output
print("\nSubmission created successfully.")
print("Output file:", OUTPUT_PATH.resolve())
print("Submission shape:", submission.shape)
print(submission.head())

print("\nPredicted model distribution:")
print(submission["model"].value_counts())