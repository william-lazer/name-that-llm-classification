
# ============================================================
# Project: Name That LLM
# Experiment 13: Bidirectional LSTM
# Objective: Classify text into one of 18 language models
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

from keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)

from keras.models import Sequential
from keras.callbacks import EarlyStopping


# ------------------------------------------------------------
# 1. Set random seeds
# ------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------
# 2. Load the dataset
# ------------------------------------------------------------

train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

print("========== DATASET INFORMATION ==========")
print("Dataset shape:", train.shape)
print("Number of classes:", y.nunique())
print("Number of unique prompts:", groups.nunique())


# ------------------------------------------------------------
# 3. Create a prompt-grouped validation split
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, valid_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train_text = X.iloc[train_idx]
X_valid_text = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(X_train_text))
print("Validation rows:", len(X_valid_text))


# ------------------------------------------------------------
# 4. Encode target labels
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

num_classes = len(label_encoder.classes_)

print("Number of target classes:", num_classes)


# ------------------------------------------------------------
# 5. Convert text into integer sequences
# ------------------------------------------------------------

MAX_WORDS = 20000
MAX_LENGTH = 300

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
    standardize="lower_and_strip_punctuation"
)

# Learn vocabulary using training text only
vectorizer.adapt(
    X_train_text.to_numpy()
)

vocab_size = len(vectorizer.get_vocabulary())

print("\n========== TEXT VECTORIZATION ==========")
print("Vocabulary size:", vocab_size)
print("Maximum sequence length:", MAX_LENGTH)


# Keep the text as strings.
# TextVectorization converts strings into padded sequences
# when they enter the model.

X_train_sequences = X_train_text.to_numpy()
X_valid_sequences = X_valid_text.to_numpy()


# ------------------------------------------------------------
# 6. Build the Bidirectional LSTM model
# ------------------------------------------------------------

model = Sequential([
    vectorizer,

    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),

    Bidirectional(
        LSTM(32)
    ),

    Dropout(0.4),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MODEL ARCHITECTURE ==========")
model.summary()


# ------------------------------------------------------------
# 7. Configure early stopping
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ------------------------------------------------------------
# 8. Train the model
# ------------------------------------------------------------

print("\n========== MODEL TRAINING ==========")

history = model.fit(
    X_train_sequences,
    y_train_encoded,
    validation_data=(
        X_valid_sequences,
        y_valid_encoded
    ),
    epochs=8,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)


# ------------------------------------------------------------
# 9. Generate validation predictions
# ------------------------------------------------------------

print("\n========== MODEL PREDICTION ==========")

probabilities = model.predict(
    X_valid_sequences,
    batch_size=16,
    verbose=0
)

predicted_encoded = np.argmax(
    probabilities,
    axis=1
)

predictions = label_encoder.inverse_transform(
    predicted_encoded
)


# ------------------------------------------------------------
# 10. Evaluate the model
# ------------------------------------------------------------

score = balanced_accuracy_score(
    y_valid,
    predictions
)

print("\n========== FINAL RNN RESULTS ==========")

print(f"Validation balanced accuracy: {score:.4f}")
print(f"Validation balanced accuracy: {score * 100:.2f}%")

print("\nValidation accuracy:")
print(f"{np.mean(predictions == y_valid.to_numpy()) * 100:.2f}%")

print("\nPredicted class distribution:")
print(
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

print("\nActual validation class distribution:")
print(
    y_valid
    .value_counts()
    .sort_index()
)

print("\nRNN experiment completed.")