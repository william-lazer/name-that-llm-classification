import pandas as pd
from pathlib import Path
import zipfile

# 1. Inspect available dataset files
data_folder = Path("data")

print("=== DATA FILES ===")

for file in data_folder.iterdir():
    print(file.name, "-", round(file.stat().st_size / (1024 * 1024), 2), "MB")


# 2. Inspect training data
train = pd.read_csv("data/train.csv")

print("\n=== TRAIN DATA ===")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())


# 3. Inspect local test data
test = pd.read_csv("data/test.csv")

print("\n=== TEST DATA ===")
print("Shape:", test.shape)
print("Columns:", test.columns.tolist())
print(test.head())


# 4. Inspect the sample submission
sample = pd.read_parquet("data/sample_submission.parquet")

print("\n=== SAMPLE SUBMISSION ===")
print("Shape:", sample.shape)
print("Columns:", sample.columns.tolist())
print(sample.head())
print(sample.dtypes)

zip_path = "data/name-that-llm.zip"

print("\n=== ZIP FILE CONTENTS ===")

with zipfile.ZipFile(zip_path, "r") as archive:
    for item in archive.infolist():
        print(
            f"File: {item.filename} | "
            f"Size: {item.file_size:,} bytes"
        )
