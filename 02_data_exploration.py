
import pandas as pd

# Define dataset paths
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 1. Dataset dimensions
print("=" * 50)
print("DATASET DIMENSIONS")
print("=" * 50)

print("Training shape:", train.shape)
print("Test shape:", test.shape)

# 2. Column information
print("\nTraining columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# 3. First five training records
print("\nFIRST FIVE TRAINING RECORDS")
print(train.head().to_string(index=False))

# 4. Missing values
print("\nMISSING VALUES")
print(train.isnull().sum())

# 5. Duplicate checks
print("\nDUPLICATE CHECKS")
print("Duplicate training IDs:", train["id"].duplicated().sum())
print("Duplicate passages:", train["text"].duplicated().sum())
print("Duplicate test IDs:", test["id"].duplicated().sum())

# 6. Unique prompts and models
print("\nDATASET DIVERSITY")
print("Unique prompts:", train["prompt_id"].nunique())
print("Unique models:", train["model"].nunique())

# 7. Model distribution
print("\nPASSAGES PER MODEL")
print(train["model"].value_counts().sort_index())

print("\nDataset inspection completed.")

# 8. Investigate prompt coverage

print("\nPROMPT COVERAGE ANALYSIS")

# Count passages associated with each prompt
prompt_counts = train.groupby("prompt_id").size()

print("Number of prompts:", len(prompt_counts))
print("Minimum passages per prompt:", prompt_counts.min())
print("Maximum passages per prompt:", prompt_counts.max())
print("Average passages per prompt:", round(prompt_counts.mean(), 2))

# Check the number of distinct models represented per prompt
models_per_prompt = train.groupby("prompt_id")["model"].nunique()

print("\nModels represented per prompt:")
print(models_per_prompt.value_counts().sort_index())

# Verify whether each prompt has all 18 models
complete_prompts = (models_per_prompt == 18).sum()

print(
    "\nPrompts containing all 18 models:",
    complete_prompts
)