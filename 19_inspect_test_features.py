
import pandas as pd

# 1. Load the training dataset
train = pd.read_csv("data/train.csv")

# 2. Basic dataset information
print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", train.shape)
print("Number of LLM classes:", train["model"].nunique())
print("Number of unique prompts:", train["prompt_id"].nunique())

# 3. Analyze text length
train["text"] = train["text"].fillna("").astype(str)

train["character_count"] = train["text"].str.len()
train["word_count"] = train["text"].str.split().str.len()
train["line_count"] = train["text"].str.count("\n") + 1
train["question_marks"] = train["text"].str.count(r"\?")
train["exclamation_marks"] = train["text"].str.count("!")
train["code_markers"] = train["text"].str.count("```")

# 4. Overall text statistics
print("\n========== OVERALL TEXT STATISTICS ==========")

print(
    train[
        [
            "character_count",
            "word_count",
            "line_count",
            "question_marks",
            "exclamation_marks",
            "code_markers",
        ]
    ].describe().round(2)
)

# 5. Compare text characteristics across LLM classes
features = [
    "character_count",
    "word_count",
    "line_count",
    "question_marks",
    "exclamation_marks",
    "code_markers",
]

print("\n========== STATISTICS BY LLM CLASS ==========")

class_stats = train.groupby("model")[features].mean().round(2)

print(class_stats.to_string())

# 6. Save results for later analysis
class_stats.to_csv("llm_text_feature_statistics.csv")

print("\nSaved statistics to llm_text_feature_statistics.csv")