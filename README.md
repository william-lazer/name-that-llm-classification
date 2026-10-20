# Name That LLM — Machine Learning Classification

## Project Overview

This project explores machine learning techniques for identifying which Large Language Model (LLM) generated a given text sample.

The project evaluates different text representations, classification algorithms, and feature-engineering approaches to understand which techniques are most effective for multiclass text classification.

The primary focus is on character-level TF-IDF, traditional machine learning classifiers, numerical text-statistics features, and validation strategies designed to reduce prompt leakage.

**Current best local validation result: 31.42% mean balanced accuracy.**

## Problem Statement

Given a text sample, predict which of 18 LLM classes generated it.

This is a multiclass text-classification problem evaluated using balanced accuracy, which accounts for class-level performance rather than relying solely on overall accuracy.

## Dataset

The training dataset available during development contains:

* **2,699** text samples
* **18** target classes
* **150** unique prompt IDs
* Sample IDs, prompt IDs, text, and target model labels

The local test file used during development contains three samples. This local file is not assumed to represent the complete official competition test dataset.

Raw competition data is excluded from this repository. Obtain the appropriate dataset and verify the competition's submission requirements through Kaggle before reproducing a competition submission.

## Technologies Used

* Python
* Pandas and NumPy
* SciPy
* Scikit-learn
* TensorFlow
* TF-IDF feature extraction
* Character-level and word-level text representations
* Logistic Regression
* Linear Support Vector Classification (LinearSVC)
* Multinomial Naive Bayes
* Random Forest
* Bidirectional LSTM
* Feature engineering and hyperparameter tuning
* Grouped validation and error analysis

## Methodology

The project follows an iterative model-development process:

1. Inspect the dataset and examine its class distribution.
2. Establish a baseline using TF-IDF and Logistic Regression.
3. Evaluate character-level and word-level text representations.
4. Compare alternative classification algorithms.
5. Tune model hyperparameters.
6. Validate models using grouped train-validation splits based on prompt IDs.
7. Investigate misclassifications and difficult-to-distinguish LLM classes.
8. Engineer numerical text-statistics features.
9. Compare the character-only model against the enhanced feature model using identical validation splits.
10. Document results and retain the experiment scripts for reproducibility.

## Experiments

The repository contains scripts covering the following experiments:

| Script                           | Experiment                                                    |
| -------------------------------- | ------------------------------------------------------------- |
| `01_download_dataset.py`         | Dataset download workflow                                     |
| `02_data_exploration.py`         | Initial dataset inspection                                    |
| `03_baseline_model.py`           | Word-level TF-IDF with Logistic Regression                    |
| `04_character_model.py`          | Character-level TF-IDF                                        |
| `05_combined_model.py`           | Combined word and character features                          |
| `06_tune_character_model.py`     | Logistic Regression hyperparameter tuning                     |
| `07_validate_character_model.py` | Grouped validation                                            |
| `08_inspect_submission.py`       | Local submission-data inspection                              |
| `09_final_model_submission.py`   | Local prediction generation                                   |
| `10_compare_classifiers.py`      | Classifier comparison                                         |
| `11_combined_features_svm.py`    | Combined features with LinearSVC                              |
| `12_naivebayes.py`               | Word-level TF-IDF with Naive Bayes                            |
| `13_rnn_model.py`                | LSTM-based text classification                                |
| `14_character_naive_bayes.py`    | Character-level TF-IDF with Naive Bayes                       |
| `15_tune_linear_svc.py`          | LinearSVC hyperparameter tuning                               |
| `16_analyze_model_errors.py`     | Classification errors and class-level analysis                |
| `17_word_svc.py`                 | Word-level TF-IDF with LinearSVC                              |
| `18_random_forest.py`            | Random Forest evaluation                                      |
| `19_inspect_test_features.py`    | Test-text feature inspection                                  |
| `20_character_plus_stats.py`     | Character TF-IDF plus numerical text statistics               |
| `21_tune_statistics_weight.py`   | Tuning the contribution of numerical features                 |
| `22_compare_features.py`         | Controlled comparison of character-only and enhanced features |

The scripts document the experimentation process, including approaches that did not outperform the leading model.

## Best Local Validation Model

The strongest recorded local validation result was obtained using:

**Character-level TF-IDF + Numerical Text Statistics + LinearSVC**

### Model configuration

| Parameter                 | Configuration                          |
| ------------------------- | -------------------------------------- |
| Text representation       | Character-level TF-IDF                 |
| Character n-grams         | 3–5                                    |
| Maximum TF-IDF features   | 100,000                                |
| Classifier                | LinearSVC                              |
| Regularization parameter  | C = 0.5                                |
| Class weighting           | `balanced`                             |
| Maximum iterations        | 10,000                                 |
| Additional features       | Numerical text statistics              |
| Numerical feature scaling | StandardScaler fitted on training data |
| Numerical feature weight  | 0.5                                    |

The numerical features capture characteristics such as text length, word count, punctuation counts, and other text statistics.

The numerical features are scaled using training data only before being combined with the sparse character TF-IDF matrix.

## Validation Strategy

The experiments use `GroupShuffleSplit` with `prompt_id` as the grouping variable.

This approach helps reduce the risk of prompt leakage by keeping samples associated with the same prompt together within a split, rather than allowing related prompt samples to appear on both sides of the train-validation boundary.

The reported scores are balanced accuracy percentages calculated on three grouped validation splits.

### Best model: validation results

| Validation split | Balanced accuracy |
| ---------------- | ----------------: |
| Split 1          |            33.52% |
| Split 2          |            30.93% |
| Split 3          |            29.81% |
| **Mean**         |        **31.42%** |

These results represent local validation performance. They are not official Kaggle leaderboard results.

## Model Comparison

The following table summarizes the mean balanced accuracy of selected experiments.

| Model                                               | Mean balanced accuracy |
| --------------------------------------------------- | ---------------------: |
| Character TF-IDF + Logistic Regression              |                 29.38% |
| Character TF-IDF + LinearSVC                        |                 29.57% |
| Character TF-IDF + Numerical Statistics + LinearSVC |             **31.42%** |

The character-only LinearSVC baseline and the statistics-enhanced model were evaluated using the same three validation splits in the controlled feature comparison.

### Feature comparison by split

| Validation split | Character-only LinearSVC | Character TF-IDF + statistics |                      Change |
| ---------------- | -----------------------: | ----------------------------: | --------------------------: |
| Split 1          |                   31.67% |                        33.52% |     +1.85 percentage points |
| Split 2          |                   30.37% |                        30.93% |     +0.56 percentage points |
| Split 3          |                   27.78% |                        29.81% |     +2.03 percentage points |
| **Mean**         |               **29.94%** |                    **31.42%** | **+1.48 percentage points** |

The enhanced model performed better on all three recorded validation splits.

This is encouraging evidence that numerical text statistics provide additional predictive information for this dataset. However, three splits do not establish statistical significance or guarantee that the improvement will generalize to unseen datasets.

### Numerical feature-weight experiment

The contribution of the numerical statistics was evaluated using weights of 0.25, 0.5, 0.75, and 1.0.

| Numerical feature weight | Mean balanced accuracy |
| ------------------------ | ---------------------: |
| 0.25                     |                 31.23% |
| **0.50**                 |             **31.42%** |
| 0.75                     |                 31.23% |
| 1.00                     |                 31.05% |

Among the tested configurations, a numerical feature weight of **0.5** achieved the highest mean balanced accuracy.

## Key Findings

* Character-level TF-IDF performed better than the tested word-only TF-IDF configurations.
* LinearSVC performed better than the tested Random Forest and Naive Bayes configurations.
* Numerical text-statistics features improved the recorded mean balanced accuracy from 29.94% to 31.42% in a controlled comparison.
* A numerical feature weight of 0.5 produced the best mean result among the tested weights.
* Error analysis showed that some related LLM classes were frequently confused.
* The LSTM experiment did not outperform the leading TF-IDF-based approach.
* Validation results varied across grouped splits, demonstrating the importance of evaluating performance across more than one split.

These conclusions apply to the tested configurations and recorded validation splits.

## Project Structure

```text
Name_That_LLM/
├── 01_download_dataset.py
├── 02_data_exploration.py
├── 03_baseline_model.py
├── 04_character_model.py
├── 05_combined_model.py
├── 06_tune_character_model.py
├── 07_validate_character_model.py
├── 08_inspect_submission.py
├── 09_final_model_submission.py
├── 10_compare_classifiers.py
├── 11_combined_features_svm.py
├── 12_naivebayes.py
├── 13_rnn_model.py
├── 14_character_naive_bayes.py
├── 15_tune_linear_svc.py
├── 16_analyze_model_errors.py
├── 17_word_svc.py
├── 18_random_forest.py
├── 19_inspect_test_features.py
├── 20_character_plus_stats.py
├── 21_tune_statistics_weight.py
├── 22_compare_features.py
├── requirements.txt
├── .gitignore
└── README.md
```

The local `data/` directory is excluded from Git and is not included in the structure above.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/william-lazer/name-that-llm-classification.git
cd name-that-llm-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows — Command Prompt:**

```bat
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Prepare the dataset

Obtain the competition data through Kaggle and place the required files in the expected local `data/` directory. Verify each script's input paths and the current competition submission requirements before running the workflow.

## Limitations

* The available training dataset is relatively small for an 18-class classification problem.
* Some LLM classes may produce similar writing styles.
* Validation scores vary across prompt-group splits.
* The current best result is based on three grouped validation splits.
* Local validation performance may differ from the official Kaggle score.
* The official Kaggle performance of the current best local model has not been confirmed.
* The local development test file contains only three samples and should not be treated as the full official competition test set.

## Future Improvements

* Perform additional feature engineering and model evaluation.
* Investigate frequently confused LLM class pairs.
* Evaluate more validation splits and alternative validation strategies.
* Explore additional text representations and classifier configurations.
* Improve the reproducibility of experiments through centralized configuration and documented random seeds.
* Verify the final workflow against the official Kaggle dataset and submission format.

## Author

Developed as a practical machine learning portfolio project focused on multiclass text classification, feature engineering, model comparison, grouped validation, and error analysis.

**GitHub repository:**
https://github.com/william-lazer/name-that-llm-classification
