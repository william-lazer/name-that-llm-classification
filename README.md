# Name That LLM — Machine Learning Classification

## Project Overview

This project explores machine learning techniques for classifying text according to its generating Large Language Model (LLM).

The objective is to evaluate different text representations, classification algorithms, and feature-engineering approaches to identify which techniques perform best on the available dataset.

The project uses Python and scikit-learn, with TensorFlow for an LSTM-based experiment.

## Problem Statement

Given a text sample, predict which of 18 LLM classes generated it.

This is a multiclass text-classification problem evaluated using balanced accuracy.

## Dataset

The training dataset contains:

* 2,699 text samples
* 18 target classes
* 150 unique prompt IDs
* Text, prompt ID, sample ID, and target model columns

The test dataset available in the local development files contains three samples. The competition dataset and its submission requirements should be obtained and verified through Kaggle.

Raw competition data is not included in this repository.

## Technologies Used

* Python
* Pandas and NumPy
* SciPy
* Scikit-learn
* TensorFlow
* TF-IDF text feature extraction
* LinearSVC
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest
* Bidirectional LSTM
* Model evaluation and error analysis

## Experiments

The project includes experiments involving:

1. Initial TF-IDF exploration and data inspection
2. Baseline Logistic Regression
3. Character-level TF-IDF
4. Combined word-level and character-level features
5. Logistic Regression hyperparameter tuning
6. Grouped validation
7. Classifier comparison
8. Naive Bayes
9. LinearSVC hyperparameter tuning
10. Confusion matrix and classification-report analysis
11. Word-level TF-IDF with LinearSVC
12. Random Forest
13. LSTM-based text classification
14. Numeric text-statistics feature engineering

The experiment scripts are retained to document the model-development process and compare alternative approaches.

## Best Local Validation Result

The strongest result observed in the recorded experiments was obtained using:

**Character-level TF-IDF + Numeric Text Statistics + LinearSVC**

Configuration:

* Character n-grams: 3–5
* Maximum TF-IDF features: 100,000
* LinearSVC regularization parameter: C = 0.5
* Class weighting: balanced
* Maximum iterations: 10,000
* Numeric features: text length, word count, punctuation counts, and other text statistics

### Validation Results

| Split    | Balanced Accuracy |
| -------- | ----------------: |
| Split 1  |            33.52% |
| Split 2  |            30.93% |
| Split 3  |            29.81% |
| **Mean** |        **31.42%** |

Validation was performed using GroupShuffleSplit grouped by prompt ID to reduce the risk of prompt leakage between training and validation samples.

These are local validation results, not the official Kaggle public score.

## Key Findings

* Character-level TF-IDF outperformed the tested word-only TF-IDF configuration.
* LinearSVC outperformed the tested Random Forest and Naive Bayes configurations.
* Adding numeric text statistics improved the mean balanced accuracy from 29.94% to 31.42% in the recorded experiments.
* The confusion matrix revealed that several related LLM classes were frequently misclassified.
* The LSTM experiment did not outperform the leading TF-IDF-based approach.

These findings apply to the configurations and validation splits tested in this project.

## Project Structure

```text
Name_That_LLM/
├── 00_testtfidf.py
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
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a Python virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Obtain the competition dataset through Kaggle and place the files in the expected local data directory before running the relevant scripts.

## Limitations

* The training dataset is relatively small for an 18-class classification task.
* Several LLM classes have similar writing patterns.
* Validation scores vary between splits.
* Local validation performance may differ from the official Kaggle score.
* The improved model's official Kaggle score has not yet been confirmed.

## Future Improvements

* Further feature engineering and model evaluation
* Improved analysis of difficult-to-classify model pairs
* More robust validation across prompt groups
* Evaluation of additional text representations
* Reproducible final Kaggle submission

## Author

Developed as a practical machine learning project to explore text classification, feature engineering, model comparison, and validation methodology.
