# UK Police Crime Data Analysis

This project analyzes City of London street-level crime records for August 2026 and compares five supervised machine learning classifiers for predicting the recorded crime type.

## Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)

Each model has its own train-test split. The script reports Accuracy, Precision, Recall, F1 Score, Log Loss, and a classification report for each model.

## Model Comparison

The comparison graph displays four evaluation metrics for all five classifiers:

- Accuracy
- Precision
- Recall
- F1 Score

The metrics are also saved in `model_comparison.csv`, and the graph is saved as `model_comparison.png`.

**Note:** Each classifier uses a separate test split, so the scores are useful for an initial comparison but are not a strict head-to-head comparison on identical test records.

## Dataset

Place the following CSV file in the same folder as the Python script:

`2026-08-city-of-london-street.csv`

The dataset should contain columns such as `Crime type`, `Longitude`, and `Latitude`.

Data cleaning includes:
- Handling missing values in required columns.
- Converting longitude and latitude to numeric values.
- Handling duplicate Crime IDs.
- Checking broad UK coordinate limits.
- Combining rare crime types into `Other crime`.

## Requirements

- Python 3.9 or later
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Keep the Python script and dataset CSV in the same folder.
2. Open a terminal in that folder.
3. Run:

```bash
python crime_project_all_metrics_comparison.py
```

If you rename the Python file, use its actual name in the command.

## Output Files

- `cleaned_crime_data.csv` — cleaned dataset.
- `model_comparison.csv` — Accuracy, Precision, Recall, and F1 Score for the five classifiers.
- `model_comparison.png` — comparison chart for all four metrics.

## Notes

- The available features are mainly location-related, so they may not be sufficient to classify crime type accurately.
- The models are evaluated on separate train-test splits.
- Results describe performance on this dataset and should not be treated as proof that crime can be predicted or prevented in real-world situations.
