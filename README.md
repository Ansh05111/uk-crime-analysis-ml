# UK Police Crime Data Analysis

This project analyzes City of London street-level crime records for August 2026 and compares five machine-learning classifiers for predicting the recorded crime type.

## Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)

Each model has its own training and testing split. The script reports Accuracy, Precision, Recall, F1 Score, Log Loss, and a classification report for each model. It then compares model Accuracy in a chart.

## Dataset

Place the following CSV file in the same folder as the Python script:

`2026-08-city-of-london-street.csv`

The script expects columns such as `Crime type`, `Longitude`, and `Latitude`.

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

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Keep the Python script and dataset CSV in the same folder.
2. Open a terminal in that folder.
3. Run the script:

```bash
python crime_project_without_section7.py
```

If you have renamed the Python file, replace the filename in the command with your actual filename.

## Output Files

After running the script, the following files are generated:

- `cleaned_crime_data.csv` — the cleaned dataset.
- `model_comparison.csv` — comparison of Accuracy and F1 Score for the five models.
- `model_comparison.png` — bar chart comparing model Accuracy.

## Notes

- This is a machine-learning classification project using the available dataset features, mainly location-related columns.
- Each model uses a separate train-test split, so their scores are not a strict like-for-like comparison on identical test records.
- Results describe performance on this dataset and should not be treated as proof that crime can be predicted or prevented in real-world situations.
