# UK Crime Analysis and Machine Learning

A Python-based machine learning project that explores UK police street-level crime data through exploratory data analysis, classification algorithms, and geographic clustering.

## About the Project

This project analyzes crime records to understand crime type distributions and geographic patterns. It implements four supervised machine learning classifiers and K-Means clustering, then evaluates the classification models using standard performance metrics.

## Objectives

- Explore crime type distributions and geographic patterns.
- Clean and preprocess crime data for machine learning.
- Train and evaluate multiple classification algorithms.
- Compare models using Accuracy, Precision, Recall, and F1 Score.
- Identify geographic clusters using K-Means clustering.

## Algorithms Implemented

### Classification Models

1. **Logistic Regression** — a baseline classification algorithm.
2. **Decision Tree** — classifies records using decision rules.
3. **Random Forest** — combines multiple decision trees.
4. **Support Vector Machine (SVM)** — classifies data using decision boundaries.

### Clustering Algorithm

5. **K-Means Clustering** — groups crime records based on geographic coordinates.

## Dataset

**Source:** UK Police street-level crime data.

The dataset includes fields such as:

- Crime ID
- Month
- Longitude and Latitude
- Location
- LSOA area name
- Crime type
- Last outcome category

Data cleaning removes records without valid geographic coordinates or crime types. Rare crime categories may be grouped to support classification.

## Exploratory Data Analysis

The project includes three main visualizations:

- **Crime Type Distribution:** displays the frequency of different crime categories.
- **Crime by Area:** compares crime types across selected areas.
- **Geographic Distribution:** plots crime locations using longitude and latitude.

## Model Evaluation Metrics

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1 Score
- Log Loss
- Classification Report

Each classification model is trained and evaluated in its own section. A final comparison table and bar chart compare the four classifiers. Confusion matrices are not included in the current implementation.

## Results

The following results were obtained during the current run:

| Model | Accuracy | Weighted F1 Score |
|---|---:|---:|
| Logistic Regression | 12.41% | 0.1404 |
| Decision Tree | 20.00% | 0.1993 |
| Random Forest | 22.07% | 0.2279 |
| Support Vector Machine | 26.21% | 0.1513 |

**Best classifier by weighted F1 Score:** Random Forest, with a score of **0.2279**.

SVM achieved the highest accuracy at 26.21%, while Random Forest achieved the highest weighted F1 Score. Overall classification performance was limited, indicating that the selected location-based features alone may not adequately distinguish all crime categories.

### K-Means Clustering Results

- Tested cluster counts: K = 2, 3, 4, 5.
- Selected K: 2.
- Silhouette Score for K = 2: 0.443.

The Elbow Plot displays inertia for each tested K value. The Silhouette Score Plot displays valid scores for cluster configurations satisfying the minimum cluster-size condition.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Spyder / IPython

## Repository

GitHub: [Ansh05111/uk-crime-analysis-ml](https://github.com/Ansh05111/uk-crime-analysis-ml)

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Ansh05111/uk-crime-analysis-ml.git
```

### 2. Open the project directory

```bash
cd uk-crime-analysis-ml
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Add the dataset

Place the required UK Police crime CSV file in the project directory. Ensure its filename matches the filename used in `project.py`.

### 5. Run the project

```bash
python project.py
```

The script can also be executed using Spyder or another Python IDE.

## Generated Outputs

Depending on successful execution, the project generates:

- Crime distribution and geographic plots
- Individual classification evaluation reports
- K-Means geographic cluster visualization
- K-Means Elbow Plot
- K-Means Silhouette Score Plot
- `model_comparison.csv`
- `crime_data_with_clusters.csv`
- `cluster_profile.csv`

## Limitations

- The dataset covers a limited time period and geographic region.
- Classification uses a limited set of input features.
- Some crime categories have relatively few records.
- Model performance is measured on a held-out test split and should not be interpreted as reliable crime forecasting.

## Future Improvements

- Include additional informative features.
- Explore methods for handling imbalanced classes.
- Evaluate more classification algorithms.
- Test models on crime data from additional months and regions.
- Experiment with alternative clustering configurations.

## Disclaimer

This project is intended for educational and academic purposes. Its outputs should not be used as the sole basis for real-world policing decisions.

---

**Developed as a machine learning and exploratory data analysis project.**
