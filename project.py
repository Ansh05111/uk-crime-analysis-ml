# ============================================================
# UK POLICE CRIME DATA ANALYSIS
# Dataset: City of London Street-Level Crime, August 2026
# Models: Logistic Regression, Decision Tree, Random Forest,
#         Gradient Boosting, and SVM (five supervised classifiers)
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report,
    make_scorer
)

sns.set_theme(style="whitegrid")
MIN_CLASS = 10  # Rare crime types are grouped into "Other crime".


def finish(name):
    """Tidy, save, and display the current plot."""
    plt.tight_layout()
    plt.savefig(name, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
FILE = "2026-08-city-of-london-street.csv"
try:
    df = pd.read_csv(FILE)
except FileNotFoundError:
    raise SystemExit(
        f"{FILE} not found. Put the CSV file in the same folder as this script."
    )

print("========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values before cleaning:\n", df.isnull().sum())
print("Duplicate rows before cleaning:", df.duplicated().sum())

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------
df.columns = df.columns.str.strip()
df = df.replace(r"^\\s*$", np.nan, regex=True)

for col in ["Longitude", "Latitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Crime type", "Longitude", "Latitude"]).copy()

# Remove duplicate Crime IDs only; keep records without an ID.
if "Crime ID" in df.columns:
    has_id = df["Crime ID"].notna()
    df = pd.concat([
        df.loc[has_id].drop_duplicates(subset=["Crime ID"]),
        df.loc[~has_id]
    ], ignore_index=True)
else:
    df = df.drop_duplicates()

# Keep valid broad UK coordinates.
df = df[
    df["Longitude"].between(-8.5, 2.5)
    & df["Latitude"].between(49.5, 61.5)
].copy()

# Merge rare crime types into "Other crime".
rare = df["Crime type"].value_counts()
rare = rare[rare < MIN_CLASS].index
df["Crime type"] = df["Crime type"].replace(dict.fromkeys(rare, "Other crime"))

df = df.reset_index(drop=True)
df.to_csv("cleaned_crime_data.csv", index=False)

print("\\n========== CLEANED DATASET ==========")
print("Cleaned shape:", df.shape)
print("Crime types:", df["Crime type"].nunique())
print(df["Crime type"].value_counts())
print("\\nMissing values after cleaning:\\n", df.isnull().sum())

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
# Plot 1: distribution of crime types.
plt.figure(figsize=(11, 5))
df["Crime type"].value_counts().plot(kind="bar", title="Distribution of Crime Types")
plt.xlabel("Crime type")
plt.ylabel("Number of records")
plt.xticks(rotation=45, ha="right")
finish("eda_crime_types.png")

# Plot 2: outcomes, when the dataset contains this column.
if "Last outcome category" in df.columns:
    plt.figure(figsize=(11, 5))
    df["Last outcome category"].fillna("Unknown").value_counts().head(10).plot(
        kind="bar", title="Top 10 Crime Outcomes"
    )
    plt.xlabel("Outcome category")
    plt.ylabel("Number of records")
    plt.xticks(rotation=45, ha="right")
    finish("eda_outcomes.png")

# Plot 3: geographic distribution, zoomed to the London region for readability.
plt.figure(figsize=(8, 6))
plt.scatter(df["Longitude"], df["Latitude"], alpha=0.5, s=15)
plt.title("Geographic Distribution of Crime Records")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(-0.20, 0.05)
plt.ylim(51.45, 51.60)
finish("eda_map.png")

# Plot 4: crime type by the eight most frequent LSOA areas.
if "LSOA name" in df.columns:
    top_areas = df["LSOA name"].dropna().value_counts().head(8).index
    if len(top_areas):
        cross_tab = pd.crosstab(
            df.loc[df["LSOA name"].isin(top_areas), "LSOA name"],
            df.loc[df["LSOA name"].isin(top_areas), "Crime type"]
        )
        plt.figure(figsize=(12, 6))
        sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Crime Type by Top 8 LSOA Areas")
        plt.xlabel("Crime type")
        plt.ylabel("LSOA area")
        finish("eda_lsoa_by_type.png")

# ------------------------------------------------------------
# 4. PREPARE DATA FOR THE MODELS
# ------------------------------------------------------------
possible_features = ["Month", "Longitude", "Latitude", "Reported by", "LSOA name"]
features = [
    col for col in possible_features
    if col in df.columns and df[col].nunique(dropna=True) > 1
]
dropped_features = [
    col for col in possible_features
    if col in df.columns and col not in features
]

print("\nFeatures used:", features)
print("Columns dropped (constant or empty):", dropped_features)

X = df[features].copy()
y = df["Crime type"].copy()

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [col for col in features if col not in numeric_cols]

for col in categorical_cols:
    X[col] = X[col].fillna("Unknown").astype(str)

# Scale numeric columns and encode text columns for the classifiers.
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    sparse_threshold=0
)

baseline = y.value_counts(normalize=True).max()
print(f"Majority-class baseline (full dataset): {baseline:.3f}")

# ------------------------------------------------------------
# 5. TRAIN AND EVALUATE THE CLASSIFIERS
# ------------------------------------------------------------

# 5.1 Logistic Regression
print("\n========== 5.1 LOGISTIC REGRESSION ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
log_model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, log_pred), 4))
print("Precision:", round(precision_score(y_test, log_pred, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, log_pred, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, log_pred, average="weighted", zero_division=0), 4))
print("Log Loss:", round(log_loss(y_test, log_prob, labels=log_model.steps[-1][1].classes_), 4))
print(classification_report(y_test, log_pred, zero_division=0))
log_accuracy = accuracy_score(y_test, log_pred)
log_f1 = f1_score(y_test, log_pred, average="weighted", zero_division=0)

# 5.2 Decision Tree
print("\n========== 5.2 DECISION TREE ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=43, stratify=y
)
tree_model = make_pipeline(
    preprocessor,
    DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_prob = tree_model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, tree_pred), 4))
print("Precision:", round(precision_score(y_test, tree_pred, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, tree_pred, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, tree_pred, average="weighted", zero_division=0), 4))
print("Log Loss:", round(log_loss(y_test, tree_prob, labels=tree_model.steps[-1][1].classes_), 4))
print(classification_report(y_test, tree_pred, zero_division=0))
tree_accuracy = accuracy_score(y_test, tree_pred)
tree_f1 = f1_score(y_test, tree_pred, average="weighted", zero_division=0)

# 5.3 Random Forest
print("\n========== 5.3 RANDOM FOREST ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=44, stratify=y
)
forest_model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
forest_prob = forest_model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, forest_pred), 4))
print("Precision:", round(precision_score(y_test, forest_pred, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, forest_pred, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, forest_pred, average="weighted", zero_division=0), 4))
print("Log Loss:", round(log_loss(y_test, forest_prob, labels=forest_model.steps[-1][1].classes_), 4))
print(classification_report(y_test, forest_pred, zero_division=0))
forest_accuracy = accuracy_score(y_test, forest_pred)
forest_f1 = f1_score(y_test, forest_pred, average="weighted", zero_division=0)

# 5.4 Gradient Boosting
print("\n========== 5.4 GRADIENT BOOSTING ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=45, stratify=y
)
boost_model = make_pipeline(
    preprocessor,
    GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
)
boost_model.fit(X_train, y_train)
boost_pred = boost_model.predict(X_test)
boost_prob = boost_model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, boost_pred), 4))
print("Precision:", round(precision_score(y_test, boost_pred, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, boost_pred, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, boost_pred, average="weighted", zero_division=0), 4))
print("Log Loss:", round(log_loss(y_test, boost_prob, labels=boost_model.steps[-1][1].classes_), 4))
print(classification_report(y_test, boost_pred, zero_division=0))
boost_accuracy = accuracy_score(y_test, boost_pred)
boost_f1 = f1_score(y_test, boost_pred, average="weighted", zero_division=0)

# 5.5 Support Vector Machine (SVM)
print("\n========== 5.5 SUPPORT VECTOR MACHINE ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=46, stratify=y
)
svm_model = make_pipeline(
    preprocessor,
    CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", class_weight="balanced"),
        method="sigmoid",
        cv=3
    )
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)
print("Accuracy:", round(accuracy_score(y_test, svm_pred), 4))
print("Precision:", round(precision_score(y_test, svm_pred, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, svm_pred, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, svm_pred, average="weighted", zero_division=0), 4))
print("Log Loss:", round(log_loss(y_test, svm_prob, labels=svm_model.steps[-1][1].classes_), 4))
print(classification_report(y_test, svm_pred, zero_division=0))
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average="weighted", zero_division=0)

results = [
    {"Model": "Logistic Regression", "Accuracy": log_accuracy, "F1 Score": log_f1},
    {"Model": "Decision Tree", "Accuracy": tree_accuracy, "F1 Score": tree_f1},
    {"Model": "Random Forest", "Accuracy": forest_accuracy, "F1 Score": forest_f1},
    {"Model": "Gradient Boosting", "Accuracy": boost_accuracy, "F1 Score": boost_f1},
    {"Model": "SVM", "Accuracy": svm_accuracy, "F1 Score": svm_f1},
]
# ------------------------------------------------------------
# 6. COMPARE CLASSIFIERS
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of Five Classifiers")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()
