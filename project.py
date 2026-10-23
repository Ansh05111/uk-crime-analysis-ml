# ============================================================
# UK POLICE CRIME DATA ANALYSIS - City of London Street-Level Crime, August 2026
# Models: Logistic Regression, Decision Tree, Random Forest, SVM (classification) + K-Means (clustering)
# ============================================================
import os
os.environ["OMP_NUM_THREADS"] = "3"                                                   
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss,
                             classification_report, ConfusionMatrixDisplay, silhouette_score)

sns.set_theme(style="whitegrid")
MIN_CLASS = 10                                # crime types with fewer rows are merged into "Other crime"


def finish(name):
    plt.tight_layout(); plt.savefig(name, dpi=200); plt.show()


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
FILE = "2026-08-city-of-london-street.csv"
try:
    df = pd.read_csv(FILE)
except FileNotFoundError:
    raise SystemExit(f"{FILE} not found - put the CSV in the same folder as this script.")
print(df.info())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())


# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert blank strings into missing values
df = df.replace(r"^\s*$", np.nan, regex=True)

# Convert longitude and latitude into numeric values
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")

# Remove rows with missing essential values
df = df.dropna(
    subset=["Crime type", "Longitude", "Latitude"]
)

# Remove duplicate Crime IDs while retaining records without IDs
df = pd.concat([
    df[df["Crime ID"].notna()].drop_duplicates(subset=["Crime ID"]),
    df[df["Crime ID"].isna()]
]).reset_index(drop=True)

# Keep coordinates within broad UK geographic limits
df = df[
    df["Longitude"].between(-8.5, 2.5)
    & df["Latitude"].between(49.5, 61.5)
].copy()

# Identify rare crime categories
crime_counts = df["Crime type"].value_counts()
rare_crimes = crime_counts[crime_counts < MIN_CLASS].index

# Merge rare categories into "Other crime"
df["Crime type"] = df["Crime type"].where(
    ~df["Crime type"].isin(rare_crimes),
    "Other crime"
)

# Remove categories that still have fewer than MIN_CLASS records
crime_counts = df["Crime type"].value_counts()

df = df[
    df["Crime type"].map(crime_counts) >= MIN_CLASS
].reset_index(drop=True)

# Save the cleaned dataset
df.to_csv("cleaned_crime_data.csv", index=False)

print("\nCleaned shape:", df.shape)
print("Number of crime types:", df["Crime type"].nunique())
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
print("\n========== EXPLORATORY DATA ANALYSIS ==========")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y="Crime type", order=df["Crime type"].value_counts().index)
plt.title("Distribution of Crime Types")
plt.xlabel("Number of Records")
plt.ylabel("Crime Type")
finish("eda_crime_type_distribution.png")

top_areas = df["LSOA name"].value_counts().head(8).index
area_crime = pd.crosstab(df.loc[df["LSOA name"].isin(top_areas), "LSOA name"], df.loc[df["LSOA name"].isin(top_areas), "Crime type"])

plt.figure(figsize=(12, 7))
sns.heatmap(area_crime, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
plt.title("Crime Type Distribution Across Top 8 Areas")
plt.xlabel("Crime Type")
plt.ylabel("LSOA Area")
plt.xticks(rotation=45, ha="right")
finish("eda_crime_area_heatmap.png")


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Crime type", s=55, alpha=0.8)
plt.xlim(-0.20, 0.05)
plt.ylim(51.45, 51.60)
plt.title("Geographic Distribution of Crime Records")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Crime Type", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("eda_geographic_distribution.png", dpi=200)
plt.show()
print("EDA completed successfully.")

# ------------------------------------------------------------
# 4. PREPARE DATA FOR CLASSIFICATION
# ------------------------------------------------------------

candidates = ["Month", "Longitude", "Latitude", "Reported by", "LSOA name"]
features = [c for c in candidates if c in df.columns and df[c].nunique() > 1]

X = df[features].copy()
y = df["Crime type"]

num = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat = [c for c in features if c not in num]
X[cat] = X[cat].fillna("Unknown").astype(str)

print("\n========== DATA PREPARATION ==========")
print("Features:", features)
print("Total records:", len(X))
print("Crime categories:", y.nunique())

model_results = []


# ------------------------------------------------------------
# 5.1 LOGISTIC REGRESSION
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_lr = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr_lr = pre_lr.fit_transform(X_train)
Xte_lr = pre_lr.transform(X_test)

lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr.fit(Xtr_lr, y_train)

lr_pred = lr.predict(Xte_lr)
lr_prob = lr.predict_proba(Xte_lr)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, average="weighted", zero_division=0)
lr_recall = recall_score(y_test, lr_pred, average="weighted", zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, average="weighted", zero_division=0)
lr_loss = log_loss(y_test, lr_prob, labels=lr.classes_)

print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy:", round(lr_accuracy, 4))
print("Precision:", round(lr_precision, 4))
print("Recall:", round(lr_recall, 4))
print("F1 Score:", round(lr_f1, 4))
print(classification_report(y_test, lr_pred, zero_division=0))

model_results.append({"Model": "Logistic Regression", "Accuracy": lr_accuracy, "Precision": lr_precision, "Recall": lr_recall, "F1 Score": lr_f1, "Log Loss": lr_loss})


# ------------------------------------------------------------
# 5.2 DECISION TREE
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_dt = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr_dt = pre_dt.fit_transform(X_train)
Xte_dt = pre_dt.transform(X_test)

dt = DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
dt.fit(Xtr_dt, y_train)

dt_pred = dt.predict(Xte_dt)
dt_prob = dt.predict_proba(Xte_dt)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, average="weighted", zero_division=0)
dt_recall = recall_score(y_test, dt_pred, average="weighted", zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, average="weighted", zero_division=0)
dt_loss = log_loss(y_test, dt_prob, labels=dt.classes_)

print("\n========== DECISION TREE ==========")
print("Accuracy:", round(dt_accuracy, 4))
print("Precision:", round(dt_precision, 4))
print("Recall:", round(dt_recall, 4))
print("F1 Score:", round(dt_f1, 4))
print(classification_report(y_test, dt_pred, zero_division=0))

model_results.append({"Model": "Decision Tree", "Accuracy": dt_accuracy, "Precision": dt_precision, "Recall": dt_recall, "F1 Score": dt_f1, "Log Loss": dt_loss})


# ------------------------------------------------------------
# 5.3 RANDOM FOREST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_rf = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr_rf = pre_rf.fit_transform(X_train)
Xte_rf = pre_rf.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(Xtr_rf, y_train)

rf_pred = rf.predict(Xte_rf)
rf_prob = rf.predict_proba(Xte_rf)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average="weighted", zero_division=0)
rf_loss = log_loss(y_test, rf_prob, labels=rf.classes_)

print("\n========== RANDOM FOREST ==========")
print("Accuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print(classification_report(y_test, rf_pred, zero_division=0))

model_results.append({"Model": "Random Forest", "Accuracy": rf_accuracy, "Precision": rf_precision, "Recall": rf_recall, "F1 Score": rf_f1, "Log Loss": rf_loss})


# ------------------------------------------------------------
# 5.4 SUPPORT VECTOR MACHINE
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre_svm = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr_svm = pre_svm.fit_transform(X_train)
Xte_svm = pre_svm.transform(X_test)

svm = CalibratedClassifierCV(estimator=SVC(kernel="rbf", class_weight="balanced"), method="sigmoid", cv=3)
svm.fit(Xtr_svm, y_train)

svm_pred = svm.predict(Xte_svm)
svm_prob = svm.predict_proba(Xte_svm)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_recall = recall_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, average="weighted", zero_division=0)
svm_loss = log_loss(y_test, svm_prob, labels=svm.classes_)

print("\n========== SUPPORT VECTOR MACHINE ==========")
print("Accuracy:", round(svm_accuracy, 4))
print("Precision:", round(svm_precision, 4))
print("Recall:", round(svm_recall, 4))
print("F1 Score:", round(svm_f1, 4))
print(classification_report(y_test, svm_pred, zero_division=0))

model_results.append({"Model": "SVM", "Accuracy": svm_accuracy, "Precision": svm_precision, "Recall": svm_recall, "F1 Score": svm_f1, "Log Loss": svm_loss})


# ------------------------------------------------------------
# 6. K-MEANS CLUSTERING
# ------------------------------------------------------------

print("\n========== K-MEANS CLUSTERING ==========")

geo = df.loc[df["Longitude"].between(-0.20, 0.05) & df["Latitude"].between(51.45, 51.60), ["Longitude", "Latitude"]].dropna().copy()

geo_scaler = StandardScaler()
geo_scaled = geo_scaler.fit_transform(geo)

cluster_results = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(geo_scaled)
    sizes = pd.Series(labels).value_counts()
    silhouette = silhouette_score(geo_scaled, labels) if len(sizes) > 1 and sizes.min() >= 5 else np.nan
    cluster_results.append({"K": k, "Inertia": km.inertia_, "Silhouette Score": silhouette})
cluster_scores = pd.DataFrame(cluster_results)

print("\nK-Means Evaluation:")
print(cluster_scores.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(cluster_scores["K"], cluster_scores["Inertia"], marker="o")
plt.xticks(cluster_scores["K"])
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Plot")
finish("kmeans_elbow.png")
valid_scores = cluster_scores.dropna(subset=["Silhouette Score"])

if not valid_scores.empty:
    plt.figure(figsize=(8, 5))
    plt.plot(valid_scores["K"], valid_scores["Silhouette Score"], marker="o")
    plt.xticks(valid_scores["K"])
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Scores")
    finish("kmeans_silhouette.png")
    best_k = int(valid_scores.loc[valid_scores["Silhouette Score"].idxmax(), "K"])
    print("\nBest K:", best_k)
else:
    print("No valid K found for Silhouette Score.")
    
    
    

# ------------------------------------------------------------
# 7. FINAL CLASSIFICATION MODEL COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame(model_results)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))
comparison.to_csv("model_comparison.csv", index=False)

ax = comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot.bar(figsize=(10, 6), rot=15, ylim=(0, 1))
ax.set_title("Classification Model Comparison")
ax.set_ylabel("Score")
finish("model_comparison.png")

best_model = comparison.loc[comparison["F1 Score"].idxmax()]

print("\nBest Model by Weighted F1 Score:", best_model["Model"])
print("Best Weighted F1 Score:", round(best_model["F1 Score"], 4))
