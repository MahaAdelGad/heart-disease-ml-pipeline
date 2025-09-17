import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)

# Missing values
print(df.isnull().sum())

# Quick stats
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Target distribution
sns.countplot(x="target", data=df)
plt.show()


# --- Preprocessing ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

# Encode categorical
X = pd.get_dummies(X, columns=['cp', 'thal', 'slope'], drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)


# --- Supervised Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))

# SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))


# --- Model Comparison ---
from sklearn.metrics import precision_score, recall_score, f1_score

results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}

def evaluate_model(name, y_test, y_pred):
    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_test, y_pred))
    results["Precision"].append(precision_score(y_test, y_pred))
    results["Recall"].append(recall_score(y_test, y_pred))
    results["F1"].append(f1_score(y_test, y_pred))

evaluate_model("LogReg", y_test, log_pred)
evaluate_model("RandomForest", y_test, rf_pred)
evaluate_model("KNN", y_test, knn_pred)
evaluate_model("SVM", y_test, svm_pred)
evaluate_model("DecisionTree", y_test, dt_pred)

results_df = pd.DataFrame(results)
print("\nFinal Model Comparison:")
print(results_df)


# --- PCA ---
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_)
print("\nExplained variance ratio (cumulative):", explained_var)

n_components = np.argmax(explained_var >= 0.95) + 1
print(f"Number of components to keep ~95% variance: {n_components}")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print("Original shape:", X_scaled.shape, " Reduced shape:", X_pca.shape)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Variance Explained")
plt.grid()
plt.show()


# --- Feature Selection ---
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler

# Feature importance (RF)
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance (Random Forest):\n", feature_importance.head(10))

plt.figure(figsize=(10,6))
feature_importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# RFE
log_reg_rfe = LogisticRegression(max_iter=1000)
rfe = RFE(log_reg_rfe, n_features_to_select=8)
rfe.fit(X_train, y_train)
print("\nRFE Selected Features:", X.columns[rfe.support_])

# Chi-Square
X_minmax = MinMaxScaler().fit_transform(X)
chi2_selector = SelectKBest(score_func=chi2, k=8)
chi2_selector.fit(X_minmax, y)
print("\nChi-Square Selected Features:", X.columns[chi2_selector.get_support()])


# --- ROC Curves ---
from sklearn.metrics import roc_curve, roc_auc_score

log_reg_proba = log_reg.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, log_reg_proba)
plt.plot(fpr, tpr, label=f"LogReg (AUC={roc_auc_score(y_test, log_reg_proba):.2f})")

rf_proba = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, rf_proba)
plt.plot(fpr, tpr, label=f"RandomForest (AUC={roc_auc_score(y_test, rf_proba):.2f})")

knn_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, knn_proba)
plt.plot(fpr, tpr, label=f"KNN (AUC={roc_auc_score(y_test, knn_proba):.2f})")

svm_proba = svm.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, svm_proba)
plt.plot(fpr, tpr, label=f"SVM (AUC={roc_auc_score(y_test, svm_proba):.2f})")

dt_proba = dt.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, dt_proba)
plt.plot(fpr, tpr, label=f"DecisionTree (AUC={roc_auc_score(y_test, dt_proba):.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# --- Hyperparameter Tuning (Random Forest) ---
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\nBest Parameters for Random Forest:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
rf_tuned_pred = best_rf.predict(X_test)
print("\nRandom Forest (Tuned)")
print("Accuracy:", accuracy_score(y_test, rf_tuned_pred))
print(classification_report(y_test, rf_tuned_pred))
print(confusion_matrix(y_test, rf_tuned_pred))


# --- Unsupervised Learning ---
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score

# K-Means Elbow
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.show()

# KMeans with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
print("\nK-Means Clustering Crosstab:\n", pd.crosstab(y, clusters))
print("ARI Score:", adjusted_rand_score(y, clusters))

# Hierarchical Clustering (sample of 100)
linked = linkage(X_scaled[:100], method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (sample of 100)")
plt.show()


# --- Save Best Model ---
import joblib
joblib.dump((best_rf, scaler, X.columns), "heart_disease_model.pkl")
print("\nBest model saved as heart_disease_model.pkl")
