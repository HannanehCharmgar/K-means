import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

# ==============================
# 1️⃣ LOAD DATA
# ==============================

print("\n[INFO] Loading Iris dataset from sklearn...\n")

iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)
print("[INFO] Dataset loaded successfully.")
print("[INFO] First 5 rows:")
print(df.head())

# ==============================
# 2️⃣ DATA CLEANING
# ==============================

print("\n[INFO] Checking for missing values...\n")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\n[INFO] No missing values found. Dataset is clean.")
else:
    print("\n[WARNING] Missing values detected!")

# ==============================
# 3️⃣ NORMALIZATION
# ==============================

print("\n[INFO] Applying StandardScaler normalization...\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[INFO] Normalization completed.")
print(f"[INFO] Mean after scaling (approx): {np.mean(X_scaled, axis=0)}")
print(f"[INFO] Std after scaling (approx): {np.std(X_scaled, axis=0)}")

# ==============================
# 4️⃣ CUSTOM K-MEANS IMPLEMENTATION
# ==============================

class KMeansCustom:
    def __init__(self, k=3, max_iters=10):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        self.history = []

        for iteration in range(self.max_iters):
            print(f"\n[INFO] Iteration {iteration+1} started.")

            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Save state for visualization
            self.history.append((X.copy(), labels.copy(), self.centroids.copy()))

            # Update centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) for i in range(self.k)
            ])

            # Check convergence
            
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print("[INFO] Convergence reached.")
                break

            self.centroids = new_centroids

        self.labels_ = labels
        print("\n[INFO] Training completed successfully.")


# ==============================
# 5️⃣ TRAIN MODEL
# ==============================

print("\n[INFO] Training K-Means model...\n")

kmeans = KMeansCustom(k=3, max_iters=8)
kmeans.fit(X_scaled)

labels = kmeans.labels_

# ==============================
# 6️⃣ VISUALIZE ITERATIONS (PCA 2D)
# ==============================

print("\n[INFO] Visualizing algorithm steps...\n")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for i, (X_step, labels_step, centroids_step) in enumerate(kmeans.history):
    plt.figure()
    X_step_pca = pca.transform(X_step)
    centroids_pca = pca.transform(centroids_step)

    plt.scatter(X_step_pca[:, 0], X_step_pca[:, 1], c=labels_step)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200)
    plt.title(f"K-Means Iteration {i+1}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# ==============================
# 7️⃣ EVALUATION METRICS
# ==============================

print("\n[INFO] Evaluating model performance...\n")

sil_score = silhouette_score(X_scaled, labels)
print(f"[RESULT] Silhouette Score: {sil_score:.4f}")

conf_matrix = confusion_matrix(y_true, labels)
print("\n[RESULT] Confusion Matrix:")
print(conf_matrix)

# ==============================
# 8️⃣ ELBOW METHOD
# ==============================

print("\n[INFO] Computing Elbow Method...\n")

inertias = []
k_range = range(1, 10)

for k in k_range:
    model = KMeansCustom(k=k, max_iters=10)
    model.fit(X_scaled)
    inertia = 0
    for i in range(k):
        cluster_points = X_scaled[model.labels_ == i]
        centroid = model.centroids[i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    inertias.append(inertia)

plt.figure()
plt.plot(k_range, inertias)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()
 #====================================
 # metric show
 #===================================   
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from scipy.optimize import linear_sum_assignment

print("\n==============================")
print("      MODEL EVALUATION")
print("==============================\n")

# -------- Internal Metrics --------
inertia = 0
for i in range(kmeans.k):
    cluster_points = X_scaled[labels == i]
    centroid = kmeans.centroids[i]
    inertia += np.sum((cluster_points - centroid) ** 2)

sil_score = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)
ch_score = calinski_harabasz_score(X_scaled, labels)

print("🔹 Internal Evaluation Metrics (No ground truth used):")
print(f"Inertia (WCSS): {inertia:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

# -------- External Metrics --------
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)
homogeneity = homogeneity_score(y_true, labels)
completeness = completeness_score(y_true, labels)
v_measure = v_measure_score(y_true, labels)

print("\n🔹 External Evaluation Metrics (Using ground truth for analysis only):")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Homogeneity: {homogeneity:.4f}")
print(f"Completeness: {completeness:.4f}")
print(f"V-Measure: {v_measure:.4f}")

# -------- Accuracy with Optimal Label Mapping --------
conf_matrix = confusion_matrix(y_true, labels)
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
accuracy = conf_matrix[row_ind, col_ind].sum() / np.sum(conf_matrix)

print(f"\nCluster Accuracy (after optimal mapping): {accuracy:.4f}")

import matplotlib.pyplot as plt
import numpy as np

print("\n[INFO] Plotting evaluation metrics properly...\n")

# ---------- INTERNAL METRICS ----------
internal_names = [
    "Silhouette",
    "1 / Davies-Bouldin",
    "Calinski-Harabasz (scaled)"
]


db_inverse = 1 / db_score
ch_scaled = ch_score / max(ch_score, 1)  

internal_values = [
    sil_score,
    db_inverse,
    ch_scaled
]

plt.figure(figsize=(8,5))
plt.bar(internal_names, internal_values)
plt.title("Internal Evaluation Metrics")
plt.ylabel("Score (Higher is Better)")
plt.xticks(rotation=20)
plt.show()


# ---------- EXTERNAL METRICS ----------
external_names = [
    "ARI",
    "NMI",
    "Homogeneity",
    "Completeness",
    "V-Measure",
    "Accuracy"
]

external_values = [
    ari,
    nmi,
    homogeneity,
    completeness,
    v_measure,
    accuracy
]

plt.figure(figsize=(8,5))
plt.bar(external_names, external_values)
plt.title("External Evaluation Metrics")
plt.ylabel("Score (Higher is Better)")
plt.xticks(rotation=30)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

print("\n[INFO] Generating Radar Chart for overall performance...\n")

# معیارهایی که بین 0 و 1 هستند بهترن برای رادار
radar_metrics = {
    "Silhouette": sil_score,
    "ARI": ari,
    "NMI": nmi,
    "Homogeneity": homogeneity,
    "Completeness": completeness,
    "V-Measure": v_measure,
    "Accuracy": accuracy
}

labels = list(radar_metrics.keys())
values = list(radar_metrics.values())


values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("K-Means Overall Performance Radar Chart", pad=20)

plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print("\n[INFO] Evaluating different K values...\n")

k_values = range(2, 10)

sil_scores = []
db_scores = []
ch_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = model.fit_predict(X_scaled)
    
    sil_scores.append(silhouette_score(X_scaled, labels_k))
    db_scores.append(davies_bouldin_score(X_scaled, labels_k))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))

# ----- Silhouette -----
plt.figure()
plt.plot(k_values, sil_scores)
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# ----- Davies-Bouldin -----
plt.figure()
plt.plot(k_values, db_scores)
plt.title("Davies-Bouldin Index vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index (Lower is Better)")
plt.show()

# ----- Calinski-Harabasz -----
plt.figure()
plt.plot(k_values, ch_scores)
plt.title("Calinski-Harabasz Index vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Calinski-Harabasz Score")
plt.show()