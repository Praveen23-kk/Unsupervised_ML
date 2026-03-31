# Unsupervised Machine Learning — Clustering & Dimensionality Reduction

A collection of hands-on Jupyter notebooks covering the core unsupervised learning algorithms using **scikit-learn**, **scipy**, and real-world datasets.

---

## 📁 Notebooks

### 1. `K_Means_Clustering.ipynb`
**Algorithm:** K-Means Clustering  
**Dataset:** Synthetic blob data (`make_blobs` — 1000 samples, 3 clusters)

Covers the full K-Means pipeline from data generation to cluster validation:
- Feature scaling with `StandardScaler`
- Train/test split before clustering
- **Elbow Method** (WCSS plot) to find the optimal number of clusters
- **Knee/Elbow detection** using the `kneed` library (`KneeLocator`)
- **Silhouette Score** analysis to validate cluster quality
- Final cluster visualization with scatter plots

---

### 2. `DBSCAN.ipynb`
**Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
**Dataset:** Synthetic moon-shaped data (`make_moons` — 250 samples)

Demonstrates how DBSCAN handles non-linearly separable clusters that K-Means fails on:
- Why DBSCAN is better than K-Means for non-convex shapes
- Feature scaling with `StandardScaler`
- Fitting DBSCAN with `eps=0.3`
- Visualizing cluster labels vs. true labels side-by-side

---

### 3. `Hierarchical_Clustering.ipynb`
**Algorithm:** Agglomerative (Hierarchical) Clustering  
**Dataset:** Iris dataset (sklearn built-in)

Covers bottom-up hierarchical clustering with a PCA preprocessing step:
- Standardization → PCA (2 components) for 2D visualization
- **Dendrogram** construction using `scipy.cluster.hierarchy` (Ward linkage)
- Cutting the dendrogram at `n_clusters=2` with `AgglomerativeClustering`
- Euclidean distance + Ward linkage strategy
- Scatter plot of final cluster assignments in PCA space

---

### 4. `PCA_Principal_Component_Analysis.ipynb`
**Algorithm:** Principal Component Analysis (PCA)  
**Dataset:** Breast Cancer Wisconsin dataset (sklearn built-in — 30 features)

Demonstrates dimensionality reduction from 30 features to 4 principal components:
- Standardization of high-dimensional data
- Fitting `PCA(n_components=4)` with scikit-learn
- Inspecting `explained_variance_` for each component
- 2D visualization of the first two principal components colored by cancer target (malignant vs. benign)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Clustering algorithms, PCA, datasets, preprocessing, metrics |
| `scipy` | Hierarchical clustering dendrogram |
| `matplotlib` | Visualizations and plots |
| `pandas` / `numpy` | Data manipulation |
| `seaborn` | Statistical plots |
| `kneed` | Automated elbow/knee detection |

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Praveen23-kk/Unsupervised_ML/.git
cd Unsupervised_ML
```

**2. Install dependencies**
```bash
pip install scikit-learn scipy matplotlib pandas numpy seaborn kneed
```

**3. Launch Jupyter**
```bash
jupyter notebook
```

Then open any `.ipynb` file and run all cells.

---

## 📊 Algorithms at a Glance

| Notebook | Algorithm | Dataset | Key Concept |
|---|---|---|---|
| K-Means | Centroid-based | Blobs (synthetic) | Elbow + Silhouette |
| DBSCAN | Density-based | Moons (synthetic) | Noise-robust clustering |
| Hierarchical | Agglomerative | Iris | Dendrogram + Ward linkage |
| PCA | Dimensionality reduction | Breast Cancer | Variance explained |

---

## 📌 Key Concepts Covered

- **Feature Scaling** — Why standardization is critical before clustering
- **Choosing K** — Elbow method, KneeLocator, and Silhouette scores
- **Non-linear Clusters** — Why DBSCAN outperforms K-Means on moon/ring shapes
- **Dendrograms** — How to read and cut a hierarchical tree
- **Dimensionality Reduction** — Compressing 30 features into 2D while preserving variance

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
