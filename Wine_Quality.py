import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Make sure the CSV file `wine_quality.csv` is in the same directory
file = pd.read_csv(r"C:\Users\OMEN\Downloads\Wine_Quality\winequality-red.csv")

# Step 2: Preprocess the data
X = file.drop(columns=['quality'])  # Features
y = file['quality']  # Target variable (quality)

# Standardize the features for PCA, t-SNE, and UMAP
#scaler = StandardScaler()
X_scaled = StandardScaler().fit_transform(X)

# Step 3: Apply dimensionality reduction
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, n_jobs=1)
X_umap = umap_reducer.fit_transform(X_scaled)

# Step 4: Plot the results
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# PCA Plot
scatter_pca = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolor='k', s=30)
axs[0].set_title("PCA", fontsize=16)
axs[0].set_xlabel("PC_1", fontsize=10)
axs[0].set_ylabel("PC_2", fontsize=10)
colorbr_pca = fig.colorbar(scatter_pca, ax=axs[0], label='Quality', orientation="vertical")

# t-SNE Plot
scatter_tsne = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolor='k', s=30)
axs[1].set_title("t-SNE", fontsize=16)
axs[1].set_xlabel("t-SNE_1", fontsize=10)
axs[1].set_ylabel("t-SNE_2", fontsize=10)
colorbr_tsne = fig.colorbar(scatter_tsne, ax=axs[1], label='Quality', orientation="vertical")

# UMAP Plot
scatter_umap = axs[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolor='k', s=30)
axs[2].set_title("UMAP", fontsize=16)
axs[2].set_xlabel("UMAP_1", fontsize=10)
axs[2].set_ylabel("UMAP_2", fontsize=10)
colorbr_umap = fig.colorbar(scatter_umap, ax=axs[2], label='Quality', orientation="vertical")

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

from scipy import stats

# Group the data by 'quality' and extract the 'residual sugar' for each quality level
groups = [file[file['quality'] == q]['residual sugar'] for q in file['quality'].unique()]

# Perform the one-way ANOVA test
f_statistic,p_value = stats.f_oneway(*groups)

# Print the results
#print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
