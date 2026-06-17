import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_pca_analysis(scaled_features, n_components=2):
    """Applies PCA to reduce dimensionality."""
    print("\n--- Running Principal Component Analysis ---")
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(
        data=pca_result, 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=scaled_features.index
    )
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return pca, pca_df

def find_optimal_clusters(scaled_features, max_k=8):
    """Evaluates K-Means configurations iteratively."""
    print("\n--- Tuning K-Means Clustering ---")
    inertia_scores = []
    silhouette_avg_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(scaled_features, cluster_labels)
        
        inertia_scores.append(inertia)
        silhouette_avg_scores.append(sil_score)
        
        print(f"  k={k} | Inertia: {inertia:.2f} | Silhouette Score: {sil_score:.3f}")
        
    return list(k_range), inertia_scores, silhouette_avg_scores
  
  
