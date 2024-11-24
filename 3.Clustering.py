import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys


def load_processed_data(file_path):
    """Loads the processed data from CSV"""
    df = pd.read_csv(file_path)
    return df


def apply_kmeans_clustering(df, n_clusters=5):
    """Applies K-means clustering to the data"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)
    return labels, kmeans


def apply_dbscan_clustering(df, eps=0.5, min_samples=5):
    """Applies DBSCAN clustering to the data"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    return labels, dbscan


def apply_hierarchical_clustering(df, n_clusters=5):
    """Applies hierarchical clustering to the data"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(df)
    return labels, hierarchical


def evaluate_clustering(df, labels):
    """Evaluates clustering performance using different metrics"""
    if len(set(labels)) > 1:  # Ensure there is more than one cluster
        silhouette = silhouette_score(df, labels)
        calinski_harabasz = calinski_harabasz_score(df, labels)
        davies_bouldin = davies_bouldin_score(df, labels)
        return (silhouette , calinski_harabasz , davies_bouldin)
    else:
        print("Clustering resulted in a single cluster or noise; evaluation metrics are not meaningful.")


def visualize_clusters(df, labels, title, output_image):
    """Visualizes the clustering results using PCA"""
    # Dimensionality reduction
    reducer = PCA(n_components=3)
    reduced_data = reducer.fit_transform(df)

    fig = plt.figure(figsize=(10, 8))

    # Create 3D axis
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Principal Component 1')
    axis.set_ylabel('Principal Component 2')
    axis.set_zlabel('Principal Component 3')

    # Plot data points in 3D space
    scatter = axis.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap='viridis',
                           alpha=0.8)

    # Use colorbar for cluster label legend
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks(np.arange(np.min(labels), np.max(labels) + 1))  # Set ticks to cluster labels
    colorbar.set_label('Cluster Label')

    plt.title(title)
    plt.savefig(output_image)
    plt.close()


def tune_kmeans(df):
    """Tunes K-means clustering"""
    print("\nTuning K-means Clustering...")
    best_silhouette = 0
    best_k = 0
    for k in range(2, 10):
        kmeans_labels, _ = apply_kmeans_clustering(df, n_clusters=k)
        silhouette, calinski, davies = evaluate_clustering(df, kmeans_labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    print(f"Best K for K-means: {best_k}, Best Silhouette Score: {best_silhouette:.3f}")
    kmeans_labels, _ = apply_kmeans_clustering(df, n_clusters=best_k)
    visualize_clusters(df, kmeans_labels, 'KMeans Clustering', './visualizations/clustering/clustering_kmeans.jpeg')


def tune_dbscan(df):
    """Tunes DBSCAN clustering"""
    print("\nTuning DBSCAN Clustering...")
    best_silhouette = -1
    best_eps = 0
    best_min_samples = 0
    for eps in np.arange(0.5, 10.0, 0.1):  # Test a range of `eps` values
        for min_samples in range(2, 10):   # Test a range of `min_samples`
            dbscan_labels, _ = apply_dbscan_clustering(df, eps=eps, min_samples=min_samples)
            if len(set(dbscan_labels)) > 1:  # Ensure meaningful clusters
                silhouette, calinski, davies = evaluate_clustering(df, dbscan_labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples
    print(f"Best parameters for DBSCAN: eps={best_eps}, min_samples={best_min_samples}, Best Silhouette Score: {best_silhouette:.3f}")
    dbscan_labels, _ = apply_dbscan_clustering(df, eps=best_eps, min_samples=best_min_samples)
    visualize_clusters(df, dbscan_labels, 'DBSCAN Clustering', './visualizations/clustering/clustering_dbscan.jpeg')


def tune_hierarchical(df):
    """Tunes Hierarchical clustering"""
    print("\nTuning Hierarchical Clustering...")
    best_silhouette = 0
    best_n_clusters = 0
    for n_clusters in range(2, 10):  # Test different numbers of clusters
        hierarchical_labels, _ = apply_hierarchical_clustering(df, n_clusters=n_clusters)
        silhouette, calinski, davies = evaluate_clustering(df, hierarchical_labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_n_clusters = n_clusters
    print(f"Best number of clusters for Hierarchical Clustering: {best_n_clusters}, Best Silhouette Score: {best_silhouette:.3f}")
    hierarchical_labels, _ = apply_hierarchical_clustering(df, n_clusters=best_n_clusters)
    visualize_clusters(df, hierarchical_labels, 'Hierarchical Clustering', './visualizations/clustering/clustering_hierarchical.jpeg')

def main():
    file_path = sys.argv[1]
    # Load the processed data
    df = load_processed_data(file_path)
    df.drop(columns=['price_bucket_equidepth' , 'price_bucket_equiwidth', 'price'], inplace=True)

    kmeans_labels, _ = apply_kmeans_clustering(df, n_clusters=2)
    visualize_clusters(df, kmeans_labels, 'KMeans Clustering', './visualizations/clustering/clustering_kmeans.jpeg')
    silhouette, calinski, davies  = evaluate_clustering(df, kmeans_labels)
    print("\nApplying Kmeans:...")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies:.3f}")


    dbscan_labels, _ = apply_dbscan_clustering(df, eps=9.699999999999998, min_samples=4)
    visualize_clusters(df, dbscan_labels, 'DBSCAN Clustering', './visualizations/clustering/clustering_dbscan.jpeg')
    silhouette, calinski, davies = evaluate_clustering(df, dbscan_labels)
    print("\nApplying DBSCAN:...")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies:.3f}")


    hierarchical_labels, _ = apply_hierarchical_clustering(df, n_clusters=2)
    visualize_clusters(df, hierarchical_labels, 'Hierarchical Clustering', './visualizations/clustering/clustering_hierarchical.jpeg')
    silhouette, calinski, davies = evaluate_clustering(df, hierarchical_labels)
    print("\nApplying Hierarchical:...")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies:.3f}")


    tune_kmeans(df)
    tune_dbscan(df)
    tune_hierarchical(df)



if __name__ == "__main__":
    main()
