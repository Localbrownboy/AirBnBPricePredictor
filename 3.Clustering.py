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
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
    else:
        print("Clustering resulted in a single cluster or noise; evaluation metrics are not meaningful.")


def visualize_clusters(df, labels):
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

    plt.title('AirBnB Data 3D Cluster Visualization')
    plt.show()


def main():
    file_path = sys.argv[1]
    # Load the processed data
    df = load_processed_data(file_path)
    df.drop(columns=['price_bucket'], inplace=True)

    # Apply K-means clustering
    print("\nApplying K-means Clustering...")
    kmeans_labels, kmeans_model = apply_kmeans_clustering(df, n_clusters=5)
    evaluate_clustering(df, kmeans_labels)
    visualize_clusters(df, kmeans_labels)

    # Apply DBSCAN clustering
    print("\nApplying DBSCAN Clustering...")
    dbscan_labels, _ = apply_dbscan_clustering(df, eps=0.3, min_samples=5)
    unique_labels = len(set(dbscan_labels))
    print(f"Number of clusters found by DBSCAN: {unique_labels}")
    evaluate_clustering(df, dbscan_labels)
    visualize_clusters(df, dbscan_labels)

    # Apply Hierarchical Clustering
    print("\nApplying Hierarchical Clustering...")
    hierarchical_labels, _ = apply_hierarchical_clustering(df, n_clusters=5)
    evaluate_clustering(df, hierarchical_labels)
    visualize_clusters(df, hierarchical_labels)


if __name__ == "__main__":
    main()
