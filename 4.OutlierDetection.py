import pandas as pd
import sys
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import numpy as np 
# Load data function
def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

# train a LOF model and return the index of outliers in dataset
def local_outlier_factor(df):
    model = LocalOutlierFactor(n_neighbors=400, contamination=0.06)
    predictions = model.fit_predict(df)
    
    # model.fit_predict(df) returns either 1 (inliners) or -1 (outliers)
    df['outliers'] = predictions
    outliers = df[df['outliers'] == -1]

    print(f'Detected {len(outliers)} outliers from Local Outlier Factor.')

    outlier_index = outliers.index
    return outlier_index

# train an isolation forest model and return the index of outliers in dataset
def isolation_forest(df):
    model = IsolationForest(n_estimators=300, contamination=0.02)
    predictions = model.fit_predict(df)
    
    df['outliers'] = predictions
    outliers = df[df['outliers'] == -1]

    print(f'Detected {len(outliers)} outliers from Isolation Forest.')

    outlier_index = outliers.index
    return outlier_index

# train an one class SVM model and return the index of outliers in dataset
def one_class_svm(df):
    model = OneClassSVM(nu=0.02, gamma='scale', kernel='rbf')
    predictions = model.fit_predict(df)
    
    df['outliers'] = predictions
    outliers = df[df['outliers'] == -1]

    print(f'Detected {len(outliers)} outliers from One Class SVM.')

    outlier_index = outliers.index
    return outlier_index

# visualize the dataset by 3d plot with outliers marked
def visualize_outliers(df, outliers, title, filepath):
    # Dimensionality reduction to 3 components
    reducer = PCA(n_components=3)
    reduced_data = reducer.fit_transform(df)

    fig = plt.figure(figsize=(10, 8))

    # Create 3D axis
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Principal Component x')
    axis.set_ylabel('Principal Component y')
    axis.set_zlabel('Principal Component z')

    # Plot normal points
    normal_points = np.setdiff1d(np.arange(len(df)), outliers)
    axis.scatter(reduced_data[normal_points, 0], reduced_data[normal_points, 1], reduced_data[normal_points, 2],
                 c='cyan', alpha=0.8, label='Data Points')

    # Plot outliers
    axis.scatter(reduced_data[outliers, 0], reduced_data[outliers, 1], reduced_data[outliers, 2],
                 c='red', edgecolors='black', alpha=1.0, label='Outliers')

    # Add legend
    axis.legend(loc='upper left', edgecolor='blue')

    plt.title(title)
    plt.savefig(filepath)


def main():
    file_path = sys.argv[1]
    # Load the dataset
    df = load_data(file_path)

    # remove unnecessary features before training anomaly detection models
    df = df.drop(['price_bucket'], axis=1)

    outliers = isolation_forest(df)
    visualize_outliers(df, outliers, 'Isolation Forest', './visualizations/outliers_isolation_forest.jpeg')

    # Elliptic Envelope assumes Gaussian distribution and computationally expensive

    outliers = one_class_svm(df)
    visualize_outliers(df, outliers, 'One Class SVM', './visualizations/outliers_1class_svm.jpeg')

    outliers = local_outlier_factor(df)
    visualize_outliers(df, outliers, 'Local Outlier Factor', './visualizations/outliers_lof.jpeg')

    df.drop(index=outliers, inplace=True)
    df.to_csv('./data/listings_outliers_removed.csv' , index=False)


if __name__ == "__main__":
    main()