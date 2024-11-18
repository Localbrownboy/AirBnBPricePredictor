import pandas as pd
import sys
from sklearn.neighbors import LocalOutlierFactor
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
    model = LocalOutlierFactor(n_neighbors=400, contamination=0.05)
    predictions = model.fit_predict(df)
    
    # model.fit_predict(df) returns either 1 (inliners) or -1 (outliers)
    df['outliers'] = predictions
    outliers = df[df['outliers'] == -1]

    print(f'Detected {len(outliers)} outliers from Local Outlier Factor.')

    outlier_index = outliers.index
    return outlier_index

# visualize the dataset by scatterplot with outliers marked
def visualize_outliers(df, outliers , title , filepath):
    """
    Visualizes the data points and outliers using PCA in a 3D scatter plot.
    
    Parameters:
    - df: DataFrame or array-like object containing the data.
    - outliers: List or array of indices for the detected outliers.
    """
    # Dimensionality reduction to 3 components
    reducer = PCA(n_components=3)
    reduced_data = reducer.fit_transform(df)

    fig = plt.figure(figsize=(10, 8))

    # Create 3D axis
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Principal Component 1')
    axis.set_ylabel('Principal Component 2')
    axis.set_zlabel('Principal Component 3')

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

    # Store the target variable price_bucket separately
    price_bucket = df['price_bucket']
    df = df.drop(columns=['price_bucket'])

    outliers = local_outlier_factor(df)

    visualize_outliers(df, outliers, 'Local Outlier Factor', './visualizations/outliers_lof.jpeg')

    # Drop outliers
    df.drop(index=outliers, inplace=True)
    price_bucket.drop(index=outliers, inplace=True)

    # Add price_bucket back to df
    df['price_bucket'] = price_bucket

    # Drop the 'outliers' column
    df.drop(columns=['outliers'], inplace=True)

    df.to_csv('./data/listings_outliers_removed.csv' , index=False)


if __name__ == "__main__":
    main()