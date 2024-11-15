import pandas as pd
import sys
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

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
def visualize_outliers(df, outliers, model_name, output_image):
    # reduce dimensionality to 2 for visualization
    pca = PCA(2)
    df = pd.DataFrame(pca.fit_transform(df))

    plt.figure(figsize=(10,6))

    # plot normal points
    plt.scatter(df[0], df[1], c='cyan', label='data points')

    # plot outliers
    plt.scatter(df.iloc[outliers, 0], df.iloc[outliers, 1], c='red', edgecolors='black', label='outliers')

    plt.title(f'Anomaly Detection using {model_name}')
    plt.xlabel('PCA x')
    plt.ylabel('PCA y')
    plt.legend(loc="upper left", markerfirst=False, edgecolor='blue')
    plt.savefig(output_image)

def main():
    file_path = sys.argv[1]
    # Load the dataset
    df = load_data(file_path)

    # remove unnecessary features before training anomaly detection models
    df = df.drop(['price_bucket'], axis=1)

    outliers = local_outlier_factor(df)

    visualize_outliers(df, outliers, 'Local Outlier Factor', './visualizations/outliers_lof.jpeg')

if __name__ == "__main__":
    main()