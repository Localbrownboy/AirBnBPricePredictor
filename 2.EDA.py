import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import sys
import numpy as np
from sklearn.decomposition import PCA

# Load data function
def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

# Function to plot price distribution
def plot_price_distribution(df, output_html, output_image, x_axis_limit=500):
    fig = px.histogram(df, x='price', title="Distribution of Price")
    fig.update_xaxes(range=[0, x_axis_limit])
    fig.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency")
    fig.write_html(output_html)
    fig.write_image(output_image, format='jpeg')

# Function to plot listing count per host
def plot_listing_count_per_host(df, output_image):
    host_counts = df['host_id'].value_counts()
    fig = px.histogram(host_counts, title="Listing Count per Host")
    fig.update_xaxes(range=[0, 10])

    fig.update_layout(xaxis_title="Number of Listings", yaxis_title="Frequency", showlegend=False)
    fig.write_image(output_image, format='jpeg')

# Function to plot property type distribution
def plot_property_type_distribution(df, output_image):
    fig = px.histogram(df, x='property_type', title="Distribution of Property Types")
    fig.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Frequency",
        xaxis=dict(tickmode="linear")  # Ensures every x-tick is displayed
    )    
    fig.write_image(output_image, format='jpeg')

# Function to plot room type distribution
def plot_room_type_distribution(df, output_image):
    fig = px.histogram(df, x='room_type', title="Distribution of Room Types")
    fig.update_layout(xaxis_title="Room Type", 
                      yaxis_title="Frequency", 
                      xaxis=dict(tickmode="linear") )
    fig.write_image(output_image, format='jpeg')

# Function to plot listings on map based on neighborhood
def plot_map(df, output_html):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in df.iterrows():
        folium.Marker([row['latitude'], row['longitude']]).add_to(marker_cluster)
    m.save(output_html)

# Function to plot distribution for accommodates, beds, bedrooms, bathrooms
def plot_distribution(df, column, output_image):
    fig = px.histogram(df, x=column, title=f"Distribution of {column.capitalize()}")
    fig.update_layout(xaxis_title=column.capitalize(), yaxis_title="Frequency")
    fig.write_image(output_image, format='jpeg')

# Function to plot review scores rating distribution
def plot_review_scores_rating(df, output_image):
    fig = px.histogram(df, x='review_scores_rating', title="Distribution of Review Scores Rating")
    fig.update_layout(xaxis_title="Review Scores Rating", yaxis_title="Frequency")
    fig.write_image(output_image, format='jpeg')

# Function to plot top 5 amenities
def plot_top_amenities(df, output_image, count=5):
    amenity_columns = df.filter(like='has_').drop('has_availability', axis=1)
    amenity_counts = amenity_columns.sum().sort_values(ascending=False).head(count)
    fig = px.bar(amenity_counts, x=amenity_counts.index, y=amenity_counts.values, title=f"Top {count} Most Frequent Amenities")
    fig.update_layout(xaxis_title="Amenity", yaxis_title="Count")
    fig.write_image(output_image, format='jpeg')

# Function to plot correlation heatmap
# Function to plot correlation heatmap for specific attributes
def plot_correlation_heatmap(df, attributes, output_image):
    """
    Plots a correlation heatmap for the specified attributes.
    
    Parameters:
    - df: The DataFrame containing the data.
    - attributes: A list of column names for which to display correlations.
    - output_image: Path to save the heatmap image.
    """
    # Filter DataFrame to only include specified attributes
    df_selected = df[attributes]
    
    # Calculate correlation matrix for the selected attributes
    correlation_matrix = df_selected.corr()
    
    # Plot the correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()


# Functions to plot relationships with price
def plot_relationship_with_price(df, x_column, output_image):
    fig = px.scatter(df, x=x_column, y='price', title=f"Price vs {x_column.capitalize()}")
    fig.update_layout(xaxis_title=x_column.capitalize(), yaxis_title="Price ($)")
    fig.write_image(output_image, format='jpeg')


def plot_price_bucket_distribution(df, output_image, bucket_column):
    """
    Plots the distribution of the price buckets, sorted numerically by the bucket ranges.
    """
    # Extract the lower bound of the price range for sorting
    df['price_bucket_order'] = df[bucket_column].str.extract(r'(\d+)', expand=False).astype(int)

    # Sort the DataFrame by the lower bound
    df = df.sort_values(by='price_bucket_order')

    # Plot the histogram
    fig = px.histogram(df, x=bucket_column, title="Distribution of Price Buckets")
    fig.update_layout(
        xaxis_title="Price Bucket",
        yaxis_title="Frequency",
        xaxis=dict(categoryorder="array", categoryarray=df[bucket_column].unique())
    )
    fig.write_image(output_image, format='jpeg')

    # Drop the temporary column to keep the original DataFrame intact
    df.drop(columns=['price_bucket_order'], inplace=True, errors='ignore')


def plot_highest_correlations_with_price(df, target_column, top_n, output_image):
    """
    Finds and plots the top N features with the highest correlation to a target column.

    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: The column to compute correlations against (e.g., 'price').
    - top_n: Number of top correlations to include in the heatmap.
    - output_image: Path to save the heatmap image.
    """
    # Calculate correlations of all numeric columns with the target column
    correlations = df.corr()[target_column].drop(target_column)
    
    # Sort correlations by absolute value
    top_correlations = correlations.abs().sort_values(ascending=False).head(top_n)
    
    # Get the top correlated features
    top_features = top_correlations.index.tolist()
    
    # Create a smaller correlation matrix
    filtered_corr = df.corr().loc[top_features, [target_column]]
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f" , annot_kws={"fontsize": 10})
    plt.title(f'Top {top_n} Features Correlated with {target_column.capitalize()}')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()


def visualize_pca_with_buckets(df, bucket_labels, title, output_image):
    """
    Visualizes the PCA scatterplot with 3D points labeled by sorted bucket labels.

    Parameters:
    - df: DataFrame containing the dataset.
    - bucket_labels: Labels (e.g., equiwidth or equidepth buckets) for coloring the points.
    - title: Title for the plot.
    - output_image: Path to save the 3D PCA scatterplot.
    """
    # Drop price-related columns before PCA
    features = df.drop(columns=['price', 'price_bucket_equiwidth', 'price_bucket_equidepth'], errors='ignore')

    # Dimensionality reduction to 3 components
    reducer = PCA(n_components=3)
    reduced_data = reducer.fit_transform(features.select_dtypes(include=[np.number]))

    # Sort bucket labels
    unique_labels = sorted(np.unique(bucket_labels), key=lambda x: int(x.split('-')[0]))
    label_to_number = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_number[label] for label in bucket_labels]

    fig = plt.figure(figsize=(10, 8))

    # Create a 3D axis
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Principal Component 1')
    axis.set_ylabel('Principal Component 2')
    axis.set_zlabel('Principal Component 3')

    # Plot data points in 3D space
    scatter = axis.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=numeric_labels,
        cmap='viridis',
        alpha=0.8
    )

    # Use colorbar for bucket labels
    colorbar = plt.colorbar(scatter)
    colorbar.set_ticks(range(len(unique_labels)))  # Set ticks to match numeric encoding
    colorbar.set_ticklabels(unique_labels)  # Map ticks back to sorted original labels
    colorbar.set_label('Bucket Label')

    plt.title(title)
    plt.savefig(output_image)
    plt.close()

# Main function to execute all EDA steps
def main():
    file_path = sys.argv[1]
    file_path_scaled = sys.argv[2]
    file_path_processed = sys.argv[3]
    # Load the dataset
    df = load_data(file_path)
    df_scaled = load_data(file_path_scaled).drop( ['price_bucket_equidepth' , 'price_bucket_equiwidth'] , axis=1) # drop price bucket column 
    df_processed = load_data(file_path_processed)
    # Plot price distribution
    plot_price_distribution(df, './visualizations/eda/price_histogram.html', './visualizations/eda/price_histogram.jpeg', x_axis_limit=2000)
    
    # Plot listing count per host
    plot_listing_count_per_host(df, './visualizations/eda/listing_count_per_host.jpeg')
    
    # Plot property type distribution
    plot_property_type_distribution(df, './visualizations/eda/property_type_distribution.jpeg')
    
    # Plot room type distribution
    plot_room_type_distribution(df, './visualizations/eda/room_type_distribution.jpeg')
    
    # Plot map of listings
    plot_map(df, './visualizations/eda/map.html')
    
    # Plot distribution for accommodates, beds, bedrooms, bathrooms
    for feature in ['accommodates', 'beds', 'bedrooms', 'bathrooms', 'host_acceptance_rate']:
        plot_distribution(df, feature, f'./visualizations/eda/{feature}_distribution.jpeg')
    
    # Plot review scores rating distribution
    plot_review_scores_rating(df, './visualizations/eda/review_scores_rating_distribution.jpeg')
    
    # Plot top 5 amenities
    plot_top_amenities(df, './visualizations/eda/top_amenities.jpeg')
        
    selected_attributes = ['price', 'accommodates', 'bedrooms', 'bathrooms', 'review_scores_rating', 'amenity_count']

    # Generate the heatmap
    plot_correlation_heatmap(df, selected_attributes, './visualizations/eda/selected_correlation_heatmap.jpeg')
    plot_highest_correlations_with_price(
        df=df_scaled,
        target_column='price',
        top_n=20,
        output_image='./visualizations/eda/top_correlations_with_price.jpeg'
    )

    plot_price_bucket_distribution(df, './visualizations/eda/price_bucket_equiwidth_distribution.jpeg', bucket_column='price_bucket_equiwidth')
    plot_price_bucket_distribution(df, './visualizations/eda/price_bucket_equidepth_distribution.jpeg', bucket_column='price_bucket_equidepth')

    # PCA scatterplot with equiwidth bucket labels
    visualize_pca_with_buckets(
        df=df_processed,
        bucket_labels=df['price_bucket_equiwidth'],
        title='PCA Scatterplot with Equiwidth Buckets',
        output_image='./visualizations/eda/pca_equiwidth_buckets.jpeg'
    )

    # PCA scatterplot with equidepth bucket labels
    visualize_pca_with_buckets(
        df=df_processed,
        bucket_labels=df['price_bucket_equidepth'],
        title='PCA Scatterplot with Equidepth Buckets',
        output_image='./visualizations/eda/pca_equidepth_buckets.jpeg'
    )

    # Plot relationships between price and other features
    for feature in ['accommodates', 'bedrooms', 'neighbourhood_cleansed', 'amenity_count', ]:
        plot_relationship_with_price(df, feature, f'./visualizations/eda/price_vs_{feature}.jpeg')

if __name__ == "__main__":
    main()
