import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
import re
import sys


def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocesses the DataFrame by dropping irrelevant columns and cleaning data."""
    # Dropping irrelevant columns
    irrelevant_columns = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview',
        'picture_url', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
        'host_thumbnail_url', 'host_picture_url', 'host_has_profile_pic', 'host_verifications', 'neighbourhood',
        'neighbourhood_group_cleansed', 'bathrooms_text', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated',
        'calendar_last_scraped', 'first_review', 'last_review', 'license'
    ]
    df.drop(columns=irrelevant_columns, inplace=True)



    # Convert 'price' to float
    df['price'] = df['price'].replace('[$,]', '', regex=True).astype(float)

    # Process 'amenities' column
    df['amenities'] = df['amenities'].apply(json.loads)
    df['amenity_count'] = df['amenities'].apply(len) # Add new feature for length of array 

    # Convert rates to decimals
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '').astype(float) / 100

    # Drop rows where 'price' is empty
    df.dropna(subset=['price'], inplace=True)
    df = df[df['price'] < 1000]

    # Save pruned data to CSV
    df.to_csv('./data/intermediate_listings_pruned_columns.csv', index=False)

    return df

def impute_missing_values(df):
    """Imputes missing values for numerical and categorical columns."""
    # Save a copy of the original DataFrame for comparison
    original_df = df.copy()

    # Identify numerical and categorical columns to impute
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['price'])
    cat_cols = df.select_dtypes(include=['object']).columns.difference(['amenities'])

    # Impute numerical columns
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Count the number of imputed values per attribute
    num_imputed_counts = (original_df[num_cols].isnull() & ~df[num_cols].isnull()).sum()
    cat_imputed_counts = (original_df[cat_cols].isnull() & ~df[cat_cols].isnull()).sum()

    # Combine numerical and categorical counts into a single DataFrame
    imputed_counts = pd.concat([num_imputed_counts, cat_imputed_counts], axis=0)
    imputed_counts = imputed_counts.rename("Imputed Count")

    # Save imputed data to CSV
    df.to_csv('./data/intermediate_listings_imputed.csv', index=False)

    print("Number of imputed values per attribute:")
    print(imputed_counts)

    return df, imputed_counts

def normalize_numerical_features(df):
    """Normalizes numerical features using StandardScaler."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['price','host_id'])
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save normalized data to CSV
    df.to_csv('./data/intermediate_listings_normalized.csv', index=False)

    return df

def encode_amenities(df, min_usage=0.05):
    """
    Encodes the 'amenities' column using MultiLabelBinarizer,
    cleans column names, and removes columns used in less than min_usage percentage of the rows.
    """
    # One-hot encode the 'amenities' column
    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(df['amenities'])

    # Clean column names
    def clean_column_name(name):
        clean_name = name.replace(' ', '_').lower()
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        return clean_name

    amenity_cols = ['has_' + clean_column_name(amenity) for amenity in mlb.classes_]
    amenities_df = pd.DataFrame(amenities_encoded, columns=amenity_cols, index=df.index)

    # Merge new columns
    df = pd.concat([df, amenities_df], axis=1)

    # Remove infrequent 'has_' columns
    has_cols = amenity_cols
    col_usage = df[has_cols].mean()
    cols_to_keep = col_usage[col_usage >= min_usage].index.tolist()
    cols_to_drop = [col for col in has_cols if col not in cols_to_keep]
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"Removed {len(cols_to_drop)} 'has_' columns.")

    return df

def label_encode_features(df, label_cols):
    """Applies Label Encoding to specified ordinal features."""
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def one_hot_encode_features(df, label_cols):
    """Applies One Hot Encoding to specified nominal features."""
    return pd.get_dummies(df, columns=label_cols, dtype=int)

def bin_price(df):
    """Bins the 'price' column into buckets."""
    price_max = df['price'].max()
    price_bins = np.arange(0, price_max + 10, 10)
    bin_labels = [f"{int(price_bins[i])}-{int(price_bins[i+1])}" for i in range(len(price_bins) - 1)]

    df['price_bucket'] = pd.cut(
        df['price'],
        bins=price_bins,
        labels=bin_labels,
        include_lowest=True,
        right=True
    )
    return df


def save_data(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def main():
    file_path = sys.argv[1]
    # Load the dataset
    df = load_data(file_path)

    # Preprocess data
    df = preprocess_data(df)

    # Impute missing values
    (df, impute_count) = impute_missing_values(df)

    # One Hot Encode amenities
    df = encode_amenities(df)
    columns_to_drop = ['amenities']
    df.drop(columns=columns_to_drop , inplace=True)

    save_data(df, './data/intermediate_listings_encoded.csv') # used to visualize data in EDA.py

    # convert amenities dtypes to object (prevent normalization)
    amenities_columns = df.columns[df.columns.str.startswith('has_') & (df.columns != 'has_availability')]
    df[amenities_columns] = df[amenities_columns].astype('object')

    # Normalize numerical features
    df = normalize_numerical_features(df)

    # Label Encode host response time (ordinal)
    df = label_encode_features(df, ['host_response_time'])

    # One Hot Encode nominal features
    label_cols = ['host_is_superhost', 'host_neighbourhood', 'host_identity_verified', 
                  'neighbourhood_cleansed', 'property_type', 'room_type', 'has_availability', 'instant_bookable']
    df = one_hot_encode_features(df, label_cols)

    # Save encoded data to CSV
    save_data(df, './data/intermediate_listings_encoded_and_scaled.csv')

    # Bin price into buckets
    df = bin_price(df)

    # Drop unnecessary columns
    columns_to_drop = ['price' , 'host_id']
    df.drop(columns=columns_to_drop , inplace=True)

    # Save the final processed data
    save_data(df, './data/processed_listings.csv')

if __name__ == "__main__":
    main()
