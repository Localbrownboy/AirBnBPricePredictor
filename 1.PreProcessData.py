import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
import re


def encode_amenities(df, min_usage=0.05):
    """
    Encodes the 'amenities' column using MultiLabelBinarizer,
    cleans column names, and removes columns used in less than min_usage percentage of the rows.
    """
    # One-hot encode the 'amenities' column using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    amenities_encoded = mlb.fit_transform(df['amenities'])
    
    # Clean column names to remove invalid characters and replace spaces with underscores
    def clean_column_name(name):
        # Replace spaces with underscores and convert to lowercase
        clean_name = name.replace(' ', '_').lower()
        # Only numbers, letters, and underscores in name (clean out other characters)
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        return clean_name
    
    amenity_cols = ['has_' + clean_column_name(amenity) for amenity in mlb.classes_]
    amenities_df = pd.DataFrame(amenities_encoded, columns=amenity_cols, index=df.index)
    
    # Merge the new columns back into the original DataFrame
    df = pd.concat([df, amenities_df], axis=1)
    
    # Remove 'has_' columns used in less than min_usage percentage of the rows
    has_cols = amenity_cols
    col_usage = df[has_cols].mean()
    cols_to_keep = col_usage[col_usage >= min_usage].index.tolist()
    cols_to_drop = [col for col in has_cols if col not in cols_to_keep]
    df.drop(columns=cols_to_drop, inplace=True)
    
    print(f"Removed {len(cols_to_drop)} 'has_' columns.")
    
    return df



def main():
    # Load the dataset
    df = pd.read_csv('./data/listings.csv')
    # Dropping columns that are irrelevant
    irrelevant_columns = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview',
        'picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about',
        'host_thumbnail_url', 'host_picture_url', 'host_has_profile_pic', 'host_verifications', 'neighbourhood',
        'neighbourhood_group_cleansed', 'bathrooms_text', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated',
        'calendar_last_scraped', 'first_review', 'last_review', 'license'
    ]
    df.drop(columns=irrelevant_columns, inplace=True)

    df['price'] = df['price'].replace('[$,]', '', regex=True).astype(float) # the price field is text for some reason

    df['amenities'] = df['amenities'].apply(json.loads) # turn the string into a python list
    df['amenity_count'] = df['amenities'].apply(len) # add a feature for the total lenght of the amenity list
    # Convert host_response_rate and host_acceptance_rate to decimals
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float) / 100
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '').astype(float) / 100


    # Drop rows where the price field is empty
    df.dropna(subset=['price'], inplace=True)

    df.to_csv('./data/intermediate_listings_pruned_columns.csv' , index=False) # save pruned csv 


    # Impute missing values
    # Fill numerical columns with the median, categorical columns with the mode
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['price']) # Remove price from this list 
    cat_cols = df.select_dtypes(include=['object']).columns.difference(['amenities']) # and amenities from this list 

    num_imputer = SimpleImputer(strategy='mean') # imputation rule for number features 
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent') # imputation rule for categorical features
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    df.to_csv('./data/intermediate_listings_imputed.csv' , index=False) # save imputed csv 

    # Normalize numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df.to_csv('./data/intermediate_listings_normalized.csv' , index=False) # save normalized csv 

    # One Hot Encode amenities
    df = encode_amenities(df)

    label_cols = [
        'host_is_superhost', 'host_neighbourhood', 'host_identity_verified',
        'has_availability', 'instant_bookable', 'neighbourhood_cleansed',
        'property_type', 'room_type'
    ]
    
    # Apply Label Encoding
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df.to_csv('./data/intermediate_listings_encoded.csv', index=False)


    price_max = df['price'].max()
    # Create bins from 0 to max price in $10 increments
    price_bins = np.arange(0, price_max + 10, 10)
    # Create labels for the bins, e.g., '150-160'
    bin_labels = [f"{int(price_bins[i])}-{int(price_bins[i+1])}" for i in range(len(price_bins) - 1)]
    # Bin the prices and assign labels
    df['price_bucket'] = pd.cut(
        df['price'],
        bins=price_bins,
        labels=bin_labels,
        include_lowest=True,
        right=True # make the bins right inclusive ie: {150-160 | x > 150 && x <= 160} 
    )
    df.drop(columns=['amenities','price'], inplace=True) # we can remove these now that its info is saved elsewhere

    df.to_csv('./data/processed_listings.csv', index=False)


if __name__ == "__main__":
    main()