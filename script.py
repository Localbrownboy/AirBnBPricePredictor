import os

print(f"\nYou are in Data Preprocessing:\n")
os.system('python3 ./1.PreProcessData.py ./data/listings.csv')

print(f"\nYou are in EDA:\n")
os.system('python3 ./2.EDA.py ./data/intermediate_listings_encoded.csv ./data/intermediate_listings_encoded_and_scaled.csv ./data/processed_listings.csv')

print(f"\nYou are in Clustering:\n")
os.system('python3 ./3.Clustering.py ./data/processed_listings.csv')

print(f"\nYou are in Outlier Detection:\n")
os.system('python3 ./4.OutlierDetection.py ./data/processed_listings.csv')

print(f"\nYou are in Feature Selection:\n")
os.system('python3 ./5.FeatureSelection.py ./data/listings_outliers_removed.csv')

print(f"\nYou are in Classification:\n")
os.system('python3 ./6.Classification.py ./data/listings_mi_selected_features_price_bucket_equiwidth.csv')