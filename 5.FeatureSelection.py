import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import sys

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

def encode_target_variable(df: pd.DataFrame, target_col: str):
    """Encodes the target variable categories into integers using LabelEncoder."""
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    return df, le

def select_features_rfe(df: pd.DataFrame, target_col: str, num_features: int = 20):
    """Select num_features features using Recursive Feature Elimination with a Decision Tree classifier"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = DecisionTreeClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X, y)

    selected_features = X.columns[rfe.support_].tolist()
    print("\nSelected Features using RFE:", selected_features)

    return selected_features

def select_features_mi(df: pd.DataFrame, target_col: str, num_features: int = 20):
    """Select num_features features using Mutual Information scores"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)

    # Select the top features based on the scores
    selected_features = mi_scores.nlargest(num_features).index.tolist()
    print("Selected Features using Mutual Information:", selected_features)

    return selected_features
def main():
    file_path = sys.argv[1]
    target_vars = ['price_bucket_equiwidth', 'price_bucket_equidepth']
    num_features = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    df = load_data(file_path)

    for target_var in target_vars:
        # Create a copy of the dataframe and drop the other target variable
        temp_df = df.copy().drop(columns=[var for var in target_vars if var != target_var])
        print(f"\nProcessing target variable: {target_var}")

        # Encode the target variable
        temp_df, label_encoding = encode_target_variable(temp_df, target_var)

        # Perform feature selection using Mutual Information
        mi_selected_features = select_features_mi(temp_df, target_var, num_features)

        # Perform feature selection using RFE
        rfe_selected_features = select_features_rfe(temp_df, target_var, num_features)

        # Restore original target variable labels
        temp_df[target_var] = label_encoding.inverse_transform(temp_df[target_var])

        # Save dataframe with selected features to CSV files
        df_mi_selected_features = temp_df[mi_selected_features + [target_var]]
        mi_output_file = f'./data/listings_mi_selected_features_{target_var}.csv'
        df_mi_selected_features.to_csv(mi_output_file, index=False)
        print(f"Saved Mutual Information selected features to {mi_output_file}")

        df_rfe_selected_features = temp_df[rfe_selected_features + [target_var]]
        rfe_output_file = f'./data/listings_rfe_selected_features_{target_var}.csv'
        df_rfe_selected_features.to_csv(rfe_output_file, index=False)
        print(f"Saved RFE selected features to {rfe_output_file}")

if __name__ == "__main__":
    main()