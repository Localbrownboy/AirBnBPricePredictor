import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df


def cross_validate_model(model, X_train, y_train, cv):
    """
    Evaluate the model consistency by performing cross-validation.
    
    Parameters:
    - model: the regression model to evaluate
    - X_train: training data of features
    - y_train: training data of target variable
    - cv: cross-validation splitting strategy
    
    Returns:
    - mean_score: average cross-validation score
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    mean_score = np.mean(scores)
    return mean_score


def evaluate_model_metrics(y_test, y_pred):
    """
    Evaluate the regression model using MSE and R2 metrics.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: predicted values of target variable
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}\n')


def visualize_residuals(y_test, y_pred, model_name):
    """
    Visualize residuals for regression models.
    
    Parameters:
    - y_test: true values of target variable
    - y_pred: predicted values of target variable
    - model_name: name of the regression model
    """
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Residual Plot of {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f'./visualizations/regression/residual_plot_{model_name.lower().replace(" ", "_")}.jpeg')
    plt.close()


def main():
    file_path = sys.argv[1]
    df = load_data(file_path)

    # Split data into features (X) and target variable (y)
    X = df.drop(['price', 'price_bucket_equiwidth' , 'price_bucket_equidepth'], axis=1)  # Remove price variables
    y = df['price']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=55
    )

    # Initialize regression models
    regressors = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=55),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=55
        ),
        "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }

    for name, model in regressors.items():
        # Evaluate the model using cross-validation
        print(f'\nCross-Validation Score of {name} is: {cross_validate_model(model, X_train, y_train, cv=5)}')

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model metrics
        evaluate_model_metrics(y_test, y_pred)

        # Visualize residuals
        visualize_residuals(y_test, y_pred, model_name=name)


if __name__ == "__main__":
    main()
