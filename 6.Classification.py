import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,  mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df


def cross_validate_model(model, X_train, y_train, cv):
    """
    Evaluate the model consistency by performing cross-validation.
    
    Parameters:
    - model: the classification model we evaluate
    - X_train: the training data of features
    - y_train: the training data of target variable
    - cv: the cross-validation splitting strategy
    
    Return:
    - mean_score: the average of all folds of cross validation score
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    mean_score = np.mean(scores)

    return mean_score


def evaluate_model_metrics(y_test, y_pred):
    """
    Evalute the model using various metrics, including accuracy, precision, recall and f1-score.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: prediction of target variable by model
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'f1-score: {f1}\n')

def evaluate_regression_metrics(y_test, y_pred):
    """
    Evaluate the regression model using various metrics, including MSE and R-squared.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: prediction of target variable by model
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}\n')



def visualize_confusion_matrix(y_test, y_pred, model_name, outout_image):
    """
    Compute confusion matrix of a classification model and visualize it.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: prediction of target variable by model
    - model_name: the name of the classification model
    - output_image: the output filepath of the confusion matrix image
    """
    # all class labels in order (easier to compare in confusion matrix)    
    class_labels = ['20-105', '105-138', '138-172', '172-211', '211-260', '260-350', '350-999']

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot()

    plt.title(f'Confusion Matrix of {model_name}')
    plt.xticks(rotation=30)
    plt.savefig(outout_image)
    plt.close()

def plot_residuals(y_test, y_pred, model_name, output_image):
    """
    Plot residuals of the regression model.
    
    Parameters:
    - y_test: True values of the target variable
    - y_pred: Predicted values of the target variable
    - model_name: Name of the regression model
    - output_image: File path to save the residual plot
    """
    residuals = y_test - y_pred
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Residual Plot: {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(output_image)
    plt.close()



def main():
    file_path = sys.argv[1]
    df = load_data(file_path)

    # split data to features (X) and target variable (y)
    X = df.drop(['price_bucket', 'price'], axis=1)  # Remove price_bucket (classification target) and price
    y_classification = df['price_bucket']

    # Prepare data for regression
    y_regression = df['price']

    # split the dataset into training and testing sets
    # Split data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )

    # Split data for regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    # initialize stratified k-fold for cross validation
    sk_fold = StratifiedKFold(n_splits=10)

    # initialize three classification models
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(50,10,6),
            max_iter=300,
            random_state=42,
            early_stopping=True,        # Enable early stopping
            validation_fraction=0.2,   # Fraction of training data used for validation
            n_iter_no_change=10,         
        )
        # "SVM": SVC(random_state=42, ),
        # "kNN": KNeighborsClassifier(n_neighbors=16)
    }

    for name, model in classifiers.items():
        # evaluate the model consistency by cross-validation score of all folds
        print(f'\nCross Validation Score of {name} is: {cross_validate_model(model, X_train_clf, y_train_clf, sk_fold)}')

        # train the model and predict for new class labels
        model.fit(X_train_clf, y_train_clf)
        y_pred_clf = model.predict(X_test_clf)

        # evaluate the model by accuracy, precision, recall, and f1-score 
        evaluate_model_metrics(y_test_clf, y_pred_clf)

        # visualize confusion matrix
        visualize_confusion_matrix(y_test_clf, y_pred_clf, name, f'./visualizations/confusion_matrix_{name}.jpeg')

    # tune hyperparameters for random forest classifier by grid search
    forest_model = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [50, 75, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10, 20, 30, 50],
        "bootstrap": [True, False]
    }
    # rf = GridSearchCV(estimator=forest_model, param_grid=param_dist, cv=sk_fold, n_jobs=-1)
    # rf.fit(X_train_clf, y_train_clf)

    # # retrieve the best parameter for the model and use that to predict new labels
    # print(f"Best Parameters for Random Forest: {rf.best_params_}")
    # best_rf = rf.best_estimator_
    # y_pred_clf = best_rf.predict(X_test_clf)

    # # evaluate the model by accuracy, precision, recall, and f1-score 
    # evaluate_model_metrics(y_test_clf, y_pred_clf)

    # # visualize confusion matrix
    # visualize_confusion_matrix(y_test_clf, y_pred_clf, 'Random Forest', './visualizations/confusion_matrix_tuned_random_forest.jpeg')

    # Regression model
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)

    evaluate_regression_metrics(y_test_reg, y_pred_reg)

    # Plot residuals
    plot_residuals(
        y_test=y_test_reg, 
        y_pred=y_pred_reg, 
        model_name="Random Forest Regressor", 
        output_image="./visualizations/residual_plot_random_forest.jpeg"
    )




if __name__ == "__main__":
    main()