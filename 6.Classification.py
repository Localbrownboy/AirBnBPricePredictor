import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
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


def main():
    file_path = sys.argv[1]
    df = load_data(file_path)

    # split data to features (X) and target variable (y)
    X = df.drop('price_bucket', axis=1)
    y = df['price_bucket']

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize stratified k-fold for cross validation
    sk_fold = StratifiedKFold(n_splits=10)

    # initialize three classification models
    classifiers = {
        # "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # "SVM": SVC(random_state=42, ),
        "kNN": KNeighborsClassifier(n_neighbors=16)
    }

    for name, model in classifiers.items():
        # evaluate the model consistency by cross-validation score of all folds
        print(f'\nCross Validation Score of {name} is: {cross_validate_model(model, X_train, y_train, sk_fold)}')

        # train the model and predict for new class labels
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # evaluate the model by accuracy, precision, recall, and f1-score 
        evaluate_model_metrics(y_test, y_pred)

        # visualize confusion matrix
        visualize_confusion_matrix(y_test, y_pred, name, f'./visualizations/confusion_matrix_{name}.jpeg')

    # tune hyperparameters for random forest classifier by grid search
    forest_model = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10, 20, 30, 50],
        "bootstrap": [True, False]
    }
    rf = GridSearchCV(estimator=forest_model, param_grid=param_dist, cv=sk_fold, n_jobs=-1)
    rf.fit(X_train, y_train)

    # retrieve the best parameter for the model and use that to predict new labels
    print(f"Best Parameters for Random Forest: {rf.best_params_}")
    best_rf = rf.best_estimator_
    y_pred = best_rf.predict(X_test)

    # evaluate the model by accuracy, precision, recall, and f1-score 
    evaluate_model_metrics(y_test, y_pred)

    # visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred, 'Random Forest', './visualizations/confusion_matrix_random_forest.jpeg')



if __name__ == "__main__":
    main()