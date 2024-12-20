import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


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


def evaluate_model_metrics(y_test, y_pred, y_pred_prob=None):
    """
    Evalute the model using various metrics.
    For classification: accuracy, precision, recall, f1-score, and ROC-AUC.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: prediction of target variable by model
    - y_pred_prob: predicted probability of each class in target variable
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc_score = roc_auc_score(y_test, y_pred_prob, average='weighted', multi_class='ovr')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'f1-score: {f1}')
    print(f'ROC-AUC: {auc_score}\n')



def visualize_model_results(y_test, y_pred, model_name, y_pred_prob=None):
    """
    Visualize the model results.
    For classification: confusion matrix and ROC curve
    
    Parameters:
    - y_test: true values of target variable
    - y_pred: predicted values of target variable
    - model_name: the name of the model
    - y_pred_prob: predicted probability of each class in target variable (only in categorical)
    """
    # Confusion Matrix
    class_labels = sorted(np.unique(y_test), key=lambda x: int(x.split('-')[0]))

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot()

    plt.title(f'Confusion Matrix of {model_name}')
    plt.xticks(rotation=30)
    plt.savefig(f'./visualizations/classification/confusion_matrix_{model_name.lower().replace(" ","_")}.jpeg')
    plt.close()

    # ROC curves
    classes = np.unique(y_test)  # List of unique classes
    y_true_binary = label_binarize(y_test, classes=classes)

    plt.figure()

    for i, class_label in enumerate(sorted(classes, key=lambda x: int(x.split('-')[0]))):
        # Compute ROC for each class
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {class_label}")

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.savefig(f'./visualizations/classification/roc_curves_{model_name.lower().replace(" ","_")}.jpeg')
    plt.close()



def main():
    file_path = sys.argv[1]
    df = load_data(file_path)

    # split data to features (X) and target variable (y)
    X = df.drop(['price_bucket_equiwidth', 'price',], axis=1)  # Remove price_bucket (classification target) and price
    y_classification = df['price_bucket_equiwidth']

    # split the dataset into training and testing sets
    # Split data for classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )


    # initialize stratified k-fold for cross validation
    sk_fold = StratifiedKFold(n_splits=5)

    # initialize classification models
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Neural Network Classifier": MLPClassifier(
            hidden_layer_sizes=(20,20,10),
            max_iter=5000,
            random_state=42,
            # early_stopping=True,       # Enable early stopping
            # validation_fraction=0.2,   # Fraction of training data used for validation
            n_iter_no_change=10,
            activation='logistic',
            alpha=0.001,
            solver='adam',
            learning_rate='adaptive',
                   
        ),
        "kNN": KNeighborsClassifier(n_neighbors=16)

    }

    for name, model in classifiers.items():
        # evaluate the model consistency by cross-validation score of all folds
        print(f'\nCross Validation Score of {name} is: {cross_validate_model(model, X_train_clf, y_train_clf, sk_fold)}')

        # train the model and predict for new class labels
        model.fit(X_train_clf, y_train_clf)
        y_pred_clf = model.predict(X_test_clf)
        y_pred_prob = model.predict_proba(X_test_clf)

        # evaluate the model by accuracy, precision, recall, and f1-score 
        evaluate_model_metrics(y_test_clf, y_pred_clf, y_pred_prob)

        # visualize confusion matrix
        visualize_model_results(y_test_clf, y_pred_clf, name, y_pred_prob)

    # -------------------------------------------------------------------------------------

    print(f"Performing hyperparameter tuning in Random Forest Classifier:")

    # tune hyperparameters for random forest classifier by grid search
    forest_model = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [50, 75, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10, 20, 30],
        "bootstrap": [True, False]
    }
    rf = GridSearchCV(estimator=forest_model, param_grid=param_dist, cv=sk_fold, n_jobs=-1)
    rf.fit(X_train_clf, y_train_clf)

    # retrieve the best parameter for the model and use that to predict new labels
    print(f"Best Parameters for Tuned Random Forest Classifier: {rf.best_params_}")
    best_rf = rf.best_estimator_
    y_pred_clf = best_rf.predict(X_test_clf)
    y_pred_prob = best_rf.predict_proba(X_test_clf)

    # evaluate the model by accuracy, precision, recall, and f1-score 
    evaluate_model_metrics(y_test_clf, y_pred_clf, y_pred_prob)

    # visualize confusion matrix
    visualize_model_results(y_test_clf, y_pred_clf, 'Tuned Random Forest Classifier', y_pred_prob)




if __name__ == "__main__":
    main()
