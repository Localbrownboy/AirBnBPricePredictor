import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,  mean_squared_error, r2_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


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
    Evalute the model using various metrics.
    For classification: accuracy, precision, recall and f1-score.
    For regression: mean squared error (MSE) and r2 score.
    
    Parameters:
    - y_test: test data of target variable
    - y_pred: prediction of target variable by model
    """
    if y_test.name == 'price_bucket_equiwidth': # classification
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'f1-score: {f1}\n')

    else:                             # regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}\n')


def visualize_model_results(y_test, y_pred, model_name, y_pred_prob=None):
    """
    Visualize the model results.
    For classification: confusion matrix and ROC curve
    For regression: residuals plot
    
    Parameters:
    - y_test: true values of target variable
    - y_pred: predicted values of target variable
    - model_name: the name of the model
    - y_pred_prob: predicted probability of each class in target variable (only in categorical)
    """
    if y_test.name == 'price_bucket_equiwidth': # classification

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

        for i, class_label in enumerate(classes):
            # Compute ROC for each class
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred_prob[:, i])
            plt.plot(fpr, tpr, label=f"Class {class_label}")

        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.savefig(f'./visualizations/classification/roc_curves_{model_name.lower().replace(" ","_")}.jpeg')
        plt.close()

    else:                            # regression
        residuals = y_test - y_pred

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f'Residual Plot of {model_name}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.savefig(f'./visualizations/classification/residual_plot_{model_name.lower().replace(" ","_")}.jpeg')
        plt.close()



def main():
    file_path = sys.argv[1]
    df = load_data(file_path)

    # split data to features (X) and target variable (y)
    X = df.drop(['price_bucket_equiwidth', 'price',], axis=1)  # Remove price_bucket (classification target) and price
    y_classification = df['price_bucket_equiwidth']
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
    sk_fold = StratifiedKFold(n_splits=5)

    # initialize classification models
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=5, max_depth=20, bootstrap=False, random_state=42, ),
        "Neural Network Classifier": MLPClassifier(
            hidden_layer_sizes=(20,20,10),
            max_iter=5000,
            random_state=42,
            # early_stopping=True,       # Enable early stopping
            validation_fraction=0.2,   # Fraction of training data used for validation (to determine when to stop)
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
        evaluate_model_metrics(y_test_clf, y_pred_clf)

        # visualize confusion matrix
        visualize_model_results(y_test_clf, y_pred_clf, name, y_pred_prob)

    # -------------------------------------------------------------------------------------
    # Regression model
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    print(f'\nCross Validation Score of Random Forest Regressor is: {cross_validate_model(regressor, X_train_reg, y_train_reg, sk_fold)}')

    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)

    # evaluate the model by mean squared error and r2-score
    evaluate_model_metrics(y_test_reg, y_pred_reg)

    # Plot residuals
    visualize_model_results(y_test_reg, y_pred_reg, model_name="Random Forest Regressor")

    # -------------------------------------------------------------------------------------
    # Second Regression Model
    gbr = GradientBoostingRegressor(
        n_estimators=100,    # Number of boosting stages
        learning_rate=0.1,   # Shrinks contribution of each tree
        max_depth=3,         # Maximum depth of the trees
        random_state=42
    )

    # Evaluate using cross-validation
    print(f'\nCross Validation Score of Gradient Boosting Regressor is: {cross_validate_model(gbr, X_train_reg, y_train_reg, sk_fold)}')

    # Train and predict
    gbr.fit(X_train_reg, y_train_reg)
    y_pred_gbr = gbr.predict(X_test_reg)

    # Evaluate the model
    evaluate_model_metrics(y_test_reg, y_pred_gbr)

    # Plot residuals
    visualize_model_results(y_test_reg, y_pred_gbr, model_name="Gradient Boosting Regressor")

    # -------------------------------------------------------------------------------------
    # Third Regression Model
    svr = SVR(
        kernel='rbf',         # Use the RBF kernel for non-linear relationships
        C=1.0,                # Regularization parameter
        epsilon=0.1           # Margin of tolerance for the loss function
    )

    # Cross-validation
    print(f'\nCross Validation Score of SVR is: {cross_validate_model(svr, X_train_reg, y_train_reg, sk_fold)}')

    # Train and predict
    svr.fit(X_train_reg, y_train_reg)
    y_pred_svr = svr.predict(X_test_reg)

    # Evaluate and visualize
    evaluate_model_metrics(y_test_reg, y_pred_svr)
    visualize_model_results(y_test_reg, y_pred_svr, model_name="Support Vector Regressor")
    # -------------------------------------------------------------------------------------


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
    evaluate_model_metrics(y_test_clf, y_pred_clf)

    # visualize confusion matrix
    visualize_model_results(y_test_clf, y_pred_clf, 'Tuned Random Forest Classifier', y_pred_prob)




if __name__ == "__main__":
    main()
