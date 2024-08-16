import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
#--------------------------Regression-----------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn import neighbors
from lightgbm import LGBMRegressor
import pickle
#-------------------------classification-------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


#----------------------------- Regression ------------------------------

class Models:
    @staticmethod
    def all_models(data):
        x = data.loc[:, data.columns != 'res']
        y = data.loc[:, 'res']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        sc = StandardScaler()
        sc.fit(X_train)
        xds_train = sc.transform(X_train)
        xds_test = sc.transform(X_test)

        models = {
            'DecisionTree': DecisionTreeRegressor(),
            'XGBoost': XGBRegressor(base_score=0.5, objective="reg:linear", random_state=0),
            'ExtraTrees': ExtraTreesRegressor(n_jobs=16),
            'RandomForest': RandomForestRegressor(),
            'LGBM': LGBMRegressor(n_jobs=16),
            'KNN': neighbors.KNeighborsRegressor(n_neighbors=23, p=1)
        }

        results = {}
        plots = {}

        for name, model in models.items():
            model.fit(xds_train, y_train)
            y_pred = model.predict(xds_test)
            y_p_train = model.predict(xds_train)

            r2_train = r2_score(y_train, y_p_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_p_train))
            r2_test = r2_score(y_test, y_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                'r2_train': r2_train,
                'rmse_train': rmse_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test
            }

            # Create plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='blue', label='Predicted vs. Actual')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Actual vs. Predicted Plot - {name}')
            ax.legend()

            # Save plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots[name] = buf

            plt.close(fig)

        return results, plots

    @staticmethod
    def cross_validation(model_class, data, n_fold=5, random_state=0):
        X = np.asarray(data.loc[:, data.columns != 'res'])
        y = np.asarray(data.loc[:, 'res'])
        avg_r2 = 0
        avg_rmse = 0
        counter = 1

        r2_scores = []
        rmse_scores = []
        plot_folds = []

        cv = KFold(n_splits=n_fold, random_state=random_state, shuffle=True)

        if model_class == DecisionTreeRegressor:
            model = model_class()
        else:
            model = model_class(n_jobs=16)

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            sc = StandardScaler()
            sc.fit(X_train)
            xds_train = sc.transform(X_train)
            xds_test = sc.transform(X_test)

            model.fit(xds_train, y_train)
            pred = model.predict(xds_test)

            r2 = r2_score(y_test, pred)
            r2_scores.append(r2)
            avg_r2 += r2

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            rmse_scores.append(rmse)
            avg_rmse += rmse

            # Create fold plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, pred, color='blue', label='Predicted vs. Actual')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Fold {counter} - Actual vs. Predicted Plot')
            ax.legend()

            # Save plot to a BytesIO object
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_folds.append(buf)

            plt.close(fig)
            counter += 1

        avg_r2 /= n_fold
        avg_rmse /= n_fold

        return avg_r2, avg_rmse, r2_scores, rmse_scores, plot_folds

    @staticmethod
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)


#----------------------------- Classification ------------------------------
class Models_clf:
    @staticmethod
    def all_models(data):
        """
        Build and evaluate multiple classification models.

        Args:
        data (pd.DataFrame): The dataset to use for model building.

        Returns:
        dict: Evaluation results of the models.
        dict: Plots of model evaluations.
        """
        X = data.drop(columns='target')
        y = data['target']
        
        models = {
            'DecisionTree': DecisionTreeClassifier(),
            'XGBoost': XGBClassifier(),
            'ExtraTrees': ExtraTreesClassifier(),
            'RandomForest': RandomForestClassifier(),
            'LGBM': LGBMClassifier(),
            'KNN': KNeighborsClassifier()
        }

        results = {}
        plots = {}

        for model_name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            results[model_name] = {
                'accuracy_train': accuracy_score(y, y_pred),
                'conf_matrix_train': confusion_matrix(y, y_pred),
                'class_report_train': classification_report(y, y_pred, output_dict=True)
            }

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(results[model_name]['conf_matrix_train'], annot=True, cmap="Blues", fmt='d')
            plt.title(f'Confusion Matrix for {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f"{model_name}_confusion_matrix.png")
            plots[model_name] = f"{model_name}_confusion_matrix.png"
        
        return results, plots

    @staticmethod
    def cross_validation(model_class, data, n_folds):
        """
        Perform Stratified K-Fold cross-validation for a given classification model.

        Args:
        model_class (class): The classification model class.
        data (pd.DataFrame): The dataset to use for cross-validation.
        n_folds (int): Number of folds for cross-validation.

        Returns:
        tuple: Average accuracy, fold-wise accuracies, and plots for each fold.
        """
        X = data.drop(columns='target')
        y = data['target']
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies = []
        fold_plots = []
        
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # Plot confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
            plt.title(f'Confusion Matrix for Fold {fold + 1}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f"fold_{fold + 1}_confusion_matrix.png")
            fold_plots.append(f"fold_{fold + 1}_confusion_matrix.png")
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        return avg_accuracy, accuracies, fold_plots