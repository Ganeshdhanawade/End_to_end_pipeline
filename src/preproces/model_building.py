import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
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