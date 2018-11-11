#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #6: Linear Regressions
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
plt.rcParams['figure.figsize'] = (15, 5)
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 50)
FEATURES = ['Average Income',
            'Housing Average Age',
            'Average Rooms',
            'Average Bedrooms',
            'Population',
            'Average Occupation',
            'Latitude',
            'Longitude']


def main():
    frame, full_frame = build_frames()
    training_frame, testing_frame = partition_frame(full_frame)

    # > PROBLEM 1 < #
    print('> Displaying Univariate Boxplots')
    univariate_boxplots(frame)
    print('> Displaying Univariate Histograms')
    univariate_histograms(frame)

    # > PROBLEM 2 < #
    print('> Displaying Scatterplots')
    scatter_board(full_frame)

    # > PROBLEM 3 < #
    print('> Applying Regressions')
    apply_regression(training_frame, testing_frame, 'OLS')
    apply_regression(training_frame, testing_frame, 'RIDGE')
    apply_regression(training_frame, testing_frame, 'LASSO')
    for i in range(len(FEATURES)):
        scale_feature(training_frame, FEATURES[i])
        scale_feature(testing_frame, FEATURES[i])
    print('> Successfully applied StandardScaler to dataset')
    ols_model = apply_regression(training_frame, testing_frame, 'OLS')
    rid_model = apply_regression(training_frame, testing_frame, 'RIDGE')
    apply_regression(training_frame, testing_frame, 'LASSO')

    # > PROBLEM 4 < #
    print('> Preparing to apply GridSearchCV to tune parameters')
    param_grid = {'OLS': {'normalize': [True, False]},
                  'RIDGE': {'alpha': [0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0, 10.0, 100.0],
                            'normalize': [True, False],
                            'tol': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9],
                            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                            },
                  'LASSO': {'alpha': [0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 5.0],
                            'normalize': [True, False],
                            'tol': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9],
                            'positive': [True, False],
                            'selection': ['cyclic', 'random']
                            },
                  }
    apply_grid_search(LinearRegression(), param_grid, 'OLS', training_frame, testing_frame)
    apply_grid_search(Ridge(), param_grid, 'RIDGE', training_frame, testing_frame)
    apply_grid_search(Lasso(), param_grid, 'LASSO', training_frame, testing_frame)
    las_model = apply_regression(training_frame, testing_frame, 'LASSO', silent=True, default_alpha=0.1)  # Optimize

    # > PROBLEM 5 < #
    print('> Displaying resulting coefficients...')
    coefficients = {'OLS': ols_model.coef_, 'RIDGE': rid_model.coef_, 'LASSO': las_model.coef_}
    plot_coefficients(coefficients)
    plot_coefficient_magnitudes(coefficients)


def build_frames():
    frame = pd.DataFrame(fetch_california_housing().data)
    frame.columns = FEATURES
    full_frame = pd.DataFrame(fetch_california_housing().data)
    full_frame.columns = FEATURES
    full_frame['Value'] = fetch_california_housing().target
    return frame, full_frame


def partition_frame(frame):
    training_flags = []
    for i in range(len(frame)):
        training_flags.append(random.choice([True, True, True, True, True, True, False]))
    frame['TRAINING_DATA_FLAG'] = training_flags
    training_frame = frame[frame['TRAINING_DATA_FLAG'] == True].drop('TRAINING_DATA_FLAG', 1)
    testing_frame = frame[frame['TRAINING_DATA_FLAG'] == False].drop('TRAINING_DATA_FLAG', 1)
    return training_frame, testing_frame


def scale_feature(frame, feature):
    x = frame[[feature, 'Value']]
    scaler = StandardScaler().fit(x)
    frame[feature] = scaler.transform(x)


def univariate_boxplots(frame):
    num_features = len(frame.columns)
    width = int(num_features ** 0.5) + 1
    fig = plt.figure()
    cols = list(frame.columns)

    for feature, count in zip(frame.columns, range(1, num_features + 1)):
        ax = fig.add_subplot(width, width, count)
        ax.boxplot(frame[cols[count - 1]])
        ax.set_title(cols[count - 1])
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_xticks([])
    plt.tight_layout()
    plt.show()


def univariate_histograms(frame):
    num_features = len(frame.columns)
    width = int(num_features ** 0.5) + 1
    fig = plt.figure()
    cols = list(frame.columns)

    for feature, count in zip(frame.columns, range(1, num_features + 1)):
        ax = fig.add_subplot(width, width, count)
        ax.hist(frame[cols[count - 1]], bins=40)
        ax.set_title(cols[count - 1])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def scatter_board(frame):
    num_features = len(frame.columns)
    width = int(num_features ** 0.5)
    fig = plt.figure()
    cols = list(frame.columns)

    for feature, count in zip(frame.columns, range(1, num_features)):
        ax = fig.add_subplot(width, width, count)
        ax.scatter(frame[cols[count - 1]], frame['Value'], c='k', s=1)
        ax.set_title(cols[count - 1])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def apply_regression(training_frame, testing_frame, model_type, silent=False, default_alpha=1.0):
    if model_type.upper() == 'OLS':
        reg = LinearRegression()
    elif model_type.upper() == 'RIDGE':
        reg = Ridge(alpha=default_alpha)
    elif model_type.upper() == 'LASSO':
        reg = Lasso(alpha=default_alpha)
    else:
        print("Invalid Training Model")
        return None
    model = reg.fit(training_frame[FEATURES], training_frame['Value'])
    score = model.score(testing_frame[FEATURES], testing_frame['Value'])
    if not silent:
        print('%s Regression Complete: Score of %f' % (model_type, score))
    return model


def apply_grid_search(model, param_grid, model_name, training_frame, testing_frame):
    grid_search = GridSearchCV(model, param_grid[model_name], cv=7)
    grid_search.fit(training_frame[FEATURES], training_frame['Value'])
    print('For the %s Model, optimal parameters are:' % model_name, grid_search.best_params_)
    print('Resulting score:', grid_search.score(testing_frame[FEATURES], testing_frame['Value']))


def plot_coefficients(coef):
    labels = list(coef.keys())
    s1 = np.arange(len(coef[labels[0]]))
    s2 = [x + 0.25 for x in s1]
    s3 = [x + 0.25 for x in s2]
    v1 = coef[labels[0]]
    v2 = coef[labels[1]]
    v3 = coef[labels[2]]
    plt.bar(s1, v1, color='red', width=0.25, edgecolor='white', label='OLS')
    plt.bar(s2, v2, color='blue', width=0.25, edgecolor='white', label='RIDGE')
    plt.bar(s3, v3, color='green', width=0.25, edgecolor='white', label='LASSO')
    plt.xlabel('Coefficients', fontweight='bold')
    plt.xticks([r + 0.25 for r in range(len(coef[labels[0]]))], ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'])
    plt.axhline(0, color='black')
    plt.legend()
    plt.show()


def plot_coefficient_magnitudes(coef):
    labels = list(coef.keys())
    s1 = np.arange(len(coef[labels[0]]))
    s2 = [x + 0.25 for x in s1]
    s3 = [x + 0.25 for x in s2]
    v1 = coef[labels[0]]
    v2 = coef[labels[1]]
    v3 = coef[labels[2]]
    nv1 = [abs(x) / max(abs(v1)) for x in v1]
    nv2 = [abs(x) / max(abs(v2)) for x in v2]
    nv3 = [abs(x) / max(abs(v3)) for x in v3]
    plt.bar(s1, nv1, color='red', width=0.25, edgecolor='white', label='OLS')
    plt.bar(s2, nv2, color='blue', width=0.25, edgecolor='white', label='RIDGE')
    plt.bar(s3, nv3, color='green', width=0.25, edgecolor='white', label='LASSO')
    plt.xlabel('Coefficient Magnitudes', fontweight='bold')
    plt.xticks([r + 0.25 for r in range(len(coef[labels[0]]))], ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'])
    plt.axhline(0, color='black')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
