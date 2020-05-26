import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np


def load_data():
    data = pd.read_csv('../data/breast-cancer-wisconsin-data/data.csv')
    data = data.drop(columns='id')
    data = pd.get_dummies(data, columns=['diagnosis'], drop_first=True)

    print(data.head())

    return data


def init_data(data):
    target = data['diagnosis_M']
    data = data.drop(columns='diagnosis_M')

    print(data.shape)
    print(data.head())

    train_x, test_x, train_y, test_y = train_test_split(data, target, stratify=target, random_state=1)
    print('train size: {}'.format(train_x.shape))
    print('test size: {}'.format(test_x.shape))
    print('train counts: {}'.format(train_y.value_counts()))
    print('test counts: {}'.format(test_y.value_counts()))

    return train_x, test_x, train_y, test_y


def q01_02(train_x, test_x, train_y, test_y):
    parameters = {'n_estimators': (10, 20, 50, 100, 200, 500),
                  'max_depth': (2, 3, 4, 5, 6, 7, 8)
                  }
    parameters = {'n_estimators': (20, 50),
                  'max_depth': (4, 6, 8)
                  }
    clf = ensemble.RandomForestClassifier(random_state=1)
    classifiers = GridSearchCV(clf, parameters, scoring='accuracy', return_train_score=True)
    classifiers.fit(train_x, train_y)

    train_process_results = pd.DataFrame(list(zip(classifiers.cv_results_['params'],
                                                  classifiers.cv_results_['mean_train_score'],
                                                  classifiers.cv_results_['mean_test_score'])),
                                         columns=["Parameter", "mean train score", "mean validation score"])
    print(train_process_results)
    print(classifiers.best_params_)

    clf = ensemble.RandomForestClassifier(random_state=2)
    classifiers = GridSearchCV(clf, parameters, scoring='accuracy', return_train_score=True)
    classifiers.fit(train_x, train_y)

    train_process_results = pd.DataFrame(list(zip(classifiers.cv_results_['params'],
                                                  classifiers.cv_results_['mean_train_score'],
                                                  classifiers.cv_results_['mean_test_score'])),
                                         columns=["Parameter", "mean train score", "mean validation score"])
    print(train_process_results)
    print(classifiers.best_params_)

    y_pred = classifiers.best_estimator_.score(train_x, train_y)
    y_test_pred = classifiers.best_estimator_.score(test_x, test_y)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    best_rf = classifiers.best_estimator_

    return best_rf


def q03(best_rf, train_x):
    importances = best_rf.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Top 5 feature ranking:")

    for i in range(5):
        print("%d. feature %s (%f)" % (i, train_x.columns.values[indices[i]], importances[indices[i]]))

    # # Plot the impurity-based feature importances of the forest
    # import matplotlib.pyplot as plt
    # plt.rcParams["figure.figsize"] = (14, 6)
    # plt.rcParams["font.size"] = 16
    # plt.rcParams["lines.linewidth"] = 2.5
    #
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(train_x.shape[1]), importances[indices], color="r", align="center")
    # plt.xticks(range(train_x.shape[1]), [train_x.columns.values[i] for i in indices], rotation=90)
    # plt.xlim([-1, train_x.shape[1]])
    # plt.show()


def q04(best_rf, test_x):
    test_x = test_x[:6]
    import warnings
    warnings.filterwarnings('ignore')

    shap_values = shap.KernelExplainer(best_rf.predict, test_x).shap_values(test_x)
    shap.summary_plot(shap_values, test_x)

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([test_x.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    print(importance_df)


def q05(best_rf, test_x):
    shap_values = shap.TreeExplainer(best_rf).shap_values(test_x)
    shap_values = shap_values[1]  # shap values of the 1-class
    shap.summary_plot(shap_values, test_x, plot_type="dot")

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([test_x.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    print(importance_df)


if __name__ == '__main__':
    data = load_data()
    train_x, test_x, train_y, test_y = init_data(data)

    best_rf = q01_02(train_x, test_x, train_y, test_y)
    q03(best_rf, train_x)
    q04(best_rf, test_x)
    q05(best_rf, test_x)





