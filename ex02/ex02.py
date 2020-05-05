import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from pprint import pprint


def q1():
    print("Q1:")
    log_reg = LogisticRegression(max_iter=10000, random_state=0)
    # the concatenate here foe keeping the randomness of the samples
    results = cross_validate(log_reg,
                             X=np.concatenate((X_train, X_test), axis=0),
                             y=np.concatenate((y_train, y_test), axis=0),
                             cv=5,
                             scoring=['accuracy', 'neg_mean_squared_error'], return_train_score=True)

    train_accuracy = results['train_accuracy']
    test_accuracy = results['test_accuracy']
    train_neg_mean_squared_err = results['train_neg_mean_squared_error']
    test_neg_mean_squared_err = results['test_neg_mean_squared_error']
    print("Train folds acc: {0} (mean: {1})".format(train_accuracy, np.mean(train_accuracy)))
    print("Test folds acc: {0} (mean: {1})".format(test_accuracy, np.mean(test_accuracy)))
    print("Train folds loss: {0} (mean: {1})".format(train_neg_mean_squared_err, np.mean(train_neg_mean_squared_err)))
    print("Test folds loss: {0} (mean: {1})".format(test_neg_mean_squared_err, np.mean(test_neg_mean_squared_err)))

    log_reg.fit(X_train, y_train)
    y_pred = log_reg.score(X_train, y_train)
    y_test_pred = log_reg.score(X_test, y_test)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    return log_reg


def q2():
    log_reg_cv1 = LogisticRegressionCV(solver='liblinear', penalty='l1', cv=5, max_iter=10000, scoring='accuracy')
    log_reg_cv1.fit(X_train, y_train)
    # pprint(log_reg_cv1.scores_)
    y_pred = log_reg_cv1.score(X_train, y_train)
    y_test_pred = log_reg_cv1.score(X_test, y_test)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    print("Q2:")
    log_reg_cv2 = LogisticRegressionCV(penalty='l2', cv=5, max_iter=10000, scoring='accuracy')
    log_reg_cv2.fit(X_train, y_train)
    # pprint(log_reg_cv2.scores_)
    y_pred = log_reg_cv2.score(X_train, y_train)
    y_test_pred = log_reg_cv2.score(X_test, y_test)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    log_reg_cv3 = LogisticRegressionCV(solver='saga', penalty='elasticnet', l1_ratios=[.1, .5, .7, .9, .95, .99],
                                       cv=5, max_iter=10000, scoring='accuracy')
    log_reg_cv3.fit(X_train, y_train)
    # pprint(log_reg_cv3.scores_)
    y_pred = log_reg_cv3.score(X_train, y_train)
    y_test_pred = log_reg_cv3.score(X_test, y_test)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    return log_reg_cv2


def q3():
    print("Q3:")
    rfc_model = RandomForestClassifier(random_state=0)
    model_params = {
        'n_estimators': [10, 20, 30, 50, 100, 200],
        'max_features': ['sqrt', 3, 4, 5],
        'max_depth': [2, 3, 4, 5, 6, 7, 8]
    }
    cv_rfc = GridSearchCV(estimator=rfc_model, param_grid=model_params, cv=5)
    cv_rfc.fit(X_train, y_train)
    pprint(cv_rfc.best_estimator_.get_params())

    y_pred = cv_rfc.best_estimator_.score(X_train, y_train)
    y_test_pred = cv_rfc.best_estimator_.score(X_test, y_test)
    print("Train data acc: {0}".format(y_pred))
    print("Test data acc: {0}".format(y_test_pred))

    return cv_rfc


def q4(model1, model2, model3):
    def make_roc_caliboration_curves(model, label):
        prob_pos = model.predict_proba(X_test)[:, 1]  # probs for 1-class
        fpr, tpr, _ = roc_curve(y_test, prob_pos)
        roc_auc = auc(fpr, tpr)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s " % label)
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=label, histtype="step", lw=2)
        ax3.plot(fpr, tpr, lw=3, label='%s ROC curve (area = %0.2f)' % (label, roc_auc))

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((6, 1), (2, 0), sharex=ax1)
    ax3 = plt.subplot2grid((6, 1), (3, 0), rowspan=3)

    make_roc_caliboration_curves(model1, "LogisticRegression")
    make_roc_caliboration_curves(model2, "LogisticRegressionCV")
    make_roc_caliboration_curves(model3, "RandomForest")

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    ax3.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--', label="Random Classifier")
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver operating characteristic example')
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # load data
    data = np.loadtxt('../data/heart_disease_uci/heart.csv', delimiter=",", skiprows=1)
    X = data[:, :-1]
    Y = np.asarray(data[:, -1], dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=1)

    # model1 = q1()
    # model2 = q2()
    model3 = q3()
    # q4(model1, model2, model3)



