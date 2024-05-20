from pathlib import Path

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.svm import l1_min_c
import time


def lr_rfe(X_train, y_train, num_of_features, dataset, outcome, save=False, m4_unique=False):
    """

    :param X_train:
    :param y_train:
    :param num_of_features:
    :param dataset:
    :param outcome:
    :param save:
    :param m4_unique
    :return:
    """
    lr = LogisticRegression(max_iter=50000)
    print("num of features = ", num_of_features)
    rfe = RFE(lr, n_features_to_select=num_of_features)
    rfe = rfe.fit(X_train, y_train.values.ravel())

    # pick the predictive features
    ranks = rfe.ranking_
    cols = X_train.columns.values.tolist()
    important_features = []
    for i in range(len(cols)):
        if ranks[i] == 1:
            important_features.append(cols[i])
    print("important features are: ", important_features, len(important_features))
    important_features = list(set(important_features))
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train[important_features]).fit(maxiter=1500, method="bfgs")
    # print(model.summary())
    # print(model.summary2())
    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-rfe-nof-" + str(num_of_features) + ".txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}


def lr_all(X_train, y_train, dataset, outcome, save=False, m4_unique=False):
    """

    :param X_train:
    :param y_train:
    :param dataset:
    :param outcome:
    :param save:
    :param m4_unique
    :return:
    """
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train).fit(method="bfgs", maxiter=1500)
    # print(model.summary())
    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-all.txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}


def lr_lasso(X_train, y_train, dataset, outcome, save=False, m4_unique=False):
    """

    :param X_train:
    :param y_train:
    :param dataset:
    :param outcome:
    :param save:
    :return:
    """
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train)
    model = model.fit_regularized(method="l1", acc=1e-6, maxiter=1500, trim_mode="size")
    # model = model.fit_regularized(method="l1_cvxopt_cp", maxiter=1500)
    # print(model.summary())
    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-lasso.txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}
