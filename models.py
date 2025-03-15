from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


def lr_rfe(X_train, y_train, num_of_features, dataset, outcome, save=False, m4_unique=False):
    """
    Fits a logistic regression model with recursive feature elimination (RFE).
    :param X_train: a dataframe containing features/variables
    :param y_train: a dataframe (actually series) of the outcome labels
    :param num_of_features: number of features to be selected, i.e., how many features should be left in the model
    after the elimination
    :param dataset: name of the dataset being analyzed. important for saving the fit model.
    :param outcome: name of the outcome (i.e., either in-hospital mortality or discharge-location) being analyzed.
    important for saving the fit model.
    :param save: whether the fit model's output should be saved (i.e., written to the file system) or not.
    saving is helpful in checking the model's fit parameters such as pseudo-rsquared.
    :param m4_unique: whether the model should be fit on patients in MIMIC-IV that are not in MIMIC-III
    :return: the fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values.
    """
    lr = LogisticRegression(max_iter=50000)
    rfe = RFE(lr, n_features_to_select=num_of_features)
    rfe = rfe.fit(X_train, y_train.values.ravel())

    # pick the predictive features
    ranks = rfe.ranking_
    cols = X_train.columns.values.tolist()
    important_features = []
    for i in range(len(cols)):
        if ranks[i] == 1:
            important_features.append(cols[i])
    # print("important features are: ", important_features, len(important_features))
    important_features = list(set(important_features))
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train[important_features]).fit(maxiter=1500, method="bfgs")

    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-rfe-nof-" + str(num_of_features) + ".txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}


def lr_all(X_train, y_train, dataset, outcome, save=False, m4_unique=False):
    """
    Fits a logistic regression model with ALL the features included. (No elimination, as in the case of lr_rfe, nor
    selection, as in the case of lr_lasso).
    :param X_train: a dataframe containing features/variables
    :param y_train: a dataframe (actually series) of the outcome labels
    :param dataset: name of the dataset being analyzed. important for saving the fit model.
    :param outcome: name of the outcome (i.e., either in-hospital mortality or discharge-location) being analyzed.
    important for saving the fit model.
    :param save: whether the fit model's output should be saved (i.e., written to the file system) or not.
    saving is helpful in checking the model's fit parameters such as pseudo-rsquared.
    :param m4_unique: whether the model should be fit on patients in MIMIC-IV that are not in MIMIC-III
    :return: the fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values.
    """
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train).fit(method="bfgs", maxiter=1500)
    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-all.txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}


def lr_lasso(X_train, y_train, dataset, outcome, save=False, m4_unique=False):
    """
    Fits a logistic regression model with l1-penalty.
    :param X_train: a dataframe containing features/variables
    :param y_train: a dataframe (actually series) of the outcome labels
    :param dataset: name of the dataset being analyzed. important for saving the fit model.
    :param outcome: name of the outcome (i.e., either in-hospital mortality or discharge-location) being analyzed.
    important for saving the fit model.
    :param save: whether the fit model's output should be saved (i.e., written to the file system) or not.
    saving is helpful in checking the model's fit parameters such as pseudo-rsquared.
    :param m4_unique: whether the model should be fit on patients in MIMIC-IV that are not in MIMIC-III
    :return: the fit model's paramaters; pseudo-rsquared, parameter coefficients, and p-values.
    """
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train)
    model = model.fit_regularized(method="l1", acc=1e-6, maxiter=1500, trim_mode="size")
    if save:
        if m4_unique:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/m4-only/"
        else:
            save_dir = "./" + dataset + "/data/results/regression-model-fits/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_name = save_dir+dataset+"-"+outcome+"-lr-lasso.txt"
        with open(save_name, "w") as f:
            f.write(model.summary().as_text())
    return {"prsquared": model.prsquared, "coefs": model.params, "pvalues": model.pvalues}
