from pathlib import Path
import pandas as pd
import numpy as np
import models


def effect_on_outcome(df, outcome_colname, outcome, dataset, num_of_features=25, model="all", save=False,
                      m4_unique=False):
    """
    Runs the regression models defined in 'models.py'.
    :param df: dataframe containing features and outcomes on which the models are to be fit.
    :param outcome_colname: name of the column in the dataframe 'df' that contains the outcome variable
    :param outcome: name of the outcome (e.g., in-hospital mortality (IHM))
    :param dataset: name of the dataset for which the analysis is being conducted
    :param num_of_features: number of features to be included in the model, if the model chosen is 'lr_rfe'
    :param model: name of the model to be fit. values are; 'lr_all', 'lr_rfe', 'lr_lasso', and 'all'. 'all'
                    means that all 3 lr models will be fit. Default value is 'all'
    :param save: whether to write the model fit parameters to a file to be saved in
                <dataset>/data/results/regression-model-fits/
    :param m4_unique: whether to conduct the regression analysis only in patients in MIMIC-IV that are not in MIMIC-III
    :return: outputs of each model; i.e., fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values.
    """
    X = df.drop(columns=[outcome_colname])

    if model == "all":
        # fit lr (with all the features)
        print("1. Fitting lr_all, (LR with all features)")
        results_all = models.lr_all(X_train=X, y_train=df[outcome_colname], outcome=outcome, dataset=dataset,
                                    save=save, m4_unique=m4_unique)

        # fit lr-rfe
        print("2. Fitting lr_rfe, (LR with recursive feature elimination)")
        results_lr_rfe = models.lr_rfe(X_train=X, y_train=df[outcome_colname],
                                       num_of_features=num_of_features, outcome=outcome, dataset=dataset,
                                       save=save, m4_unique=m4_unique)

        # fit lr-lasso
        print("3. Fitting lr_lasso, (LR with l1 penalty)")
        results_lr_lasso = models.lr_lasso(X_train=X, y_train=df[outcome_colname], outcome=outcome, dataset=dataset,
                                           save=save, m4_unique=m4_unique)
        results = {"lr-all": results_all, "lr-rfe": results_lr_rfe, "lr-lasso": results_lr_lasso}
    elif model == "lr-rfe":
        results = models.lr_rfe(X_train=X, y_train=df[outcome_colname], num_of_features=num_of_features,
                                outcome=outcome, dataset=dataset, save=save, m4_unique=m4_unique)
    elif model == "lr-all":
        results = models.lr_all(X_train=X, y_train=df[outcome_colname], outcome=outcome, dataset=dataset,
                                save=save, m4_unique=m4_unique)
    elif model == "lr-lasso":
        results = models.lr_lasso(X_train=X, y_train=df[outcome_colname], outcome=outcome, dataset=dataset,
                                  save=save, m4_unique=m4_unique)
    else:
        raise ValueError("Incorrect model value passed. Acceptable values are 'all', 'lr-rfe', 'lr-lasso', 'lr-all'")
    return results


def effect_on_aumc():
    """
    Runs regression analysis on the AUMC dataset
    :return: fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values. these are written to files
    saved in aumc/data/results/regression-outcomes/
    """
    df = pd.read_csv("aumc/data/processed/features-files/aumc-ami-patients-features.csv")

    eval_df = df[['agegroup', 'gender', 'lengthofstay', 'los-icu(days)', 'died-in-hosp?', 'urgency',
                  'location', 'stemi?', 'received-analgesic?', 'received-combined-therapy?',
                  'received-ace-inhibitor?', 'received-aspirin?', 'received-beta-blocker?',
                  'received-anti-platelet?', 'received-statin?']]

    model_training_files_path = "./aumc/data/processed/model-training-files/"
    Path(model_training_files_path).mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(model_training_files_path+"aumc-cohort-1.0-training-features-raw.csv", index=False)

    categorical_cols = ["agegroup", "gender", "location"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "man", "ic"])

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    train_df.to_csv(model_training_files_path+"aumc-cohort-1.0-training-features-processed.csv", index=False)

    effect_on_ihm = effect_on_outcome(df=train_df, outcome_colname="died-in-hosp?", dataset="aumc", outcome="ihm",
                                      save=True, num_of_features=17)
    process_all_model_outcomes(effect_on_ihm, dataset="aumc", outcome="ihm")


def effect_on_eicu():
    """
    Runs regression analysis on the eICU dataset
    :return: fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values. these are written to files
    saved in eicu/data/results/regression-outcomes/
    :return:
    """
    df = pd.read_csv("./eicu/data/processed/cohorts/eicu-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv")
    df["ethnicity"] = df["ethnicity"].fillna("unknown")
    df["ethnic-group"] = df["ethnicity"].apply(lambda x: assign_ethnic_group(x))
    df["region"] = df["region"].fillna("unknown")
    df["discharge-location"] = df["hospitaldischargelocation"].apply(lambda x: consolidate_discharge_location_values(x))
    df["discharge-to-home?"] = np.where(df["discharge-location"] == "home", 1, 0)

    eval_df = df[['agegroup', 'gender', 'n-stemi?', 'shock?',
                  'los-h(days)', 'los-icu(days)', 'region', 'died-in-h?',
                  'received-analgesic?', "hospitalteachingstatus",
                  'received-combined-therapy?',
                  'received-ace-inhibitor?',
                  'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                  'ethnic-group', 'discharge-to-home?']]

    model_training_files_path = "./eicu/data/processed/model-training-files/"
    Path(model_training_files_path).mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(model_training_files_path+"eicu-cohort-2.2-training-features-raw.csv", index=False)

    categorical_cols = ["agegroup", "gender", "region", "ethnic-group", "hospitalteachingstatus"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", "northeast", "other/unknown", "f"])

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    train_df.to_csv(model_training_files_path+"eicu-cohort-2.2-training-features-processed.csv", index=False)

    ihm_df = train_df.drop(columns=["discharge-to-home?"])
    dl_df = train_df.drop(columns=["died-in-h?"])

    effect_on_ihm = effect_on_outcome(df=ihm_df, outcome_colname="died-in-h?", dataset="eicu", outcome="ihm", save=True)
    effect_on_dl = effect_on_outcome(df=dl_df, outcome_colname="discharge-to-home?", dataset="eicu", outcome="dl",
                                     save=True)
    process_all_model_outcomes(effect_on_ihm, dataset="eicu", outcome="ihm")
    process_all_model_outcomes(effect_on_dl, dataset="eicu", outcome="dl")


def effect_on_mimic(dataset, m4_unique=False):
    """
    Runs regression analysis on the MIMIC datasets
    :param dataset: name of the dataset being analyzed (i.e., mimic-iii or mimic-iv)
    :param m4_unique: whether to conduct the regression analysis only in patients in MIMIC-IV that are not in MIMIC-III
    :return: fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values. these are written to files
    saved in <mimic-dataset>/data/results/regression-outcomes/
    """
    if dataset == "mimic-iii":
        dataset_data_directory = "./mimic-iii/data/processed/"
        df = pd.read_csv(dataset_data_directory+"cohorts/mimic-iii-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv")
        race_colname = "ethnicity"
        insurance_drop_value = "self pay"
    elif dataset == "mimic-iv":
        dataset_data_directory = "./mimic-iv/data/processed/"
        race_colname = "race"
        insurance_drop_value = "other"
        if m4_unique:
            df = pd.read_csv(dataset_data_directory+"cohorts/m4-only/mimic-iv-cohort2.2-ami-is-primary-diagnosis-and"
                                                    "-healthy.csv")
        else:
            df = pd.read_csv(dataset_data_directory+"cohorts/mimic-iv-cohort2.2-ami-is-primary-diagnosis-and-healthy"
                                                    ".csv")
    else:
        raise ValueError("the expected dataset name values are either 'mimic-iii' or 'mimic-iv'")

    df[race_colname] = df[race_colname].fillna("unknown")
    df["ethnic-group"] = df[race_colname].apply(lambda x: assign_ethnic_group(x))
    df["insurance"] = df["insurance"].fillna("unknown")
    df["discharge_location"] = df["discharge_location"].fillna("unknown")
    df["discharge-location"] = df["discharge_location"].apply(lambda x: consolidate_discharge_location_values(x))
    df["discharge-to-home?"] = np.where(df["discharge-location"] == "home", 1, 0)
    df["died-in-h?"] = np.where(df["discharge-location"] == "death", 1, 0)
    df["gender"] = np.where(df["gender"] == "m", "male", "female")

    eval_df = df[['agegroup', 'gender', 'n-stemi?', 'shock?',
                  'los-h(days)', 'insurance', 'died-in-h?',
                  'received-analgesic?',
                  'received-combined-therapy?',
                  'received-ace-inhibitor?',
                  'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                  'ethnic-group', 'discharge-to-home?']]

    model_training_files_path = dataset_data_directory + "model-training-files/"
    if m4_unique:
        save_dir = model_training_files_path + "m4-only/"
    else:
        save_dir = dataset_data_directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(save_dir + dataset + "-cohort-2.2-training-features-raw.csv", index=False)

    categorical_cols = ["agegroup", "gender", "insurance", "ethnic-group"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", insurance_drop_value, "other/unknown"])

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    train_df.to_csv(save_dir + dataset + "-cohort-2.2-training-features-processed.csv", index=False)

    ihm_df = train_df.drop(columns=["discharge-to-home?"])
    dl_df = train_df.drop(columns=["died-in-h?"])

    effect_on_ihm = effect_on_outcome(df=ihm_df, outcome_colname="died-in-h?", dataset=dataset, outcome="ihm",
                                      save=True, m4_unique=m4_unique)
    effect_on_dl = effect_on_outcome(df=dl_df, outcome_colname="discharge-to-home?", dataset=dataset, outcome="dl",
                                     save=True, m4_unique=m4_unique)
    process_all_model_outcomes(effect_on_ihm, dataset=dataset, outcome="ihm", m4_unique=m4_unique)
    process_all_model_outcomes(effect_on_dl, dataset=dataset, outcome="dl", m4_unique=m4_unique)


def effect_on_combined_eicu_mimic():
    """
    Runs regression analysis on the eICU+MIMIC-III combined dataset
    :return: fit model's parameters; pseudo-rsquared, parameter coefficients, and p-values. these are written to files
    saved in <mimic-dataset>/data/results/regression-outcomes/
    :return:
    """
    read_common_path = "data/processed/model-training-files/"
    eicu_train_df = pd.read_csv("./eicu"+read_common_path+"eicu-cohort-2.2-training-features-raw.csv")
    mimic_iii_train_df = pd.read_csv("./mimic-iii/"+read_common_path+"mimic-iii-cohort-2.2-training-features-raw.csv")

    eicu_train_df["insurance"] = "unknown"
    mimic_iii_train_df["region"] = "northeast"

    combined_df = pd.concat([mimic_iii_train_df, eicu_train_df], axis=0)

    eval_df = combined_df[['agegroup', 'gender', 'n-stemi?', 'shock?', "region",
                           'los-h(days)', 'insurance', 'died-in-h?',
                           'received-analgesic?',
                           'received-combined-therapy?',
                           'received-ace-inhibitor?',
                           'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                           'ethnic-group', 'discharge-to-home?']]
    categorical_cols = ["agegroup", "gender", "insurance", "region", "ethnic-group"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", "self pay", "northeast", "other/unknown"])

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    ihm_df = train_df.drop(columns=["discharge-to-home?"])
    dl_df = train_df.drop(columns=["died-in-h?"])
    effect_on_ihm = effect_on_outcome(df=ihm_df, outcome_colname="died-in-h?", dataset="eicu-and-mimic-iii",
                                      outcome="ihm", save=True)
    effect_on_dl = effect_on_outcome(df=dl_df, outcome_colname="discharge-to-home?", dataset="eicu-and-mimic-iii",
                                     outcome="dl", save=True)
    process_all_model_outcomes(effect_on_ihm, dataset="eicu-and-mimic-iii", outcome="ihm")
    process_all_model_outcomes(effect_on_dl, dataset="eicu-and-mimic-iii", outcome="dl")


def process_all_model_outcomes(results, dataset, outcome, m4_unique=False):
    """
    Reads the fit models' output (i.e., paramater coefficients, pseudo-rsquared value, and p-values) and retrieves
    statistically significant associations (i.e., parameters where p-values <= 0.05). It also computes the
    odds ratio (OR) from the parameter coefficient values.
    :param results: model fit output
    :param dataset: name of the dataset for which the model was fit
    :param outcome: name of the outcome being analyzed (i.e., in-hospital mortality or discharge location)
    :param m4_unique: whether to conduct the regression analysis only in patients in MIMIC-IV that are not in MIMIC-III
    :return: Nothing. All outputs are saved in files in the <dataset>/data/results/regression-outcomes/ folder.
    """
    prsquared_vals = []
    save_dir = "./" + dataset + "/data/results/regression-outcomes/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for model in results.keys():
        model_res_obj = results[model]
        prsquared_vals.append(model_res_obj["prsquared"])
        variables = model_res_obj["coefs"].keys().tolist()
        coefs = model_res_obj["coefs"].values.tolist()
        pvalues = model_res_obj["pvalues"].values.tolist()
        ss_variables_indeces = [pvalues.index(x) for x in pvalues if x <= 0.05]
        ss_variables = [variables[x] for x in ss_variables_indeces]
        ss_coefs = [round(coefs[x], 5) for x in ss_variables_indeces]

        # compute odds ratios
        ors = [round(x, 5) for x in np.exp(ss_coefs)]

        # generate pdf and save as csv
        rows = list(zip(ss_variables, ss_coefs, [round(pvalues[x], 5) for x in ss_variables_indeces], ors))
        df = pd.DataFrame(data=rows, columns=["variable", "coef", "p-value", "OR"])
        if m4_unique:
            Path(save_dir+"m4-only/").mkdir(parents=True, exist_ok=True)
            df.to_csv(save_dir+"m4-only/" + dataset + "-" + outcome + "-" + model + "-ss-variables.csv", index=False)
        else:
            df.to_csv(save_dir + dataset + "-" + outcome + "-" + model + "-ss-variables.csv", index=False)
    prs_rows = list(zip(results.keys(), prsquared_vals))
    prs_df = pd.DataFrame(data=sorted(prs_rows, key=lambda tup: tup[1], reverse=True),
                          columns=["model", "pseudo-r2"])
    if m4_unique:
        prs_df.to_csv(save_dir+"m4-only/" + dataset + "-" + outcome + "-pseudo-r2-values.csv", index=False)
    else:
        prs_df.to_csv(save_dir + dataset + "-" + outcome + "-pseudo-r2-values.csv", index=False)
    print("sorted pseudo-rsquared values ===>", sorted(prs_rows, key=lambda tup: tup[1], reverse=True))


def assign_ethnic_group(ethnicity):
    """
    Helper function to group ethnicity values into race values.
    :param ethnicity: value of ethnicity recorded in the raw data
    :return: a race group value.
    """
    if ("white" in ethnicity) | (ethnicity == "portuguese"):
        group = "caucasian"
    elif ethnicity == "unknown/not specified":
        group = "other/unknown"
    elif ("hispanic" in ethnicity) | ("south american" in ethnicity):
        group = "hispanic"
    elif ("asian" in ethnicity) | ("native hawaiian" in ethnicity) | ("pacific islander" in ethnicity):
        group = "asian"
    elif "black" in ethnicity:
        group = "african american"
    elif ethnicity == "american indian/alaska native":
        group = "native american"
    else:
        group = "other/unknown"
    return group


def convert_categorical_to_numerical(df, categorical_variables, to_drop):
    """
    This function one-hot encodes categorical variables in a dataframe
    :param df: dataframe
    :param categorical_variables: a list of column names of the categorical data that need to be coded
           e.g., ["gender", "hobby"]
    :param to_drop: a list of the values of the categorical columns to be dropped. e.g. ["female", "skiing"].
           These are assumed to be the "default" values. Dropping values reduces multi-linearity,
           so by default this function assumes that some values will be dropped
    :return: a dataframe of the original categorical columns expanded using one-hot encoding
    """
    one_hot_encoded_list = []
    for i in range(len(categorical_variables)):
        one_hot_encoded_list.append(df[categorical_variables[i]].str.get_dummies().add_prefix(
            categorical_variables[i] + "-").drop(categorical_variables[i] + "-" + to_drop[i], axis=1))
    one_hot_encoded_df = pd.concat(one_hot_encoded_list, axis=1)
    return one_hot_encoded_df


def consolidate_discharge_location_values(discharge_location):
    """
    Helper function to consolidate discharge location values into a few groups.
    :param discharge_location: value of discharge location recorded in the raw/original dataset
    :return: a discharge location group value.
    """
    to_home = ["home", "home with home iv providr", "home health care"]
    to_snf = ["skilled nursing facility", "snf"]
    to_nursing_home = ["home health care", "nursing home", "assisted living", "chronic/long term acute care"]
    to_rehab_or_hospice = ["rehabilitation", "rehab/distinct part hosp", "hospice-medical facility", "hospice-home",
                           "hospice", "rehab"]
    to_other_hosp = ["other hospital", "disc-tran cancer/chldrn h", "disch-tran to psych hosp",
                     "other external", "short term hospital", "long term care hospital", "other facility",
                     "acute hospital", "psych facility"]
    to_other_or_unknown = ["unknown", "left against medical advi", "other", "against advice"]
    to_death = ["death", "dead/expired", "died"]

    if discharge_location in to_home:
        return "home"
    elif discharge_location in to_snf:
        return "snf"
    elif discharge_location in to_nursing_home:
        return "nursing home"
    elif discharge_location in to_rehab_or_hospice:
        return "rehab/hospice"
    elif discharge_location in to_other_hosp:
        return "other hospital"
    elif discharge_location in to_other_or_unknown:
        return "other/unknown"
    elif discharge_location in to_death:
        return "death"
    else:
        return discharge_location


def run_analyses():
    """
    Iterates through all datasets and runs the regression analysis.
    :return:
    """
    effect_on_aumc()
    effect_on_eicu()
    effect_on_mimic(dataset="mimic-iii")
    effect_on_mimic(dataset="mimic-iv", m4_unique=True)
    # effect_on_combined_eicu_mimic()


if __name__ == '__main__':
    run_analyses()
