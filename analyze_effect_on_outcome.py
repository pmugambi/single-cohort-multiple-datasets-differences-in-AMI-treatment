from pathlib import Path

import pandas as pd
import numpy as np

import models


def effect_on_outcome(df, outcome_colname, outcome, dataset, num_of_features=25, model="all", save=False,
                      m4_unique=False):
    X = df.drop(columns=[outcome_colname])
    # print("x.cols = ", X.columns.tolist())
    #
    # print("y-values = ", df[outcome_colname].values.tolist(), sum(df[outcome_colname].values.tolist()),
    #       len(df[outcome_colname].values.tolist()))

    if model == "all":
        # fit lr
        results_all = models.lr_all(X_train=X, y_train=df[outcome_colname], outcome=outcome, dataset=dataset,
                                    save=save, m4_unique=m4_unique)

        # fit lr-rfe
        results_lr_rfe = models.lr_rfe(X_train=X, y_train=df[outcome_colname],
                                       num_of_features=num_of_features, outcome=outcome, dataset=dataset,
                                       save=save, m4_unique=m4_unique)

        # fit lr-lasso
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
    df = pd.read_csv("aumc/data/aumc-ami-patients-features.csv")

    eval_df = df[['agegroup', 'gender', 'lengthofstay', 'los-icu(days)', 'died-in-hosp?', 'urgency',
                  'location',
                  'received-analgesic?',
                  # 'received-opioid?', 'received-non-opioid?',
                  'received-combined-therapy?',
                  'received-ace-inhibitor?',
                  'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?']]

    Path("aumc/data/training-files").mkdir(parents=True, exist_ok=True)
    eval_df.to_csv("./aumc/data/training-files/aumc-cohort-1.0-training-features-raw.csv")

    categorical_cols = ["agegroup", "gender", "location"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "man", "ic"])
    print("ohe_df.columns = ", ohe_df.columns.tolist())

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    print("train_df.columns = ", train_df.columns.tolist())
    train_df.to_csv("./aumc/data/training-files/aumc-cohort-1.0-training-features-processed.csv")

    effect_on_ihm = effect_on_outcome(df=train_df, outcome_colname="died-in-hosp?", dataset="aumc", outcome="ihm",
                                      save=True, num_of_features=20)
    process_all_model_outcomes(effect_on_ihm, dataset="aumc", outcome="ihm")


def effect_on_eicu():
    df = pd.read_csv("./eicu/data/cohorts/eicu-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv")
    df["ethnicity"] = df["ethnicity"].fillna("unknown")
    df["ethnic-group"] = df["ethnicity"].apply(lambda x: assign_ethnic_group(x))
    df["region"] = df["region"].fillna("unknown")
    print('df["region"] values = ', df["region"].value_counts())
    df["discharge-location"] = df["hospitaldischargelocation"].apply(lambda x: consolidate_discharge_location_values(x))
    df["discharge-to-home?"] = np.where(df["discharge-location"] == "home", 1, 0)

    print("df.cols = ", df.columns.tolist())
    eval_df = df[['agegroup', 'gender', 'n-stemi?', 'shock?',
                  'los-h(days)', 'los-icu(days)', 'region', 'died-in-h?',
                  'received-analgesic?',
                  # 'received-opioid?', 'received-non-opioid?',
                  'received-combined-therapy?',
                  'received-ace-inhibitor?',
                  'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                  'ethnic-group', 'discharge-to-home?']]

    eval_df.to_csv("./eicu/data/training-files/eicu-cohort-2.2-training-features-raw.csv")

    categorical_cols = ["agegroup", "gender", "region", "ethnic-group"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", "northeast", "other/unknown"])
    print("ohe_df.columns = ", ohe_df.columns.tolist())

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    print("train_df.columns = ", train_df.columns.tolist())
    train_df.to_csv("./eicu/data/training-files/eicu-cohort-2.2-training-features-processed.csv")

    ihm_df = train_df.drop(columns=["discharge-to-home?"])
    dl_df = train_df.drop(columns=["died-in-h?"])

    effect_on_ihm = effect_on_outcome(df=ihm_df, outcome_colname="died-in-h?", dataset="eicu", outcome="ihm", save=True)
    effect_on_dl = effect_on_outcome(df=dl_df, outcome_colname="discharge-to-home?", dataset="eicu", outcome="dl",
                                     save=True)
    process_all_model_outcomes(effect_on_ihm, dataset="eicu", outcome="ihm")
    process_all_model_outcomes(effect_on_dl, dataset="eicu", outcome="dl")


def effect_on_mimic(dataset, m4_unique=False):
    if dataset == "mimic-iii":
        dataset_data_directory = "./mimic-iii/data/"
        df = pd.read_csv(dataset_data_directory+"cohorts/mimic-iii-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv")
        race_colname = "ethnicity"
        insurance_drop_value = "self pay"
    elif dataset == "mimic-iv":
        dataset_data_directory = "./mimic-iv/data/"
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
    # print('df["discharge_location"] value counts = ', df["discharge_location"].value_counts())
    df["discharge-location"] = df["discharge_location"].apply(lambda x: consolidate_discharge_location_values(x))
    # print('df["discharge-location"] value counts = ', df["discharge-location"].value_counts())
    df["discharge-to-home?"] = np.where(df["discharge-location"] == "home", 1, 0)
    # print('df["discharge-to-home?"] value counts = ', df["discharge-to-home?"].value_counts())
    df["died-in-h?"] = np.where(df["discharge-location"] == "death", 1, 0)
    df["gender"] = np.where(df["gender"] == "m", "male", "female")

    print("df[discharge_location] values = ", sorted(df["discharge_location"].unique().tolist()))
    print("df[gender] values = ", sorted(df["gender"].unique().tolist()))
    print("df[insurance] values = ", sorted(df["insurance"].unique().tolist()))
    print("df[race] values = ", sorted(df[race_colname].unique().tolist()))

    eval_df = df[['agegroup', 'gender', 'n-stemi?', 'shock?',
                  'los-h(days)', 'insurance', 'died-in-h?',
                  'received-analgesic?',
                  # 'received-opioid?', 'received-non-opioid?',
                  'received-combined-therapy?',
                  'received-ace-inhibitor?',
                  'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                  'ethnic-group', 'discharge-to-home?']]

    if m4_unique:
        save_dir = dataset_data_directory + "training-files/m4-only/"
    else:
        save_dir = dataset_data_directory + "training-files/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(save_dir + dataset + "-cohort-2.2-training-features-raw.csv")

    categorical_cols = ["agegroup", "gender", "insurance", "ethnic-group"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", insurance_drop_value, "other/unknown"])

    train_df = pd.concat([eval_df.drop(columns=categorical_cols), ohe_df], axis=1)
    train_df.to_csv(save_dir + dataset + "-cohort-2.2-training-features-processed.csv")

    ihm_df = train_df.drop(columns=["discharge-to-home?"])
    dl_df = train_df.drop(columns=["died-in-h?"])

    effect_on_ihm = effect_on_outcome(df=ihm_df, outcome_colname="died-in-h?", dataset=dataset, outcome="ihm",
                                      save=True, m4_unique=m4_unique)
    effect_on_dl = effect_on_outcome(df=dl_df, outcome_colname="discharge-to-home?", dataset=dataset, outcome="dl",
                                     save=True, m4_unique=m4_unique)
    process_all_model_outcomes(effect_on_ihm, dataset=dataset, outcome="ihm", m4_unique=m4_unique)
    process_all_model_outcomes(effect_on_dl, dataset=dataset, outcome="dl", m4_unique=m4_unique)


def effect_on_combined_eicu_mimic():
    eicu_train_df = pd.read_csv("./eicu/data/training-files/eicu-cohort-2.2-training-features-raw.csv")
    mimic_iii_train_df = pd.read_csv("./mimic-iii/data/training-files/mimic-iii-cohort-2.2-training-features-raw.csv")

    eicu_train_df["insurance"] = "unknown"
    mimic_iii_train_df["region"] = "northeast"

    combined_df = pd.concat([mimic_iii_train_df, eicu_train_df], axis=0)
    print("combined_df.columns = ", combined_df.columns.tolist())

    eval_df = combined_df[['agegroup', 'gender', 'n-stemi?', 'shock?', "region",
                           'los-h(days)', 'insurance', 'died-in-h?',
                           'received-analgesic?',
                           # 'received-opioid?', 'received-non-opioid?',
                           'received-combined-therapy?',
                           'received-ace-inhibitor?',
                           'received-aspirin?', 'received-beta-blocker?', 'received-anti-platelet?', 'received-statin?',
                           'ethnic-group', 'discharge-to-home?']]
    categorical_cols = ["agegroup", "gender", "insurance", "region", "ethnic-group"]
    ohe_df = convert_categorical_to_numerical(df=eval_df, categorical_variables=categorical_cols,
                                              to_drop=["18-39", "male", "self pay", "northeast", "other/unknown"])
    print("ohe_df.columns = ", ohe_df.columns.tolist())

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
    prsquared_vals = []
    for model in results.keys():
        print("model = ", model)
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
        print("df.head() = ", df.head())
        Path("./" + dataset + "/data/outcomes/").mkdir(parents=True, exist_ok=True)
        if m4_unique:
            Path("./" + dataset + "/data/outcomes/m4-only/").mkdir(parents=True, exist_ok=True)
            df.to_csv("./" + dataset + "/data/outcomes/m4-only/" + dataset + "-" + outcome + "-" + model +
                      "-ss-variables.csv")
        else:
            df.to_csv("./" + dataset + "/data/outcomes/" + dataset + "-" + outcome + "-" + model + "-ss-variables.csv")
    prs_rows = list(zip(results.keys(), prsquared_vals))
    prs_df = pd.DataFrame(data=sorted(prs_rows, key=lambda tup: tup[1], reverse=True),
                          columns=["model", "pseudo-r2"])
    if m4_unique:
        prs_df.to_csv("./" + dataset + "/data/outcomes/m4-only/" + dataset + "-" + outcome + "-pseudo-r2-values.csv")
    else:
        prs_df.to_csv("./" + dataset + "/data/outcomes/" + dataset + "-" + outcome + "-pseudo-r2-values.csv")
    print("prs_rows=", prs_rows)
    print("prs_rows 2 =", sorted(prs_rows, key=lambda tup: tup[1], reverse=True))


def assign_ethnic_group(ethnicity):
    """

    :param ethnicity:
    :return:
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

    :param discharge_location:
    :return:
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


# todo: create a function to convert 'diagnosispriority', to a feature

# effect_on_aumc()
effect_on_eicu()
# effect_on_mimic(dataset="mimic-iii")
# effect_on_combined_eicu_mimic()
# effect_on_mimic(dataset="mimic-iv")
# effect_on_mimic(dataset="mimic-iv", m4_unique=True)
