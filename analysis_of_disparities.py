import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import stats, barnard_exact
from pathlib import Path


def create_contingency_table(mini_df, treatment_col, group_col, groups_values):
    """
    Creates a list of lists representing numbers of patients who received and who did not receive a specific
    treatment, when the patients are stratified by a specific attribute (e.g., type of insurance)
    :param mini_df: a dataframe containing records for whether patients received or did not receive a specific treatment
    :param treatment_col: name of the column in the patients' features file that contains the treatment being analyzed
    :param group_col: the name of the column in the patients' features file that is being used to stratify
    the patients (e.gs., race, sex, type of insurance, region, type of hospital)
    :param groups_values: list of values for each variable being used to stratify the cohort.
    For instance [male, female] when using 'sex' to stratify the cohort.
    :return: the created list of lists (i.e., contingency table)
    """
    table = []
    for group in groups_values:
        group_df = mini_df[mini_df[group_col].isin(group)]
        group_received_treatment = group_df[group_df[treatment_col].isin([1, "yes"])]
        group_did_not_receive_treatment = group_df[group_df[treatment_col].isin([0, "no"])]
        table.append([len(group_received_treatment), len(group_did_not_receive_treatment)])
    return table


def compute_barnard_exact_test_score(contingency_table):
    """
    Runs the Barnard exact test
    :param contingency_table: a contingency table of proportions of patients who received and those who did not receive
    a treatment. (used for variables with 2 values, e.g., sex={Female, Male})
    :return: Barnard exact test statistic and p-value
    """
    res = barnard_exact(contingency_table)
    return res.statistic, res.pvalue


def compute_chi2_contingency_score(contingency_table):
    """
    Runs the chi-squared test.
    :param contingency_table: a contingency table of proportions of patients who received and those who did not receive
    a treatment. (used for variables with more than 2 values, e.g., race={Caucasian, Black, Asian})
    :return: Chi square test statistic and p-value
    """
    statistic, p_value, dof, expected_freq = chi2_contingency(contingency_table)
    return statistic, p_value


def compute_t_test(g1_values, g2_values):
    """
    Runs the t-test
    :param g1_values: [Dosage] values of the first group
    :param g2_values: [Dosage] values of the second group
    :return: t-test statistic and p-value
    """
    statistic, pvalue = stats.ttest_ind(a=g1_values, b=g2_values, equal_var=False)
    return statistic, pvalue


def test_entire_duration(df, treatment_col, pi_variable, groups_values):  # g1_values, g2_values, group_values):
    """
    Runs exact/chi-square tests on contingency tables of patients who received/did not receive a specific treatment at
    any time during their hospitalization.
    :param df: cohort features dataframe (read from the features' file)
    :param treatment_col: the name of column in 'df' that contains names of treatments
    :param pi_variable: the SDoH (race, sex, type of insurance, region, type of hospital) variable used to stratify
    the cohort
    :param groups_values: list of values for each SDoH being used to stratify the cohort. For instance [male, female]
    when using 'sex' to stratify the cohort.
    :return: the contingency table (containing patient proportions by SDoH that received a specific treatment),
    the test (Barnard or chi-square) statistic and p-value.
    """
    treatment_mini_df = df[[pi_variable, treatment_col]]
    table = create_contingency_table(mini_df=treatment_mini_df, treatment_col=treatment_col,
                                     group_col=pi_variable, groups_values=groups_values)
    if len(table) == 2:
        statistic, p_value = compute_barnard_exact_test_score(contingency_table=table)
    else:
        statistic, p_value = compute_chi2_contingency_score(table)
    return table, round(statistic, 4), round(p_value, 4)


def run_entire_admission_duration_tests(df, pi_column_names, pi_column_values, pi_save_col_names,
                                        pi_variables, cohort_number, dataset_folder, m4_unique=False):
    """
    Iterates through the list of all treatments under study and computes the statistically significant differences
    in proportions of patients who received each of the treatments, by calling function **test_entire_duration** above.
    :param df: cohort features dataframe (read from the features' file)
    :param pi_column_names: the names of columns in 'df' that contain SDoH variables to use to stratify the cohort by.
    e.g., (sex, race, region)
    :param pi_column_values: the values of each of the SDoH variables (e.g., {sex: male, female}, {region: northeast,
    midwest, south, west}
    :param pi_save_col_names: names of the columns to be created (in the output dataframe and file) that contain
    the results of the hypothesis tests
    :param pi_variables: list of names of the variables being analyzed (e.g., sex, race-2group, race-3group).
    We can analyze each SDoH variable (e.g., race) multiple ways. This argument keeps track of that.
    :param cohort_number: the number of the cohort being analyzed. Function **create_sub_cohorts_feature_files** in
    **create_feature_files.py** creates different cohorts, adding a little more dynamism in the kinds of analysis this
    system can provide.
    :param dataset_folder: path to the folder containing files for the dataset being analyzed. Allows for reading of
    existing files and saving newly created output ones.
    :param m4_unique: a flag to check whether when running hypothesis tests on MIMIC-IV dataset, only those patients in
    MIMIC-IV but not in MIMIC-III should be extracted. This too adds to the dynamism of this tool's use.
    :return: Nothing. All output are written to <dataset>/data/results/disparities/ folder.
    """
    treatment_cols = ["received-analgesic?", "received-opioid?", "received-non-opioid?", "received-opioids-only?",
                      "received-non-opioids-only?", "received-combined-therapy?", "received-ace-inhibitor?",
                      "received-aspirin?", "received-beta-blocker?", "received-anti-platelet?", "received-statin?"]
    results = []
    for i in range(len(pi_column_names)):
        pi_results = []
        for treatment in treatment_cols:
            treatment_ct, treatment_statistic, treatment_pvalue = test_entire_duration(
                df=df,
                treatment_col=treatment,
                pi_variable=pi_column_names[i],
                groups_values=pi_column_values[i])
            treatment_results = [treatment, treatment_pvalue, treatment_statistic]
            for res in treatment_ct:
                g_received_percentage = round(res[0] / sum(res), 2)
                g_dn_receive_percentage = round(res[1] / sum(res), 2)
                treatment_results += [res[0], g_received_percentage, res[1], g_dn_receive_percentage]
            pi_results.append(treatment_results)
        results.append(pi_results)
    for i in range(len(results)):
        df = pd.DataFrame(data=results[i], columns=[["treatment", "pvalue", "statistic"] + pi_save_col_names[i]])
        if m4_unique:
            save_dir = dataset_folder + "/data/results/disparities/entire-hospitalization/p-values/cohort-" + str(
                cohort_number) + "/m4-only/"
        else:
            save_dir = dataset_folder + "/data/results/disparities/entire-hospitalization/p-values/cohort-" + str(
                cohort_number) + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir + "c-" + str(cohort_number) + "-differences-in-proportions-by-" + pi_variables[i] + ".csv",
                  index=False)


def aumc_analyses(cohort_number=1.0):
    """
    Calls the Function **run_entire_admission_duration_tests** to run hypothesis tests on the AUMC dataset.
    :param cohort_number: the cohort to be extracted (represented by the number). Default here is 1.0.
    Since aumc has a small sample, only 1 cohort is created for it.
    :return: Nothing. The results of the hypothesis tests are written out into files.
    """
    dataset_folder = "./aumc"
    pi_variables = ["sex"]
    pi_column_names = ["gender"]
    pi_column_values = [[["vrouw"], ["man"]]]
    pi_save_col_names = [["female-yes-#", "female-yes-%", "female-no-#", "female-no-%", "male-yes-#",
                          "male-yes-%", "male-no-#", "male-no-%"]]
    if cohort_number != 1.0:
        return ValueError("This dataset can only be analyzed for cohort 1.0")
    else:
        cohort_df = pd.read_csv(dataset_folder + "/data/processed/features-files/aumc-ami-patients-features.csv")
        # Unlike with MIMIC and eICU, function **obtain_cohort** below
        # is not used to extract the cohort here, because there's only 1 cohort.
        run_entire_admission_duration_tests(df=cohort_df, pi_column_names=pi_column_names,
                                            pi_column_values=pi_column_values, pi_save_col_names=pi_save_col_names,
                                            pi_variables=pi_variables, cohort_number=cohort_number,
                                            dataset_folder=dataset_folder)


def eicu_analyses(cohort_number=2.2):
    """
    Calls the Function **run_entire_admission_duration_tests** to run hypothesis tests on the eICU dataset.
    :param cohort_number: the cohort to be extracted (represented by the number). Default here is 2.2.
    :return: Nothing. The results of the hypothesis tests are written out into files.
    """
    dataset_folder = "./eicu"
    dataset = "eicu"
    ami_df = pd.read_csv(dataset_folder + "/data/processed/features-files/eicu-ami-patients-features.csv")

    # there are some records where ethnicity is left blank. i'm including them with the unknown
    unknown_ethnicity_values = ["other/unknown", np.nan]
    non_caucasian_ethnicities = ami_df[~ami_df["ethnicity"].isin(
        ["caucasian"] + unknown_ethnicity_values)]["ethnicity"].unique().tolist()
    pi_column_names = ["gender", "ethnicity", "ethnicity", "ethnicity", "region", "hospitalteachingstatus"]
    pi_column_values = [[["female"], ["male"]],
                        [["caucasian"], non_caucasian_ethnicities],
                        [["caucasian"], non_caucasian_ethnicities, unknown_ethnicity_values],
                        [["caucasian"], ["african american"], ["asian"], ["hispanic"],
                         unknown_ethnicity_values],
                        [["northeast"], ["midwest"], ["west"], ["south"], [np.nan]],
                        [["f"], ["t"]]]

    pi_variables = ["sex", "race-2G", "race-3G", "race-multiG", "region", "hospitalteachingstatus"]
    pi_save_col_names = [
        ["female-yes-#", "female-yes-%", "female-no-#", "female-no-%",
         "male-yes-#", "male-yes-%", "male-no-#", "male-no-%"],
        ["caucasian-yes-#", "caucasian-yes-%", "caucasian-no-#", "caucasian-no-%",
         "non-caucasian-yes-#", "non-caucasian-yes-%", "non-caucasian-no-#", "non-caucasian-no-%"],
        ["caucasian-yes-#", "caucasian-yes-%", "caucasian-no-#", "caucasian-no-%",
         "non-caucasian-yes-#", "non-caucasian-yes-%", "non-caucasian-no-#", "non-caucasian-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        ["caucasian-am-yes-#", "caucasian-am-yes-%", "caucasian-am-no-#", "caucasian-am-no-%",
         "african-am-yes-#", "african-am-yes-%", "african-am-no-#", "african-am-no-%",
         "asian-am-yes-#", "asian-am-yes-%", "asian-am-no-#", "asian-am-no-%",
         "hispanic-am-yes-#", "hispanic-am-yes-%", "hispanic-am-no-#", "hispanic-am-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        ["northeast-yes-#", "northeast-yes-%", "northeast-no-#", "northeast-no-%",
         "midwest-yes-#", "midwest-yes-%", "midwest-no-#", "midwest-no-%",
         "west-yes-#", "west-yes-%", "west-no-#", "west-no-%",
         "south-yes-#", "south-yes-%", "south-no-#", "south-no-%",
         "unspecified-yes-#", "unspecified-yes-%", "unspecified-no-#", "unspecified-no-%"],
        ["non-teaching-yes-#", "non-teaching-yes-%", "non-teaching-no-#", "non-teaching-no-%",
         "teaching-yes-#", "teaching-yes-%", "teaching-no-#", "teaching-no-%"]
    ]

    resp = obtain_cohort(cohort_number=cohort_number, dataset_folder=dataset_folder, dataset_name=dataset)
    if isinstance(resp, Exception):
        return ValueError(resp)
    else:
        run_entire_admission_duration_tests(df=resp[0], pi_column_names=pi_column_names,
                                            pi_column_values=pi_column_values,
                                            pi_save_col_names=pi_save_col_names, pi_variables=pi_variables,
                                            cohort_number=cohort_number, dataset_folder=dataset_folder)


def mimic_analyses(dataset, cohort_number=2.2,  m4_unique=False):
    """
    Calls the Function **run_entire_admission_duration_tests** to run hypothesis tests on the MIMIC datasets.
    :param cohort_number: the cohort to be extracted (represented by the number). Default here is 2.2.
    :param dataset: name of the dataset being analyzed - either mimic-iii or mimic-iv.
    :param m4_unique: flag to check whether patients who are only in mimic-iv (and not in mimic-iii) should be
    extracted and analyzed.
    :return: Nothing. The results of the hypothesis tests are written out into files.
    """
    if dataset == "mimic-iii":
        dataset_folder = "./mimic-iii"
        race_col_name = "ethnicity"
        insurance_categories = [["private", "self pay"], ["government"], ["medicare"], ["medicaid"]]
        insurance_save_labels = ["private-yes-#", "private-yes-%", "private-no-#", "private-no-%",
                                 "government-yes-#", "government-yes-%", "government-no-#", "government-no-%",
                                 "medicare-yes-#", "medicare-yes-%", "medicare-no-#", "medicare-no-%",
                                 "medicaid-yes-#", "medicaid-yes-%", "medicaid-no-#", "medicaid-no-%"]
    elif dataset == "mimic-iv":
        dataset_folder = "./mimic-iv"
        race_col_name = "race"
        insurance_categories = [["medicare"], ["medicaid"], ["other"]]
        insurance_save_labels = ["medicare-yes-#", "medicare-yes-%", "medicare-no-#", "medicare-no-%",
                                 "medicaid-yes-#", "medicaid-yes-%", "medicaid-no-#", "medicaid-no-%",
                                 "other-yes-#", "other-yes-%", "other-no-#", "other-no-%"]
    else:
        raise ValueError("expected mimic dataset names is either 'mimic-iii' or 'mimic-iv'")
    ami_df = pd.read_csv(dataset_folder + "/data/processed/features-files/" + dataset + "-ami-patients-features.csv")
    all_ethnicities = ami_df[race_col_name].str.lower().unique().tolist()
    white_ethnicities = ami_df[ami_df[race_col_name].str.contains("white|portuguese", case=False, na=False)][
        race_col_name].dropna().unique().tolist()
    unknown_ethnicity_values = ["unknown/not specified", "patient declined to answer", "unable to obtain", "unknown"]
    black_or_african_american_ethnicities = ami_df[ami_df[race_col_name].str.contains(
        "black|african", case=False, na=False)][race_col_name].dropna().unique().tolist()
    asian_american_ethnicities = ami_df[ami_df[race_col_name].str.contains(
        "asian|pacific islander|native hawaiian", case=False, na=False)][race_col_name].dropna().unique().tolist()
    latino_or_hispanic_american_ethnicities = ami_df[ami_df[race_col_name].str.contains(
        "latino|hispanic|south american", case=False, na=False)][race_col_name].dropna().unique().tolist()
    native_alaskan_ethnicities = ami_df[ami_df[race_col_name].str.contains(
        "alaska|american indian", case=False, na=False)][race_col_name].dropna().unique().tolist()
    other_ethnicities = list(set(all_ethnicities) - set(
        white_ethnicities + black_or_african_american_ethnicities + asian_american_ethnicities +
        latino_or_hispanic_american_ethnicities + native_alaskan_ethnicities + unknown_ethnicity_values))
    non_white_ethnicities = set(all_ethnicities) - set(white_ethnicities + unknown_ethnicity_values)
    pi_column_values = [[["f"], ["m"]], [white_ethnicities, non_white_ethnicities],
                        [white_ethnicities, non_white_ethnicities, unknown_ethnicity_values],
                        [white_ethnicities, black_or_african_american_ethnicities, asian_american_ethnicities,
                         latino_or_hispanic_american_ethnicities,
                         unknown_ethnicity_values + other_ethnicities],
                        insurance_categories]
    pi_column_names = ["gender", race_col_name, race_col_name, race_col_name, "insurance"]
    pi_variables = ["sex", "race-2G", "race-3G", "race-multiG", "insurance"]
    pi_save_col_names = [
        ["female-yes-#", "female-yes-%", "female-no-#", "female-no-%",
         "male-yes-#", "male-yes-%", "male-no-#", "male-no-%"],
        ["caucasian-yes-#", "caucasian-yes-%", "caucasian-no-#", "caucasian-no-%",
         "non-caucasian-yes-#", "non-caucasian-yes-%", "non-caucasian-no-#", "non-caucasian-no-%"],
        ["caucasian-yes-#", "caucasian-yes-%", "caucasian-no-#", "caucasian-no-%",
         "non-caucasian-yes-#", "non-caucasian-yes-%", "non-caucasian-no-#", "non-caucasian-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        ["caucasian-am-yes-#", "caucasian-am-yes-%", "caucasian-am-no-#", "caucasian-am-no-%",
         "african-am-yes-#", "african-am-yes-%", "african-am-no-#", "african-am-no-%",
         "asian-am-yes-#", "asian-am-yes-%", "asian-am-no-#", "asian-am-no-%",
         "hispanic-am-yes-#", "hispanic-am-yes-%", "hispanic-am-no-#", "hispanic-am-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        insurance_save_labels]

    resp = obtain_cohort(cohort_number=cohort_number, dataset_folder=dataset_folder, dataset_name=dataset,
                         m4_unique=m4_unique)
    if isinstance(resp, Exception):
        return ValueError(resp)
    else:
        run_entire_admission_duration_tests(df=resp[0], pi_column_names=pi_column_names,
                                            pi_column_values=pi_column_values, pi_save_col_names=pi_save_col_names,
                                            pi_variables=pi_variables,
                                            cohort_number=cohort_number, dataset_folder=dataset_folder,
                                            m4_unique=m4_unique)


def obtain_cohort(cohort_number, dataset_folder, dataset_name, m4_unique=False):
    """
    Reads and returns the dataframe associated with cohorts created by Function  **create_sub_cohorts_feature_files**
    in **create_feature_files.py**. This allows for the researcher to run hypothesis tests on different cohorts; e.g.,
    those without comorbidities vs those with.
    :param cohort_number: the number of the cohort to be extracted. Expected values are {1.0, 1.1, 1.2, 1.3, 2.0, 2.1,
    2.2 and 2.3}. More details in Function **create_sub_cohorts_feature_files** inside **create_feature_files.py**
    :param dataset_folder: path to the folder containing the files of the dataset being analyzed
    :param dataset_name: name of the dataset being analyzed
    :param m4_unique: whether to obtain patients in MIMIC-IV that are not in MIMIC-III
    :return: a dataframe of the cohort's file, and the cohort name if successful, otherwise, raises an error
    """
    if dataset_name == "mimic-iv" and m4_unique:
        cohorts_path = dataset_folder + "/data/processed/cohorts/m4-only/" + dataset_name
    else:
        cohorts_path = dataset_folder + "/data/processed/cohorts/" + dataset_name
    if cohort_number == 1.0:
        return pd.read_csv(dataset_folder + "/data/" + dataset_name + "-ami-patients-features.csv"), \
               "c1.0-any-ami-all"  # c1.0
    elif cohort_number == 1.1:
        return pd.read_csv(cohorts_path + "-cohort1.1-any-ami-diagnosis-and-young.csv"), "c1.1-any-ami-young"  # c1.1
    elif cohort_number == 1.2:
        return pd.read_csv(cohorts_path + "-cohort1.2-any-ami-diagnosis-and-healthy.csv"), \
               "c1.2-any-ami-healthy"  # c1.2
    elif cohort_number == 1.3:
        return pd.read_csv(cohorts_path + "-cohort1.3-any-ami-diagnosis-and-healthy-and-young.csv"), \
               "c1.3-any-ami-healthy-and-young"  # c1.3
    elif cohort_number == 2.0:
        return pd.read_csv(cohorts_path + "-cohort2.0-ami-is-primary-diagnosis.csv"), "c2.0-ami-primary-all"  # c2.0
    elif cohort_number == 2.1:
        return pd.read_csv(cohorts_path + "-cohort2.1-ami-is-primary-diagnosis-and-young.csv"), \
               "c2.1-any-primary-young"  # c2.1
    elif cohort_number == 2.2:
        return pd.read_csv(cohorts_path + "-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv"), \
               "c2.2-ami-primary-healthy"  # c2.2
    elif cohort_number == 2.3:
        return pd.read_csv(cohorts_path + "-cohort2.3-ami-is-primary-diagnosis-and-healthy-and-young.csv"), \
               "c2.3-ami-primary-healthy-and-young"  # c2.3
    else:
        return ValueError("The cohort number you entered is wrong, "
                          "check documentation for the acceptable cohort numbers")


def create_analysis_df():
    """
    This function puts it all together. It iterates through all datasets to obtain the statistically significant
    differences in orders for various treatments by running the required hypothesis tests.
    :return: Nothing. The results of the hypothesis tests are written out into files.
    """
    aumc_analyses()
    eicu_analyses()
    mimic_analyses(dataset="mimic-iii")
    mimic_analyses(dataset="mimic-iv", m4_unique=True)


if __name__ == '__main__':
    create_analysis_df()
