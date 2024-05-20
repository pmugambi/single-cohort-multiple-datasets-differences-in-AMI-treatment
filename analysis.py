import re

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import stats, barnard_exact
from pathlib import Path


# import sys
# import os
# from os import listdir
# from os.path import isfile, join


# generate contigency tables - DONE
# compute exact tests - DONE
# compute t-tests
# todo: run outcome association models
# -- for healthy and separately all AMI patients do: DONE
#       obtain differences in proportions for entire duration
#       obtain differences in proportions for first x days
# todo: may be run analyses for only 2 days?


def create_contingency_table(mini_df, treatment_col, group_col, groups_values):  # , group1, group2):
    """

    :param mini_df:
    :param treatment_col:
    :param group_col:
    :param groups_values:
    :return:
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
    
    :param contingency_table:
    :return:
    """
    res = barnard_exact(contingency_table)
    return res.statistic, res.pvalue


def compute_chi2_contingency_score(contingency_table):
    """

    :param contingency_table:
    :return:
    """
    statistic, p_value, dof, expected_freq = chi2_contingency(contingency_table)
    return statistic, p_value


def compute_t_test(g1_values, g2_values):
    """

    :param g1_values:
    :param g2_values:
    :return:
    """
    statistic, pvalue = stats.ttest_ind(a=g1_values, b=g2_values, equal_var=False)
    return statistic, pvalue


def test_entire_duration(df, treatment_col, pi_variable, groups_values):  # g1_values, g2_values, group_values):
    """

    :param df:
    :param treatment_col:
    :param pi_variable:
    :param groups_values: list of
    :return:
    """
    treatment_mini_df = df[[pi_variable, treatment_col]]
    table = create_contingency_table(mini_df=treatment_mini_df, treatment_col=treatment_col,
                                     group_col=pi_variable, groups_values=groups_values)
    # print("table = ", table)
    if len(table) == 2:
        statistic, p_value = compute_barnard_exact_test_score(contingency_table=table)
    else:
        statistic, p_value = compute_chi2_contingency_score(table)
    return table, round(statistic, 4), round(p_value, 4)


def test_daily(df, treatment_cols, pi_variable, treatment_name, g1_values, g2_values, los_col):  # todo: rethink this
    """

    :param df:
    :param treatment_cols:
    :param pi_variable:
    :param g1_values:
    :param g2_values:
    :return:
    """
    treatment_per_day_mini_df = df[[pi_variable, los_col] + treatment_cols]
    pvalues = []
    statistics = []
    tables = []
    g1_r = 0
    g1_dnr = 0
    g2_r = 0
    g2_dnr = 0
    table_sums = []
    for i in range(1, len(treatment_cols) + 1):
        print("i = ", i)
        print(treatment_per_day_mini_df.head()[los_col].values.tolist())
        # only filter patients who were admitted until day i
        treatment_per_day_mini_df = treatment_per_day_mini_df[treatment_per_day_mini_df[los_col] >= i]
        table = create_contingency_table(mini_df=treatment_per_day_mini_df,
                                         treatment_col=treatment_name + "-d" + str(i) + "?",
                                         group_col=pi_variable,
                                         group1=g1_values,
                                         group2=g2_values)
        print("table = ", table)
        treatment_day_statistic, treatment_day_pvalue = compute_barnard_exact_test_score(contingency_table=table)
        pvalues.append(round(treatment_day_pvalue, 4))
        statistics.append(round(treatment_day_statistic, 3))
        d_g1_r_raw = table[0][0]
        d_g1_dnr_raw = table[0][1]
        if sum(table[0]) > 0:
            d_g1_r_percentage = round((table[0][0]) / sum(table[0]), 2)
            d_g1_dnr_percentage = round((table[0][1]) / sum(table[0]), 2)
        else:
            d_g1_r_percentage = 0
            d_g1_dnr_percentage = 0
        d_g2_r_raw = table[1][0]
        d_g2_dnr_raw = table[1][1]
        if sum(table[1]) > 0:
            d_g2_r_percentage = round((table[1][0]) / sum(table[1]), 2)
            d_g2_dnr_percentage = round((table[1][1]) / sum(table[1]), 2)
        else:
            d_g2_r_percentage = 0
            d_g2_dnr_percentage = 0
        tables.append([d_g1_r_raw, d_g1_r_percentage, d_g1_dnr_raw, d_g1_dnr_percentage, d_g2_r_raw, d_g2_r_percentage,
                       d_g2_dnr_raw, d_g2_dnr_percentage])
        table_sums.append(sum(table[0]) + sum(table[1]))
        g1_r += d_g1_r_raw
        g1_dnr += d_g1_dnr_raw
        g2_r += d_g2_r_raw
        g2_dnr += d_g2_dnr_raw
    if max(table_sums) < 300:  # checking that the number of individuals per day are small so as to aggregate sums
        total_statistic, total_pvalue = compute_barnard_exact_test_score([[g1_r, g1_dnr], [g2_r, g2_dnr]])
        pvalues.append(round(total_pvalue, 4))
        statistics.append(round(total_statistic, 3))
        if g1_r + g1_dnr > 0:
            g1_r_percentage = round((g1_r / (g1_r + g1_dnr)), 2)
            g1_dnr_percentage = round((g1_dnr / (g1_r + g1_dnr)), 2)
        else:
            g1_r_percentage = 0
            g1_dnr_percentage = 0
        if (g2_r + g2_dnr) > 0:
            g2_r_percentage = round((g2_r / (g2_r + g2_dnr)), 2)
            g2_dnr_percentage = round((g2_dnr / (g2_r + g2_dnr)), 2)
        else:
            g2_r_percentage = 0
            g2_dnr_percentage = 0
        # tables.append([g1_r, round((g1_r / (g1_r + g1_dnr)), 2), g1_dnr, round((g1_dnr / (g1_r + g1_dnr)), 2),
        #                g2_r, round((g2_r / (g2_r + g2_dnr)), 2), g2_dnr, round((g2_dnr / (g2_r + g2_dnr)), 2)])
        tables.append([g1_r, g1_r_percentage, g1_dnr, g1_dnr_percentage,
                       g2_r, g2_r_percentage, g2_dnr, g2_dnr_percentage])
    return tables, pvalues, statistics


def run_entire_admission_duration_tests(df, pi_column_names, pi_column_values, pi_save_col_names,
                                        pi_variables, cohort_number, dataset_folder, m4_unique=False):
    """

    :param df:
    :param pi_column_names:
    :param pi_column_values:
    :param pi_save_col_names:
    :param pi_variables:
    :param cohort_number:
    :param dataset_folder:
    :param m4_unique:
    :return:
    """
    treatment_cols = ["received-analgesic?", "received-opioid?", "received-non-opioid?", "received-opioids-only?",
                      "received-non-opioids-only?", "received-combined-therapy?", "received-ace-inhibitor?",
                      "received-aspirin?", "received-beta-blocker?", "received-anti-platelet?", "received-statin?"]
    results = []
    for i in range(len(pi_column_names)):
        print("pi = ", pi_column_names[i])
        pi_results = []
        for treatment in treatment_cols:
            print("treatment = ", treatment)
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
            save_dir = dataset_folder + "/data/results/entire-admission-duration/p-values/cohort-" + str(
                cohort_number) + "/m4-only/"
        else:
            save_dir = dataset_folder + "/data/results/entire-admission-duration/p-values/cohort-" + str(
                cohort_number) + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir + "c-" + str(cohort_number) + "-differences-in-proportions-by-" + pi_variables[i] + ".csv")


def run_per_day_treatment_tests(df, pi_column_names, pi_column_values, los_col_name, pi_save_col_names, pi_variables,
                                cohort, cohort_number, dataset_folder, number_of_days=5):
    """

    :param df:
    :param pi_column_names:
    :param pi_column_values:
    :param los_col_name:
    :param pi_save_col_names:
    :param pi_variables:
    :param cohort:
    :param cohort_number:
    :param dataset_folder:
    :param number_of_days:
    :return:
    """
    treatment_names = ["analgesic", "opioid", "opioid-only", "non-opioid-only", "combined-therapy",
                       "ace-inhibitor", "aspirin", "anti-platelet", "beta-blocker", "statin"]
    for pi in range(len(pi_column_names)):
        for treatment in treatment_names:
            treatment_results = []
            treatment_cols = [treatment + "-d" + str(day_number) + "?"
                              for day_number in range(1, number_of_days + 1)]
            print("treatment cols = ", treatment_cols)

            treatment_tables, treatment_pvalues, treatment_statistics = test_daily(
                df=df,
                treatment_cols=treatment_cols,
                treatment_name=treatment,
                pi_variable=pi_column_names[pi],
                g1_values=pi_column_values[pi][0],
                g2_values=pi_column_values[pi][1],
                los_col=los_col_name)
            for i in range(len(treatment_tables)):
                if (len(treatment_tables) > number_of_days) & (i == len(treatment_tables) - 1):
                    day_no = "all"
                else:
                    day_no = i + 1
                treatment_results.append([day_no, *treatment_tables[i], treatment_pvalues[i], treatment_statistics[i]])
            treatment_df = pd.DataFrame(data=treatment_results, columns=[["day-no"] + pi_save_col_names[pi] +
                                                                         ["pvalue", "statistic"]])
            save_dir = dataset_folder + "/data/results/per-day/" + pi_variables[
                pi] + "/p-values/cohort-" + cohort_number
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            treatment_df.to_csv(save_dir + "/" + cohort + "-received-" + treatment + ".csv")


def test_doses_or_orders(df, pii_variables, group_col_names, pi_column_values, dataset_folder, cohort_number,
                         cohort_name, number_of_days=2):
    """

    :return:
    """

    def obtain_tt_results(t_col, g1, g2, group_col_name):
        g1_df = df[df[group_col_name].isin(g1)]
        g2_df = df[df[group_col_name].isin(g2)]
        g1_orders_or_doses = g1_df[g1_df[t_col] > 0][t_col].dropna().values.tolist()  # todo: confirm this with advisors
        g2_orders_or_doses = g2_df[g2_df[t_col] > 0][t_col].dropna().values.tolist()
        # print("len of g1 orders = ", len(g1_orders_or_doses))
        g1_mean = np.round(np.mean(g1_orders_or_doses), 2)
        g2_mean = np.round(np.mean(g2_orders_or_doses), 2)
        tt_statistic, tt_pvalue = compute_t_test(g1_values=g1_orders_or_doses, g2_values=g2_orders_or_doses)
        # print("g1 mean = ", g1_mean, " and g2 mean = ", g2_mean, " amd tt-statistic = ",
        #       tt_statistic, " and p-value = ", tt_pvalue)
        results = [len(g1_df), len(g1_orders_or_doses), g1_mean, len(g2_df), len(g2_orders_or_doses), g2_mean,
                   round(tt_pvalue, 3), round(tt_statistic, 3)]
        return results

    treatments = ["opioids", "ace-inhibitors", "aspirin", "anti-platelets", "beta-blockers", "statins"]
    treatment_abbreviations = ["opi", "ace", "asp", "naa", "bb", "st"]
    for i in range(len(pii_variables)):
        if pii_variables[i] == "sex":
            group_values = ["female", "male"]
        else:  # dangerous. I'm assuming that only sex/race can be passed.
            group_values = ["white", "non-white"]

        group1 = pi_column_values[i][0]
        group2 = pi_column_values[i][1]
        g1_v = group_values[0]
        g2_v = group_values[1]

        # entire duration
        ed_results = []
        for t in range(len(treatments)):
            if treatments[t] == "opioids":
                ed_col = "total-mme"
                treatment_cols = ["d" + str(d + 1) + "-mme" for d in range(number_of_days)]
            elif treatments[t] == "aspirin":
                ed_col = "total-aspirin-dose(mg)"
                treatment_cols = ["d" + str(d + 1) + "-aspirin-dose(mg)" for d in range(number_of_days)]
            else:
                ed_col = "total-" + treatments[t] + "-orders"
                treatment_cols = ["d" + str(d + 1) + "-" + treatments[t] + "-orders" for d in range(
                    number_of_days)]
            ed_results.append([treatments[t]] + obtain_tt_results(t_col=ed_col, g1=group1,
                                                                  g2=group2, group_col_name=group_col_names[i]))
            pd_results = []
            for treatment_col in treatment_cols:
                treatment_results = obtain_tt_results(t_col=treatment_col, g1=group1, g2=group2,
                                                      group_col_name=group_col_names[i])
                pd_results.append([treatment_col] + treatment_results)
            t_df = pd.DataFrame(data=pd_results, columns=["day-no", g1_v + "-total-#", g1_v + "-t?-yes",
                                                          g1_v + "-mean", g2_v + "-total-#", g2_v + "-t?-yes",
                                                          g2_v + "-mean", "p-value", "statistic"])
            # print("per day treatment df for treatment ", treatments[t], "= ", t_df.head())
            save_dir = dataset_folder + "/data/results/per-day/" + pii_variables[
                i] + "/p-values/cohort-" + cohort_number + "/dosage/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            t_df.to_csv(save_dir + cohort_name + "-doses-or-orders-of-" + treatments[t] + ".csv")

        ed_results_df = pd.DataFrame(data=ed_results, columns=["treatment", g1_v + "-total-#", g1_v + "-t?-yes",
                                                               g1_v + "-mean", g2_v + "-total-#", g2_v + "-t?-yes",
                                                               g2_v + "-mean", "p-value", "statistic"])
        # print(ed_results_df.head())
        save_dir = dataset_folder + "/data/results/entire-admission-duration/p-values/cohort-" + cohort_number + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ed_results_df.to_csv(save_dir + cohort_name + "-" + "differences-in-doses-or-orders-by-"
                             + pii_variables[i] + ".csv")


def run_analyses(cohort_df, pi_column_names, pi_column_values, pi_save_col_names, pi_variables, cohort_name,
                 cohort_number, dataset_folder, m4_unique=False):
    run_entire_admission_duration_tests(df=cohort_df,
                                        pi_column_names=pi_column_names,
                                        pi_column_values=pi_column_values,
                                        pi_save_col_names=pi_save_col_names,
                                        pi_variables=pi_variables,
                                        dataset_folder=dataset_folder,
                                        cohort_number=cohort_number, m4_unique=m4_unique)
    #     # run_per_day_treatment_tests(df=cohort_df,
    #     #                             pi_column_names=pi_column_names,
    #     #                             pi_column_values=pi_column_values,
    #     #                             pi_save_col_names=pi_save_col_names,
    #     #                             pi_variables=pi_variables,
    #     #                             dataset_folder=dataset_folders[i],
    #     #                             cohort=cohort_name,
    #     #                             los_col_name=los_col_names[i],
    #     #                             cohort_number=cohort_number)
    #     # test_doses_or_orders(df=cohort_df, group_col_names=pi_column_names, pi_column_values=pi_column_values,
    #     #                      pii_variables=pi_variables, dataset_folder=dataset_folders[i],
    #     #                      cohort_name=cohort_name, cohort_number=cohort_number)


def aumc_analyses(cohort_number):
    dataset_folder = "./aumc"
    pi_variables = ["sex"]
    pi_column_names = ["gender"]
    pi_column_values = [[["vrouw"], ["man"]]]
    pi_save_col_names = [["female-yes-#", "female-yes-%", "female-no-#", "female-no-%", "male-yes-#",
                          "male-yes-%", "male-no-#", "male-no-%"]]
    if cohort_number != 1.0:
        return ValueError("This dataset can only be analyzed for cohort 1.0")
    else:
        cohort_df = pd.read_csv(dataset_folder + "/data/aumc-ami-patients-features.csv")
        run_analyses(cohort_df=cohort_df, pi_column_names=pi_column_names, pi_column_values=pi_column_values,
                     pi_save_col_names=pi_save_col_names, pi_variables=pi_variables, cohort_name="c1.0-any-ami-all",
                     cohort_number=cohort_number, dataset_folder=dataset_folder)


def eicu_analyses(cohort_number):
    dataset_folder = "./eicu"
    dataset = "eicu"
    ami_df = pd.read_csv(dataset_folder + "/data/eicu-ami-patients-features.csv")

    # there are some records with blank ethnicity. i'm including them with the unknown
    unknown_ethnicity_values = ["other/unknown", np.nan]
    non_caucasian_ethnicities = ami_df[~ami_df["ethnicity"].isin(
        ["caucasian"] + unknown_ethnicity_values)]["ethnicity"].unique().tolist()
    print("eicu - non caucasian ethnicities = ", non_caucasian_ethnicities)
    pi_column_names = ["gender", "ethnicity", "ethnicity", "ethnicity", "region"]
    pi_column_values = [[["female"], ["male"]],
                        [["caucasian"], non_caucasian_ethnicities],
                        [["caucasian"], non_caucasian_ethnicities, unknown_ethnicity_values],
                        # [["caucasian"], ["african american"], ["asian"], ["hispanic"], ["native american"],
                        #  unknown_ethnicity_values],
                        [["caucasian"], ["african american"], ["asian"], ["hispanic"],
                         unknown_ethnicity_values],
                        [["northeast"], ["midwest"], ["west"], ["south"], [np.nan]]]

    pi_variables = ["sex", "race-2G", "race-3G", "race-multiG", "region"]
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
         # "native-am-yes-#", "native-am-yes-%", "native-am-no-#", "native-am-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        ["northeast-yes-#", "northeast-yes-%", "northeast-no-#", "northeast-no-%",
         "midwest-yes-#", "midwest-yes-%", "midwest-no-#", "midwest-no-%",
         "west-yes-#", "west-yes-%", "west-no-#", "west-no-%",
         "south-yes-#", "south-yes-%", "south-no-#", "south-no-%",
         "unspecified-yes-#", "unspecified-yes-%", "unspecified-no-#", "unspecified-no-%"]]

    region_values = ami_df["region"].unique().tolist()
    print("regions = ", region_values)

    resp = obtain_cohort(cohort_number=cohort_number, dataset_folder=dataset_folder, dataset_name=dataset)
    if isinstance(resp, Exception):
        return ValueError(resp)
    else:
        run_analyses(cohort_df=resp[0], pi_column_names=pi_column_names, pi_column_values=pi_column_values,
                     pi_save_col_names=pi_save_col_names, pi_variables=pi_variables, cohort_name=resp[1],
                     cohort_number=cohort_number, dataset_folder=dataset_folder)


def mimic_analyses(cohort_number, dataset, m4_unique=False):
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
    ami_df = pd.read_csv(dataset_folder + "/data/" + dataset + "-ami-patients-features.csv")
    all_ethnicities = ami_df[race_col_name].str.lower().unique().tolist()
    print("all ethnicities = ", sorted(all_ethnicities))
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
                        # [white_ethnicities, black_or_african_american_ethnicities, asian_american_ethnicities,
                        #  latino_or_hispanic_american_ethnicities, native_alaskan_ethnicities,
                        #  unknown_ethnicity_values + other_ethnicities],
                        [white_ethnicities, black_or_african_american_ethnicities, asian_american_ethnicities,
                         latino_or_hispanic_american_ethnicities,
                         unknown_ethnicity_values + other_ethnicities],
                        insurance_categories]
    pi_column_names = ["gender", race_col_name, race_col_name, race_col_name, "insurance"]
    pi_variables = ["sex", "race-2G", "race-3G", "race-multiG", "insurance"]
    pi_save_col_names = [  # todo: to cater for dosage save cols, rename this?
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
         # "native-am-yes-#", "native-am-yes-%", "native-am-no-#", "native-am-no-%",
         "unknown-yes-#", "unknown-yes-%", "unknown-no-#", "unknown-no-%"],
        insurance_save_labels]

    insurance_types = ami_df["insurance"].str.lower().unique().tolist()
    print("insurance types = ", insurance_types)

    resp = obtain_cohort(cohort_number=cohort_number, dataset_folder=dataset_folder, dataset_name=dataset,
                         m4_unique=m4_unique)
    if isinstance(resp, Exception):
        return ValueError(resp)
    else:
        run_analyses(cohort_df=resp[0], pi_column_names=pi_column_names, pi_column_values=pi_column_values,
                     pi_save_col_names=pi_save_col_names, pi_variables=pi_variables, cohort_name=resp[1],
                     cohort_number=cohort_number, dataset_folder=dataset_folder, m4_unique=m4_unique)


def obtain_cohort(cohort_number, dataset_folder, dataset_name, m4_unique=False):
    if dataset_name == "mimic-iv" and m4_unique:
        cohorts_path = dataset_folder + "/data/cohorts/m4-only/" + dataset_name
    else:
        cohorts_path = dataset_folder + "/data/cohorts/" + dataset_name
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

    :return:
    """
    # todo: P.S. add a loop here to run analyses on multiple cohorts
    # aumc_analyses(cohort_number=1.0)
    # eicu_analyses(cohort_number=2.2)
    # mimic_analyses(cohort_number=2.2, dataset="mimic-iii")
    # mimic_analyses(cohort_number=2.2, dataset="mimic-iv", m4_unique=False)
    mimic_analyses(cohort_number=2.2, dataset="mimic-iv", m4_unique=True)


if __name__ == '__main__':
    create_analysis_df()
