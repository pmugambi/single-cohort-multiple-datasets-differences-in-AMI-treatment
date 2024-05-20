import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from scipy.stats import norm, chi2
from analysis import compute_barnard_exact_test_score as barnard_exact
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency


def normalize_contingency_table(ct):
    """

    :param ct:
    :return:
    """
    N = sum(ct)
    # in case there's a zero in the ct, numpy divide would return nan making it difficult to use ct afterwards.
    # for this reason, I'm converting any nan that arise from divide back to zero using np.nan_to_num
    return np.nan_to_num(ct / N)


def convert_pvalues_to_zscores(pvalues, direction="both"):
    """

    :param pvalues:
    :param direction:
    :return:
    """

    # directions obtained from scipy documentation, here,
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html,
    # and this online p-to-z calc: https://planetcalc.com/7803/

    # P.S. The scipy documentation only provides 1 formula, i.e., inv_cdf(pvalue) because Python assumes
    # p-values are obtained from left-tailed analyses. I included the other two to be more accurate

    if direction == "both":
        # z_scores = [NormalDist().inv_cdf(pvalue / 2.) for pvalue in pvalues]
        z_scores = [abs(norm.ppf(pvalue / 2.)) for pvalue in pvalues]
    elif direction == "left":
        # z_scores = [NormalDist().inv_cdf(pvalue) for pvalue in pvalues]
        z_scores = [norm.ppf(pvalue) for pvalue in pvalues]
    elif direction == "right":
        # z_scores = [NormalDist().inv_cdf((1 - pvalue)) for pvalue in pvalues]
        z_scores = [norm.ppf(1 - pvalue) for pvalue in pvalues]
    else:
        return TypeError("The p-value can only be obtained from a left-tailed, right-tailed, or both sided analysis")
    return z_scores


def fishers_method(pvalues, sample_sizes):
    """

    :param pvalues:
    :param sample_sizes:
    :return:
    """
    # check if any p-value is zero and replace it with a value very close to zero
    pvalues = list(pvalues)
    indices = [i for i, x in enumerate(pvalues) if x == 0.0]
    if len(indices) > 0:
        for ind in indices:
            pvalues[ind] = 0.00000001  # 5 decimal places
    uw_statistic, uw_pvalue = combine_pvalues(pvalues=pvalues, method="fisher")

    # weighted version.
    """
     function obtained from eqn 2 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135688/
    """

    weights = [math.sqrt(ss) for ss in sample_sizes]
    w_statistic = 0
    # fuw_stat = 0
    for i in range(len(pvalues)):
        # fuw_stat += math.log(pvalues[i])
        w_statistic += weights[i] * chi2.ppf((1 - pvalues[i]), df=2)
    """
     to obtain the p-value of the computed statistic, I use chi2.sf (i.e., 1-cdf) as explained in this thread: 
     https://stackoverflow.com/questions/11725115/p-value-from-chi-sq-test-statistic-in-python
    """
    w_pvalue = chi2.sf(w_statistic, df=2 * len(pvalues))
    # print("fuw_stat by summation = ", -2 * fuw_stat)
    # print("unweighted statistic = ", uw_statistic, " and unweighted p-value = ", uw_pvalue)
    # print("weighted statistic = ", w_statistic, " and weighted p-value = ", w_pvalue)
    return round(uw_statistic, 4), round(uw_pvalue, 8), round(w_statistic, 4), round(w_pvalue, 8)


def stouffers_method(pvalues, sample_sizes):
    """

    :param pvalues:
    :param sample_sizes:
    :return:
    """
    # check if any p-value is zero and replace it with a value very close to zero
    pvalues = list(pvalues)
    indices = [i for i, x in enumerate(pvalues) if x == 0.0]
    if len(indices) > 0:
        for ind in indices:
            pvalues[ind] = 0.00000001  # 5 decimal places
    """
    # since our p-values are obtained from a 2-sided analysis, I'll divide all p-values by 2 before passing
    # them to the combine method. I'm guided by "... This Z-score is appropriate for one-sided right-tailed p-values;
    # minor modifications can be made if two-sided or left-tailed p-values are being analysed.
    # specifically, if two-sided p-values are being analyzed, the two-sided p-value ***(pi/2)** is used, ..." from
    # https://en.wikipedia.org/wiki/Fisher's_method#Relation_to_Stouffer.27s_Z-score_method
    """
    pvalues = [pvalue / 2.0 for pvalue in pvalues]
    """
    # I am using the squareroot of the sample size as the weight as described here:
    # "... Lipták suggested that the weights in this method “should be chosen proportional to the ‘expected’ difference
    # between the null hypothesis and the real situation and inversely proportional to the standard deviation of 
    # the statistic used in the i-th experiment” and further suggested that when nothing else is available but 
    # the sample sizes of the studies (ni), then the square root of ni can be used as a weight ..."
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3135688/
    """
    weights = [math.sqrt(ss) for ss in sample_sizes]
    w_statistic, w_pvalue = combine_pvalues(pvalues=pvalues, method="stouffer", weights=weights)
    uw_statistic, uw_pvalue = combine_pvalues(pvalues=pvalues, method="stouffer")
    # print("1. - correct -  statistic = ", statistic, " and p-value = ", pvalue)
    return round(uw_statistic, 4), round(uw_pvalue, 8), round(w_statistic, 4), round(w_pvalue, 8),


def generalize_entire_hospitalization_drug_order_results(dataset_folders, cohort_number, pii, pii_group_values):
    """

    :param dataset_folders:
    :param cohort_number:
    :param cohort_name:
    :param pii:
    :return:
    """
    treatments = []
    p_values = []
    sample_sizes = []
    received_treatment_nos = []
    did_not_receive_treatment_nos = []
    groups_treatment_numbers = []

    received_nos = []  # todo: rename
    did_not_receive_nos = []  # todo: rename

    received_nos += [group + "-yes-#" for group in pii_group_values]
    did_not_receive_nos += [group + "-no-#" for group in pii_group_values]

    group_columns = [[group + "-yes-#", group + "-no-#"] for group in pii_group_values]

    for i in range(len(dataset_folders)):
        print("dataset folder = ", dataset_folders[i])
        try:
            cohort_results_folder = dataset_folders[i] + "results/entire-admission-duration/p-values/cohort-" + str(
                cohort_number) + "/"
            p_values_df = pd.read_csv(cohort_results_folder + "c-" + str(cohort_number) +
                                      "-differences-in-proportions-by-" + pii + ".csv")
            columns = ["treatment", "pvalue"] + received_nos + did_not_receive_nos
            p_values_df = p_values_df[columns]
            received_treatment_values = p_values_df[received_nos].values
            did_not_receive_treatment_values = p_values_df[did_not_receive_nos].values

            ss = sum(received_treatment_values[0]) + sum(did_not_receive_treatment_values[0])
            group_values = [p_values_df[group].values.tolist() for group in group_columns]
            dataset_cts = []
            for j in range(len(group_values[0])):
                g_treatment_ct = []
                for x in group_values:
                    g_treatment_ct.append(x[j] / ss)
                dataset_cts.append(g_treatment_ct)
            groups_treatment_numbers.append(dataset_cts)
            treatments.append(p_values_df["treatment"].tolist())
            p_values.append(p_values_df["pvalue"].tolist())
            sample_sizes.append(p_values_df[received_nos + did_not_receive_nos].sum(axis=1).tolist())
            received_treatment_nos.append(received_treatment_values.tolist())
            did_not_receive_treatment_nos.append(did_not_receive_treatment_values.tolist())
        except FileNotFoundError:
            pass
    zipped_treatments = list(zip(*treatments))
    zipped_pvalues = list(zip(*p_values))
    zipped_sample_sizes = list(zip(*sample_sizes))
    zipped_received_treatment = list(zip(*received_treatment_nos))
    zipped_did_not_receive_treatment = list(zip(*did_not_receive_treatment_nos))
    zipped_group_cts = list(zip(*groups_treatment_numbers))

    results = []

    for i in range(len(zipped_treatments)):
        generalized_ct = np.zeros(shape=(len(pii_group_values), 2))
        for k in range(len(zipped_received_treatment[i])):
            for j in range(len(pii_group_values)):
                generalized_ct[j, 0] += zipped_received_treatment[i][k][j]
                generalized_ct[j, 1] += zipped_did_not_receive_treatment[i][k][j]
        if len(pii_group_values) > 2:
            generalized_statistic, generalized_pvalue, _, _ = chi2_contingency(generalized_ct)
        else:
            generalized_statistic, generalized_pvalue = fisher_exact(generalized_ct)
        generalized_norm_ct = np.zeros(shape=(len(pii_group_values), 2))
        for item in zipped_group_cts[i]:
            for j in range(len(pii_group_values)):
                generalized_norm_ct[j, :] += item[j]
        N1_N2 = sum(zipped_sample_sizes[i])
        generalized_norm_ct *= N1_N2
        # i'm using fisher for these because barnard's is very slow, and the differences in outcome
        # between the two tests are minimal
        if len(pii_group_values) > 2:
            generalized_normed_statistic, generalized_normed_pvalue, _, _ = chi2_contingency(
                generalized_norm_ct.astype(int))
        else:
            generalized_normed_statistic, generalized_normed_pvalue = fisher_exact(generalized_norm_ct.astype(int))

        fisher_uw_statistic, fisher_uw_pvalue, fisher_w_statistic, fisher_w_pvalue = fishers_method(
            zipped_pvalues[i], zipped_sample_sizes[i])
        stouffer_uw_statistic, stouffer_uw_pvalue, stouffer_w_statistic, stouffer_w_pvalue = stouffers_method(
            zipped_pvalues[i], zipped_sample_sizes[i])
        # todo: consider adding generalized #s to the table, e.g., % received treatment
        results.append([zipped_treatments[i], zipped_pvalues[i], fisher_uw_statistic, fisher_uw_pvalue,
                        fisher_w_statistic, fisher_w_pvalue, stouffer_uw_statistic, stouffer_uw_pvalue,
                        stouffer_w_statistic, stouffer_w_pvalue, generalized_statistic, generalized_pvalue,
                        generalized_normed_statistic, generalized_normed_pvalue])

    columns = ["treatment", "dataset pvalues", "fisher-uw-statistic", "fisher-uw-pvalue",
               "fisher-w-statistic", "fisher-w-pvalue", "stouffer-uw-statistic", "stouffer-uw-pvalue",
               "stouffer-w-statistic", "stouffer-w-pvalue", "fe-generalized-statistic", "fe-generalized-pvalue",
               "fe-normed-generalized-statistic", "fe-normed-generalized-pvalue"]

    df = pd.DataFrame(data=results, columns=columns)
    print("df.head = ", df.head())
    save_dir = "./general/results/entire-admission-duration/cohort-" + str(cohort_number) + "/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + "c-" + str(cohort_number) + "-generalized-differences-in-proportions-by-" + pii + ".csv")


def generalize_per_day_drug_order_results(dataset_folders, cohort_number, cohort_name, pii):
    """

    :param dataset_folders:
    :param cohort_number:
    :param cohort_name:
    :param pii:
    :return:
    """
    print("dataset folders = ", dataset_folders)
    treatments = ["ace-inhibitor", "analgesic", "anti-platelet", "aspirin", "beta-blocker", "combined-therapy",
                  "non-opioid-only", "opioid", "opioid-only", "statin"]

    for treatment in treatments:
        days = []
        p_values = []
        sample_sizes = []
        print("treatment = ", treatment)
        for i in range(len(dataset_folders)):
            print("dataset folder = ", dataset_folders[i])
            try:
                cohort_results_folder = dataset_folders[i] + "results/per-day/" + pii + "/p-values/cohort-" + str(
                    cohort_number) + "/"
                # print("path = ", cohort_results_folder + "c" + str(cohort_number) + "-" + cohort_name +
                # "-received-" + treatment + ".csv")
                treatment_df = pd.read_csv(cohort_results_folder + "c" + str(cohort_number) + "-" + cohort_name +
                                           "-received-" + treatment + ".csv")
                day_numbers = treatment_df["day-no"].tolist()
                days.append(day_numbers)
                treatment_p_values = treatment_df["pvalue"].tolist()
                print("day-nos = ", day_numbers, " and p-values = ", treatment_p_values)
                # sum the number of samples in the data
                if pii == "sex":
                    sample_sizes.append(treatment_df[["female-yes-#", "female-no-#",
                                                      "male-yes-#", "male-no-#"]].sum(axis=1).tolist())
                else:  # this assumes only sex/race will be passed
                    sample_sizes.append(treatment_df[["non-white-yes-#", "non-white-no-#",
                                                      "white-yes-#", "white-no-#"]].sum(axis=1).tolist())
                p_values.append(treatment_p_values)

            except FileNotFoundError:
                pass

        zipped_day_nos = list(zip(*days))
        zipped_pvalues = list(zip(*p_values))
        zipped_sample_sizes = list(zip(*sample_sizes))
        results = []
        for i in range(len(zipped_day_nos)):
            fisher_uw_statistic, fisher_uw_pvalue, fisher_w_statistic, fisher_w_pvalue = fishers_method(
                zipped_pvalues[i], zipped_sample_sizes[i])
            stouffer_uw_statistic, stouffer_uw_pvalue, stouffer_w_statistic, stouffer_w_pvalue = stouffers_method(
                zipped_pvalues[i], zipped_sample_sizes[i])
            results.append([zipped_day_nos[i], zipped_pvalues[i], fisher_uw_statistic, fisher_uw_pvalue,
                            fisher_w_statistic, fisher_w_pvalue, stouffer_uw_statistic, stouffer_uw_pvalue,
                            stouffer_w_statistic, stouffer_w_pvalue])
        df = pd.DataFrame(data=results,
                          columns=["day-no", "dataset pvalues", "fisher-uw-statistic", "fisher-uw-pvalue",
                                   "fisher-w-statistic", "fisher-w-pvalue", "stouffer-uw-statistic",
                                   "stouffer-uw-pvalue", "stouffer-w-statistic", "stouffer-w-pvalue"])
        print("df.head = ", df.head())
        save_dir = "./general/results/per-day/" + pii + "/cohort-" + str(cohort_number) + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir + "c" + str(
            cohort_number) + "-" + cohort_name + "-received-" + treatment + ".csv")


def generate_entire_hospitalization_results():
    """

    :return:
    """
    # cohort_numbers = [1.0, 2.2, 2.3]
    cohort_numbers = [2.2]
    # cohort_names = ["any-ami-all", "ami-primary-healthy", "ami-primary-healthy-and-young"]
    cohort_names = ["ami-primary-healthy"]
    # data_folders = ["./aumc/data/", "./eicu/data/", "./mimic-iii/data/"]
    data_folders = ["./eicu/data/", "./mimic-iii/data/"]
    piis = ["sex", "race-2G", "race-3G", "race-multiG"]
    pii_column_values = [["female", "male"], ["caucasian", "non-caucasian"], ["caucasian", "non-caucasian", "unknown"],
                         # ["caucasian-am", "african-am", "asian-am", "hispanic-am", "native-am", "unknown"],
                         ["caucasian-am", "african-am", "asian-am", "hispanic-am", "unknown"]]

    for i in range(len(cohort_numbers)):
        cohort_number = cohort_numbers[i]
        for j in range(len(piis)):
            print("cohort number = ", cohort_number, " and pii = ", piis[j])
            generalize_entire_hospitalization_drug_order_results(dataset_folders=data_folders,
                                                                 cohort_number=cohort_number,
                                                                 pii=piis[j], pii_group_values=pii_column_values[j])


def generate_per_day_results():
    """

    :return:
    """
    cohort_numbers = [1.0, 2.2, 2.3]
    cohort_names = ["any-ami-all", "ami-primary-healthy", "ami-primary-healthy-and-young"]
    data_folders = ["./aumc/data/", "./eicu/data/", "./mimic-iii/data/"]
    piis = ["sex", "race"]

    for i in range(len(cohort_numbers)):
        cohort_number = cohort_numbers[i]
        cohort_name = cohort_names[i]
        for pii in piis:
            print("cohort number = ", cohort_number, " and pii = ", pii)
            generalize_per_day_drug_order_results(dataset_folders=data_folders,
                                                  cohort_number=cohort_number,
                                                  cohort_name=cohort_name, pii=pii)


if __name__ == '__main__':
    generate_entire_hospitalization_results()
    # generalize_entire_hospitalization_drug_order_results(dataset_folders=folders, cohort_number=cohort_no,
    #                                                      cohort_name=cohort_nm, pii=pii)
    # generalize_per_day_drug_order_results(dataset_folders=folders, cohort_number=cohort_no, cohort_name=cohort_nm,
    # pii=pii)
