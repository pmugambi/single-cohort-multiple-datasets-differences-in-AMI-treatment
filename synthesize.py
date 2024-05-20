from pathlib import Path

import numpy as np
import pandas as pd
import operator


def obtain_ss_differences(filename, p_col_name):
    df = pd.read_csv(filename)
    df_with_ss_diffs = df[df[p_col_name] <= 0.05]
    if len(df_with_ss_diffs) > 0:  # if there are some treatments where differences are statistically significant
        # get columns with data on % of patients who received treatment
        cols_of_interest = df_with_ss_diffs.filter(regex='yes-%').columns
        df_of_interest = df_with_ss_diffs[cols_of_interest]
        for i in df_with_ss_diffs.index.tolist():
            treatment = df_with_ss_diffs.loc[i, "treatment"]
            pvalue = df_with_ss_diffs.loc[i, p_col_name]
            # max_value = max(df_of_interest.loc[i])
            # max_value_col_indeces = list(np.where(df_of_interest.loc[i] == max_value)[0])
            # groups_received_most_orders = df_of_interest.columns.tolist()[max_value_col_indeces[0]:max_value_col_indeces[-1]+1]
            print("treatment = ", treatment, " and p-value = ", pvalue)
            # ," groups = ", groups_received_most_orders,
            # " and max_value = ", max_value, max_value_col_indeces)


def obtain_number_of_admissions(filename, discharge_loc_colname, assign_race=False, normalize_count=False):
    df = pd.read_csv(filename)

    if assign_race:
        df["race"] = df["ethnicity"].apply(group_assign_race)

    N = len(df)
    sex_values = df["gender"]
    try:
        race_values = df["race"]
    except KeyError:
        race_values = df["ethnicity"]
    age_group_values = df["agegroup"]
    try:
        insurance_values = df["insurance"]
    except KeyError:
        insurance_values = None
        print("this dataset does not have insurance data")
    try:
        region_values = df["region"]
    except KeyError:
        region_values = None
        print("this dataset does not have region data")
    stemi_N = len(df[df["n-stemi?"] == 0])
    nstemi_N = len(df[df["n-stemi?"] == 1])
    shock_N = len(df[df["shock?"] == 1])
    c_shock_N = len(df[df["c-shock?"] == 1])
    min_los = min(df["los-h(days)"])
    median_los = np.median(df["los-h(days)"])
    mean_los = np.mean(df["los-h(days)"])
    max_los = max(df["los-h(days)"])
    mortality_N = len(df[df[discharge_loc_colname].str.contains("death|died|expired", case=False, na=False)])

    if insurance_values is None:
        dataset_specific_values = region_values
    else:
        dataset_specific_values = insurance_values

    if normalize_count:
        return sex_values.value_counts(normalize=True).mul(100).round(1).to_dict(), \
               race_values.value_counts(normalize=True).mul(100).round(1).to_dict(), \
               age_group_values.value_counts(normalize=True).mul(100).round(1).to_dict(), \
               round((stemi_N / N) * 100, 1), round((nstemi_N / N) * 100, 1), \
               round((shock_N / N) * 100, 1), round((c_shock_N / N) * 100, 1), np.round(mean_los, 1), \
               np.round(median_los, 1), round((mortality_N / N) * 100, 1), \
               dataset_specific_values.value_counts(normalize=True).mul(100).round(1).to_dict()

    return sex_values.value_counts().to_dict(), race_values.value_counts().to_dict(), \
           age_group_values.value_counts().to_dict()
    # , round((stemi_N / len(df)) * 100, 1), \
    #    round((nstemi_N / len(df)) * 100, 1), round((shock_N / len(df)) * 100, 1), \
    #    round((c_shock_N / len(df)) * 100, 1), median_los, round((mortality_N / len(df)) * 100, 1)


def group_assign_race(x):
    """
    Helper function to assign a patient into a racial group.
    :param x: recorded ethnicities
    :return: patient race group
    """
    if ("white" in x) | ("portuguese" in x):
        return "caucasian-american"
    elif ("black" in x) | ("african" in x):
        return "black/african-american"
    elif ("asian" in x) | ("hawaiian" in x) | ("pacific islander" in x):
        return "asian/asian-american"
    elif ("hispanic" in x) | ("latino" in x):
        return "latinx/hispanic-american"
    elif ("american indian" in x) | ("alaska native" in x) | ("alaska" in x):
        return "alaska-native/american-indian"
    # elif ("hawaiian" in x) | ("pacific islander" in x):
    #     return "polynesian/pacific-islander"
    else:
        return "unknown/unspecified"


def count_number_received_treatment(df1, df2, pii):
    if pii == "sex":
        female_received_d1 = df1["female-yes-#"]
        female_received_d2 = df2["female-yes-#"]
        female_dn_receive_d1 = df1["female-no-#"]
        female_dn_receive_d2 = df2["female-no-#"]
        male_received_d1 = df1["male-yes-#"]
        male_received_d2 = df2["male-yes-#"]
        male_dn_receive_d1 = df1["male-no-#"]
        male_dn_receive_d2 = df2["male-no-#"]
        total_female_received_no = female_received_d1 + female_received_d2
        total_female_received_perc = round(
            (total_female_received_no / (total_female_received_no + female_dn_receive_d1 + female_dn_receive_d2)) * 100,
            2)
        total_male_received_no = male_received_d1 + male_received_d2
        total_male_received_perc = round(
            (total_male_received_no / (total_male_received_no + male_dn_receive_d1 + male_dn_receive_d2)) * 100, 2)

        print("total no of female received treatments = ", total_female_received_no, " which is ",
              total_female_received_perc, " %")
        print("total no of male received treatments = ", total_male_received_no, " which is ", total_male_received_perc,
              " %")

        data = list(zip(df1["treatment"].values.tolist(), total_female_received_no.values.tolist(),
                        total_female_received_perc.values.tolist(), total_male_received_no.values.tolist(),
                        total_male_received_perc.values.tolist()))

        df = pd.DataFrame(data=data, columns=["treatment", "total-female-received-#", "total-female-received-%",
                                              "total-male-received-#", "total-male-received-%"])
        print("df.head() = ", df.head(), len(df))
        df.to_csv("./general/results/entire-admission-duration/cohort-2.2/c-2.2-combined-numbers-proportions-by-"
                  + pii + "-that-received-treatments.csv")
        # todo: generate dataframe

    elif pii == "race-multiG":
        caucasian_received_d1 = df1["caucasian-am-yes-#"]
        black_african_am_received_d1 = df1["african-am-yes-#"]
        asian_received_d1 = df1["asian-am-yes-#"]
        hispanic_received_d1 = df1["hispanic-am-yes-#"]
        unspecified_received_d1 = df1["unknown-yes-#"]
        print("mimic unspecified yes # = ", unspecified_received_d1.values.tolist())
        caucasian_dn_receive_d1 = df1["caucasian-am-no-#"]
        black_african_am_dn_receive_d1 = df1["african-am-no-#"]
        asian_dn_receive_d1 = df1["asian-am-no-#"]
        hispanic_dn_receive_d1 = df1["hispanic-am-no-#"]
        unspecified_dn_receive_d1 = df1["unknown-no-#"]
        print("mimic unspecified no # = ", unspecified_dn_receive_d1.values.tolist())
        caucasian_received_d2 = df2["caucasian-am-yes-#"]
        black_african_am_received_d2 = df2["african-am-yes-#"]
        asian_received_d2 = df2["asian-am-yes-#"]
        hispanic_received_d2 = df2["hispanic-am-yes-#"]
        unspecified_received_d2 = df2["unknown-yes-#"]
        print("eicu unspecified yes # = ", unspecified_received_d2.values.tolist())
        caucasian_dn_receive_d2 = df2["caucasian-am-no-#"]
        black_african_am_dn_receive_d2 = df2["african-am-no-#"]
        asian_dn_receive_d2 = df2["asian-am-no-#"]
        hispanic_dn_receive_d2 = df2["hispanic-am-no-#"]
        unspecified_dn_receive_d2 = df2["unknown-no-#"]
        print("eicu unspecified no # = ", unspecified_dn_receive_d2.values.tolist())
        total_caucasian_received_no = caucasian_received_d1 + caucasian_received_d2
        total_caucasian_received_perc = round((total_caucasian_received_no / (
                total_caucasian_received_no + caucasian_dn_receive_d1 + caucasian_dn_receive_d2) * 100), 2)
        total_african_am_received_no = black_african_am_received_d1 + black_african_am_received_d2
        total_african_am_received_perc = round((total_african_am_received_no / (
                total_african_am_received_no + black_african_am_dn_receive_d1 + black_african_am_dn_receive_d2) * 100),
                                               2)
        total_asian_received_no = asian_received_d1 + asian_received_d2
        total_asian_received_perc = round((total_asian_received_no / (
                total_asian_received_no + asian_dn_receive_d1 + asian_dn_receive_d2) * 100), 2)
        total_hispanic_received_no = hispanic_received_d1 + hispanic_received_d2
        total_hispanic_received_perc = round((total_hispanic_received_no / (
                total_hispanic_received_no + hispanic_dn_receive_d1 + hispanic_dn_receive_d2) * 100), 2)
        total_unspecified_received_no = unspecified_received_d1 + unspecified_received_d2
        total_unspecified_received_perc = round((total_unspecified_received_no / (
                total_unspecified_received_no + unspecified_dn_receive_d1 + unspecified_dn_receive_d2) * 100), 2)

        print("total unspecified yes # = ", total_unspecified_received_no)
        print("total unspecified = ",
              total_unspecified_received_no + unspecified_dn_receive_d1 + unspecified_dn_receive_d2)
        print("total unspecified yes % = ", total_unspecified_received_perc.values.tolist())

        data = list(zip(df1["treatment"].values.tolist(), total_caucasian_received_no.values.tolist(),
                        total_caucasian_received_perc.values.tolist(), total_african_am_received_no.values.tolist(),
                        total_african_am_received_perc.values.tolist(), total_asian_received_no.values.tolist(),
                        total_asian_received_perc.values.tolist(), total_hispanic_received_no.values.tolist(),
                        total_hispanic_received_perc.values.tolist(), total_unspecified_received_no.values.tolist(),
                        total_unspecified_received_perc.values.tolist()))
        df = pd.DataFrame(data=data, columns=["treatment", "total-caucasian-received-#", "total-caucasian-received-%",
                                              "total-african-am-received-#", "total-african-am-received-%",
                                              "total-asian-am-received-#", "total-asian-am-received-%",
                                              "total-hispanic-am-received-#", "total-hispanic-am-received-%",
                                              "total-unspecified-am-received-#", "total-unspecified-am-received-%"])

        print("df.head() = ", df.head())
        df.to_csv("./general/results/entire-admission-duration/cohort-2.2/c-2.2-combined-numbers-proportions-by-"
                  + pii + "-that-received-treatments.csv")


def combine_dicts(a, b):
    """

    :param a:
    :param b:
    :return:
    code obtained from:https://stackoverflow.com/a/11011911
    """
    return {x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}


def compute_percentages_for_dict_items(d):
    """

    :param d:
    :return:
    """
    s = sum(d.values())
    res = {}
    for k, v in d.items():
        # pct = v * 100.0 / s
        # print(k, pct)
        res[k] = round(v * 100.0 / s, 1)
    return res


def plot_differences_stacked():
    """

    :return:
    code adopted from https://plotly.com/python/bar-charts/#bar-chart-with-sorted-or-ordered-categories
    could also use code from https://gist.github.com/ctokheim/6435202a1a880cfecd71
    """
    import plotly.graph_objects as go
    eicu_f = "./eicu/data/cohorts/eicu-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv"
    mimic_f = "./mimic-iii/data/cohorts/mimic-iii-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv"

    eicu_norm_sex_counts, eicu_norm_race_counts, eicu_norm_age_group_counts, eicu_stemi_perc, eicu_nstemi_perc, \
    eicu_shock_perc, eicu_c_shock_perc, eicu_mean_los, eicu_median_los, \
    eicu_mortality_perc, eicu_norm_region_counts = obtain_number_of_admissions(
        filename=eicu_f,
        discharge_loc_colname="hospitaldischargelocation",
        normalize_count=True)
    mimic_norm_sex_counts, mimic_norm_race_counts, mimic_norm_age_group_counts, mimic_stemi_perc, mimic_nstemi_perc, \
    mimic_shock_perc, mimic_c_shock_perc, mimic_mean_los, mimic_median_los, \
    mimic_mortality_perc, mimic_norm_insurance_counts = obtain_number_of_admissions(
        filename=mimic_f, assign_race=True,
        discharge_loc_colname="discharge_location",
        normalize_count=True)
    # # print("mimic-sex-norm = ", mimic_norm_sex_counts)
    # # print("mimic-race-norm = ", mimic_norm_race_counts)
    #
    # eicu_sex_counts, eicu_race_counts, eicu_age_group_counts = obtain_number_of_admissions(
    #     filename=eicu_f,
    #     discharge_loc_colname="hospitaldischargelocation")
    # mimic_sex_counts, mimic_race_counts, mimic_age_group_counts = obtain_number_of_admissions(
    #     filename=mimic_f, assign_race=True,
    #     discharge_loc_colname="discharge_location")
    # mimic_sex_counts["male"] = mimic_sex_counts["m"]
    # mimic_sex_counts["female"] = mimic_sex_counts["f"]
    # del mimic_sex_counts["m"], mimic_sex_counts["f"]
    # mimic_race_counts["caucasian"] = mimic_race_counts["caucasian-american"]
    # mimic_race_counts["african american"] = mimic_race_counts["black/african-american"]
    # mimic_race_counts["asian"] = mimic_race_counts["asian/asian-american"]
    # mimic_race_counts["hispanic"] = mimic_race_counts["latinx/hispanic-american"]
    # mimic_race_counts["native american"] = mimic_race_counts["alaska-native/american-indian"]
    # mimic_race_counts["other/unknown"] = mimic_race_counts["unknown/unspecified"]
    # del mimic_race_counts["caucasian-american"], mimic_race_counts["black/african-american"], \
    #     mimic_race_counts["asian/asian-american"], mimic_race_counts["latinx/hispanic-american"], \
    #     mimic_race_counts["unknown/unspecified"], mimic_race_counts["alaska-native/american-indian"]
    # print("mimic-sex-values = ", mimic_sex_counts)
    # print("eicu-sex-values = ", eicu_sex_counts)
    # # print("mimic-race-norm = ", mimic_race_counts)
    #
    # x = [
    #     ["sex", "sex", "race", "race", "race", "race", "race", "race", "age-group", "age-group", "age-group",
    #      "age-group", "age-group", "age-group"],  # todo: increase the font size here
    #     ["male", "female", "Caucasian", "Black", "Asian", "Hispanic", "Native-American", "Unspecified", "18-39",
    #      "40-49", "50-59", "60-69", "70-79", "80+"]
    # ]
    # mimic_ys = [mimic_norm_sex_counts["m"], mimic_norm_sex_counts["f"], mimic_norm_race_counts["caucasian-american"],
    #             mimic_norm_race_counts["black/african-american"], mimic_norm_race_counts["asian/asian-american"],
    #             mimic_norm_race_counts["latinx/hispanic-american"],
    #             mimic_norm_race_counts["alaska-native/american-indian"],
    #             mimic_norm_race_counts["unknown/unspecified"], mimic_norm_age_group_counts["18-39"],
    #             mimic_norm_age_group_counts["40-49"], mimic_norm_age_group_counts["50-59"],
    #             mimic_norm_age_group_counts["60-69"], mimic_norm_age_group_counts["70-79"],
    #             mimic_norm_age_group_counts["80+"]]
    # eicu_ys = [eicu_norm_sex_counts["male"], eicu_norm_sex_counts["female"], eicu_norm_race_counts["caucasian"],
    #            eicu_norm_race_counts["african american"], eicu_norm_race_counts["asian"],
    #            eicu_norm_race_counts["hispanic"], eicu_norm_race_counts["native american"],
    #            eicu_norm_race_counts["other/unknown"], eicu_norm_age_group_counts["18-39"],
    #            eicu_norm_age_group_counts["40-49"], eicu_norm_age_group_counts["50-59"],
    #            eicu_norm_age_group_counts["60-69"], eicu_norm_age_group_counts["70-79"],
    #            eicu_norm_age_group_counts["80+"]]
    #
    # # combine values from mimic and eicu to get the values for combined
    # combined_sex_counts = combine_dicts(mimic_sex_counts, eicu_sex_counts)
    # combined_sex_percentages = compute_percentages_for_dict_items(combined_sex_counts)
    # combined_race_counts = combine_dicts(mimic_race_counts, eicu_race_counts)
    # combined_race_percentages = compute_percentages_for_dict_items(combined_race_counts)
    # combined_age_group_counts = combine_dicts(mimic_age_group_counts, eicu_age_group_counts)
    # combined_age_group_percentages = compute_percentages_for_dict_items(combined_age_group_counts)
    # print("combined_sex_counts = ", combined_sex_counts, combined_sex_percentages)
    # print("combined_race_counts = ", combined_race_counts)
    #
    # combined_ys = [combined_sex_percentages["male"], combined_sex_percentages["female"],
    #                combined_race_percentages["caucasian"],
    #                combined_race_percentages["african american"], combined_race_percentages["asian"],
    #                combined_race_percentages["hispanic"], combined_race_percentages["native american"],
    #                combined_race_percentages["other/unknown"], combined_age_group_percentages["18-39"],
    #                combined_age_group_percentages["40-49"], combined_age_group_percentages["50-59"],
    #                combined_age_group_percentages["60-69"], combined_age_group_percentages["70-79"],
    #                combined_age_group_percentages["80+"]]
    #
    # # fig = go.Figure()
    # # fig.add_bar(x=x, y=mimic_ys, text=[str(x) + "%" for x in mimic_ys],
    # #             name="MIMIC-III (N=2521)")  # todo: consider changing text to include #s
    # # fig.add_bar(x=x, y=eicu_ys, text=[str(x) + "%" for x in eicu_ys], name="eICU (N=644)")
    # # fig.add_bar(x=x, y=combined_ys, text=[str(x) + "%" for x in combined_ys], name="Combined (N=3165)")
    # # fig.update_layout(barmode="relative", title={'text': "Distribution of admissions by sex, race, and age-group",
    # #                                              'y': 0.9,
    # #                                              'x': 0.5,
    # #                                              'xanchor': 'center',
    # #                                              'yanchor': 'top',
    # #                                              'font': dict(size=20)},
    # #                   autosize=False,
    # #                   width=1200,
    # #                   height=600)
    save_dir = "./general/results/entire-admission-duration/plots/cohort-2.2/"
    # # Path(save_dir).mkdir(parents=True, exist_ok=True)
    # # fig.write_image(save_dir + "distribution-of-admissions-by-sex-race-and-age-group.png")
    # # fig.write_image(save_dir + "distribution-of-admissions-by-sex-race-and-age-group.svg")
    # # fig.show()
    # demographics_x = ["Female", "Race <br> unspecified", "Caucasian", "Aged 50+"]
    # mimic_demo_ys = [mimic_norm_sex_counts["f"], mimic_norm_race_counts["unknown/unspecified"],
    #                  mimic_norm_race_counts["caucasian-american"], round(mimic_norm_age_group_counts["50-59"] +
    #                  mimic_norm_age_group_counts["60-69"] + mimic_norm_age_group_counts["70-79"] +
    #                  mimic_norm_age_group_counts["80+"], 1)]
    # eicu_demo_ys = [eicu_norm_sex_counts["female"], eicu_norm_race_counts["other/unknown"],
    #                 eicu_norm_race_counts["caucasian"], round(eicu_norm_age_group_counts["50-59"] +
    #                 eicu_norm_age_group_counts["60-69"] + eicu_norm_age_group_counts["70-79"] +
    #                 eicu_norm_age_group_counts["80+"], 1)]
    # fig = go.Figure()
    # fig.add_bar(x=demographics_x, y=mimic_demo_ys, text=[str(x)+"%" for x in mimic_demo_ys],
    #             name="MIMIC-III (N=2521)", marker_color="cadetblue")
    # fig.add_bar(x=demographics_x, y=eicu_demo_ys, text=[str(x)+"%" for x in eicu_demo_ys],
    #             name="eICU (N=644)", marker_color="peru")
    # fig.update_layout(title={'text': "Distribution of admissions by sex, race, and age-group",
    #                          'y': 0.9,
    #                          'x': 0.5,
    #                          'xanchor': 'center',
    #                          'yanchor': 'top',
    #                          'font': dict(size=20)},
    #                   xaxis=dict(tickfont=dict(size=14)),
    #                   legend=dict(font=dict(size=15)),
    #                   yaxis_title="Count (%)",
    #                   autosize=False,
    #                   width=950,
    #                   height=600)
    # fig.write_image(save_dir + "distribution-of-admissions-by-sex-race-and-age-group-version-2.png")
    # fig.write_image(save_dir + "distribution-of-admissions-by-sex-race-and-age-group-version-2.svg")
    # fig.show()

    # todo: plot one for disease severity markers
    # disease_severity_markers_x = [
    #     ["diagnosis", "diagnosis", "shock", "shock", "mortality rate", "admission duration (days)",
    #      "admission duration (days)"],  # todo: increase the font size here
    #     ["STEMI", "NSTEMI", "any", "cardiogenic", "", "mean", "median"]
    # ]
    # disease_severity_markers_eicu_ys = [eicu_stemi_perc, eicu_nstemi_perc, eicu_shock_perc, eicu_c_shock_perc,
    #                                     eicu_mortality_perc,
    #                                     eicu_mean_los, eicu_median_los]
    # disease_severity_markers_mimic_ys = [mimic_stemi_perc, mimic_nstemi_perc, mimic_shock_perc, mimic_c_shock_perc,
    #                                      mimic_mortality_perc,
    #                                      mimic_mean_los, mimic_median_los]
    # text_add_ons = ["%", "%", "%", "%", "%", "", "", ""]
    # fig2 = go.Figure()
    # fig2.add_bar(x=disease_severity_markers_x, y=disease_severity_markers_mimic_ys,
    #              text=[str(disease_severity_markers_mimic_ys[i]) + text_add_ons[i] for i in range(
    #                  len(disease_severity_markers_mimic_ys))],
    #              name="MIMIC-III (N=2521)")  # todo: consider changing text to include #s
    # fig2.add_bar(x=disease_severity_markers_x, y=disease_severity_markers_eicu_ys,
    #              text=[str(disease_severity_markers_eicu_ys[i]) + text_add_ons[i] for i in range(
    #                  len(disease_severity_markers_eicu_ys))],
    #              name="eICU (N=644)")
    # fig2.update_layout(barmode="relative", title={'text': "Distribution of admissions by disease severity markers",
    #                                               'y': 0.9,
    #                                               'x': 0.5,
    #                                               'xanchor': 'center',
    #                                               'yanchor': 'top',
    #                                               'font': dict(size=20)},
    #                    autosize=False,
    #                    width=1020,
    #                    height=600)
    # fig2.write_image(save_dir + "distribution-of-admissions-by-disease-severity-markers.png")
    # fig2.write_image(save_dir + "distribution-of-admissions-by-disease-severity-markers.svg")
    # fig2.show()

    # clinical_characteristics_x = ["STEMI(%)", "Any <br> shock(%)", "Cardiogenic <br> shock(%)",
    #                               "Mortality <br> rate(%)",
    #                               "Mean <br> LOS(days)"]
    # fig = go.Figure()
    # mimic_cc_ys = [mimic_stemi_perc, mimic_shock_perc, mimic_c_shock_perc, mimic_mortality_perc, mimic_mean_los]
    # eicu_cc_ys = [eicu_stemi_perc, eicu_shock_perc, eicu_c_shock_perc, eicu_mortality_perc, eicu_mean_los]
    # text_add_ons = ["%", "%", "%", "%", ""]
    # fig.add_bar(x=clinical_characteristics_x, y=mimic_cc_ys,
    #             text=[str(mimic_cc_ys[i]) + text_add_ons[i] for i in range(len(mimic_cc_ys))],
    #             name="MIMIC-III (N=2521)", marker_color="cadetblue")
    # fig.add_bar(x=clinical_characteristics_x, y=eicu_cc_ys,
    #             text=[str(eicu_cc_ys[i]) + text_add_ons[i] for i in range(len(eicu_cc_ys))], name="eICU (N=644)",
    #             marker_color="peru")
    # fig.update_layout(title={'text': "Distribution of admissions by disease severity markers",
    #                          'y': 0.9,
    #                          'x': 0.5,
    #                          'xanchor': 'center',
    #                          'yanchor': 'top',
    #                          'font': dict(size=20)},
    #                   xaxis=dict(tickfont=dict(size=14)),
    #                   legend=dict(font=dict(size=15)),
    #                   autosize=False,
    #                   width=950,
    #                   height=600)
    # fig.write_image(save_dir + "distribution-of-admissions-by-disease-severity-markers-version-2.png")
    # fig.write_image(save_dir + "distribution-of-admissions-by-disease-severity-markers-version-2.svg")
    # fig.show()

    # todo: plot one for the individual datasets: consider using smaller single data plots
    # distribution by region
    # import plotly.express as px
    # fig3 = px.bar(x=eicu_norm_region_counts.keys(),
    #               y=eicu_norm_region_counts.values(),
    #               text=[str(x) + "%" for x in eicu_norm_region_counts.values()],
    #               labels={'x': "region", 'y': "admissions count (%)"},
    #               title="eICU (N=644): distribution of patient admissions by region")
    # fig3.write_image(save_dir + "distribution-of-admissions-by-region.png")
    # fig3.write_image(save_dir + "distribution-of-admissions-by-region.svg")
    # fig3.show()
    #
    # # distribution by insurance type
    # fig4 = px.bar(x=mimic_norm_insurance_counts.keys(),
    #               y=mimic_norm_insurance_counts.values(),
    #               text=[str(x) + "%" for x in mimic_norm_insurance_counts.values()],
    #               labels={'x': "insurance type", 'y': "admissions count (%)"},
    #               title="MIMIC-III (N=2521): distribution of patient admissions by type of insurance")
    # fig4.write_image(save_dir + "distribution-of-admissions-by-insurance.png")
    # fig4.write_image(save_dir + "distribution-of-admissions-by-insurance.svg")
    # fig4.show()

    # todo: plot one for differences in treatments, start by: by sex, see if that works (do this first tomorrow morn)
    # I'm looking through the files and generating these numbers
    # by sex
    # treatments_with_ss_differences_by_sex_x = [
    #     ["non-opioid analgesia", "non-opioid analgesia", "multimodal analgesia", "multimodal analgesia",
    #      "ACE-inhibitors", "ACE-inhibitors", "aspirin", "aspirin", "beta-blockers", "beta-blockers",
    #      "statins", "statins"],
    #     ["male", "female", "male", "female", "male", "female", "male", "female", "male", "female", "male", "female"]
    # ]
    # mimic_ss_differences_by_sex_ys = [84, 70, 0, 0, 61, 55, 83, 79, 80, 76, 77, 72]
    # eicu_ss_differences_by_sex_ys = [0, 0, 56, 65, 0, 0, 0, 0, 0, 0, 0, 0]
    # combined_ss_differences_by_sex_ys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    # fig5 = go.Figure()
    # fig5.add_bar(x=treatments_with_ss_differences_by_sex_x, y=mimic_ss_differences_by_sex_ys,
    #              text=[str(x) + "%" for x in mimic_ss_differences_by_sex_ys],
    #              name="MIMIC-III (N=2521)")  # todo: consider changing text to include #s
    # fig5.add_bar(x=treatments_with_ss_differences_by_sex_x, y=eicu_ss_differences_by_sex_ys,
    #              text=[str(x) + "%" for x in eicu_ss_differences_by_sex_ys],
    #              name="eICU (N=644)")
    # fig5.add_bar(x=treatments_with_ss_differences_by_sex_x, y=combined_ss_differences_by_sex_ys,
    #              text=[str(x) + "%" for x in combined_ss_differences_by_sex_ys],
    #              name="Combined (N=3165)")
    # fig5.update_layout(barmode="relative",
    #                    title={'text': "Significant differences in proportions of patients who received "
    #                                   "various treatment by sex",
    #                           'y': 0.9,
    #                           'x': 0.5,
    #                           'xanchor': 'center',
    #                           'yanchor': 'top',
    #                           'font': dict(size=20)},
    #                    autosize=False,
    #                    width=1150,
    #                    height=600)
    # fig5.write_image(save_dir + "significant-differences-in-treatment-by-sex.png")
    # fig5.write_image(save_dir + "significant-differences-in-treatment-by-sex.svg")
    # fig5.show()

    # # by race
    # analgesics_with_ss_differences_by_race_x = [
    #     ["any analgesia", "any analgesia", "any analgesia", "any analgesia", "any analgesia",
    #      "opioid analgesia", "opioid analgesia", "opioid analgesia", "opioid analgesia", "opioid analgesia",
    #      "non-opioid analgesia", "non-opioid analgesia", "non-opioid analgesia", "non-opioid analgesia",
    #      "non-opioid analgesia",
    #      "multimodal analgesia", "multimodal analgesia", "multimodal analgesia", "multimodal analgesia",
    #      "multimodal analgesia"],
    #     ["Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified"]
    # ]
    # ami_drugs_with_ss_differences_by_race_x = [
    #     ["ACE inhibitors", "ACE inhibitors", "ACE inhibitors", "ACE inhibitors", "ACE inhibitors",
    #      "aspirin", "aspirin", "aspirin", "aspirin", "aspirin",
    #      "beta-blockers", "beta-blockers", "beta-blockers", "beta-blockers", "beta-blockers",
    #      "non-aspirin antiplatelet", "non-aspirin antiplatelet", "non-aspirin antiplatelet", "non-aspirin antiplatelet",
    #      "non-aspirin antiplatelet",
    #      "statins", "statins", "statins", "statins", "statins"],
    #     ["Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified",
    #      "Caucasian", "Black", "Asian", "Hispanic", "Unspecified"]
    # ]
    # mimic_ss_differences_in_analgesics_by_race_ys = [87, 84, 78, 86, 73,
    #                                                  64, 57, 57, 64, 54,
    #                                                  87, 83, 78, 86, 73,
    #                                                  64, 56, 57, 64, 54]
    # mimic_ss_differences_in_ami_drugs_by_race_ys = [61, 56, 61, 48, 54,
    #                                                 86, 82, 74, 81, 72,
    #                                                 83, 77, 78, 86, 68,
    #                                                 68, 68, 70, 79, 58,
    #                                                 80, 79, 74, 74, 65]
    # eicu_ss_differences_in_analgesics_by_race_ys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # eicu_ss_differences_in_ami_drugs_by_race_ys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # combined_ss_differences_in_analgesics_by_race_ys = [85.1, 82.1, 71.1, 80.3, 72.3,
    #                                                     63.0, 58.1, 57.9, 62.3, 53.9,
    #                                                     84.9, 81.2, 71.1, 80.3, 72.1,
    #                                                     62.7, 57.3, 57.9, 62.3, 53.7]
    # combined_ss_differences_in_ami_drugs_by_race_ys = [53.7, 43.6, 39.5, 44.3, 51.5,
    #                                                    81.4, 74.4, 65.8, 77.1, 71.4,
    #                                                    79.0, 74.4, 65.8, 83.6, 67.1,
    #                                                    0, 0, 0, 0, 0,
    #                                                    71.0, 65.8, 60.5, 67.2, 63.4]
    #
    # # fig6 = go.Figure()
    # # fig6.add_bar(x=analgesics_with_ss_differences_by_race_x, y=mimic_ss_differences_in_analgesics_by_race_ys,
    # #              text=[str(x) + "%" for x in mimic_ss_differences_in_analgesics_by_race_ys],
    # #              name="MIMIC-III (N=2521)")  # todo: consider changing text to include #s
    # # fig6.add_bar(x=analgesics_with_ss_differences_by_race_x, y=eicu_ss_differences_in_analgesics_by_race_ys,
    # #              text=[str(x) + "%" for x in eicu_ss_differences_in_analgesics_by_race_ys],
    # #              name="eICU (N=644)")
    # # fig6.add_bar(x=analgesics_with_ss_differences_by_race_x, y=combined_ss_differences_in_analgesics_by_race_ys,
    # #              text=[str(x) + "%" for x in combined_ss_differences_in_analgesics_by_race_ys],
    # #              name="Combined (N=3165)")
    # # fig6.update_layout(barmode="relative",
    # #                    title={'text': "Significant differences in proportions of patients who received "
    # #                                   "analgesia by race",
    # #                           'y': 0.9,
    # #                           'x': 0.5,
    # #                           'xanchor': 'center',
    # #                           'yanchor': 'top',
    # #                           'font': dict(size=20)},
    # #                    autosize=False,
    # #                    width=1150,
    # #                    height=600)
    # # fig6.write_image(save_dir + "significant-differences-in-analgesia-by-race.png")
    # # fig6.write_image(save_dir + "significant-differences-in-analgesia-by-race.svg")
    # # fig6.show()
    #
    # # fig7 = go.Figure()
    # # fig7.add_bar(x=ami_drugs_with_ss_differences_by_race_x, y=mimic_ss_differences_in_ami_drugs_by_race_ys,
    # #              text=[str(x) + "%" for x in mimic_ss_differences_in_ami_drugs_by_race_ys],
    # #              name="MIMIC-III (N=2521)")  # todo: consider changing text to include #s
    # # fig7.add_bar(x=ami_drugs_with_ss_differences_by_race_x, y=eicu_ss_differences_in_ami_drugs_by_race_ys,
    # #              text=[str(x) + "%" for x in eicu_ss_differences_in_ami_drugs_by_race_ys],
    # #              name="eICU (N=644)")
    # # fig7.add_bar(x=ami_drugs_with_ss_differences_by_race_x, y=combined_ss_differences_in_ami_drugs_by_race_ys,
    # #              text=[str(x) + "%" for x in combined_ss_differences_in_ami_drugs_by_race_ys],
    # #              name="Combined (N=3165)")
    # # fig7.update_layout(barmode="relative",
    # #                    title={'text': "Significant differences in proportions of patients who received "
    # #                                   "AMI-related drugs by race",
    # #                           'y': 0.9,
    # #                           'x': 0.5,
    # #                           'xanchor': 'center',
    # #                           'yanchor': 'top',
    # #                           'font': dict(size=20)},
    # #                    autosize=False,
    # #                    width=1150,
    # #                    height=600)
    # # fig7.write_image(save_dir + "significant-differences-in-ami-drugs-by-race.png")
    # # fig7.write_image(save_dir + "significant-differences-in-ami-drugs-by-race.svg")
    # # fig7.show()
    #
    # # todo: plot differences by region and insurance type
    # # by region
    treatments_with_ss_differences_by_region_x = [
        # ["any analgesia", "any analgesia", "any analgesia", "any analgesia", "any analgesia",
        #  "opioid analgesia", "opioid analgesia", "opioid analgesia", "opioid analgesia", "opioid analgesia",
        #  "non-opioid analgesia", "non-opioid analgesia", "non-opioid analgesia", "non-opioid analgesia",
        #  "non-opioid analgesia",
        #  "non-opioid-only analgesia", "non-opioid-only analgesia", "non-opioid-only analgesia",
        #  "non-opioid-only analgesia", "non-opioid-only analgesia",
        #  "multimodal analgesia", "multimodal analgesia", "multimodal analgesia", "multimodal analgesia",
        #  "multimodal analgesia",
        ["ACE-inhibitors", "ACE-inhibitors", "ACE-inhibitors", "ACE-inhibitors",  # "ACE-inhibitors",
         "aspirin", "aspirin", "aspirin", "aspirin",  # "aspirin",
         "non-aspirin antiplatelets", "non-aspirin antiplatelets", "non-aspirin antiplatelets",
         "non-aspirin antiplatelets",  # "non-aspirin antiplatelets",
         "statins", "statins", "statins", "statins"],  # "statins"],
        # ["NorthEast", "MidWest", "West", "South", "Unspecified",
        #  "NorthEast", "MidWest", "West", "South", "Unspecified",
        #  "NorthEast", "MidWest", "West", "South", "Unspecified",
        #  "NorthEast", "MidWest", "West", "South", "Unspecified",
        #  "NorthEast", "MidWest", "West", "South", "Unspecified",
        ["NorthEast", "MidWest", "West", "South",  # "Unspecified",
         "NorthEast", "MidWest", "West", "South",  # "Unspecified",
         "NorthEast", "MidWest", "West", "South",  # "Unspecified",
         "NorthEast", "MidWest", "West", "South"]  # , "Unspecified"]
    ]
    eicu_ss_differences__by_region_ys = [
        # 92, 92, 77, 62, 70,
        # 51, 67, 66, 42, 66,
        # 92, 91, 77, 61, 70,
        # 41, 25, 10, 20, 4,
        # 51, 67, 66, 41, 66,
        8, 41, 28, 18,  # 42,
        84, 79, 73, 45,  # 66,
        22, 50, 39, 23,  # 26,
        30, 69, 41, 31  # , 30
    ]

    fig8 = go.Figure()
    fig8.add_bar(x=treatments_with_ss_differences_by_region_x, y=eicu_ss_differences__by_region_ys,
                 text=[str(x) + "%" for x in eicu_ss_differences__by_region_ys],
                 marker_color=["cadetblue", "peru", "DarkSlateGrey", "darkolivegreen"] * 4,  # darkcyan
                 name="eICU (N=644)")
    fig8.update_layout(barmode="relative",
                       title={'text': "Significant differences in proportions of patients who received "
                                      "AMI drugs by region",
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top',
                              'font': dict(size=20)},
                       xaxis=dict(tickfont=dict(size=14)),
                       # legend=dict(font=dict(size=15)),
                       autosize=False,
                       width=1000,
                       height=600)
    # fig8.write_image(save_dir + "significant-differences-in-AMI-drugs-treatment-by-region.png")
    # fig8.write_image(save_dir + "significant-differences-in-AMI-drugs-treatment-by-region.svg")
    # fig8.show()
    #
    # by insurance type
    treatments_with_ss_differences_by_insurance_x = [
        ["ACE-inhibitors", "ACE-inhibitors", "ACE-inhibitors", "ACE-inhibitors",
         "non-aspirin antiplatelets", "non-aspirin antiplatelets", "non-aspirin antiplatelets",
         "non-aspirin antiplatelets"],
        ["private", "government", "medicare", "medicaid",
         "private", "government", "medicare", "medicaid"]
    ]
    mimic_ss_differences_in_treatment_by_insurance_type_ys = [
        62, 74, 56, 58,
        68, 70, 63, 69
    ]
    fig9 = go.Figure()
    fig9.add_bar(x=treatments_with_ss_differences_by_insurance_x,
                 y=mimic_ss_differences_in_treatment_by_insurance_type_ys,
                 text=[str(x) + "%" for x in mimic_ss_differences_in_treatment_by_insurance_type_ys],
                 marker_color=["cadetblue", "peru", "DarkSlateGrey", "darkcyan"] * 2,
                 name="MIMIC (N=2521)")
    fig9.update_layout(barmode="relative",
                       title={'text': "Significant differences in proportions of patients who received "
                                      "treatment by insurance-type",
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top',
                              'font': dict(size=20)},
                       xaxis=dict(tickfont=dict(size=14)),
                       autosize=False,
                       width=800,
                       height=650)
    fig9.write_image(save_dir + "significant-differences-in-treatment-by-insurance.png")
    fig9.write_image(save_dir + "significant-differences-in-treatment-by-insurance.svg")
    fig9.show()
    # # todo: after, work on slides DONE


def obtain_significant_differences():
    mimic_sex = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-" \
                "differences-in-proportions-by-sex.csv"
    mimic_race_multigroup = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-" \
                            "differences-in-proportions-by-race-multiG.csv"
    mimic_insurance = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
                      "-proportions-by-insurance.csv"
    eicu_sex = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in-proportions" \
               "-by-sex.csv"
    eicu_race_multigroup = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
                           "-proportions-by-race-multiG.csv"
    eicu_region = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
                  "-proportions-by-region.csv"
    combined_sex = "./general/results/entire-admission-duration/cohort-2.2/c-2.2-combined-numbers" \
                   "-proportions-by-sex-that-received-treatment.csv"
    combined_race_multigroup = "./general/results/entire-admission-duration/cohort-2.2/c-2.2-combined-numbers" \
                               "-proportions-by-race-multiG-that-received-treatment.csv"
    # by sex
    # treatments with signific


# def plot_differences_grouped():

# eicu_region = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in-proportions-by" \
#               "-region.csv"
# eicu_sex = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in-proportions-by-sex" \
#            ".csv"
# eicu_race_multigroup = "./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
#                        "-proportions-by-race-multiG.csv"
# # obtain_ss_differences(filename=eicu_region, p_col_name="pvalue")
# # obtain_ss_differences(filename=eicu_sex, p_col_name="pvalue")
# # obtain_ss_differences(filename=eicu_race_multigroup, p_col_name="pvalue")
#
# mimic_sex = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in-proportions" \
#             "-by-sex.csv"
# mimic_race_multigroup = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
#                         "-proportions-by-race-multiG.csv"
# mimic_insurance = "./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in" \
#                   "-proportions-by-insurance.csv"
#
# # obtain_ss_differences(filename=mimic_sex, p_col_name="pvalue")
# # obtain_ss_differences(filename=mimic_race_multigroup, p_col_name="pvalue")
# # obtain_ss_differences(filename=mimic_insurance, p_col_name="pvalue")
#
# combined_race_multigroup = "./general/results/entire-admission-duration/cohort-2.2/c-2.2-generalized-differences-in" \
#                            "-proportions-by-race-multiG.csv"
# combined_sex = "./general/results/entire-admission-duration/cohort-2.2/c-2.2-generalized-differences-in-proportions" \
#                "-by-sex.csv"
# # obtain_ss_differences(filename=combined_race_multigroup, p_col_name="fe-generalized-pvalue")
# # obtain_ss_differences(filename=combined_sex, p_col_name="fe-generalized-pvalue")
#
# # get distribution of admissions
# eicu_f = "./eicu/data/cohorts/eicu-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv"
# mimic_f = "./mimic-iii/data/cohorts/mimic-iii-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv"

# a = obtain_number_of_admissions(filename=eicu_f, discharge_loc_colname="hospitaldischargelocation", pii="sex",
#                                 normalize_count=True)
# print("a = ", a)
# # obtain_number_of_admissions(filename=mimic_f, discharge_loc_colname="discharge_location", assign_race=True)
#
# eicu_sex_df = pd.read_csv("./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences-in"
#                           "-proportions-by-sex.csv")
# eicu_race_multiG_df = pd.read_csv("./eicu/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2"
#                                   "-differences-in-proportions-by-race-multiG.csv")
# # mimic_sex_df = pd.read_csv("./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2-differences"
# #                            "-in-proportions-by-sex.csv")
# mimic_race_multiG_df = pd.read_csv("./mimic-iii/data/results/entire-admission-duration/p-values/cohort-2.2/c-2.2"
#                                    "-differences-in-proportions-by-race-multiG.csv")
#
# # # count_number_received_treatment(df1=mimic_sex_df, df2=eicu_sex_df, pii="sex")
# count_number_received_treatment(df1=mimic_race_multiG_df, df2=eicu_race_multiG_df, pii="race-multiG")

plot_differences_stacked()
