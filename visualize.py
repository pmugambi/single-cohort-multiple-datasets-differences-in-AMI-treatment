import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from os import listdir
from os.path import isfile, join, isdir
from pathlib import Path


def add_text(axes, x, y, text):
    for i in range(len(x)):
        axes.text(x[i], y[i], text[i], c="blue")


def plot_two_categories(values, dataset_data_folder, pii_category, x_labels, treatment, x_axis_label, bar_labels,
                        cohort_name, write_path):
    treatments, privileged_group_percents_received, unprivileged_group_percents_received, p_values = values[0:]
    fig, axes = plt.subplots()
    barWidth = 0.25

    fig.set_figwidth(10)

    r1 = np.arange(len(treatments))
    r2 = [x + barWidth for x in r1]

    rects1 = axes.bar(r1, privileged_group_percents_received, color='r', width=barWidth, edgecolor='white',
                      label=bar_labels["privileged"])
    rects2 = axes.bar(r2, unprivileged_group_percents_received, color='g', width=barWidth, edgecolor='white',
                      label=bar_labels["unprivileged"])

    def autolabel(rects, y_val=3):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            text = height
            axes.annotate('{}'.format(text),
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, y_val),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    axes.set_xlabel(x_axis_label, fontweight='bold')
    x_ticks = [r + barWidth for r in range(len(treatments))]
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_labels)

    max_group_percents = [max(i) for i in zip(privileged_group_percents_received, unprivileged_group_percents_received)]

    if max(max_group_percents) < 60:
        text_y_ind = [i + 4 for i in max_group_percents]
    elif 60 <= max(max_group_percents) <= 80:
        text_y_ind = [i + 6 for i in max_group_percents]
    else:
        text_y_ind = [i + 8 for i in max_group_percents]  # todo: adjust this based on max height
    text_x_ind = [i for i in r1]

    add_text(axes=axes, x=text_x_ind, y=text_y_ind, text=["p=" + str(x) for x in p_values])

    axes.legend()
    if max(max_group_percents) > 90:
        plt.ylim(top=max(max_group_percents) + 28)
    elif 65 <= max(max_group_percents) <= 90:
        plt.ylim(top=max(max_group_percents) + 23)
    else:
        plt.ylim(top=max(max_group_percents) + 15)
    axes.set_ylabel("Patient proportion (%)", fontweight="bold")
    axes.set_title("Percentage of patient, by " + pii_category + ", prescribed " + treatment)
    save_dir = dataset_data_folder + "/results/" + write_path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir + cohort_name + "-percentage-prescribed-" + treatment + "-by-"+pii_category+".png")
    fig.savefig(save_dir + cohort_name + "-percentage-prescribed-" + treatment + "-by-"+pii_category+".pdf")
    # plt.show()
    plt.close()


def visualize_entire_hospitalization_results():
    """

    :return:
    """
    dataset_folders = ["./aumc", "./eicu", "./mimic-iii"]
    datasets = ["aumc", "eicu", "mimic"]
    two_category_pi_values = {"sex": {"privileged": "male", "unprivileged": "female"},
                              "race": {"privileged": "white", "unprivileged": "non-white"}}
    analgesic_treatment_values = ["received-analgesic?", "received-opioid?", "received-non-opioid?",
                                  "received-opioids-only?", "received-non-opioids-only?", "received-combined-therapy?"]
    ami_drugs_treatment_values = ["received-ace-inhibitor?", "received-aspirin?", "received-anti-platelet?",
                                  "received-beta-blocker?", "received-statin?"]
    labels = [["some-analgesic", "some-opioid", "some-non-opioid", "opioids-only", "non-opioids-only",
               "combined-therapy"],
              ["ace-inhibitor", "aspirin", "non-aspirin-antiplatelet", "beta-blocker",
               "statin"]]  # todo: change the ordering when writing the file, and then make sure it matches here
    x_axis_labels = ["Analgesic therapy", "AMI drug"]
    treatment_categories = ["analgesic therapies", "AMI drugs"]

    for i in range(len(dataset_folders)):
        print("dataset = ", datasets[i])
        try:
            results_path = dataset_folders[i] + "/data/results/entire-admission-duration/p-values/"
            directories = [d for d in listdir(results_path) if isdir(join(results_path, d))]
            for d in directories:
                d_results_path = results_path + d
                d_results_filenames = [f for f in listdir(d_results_path) if isfile(join(d_results_path, f)) &
                                       f.endswith(".csv")]
                for filename in d_results_filenames:
                    print("filename = ", filename)
                    filename_pi = re.search(r'by-(.*?).csv', filename).group(1)
                    print("filename_pi = ", filename_pi)
                    cohort_number = re.search(r'c(.*?)-', filename).group(1)
                    cohort_name = re.search(r'.\d-(.*?)-differences-', filename).group(1)
                    df = pd.read_csv(d_results_path + "/" + filename)
                    analgesics_df = df[df["treatment"].isin(analgesic_treatment_values)]
                    ami_drugs_df = df[df["treatment"].isin(ami_drugs_treatment_values)]
                    dfs = [analgesics_df, ami_drugs_df]
                    for j in range(len(dfs)):
                        treatments = dfs[j]["treatment"].tolist()
                        if filename_pi == "sex":
                            privileged_percents_received = dfs[j]["male-yes-%"].tolist()
                            unprivileged_percents_received = dfs[j]["female-yes-%"].tolist()
                        elif filename_pi == "race":
                            privileged_percents_received = dfs[j]["white-yes-%"].tolist()
                            unprivileged_percents_received = dfs[j]["non-white-yes-%"].tolist()
                        else:
                            return ValueError  # there are only sex and race piis being examined

                        p_values = dfs[j]["pvalue"].round(decimals=4).tolist()
                        values = [treatments, [round(x * 100, 2) for x in privileged_percents_received],
                                  [round(x * 100, 2) for x in unprivileged_percents_received],
                                  p_values]
                        plot_two_categories(values=values, dataset_data_folder=dataset_folders[i] + "/data",
                                            pii_category=filename_pi, x_labels=labels[j],
                                            treatment=treatment_categories[j], x_axis_label=x_axis_labels[j],
                                            bar_labels=two_category_pi_values[filename_pi],
                                            cohort_name="c" + cohort_number + "-" + cohort_name,
                                            write_path="entire-admission-duration/plots/cohort-" + cohort_number + "/")

        except FileNotFoundError as e:
            print("checking for file not found error ", e)
            pass


def visualize_per_day_results():
    """

    :return:
    """
    dataset_folders = ["./aumc", "./eicu", "./mimic-iii"]
    datasets = ["aumc", "eicu", "mimic"]
    two_category_pi_values = {"sex": {"privileged": "male", "unprivileged": "female"},
                              "race": {"privileged": "white", "unprivileged": "non-white"}}
    pi_variables = ["sex", "race"]
    for i in range(len(dataset_folders)):
        for pi in pi_variables:
            print("dataset = ", datasets[i], " and pi = ", pi)
            try:
                results_path = dataset_folders[i] + "/data/results/per-day/" + pi + "/p-values/"
                directories = [d for d in listdir(results_path) if isdir(join(results_path, d))]
                for d in directories:
                    d_results_path = results_path + d
                    d_results_filenames = [f for f in listdir(d_results_path) if isfile(join(d_results_path, f))]
                    for filename in d_results_filenames:
                        # extract treatment name
                        treatment_name = re.search(r'-received-(.*?).csv', filename).group(1)
                        cohort_number = re.search(r'c(.*?)-', filename).group(1)
                        cohort_name = re.search(r'.\d-(.*?)-received-', filename).group(1)
                        df = pd.read_csv(d_results_path + "/" + filename)
                        day_nos = df["day-no"].tolist()
                        if pi == "sex":
                            privileged_percents_received = df["male-yes-%"].tolist()
                            unprivileged_percents_received = df["female-yes-%"].tolist()
                        elif pi == "race":
                            privileged_percents_received = df["white-yes-%"].tolist()
                            unprivileged_percents_received = df["non-white-yes-%"].tolist()
                        else:
                            return ValueError  # should only take sex, race values for pii
                        p_values = df["pvalue"].round(decimals=4).tolist()
                        values = [day_nos, [round(x * 100, 2) for x in privileged_percents_received],
                                  [round(x * 100, 2) for x in unprivileged_percents_received],
                                  p_values]
                        plot_two_categories(values=values, dataset_data_folder=dataset_folders[i] + "/data/",
                                            pii_category=pi,
                                            x_labels=day_nos, treatment=treatment_name,
                                            x_axis_label="day-no",
                                            bar_labels=two_category_pi_values[pi],
                                            write_path="per-day/" + pi + "/plots/cohort-" + cohort_number + "/",
                                            cohort_name="c" + cohort_number + "-" + cohort_name)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    visualize_entire_hospitalization_results()
    visualize_per_day_results()
