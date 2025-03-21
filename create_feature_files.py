from pathlib import Path

import pandas as pd
import numpy as np
import datetime as dt

from os import listdir
from os.path import isfile, join


def process_mimic_iii_basics():
    """
    Function generates the required features for the MIMIC-III dataset.
    It reads the unprocessed master file of patients with a diagnosis of AMI, created by **obtain_ami_cohorts**
    function in **create_cohort.py** and adds features such as 'age', 'age-group', 'whether patient has comorbidities',
    and 'whether the patient had diagnoses of shock'
    :return: a dataframe of the patients with only the required features/variables
    """
    ami_cohort_df = pd.read_csv("mimic-iii/data/processed/features-files/mimic-iii-ami-patients-master-unprocessed.csv")
    ami_cohort_df[["ADMITTIME-f", "DISCHTIME-f", "DOB-f", "DOD_HOSP-f", "DOD_SSN-f"]] = ami_cohort_df[
        ["ADMITTIME", "DISCHTIME", "DOB", "DOD_HOSP", "DOD_SSN"]].apply(pd.to_datetime)
    ami_cohort_df["age"] = ((ami_cohort_df["ADMITTIME-f"].dt.to_pydatetime() - ami_cohort_df["DOB-f"].dt.
                             to_pydatetime()) / dt.timedelta(days=365)).astype(int)
    ami_cohort_df["los-h(days)"] = (ami_cohort_df["DISCHTIME-f"].dt.to_pydatetime() - ami_cohort_df["ADMITTIME-f"].dt.
                                    to_pydatetime()) / dt.timedelta(days=1)
    ami_cohort_df["died-in-h?"] = np.where(ami_cohort_df["DISCHARGE_LOCATION"].str.contains(
        "death|died|dead", case=False, na=False), 1, 0)
    ami_cohort_df["died-after-d?"] = np.where((~pd.isnull(ami_cohort_df["DOD_SSN"])) &
                                              (ami_cohort_df["died-in-h?"] == 0), 1, 0)
    # MAD stands for mortality after discharge
    ami_cohort_df["mad(days)"] = (ami_cohort_df["DOD_SSN-f"].dt.to_pydatetime() - ami_cohort_df["DISCHTIME-f"].dt.
                                  to_pydatetime()) / dt.timedelta(days=1)

    # diagnosis features, i.e., N/STEMI, shock, and cardiogenic shock and prior medical history
    ami_cohort_df["n-stemi?"] = np.where(ami_cohort_df["ICD9_CODE"].str.contains("4107", na=False), 1, 0)
    diagnoses_df = pd.read_csv("./mimic-iii/data/raw/DIAGNOSES_ICD.csv.gz")
    diagnoses_icd9code_df = pd.read_csv("./mimic-iii/data/raw/D_ICD_DIAGNOSES.csv.gz")
    shock_icd9_codes = diagnoses_icd9code_df[diagnoses_icd9code_df["LONG_TITLE"].str.contains(
        "shock", case=False, na=False)]["ICD9_CODE"].unique().tolist()
    shock_df = diagnoses_df[diagnoses_df["ICD9_CODE"].isin(shock_icd9_codes)]
    c_shock_df = diagnoses_df[diagnoses_df["ICD9_CODE"].str.contains("78551", na=False)]
    ami_cohort_df["shock?"] = np.where(ami_cohort_df["HADM_ID"].isin(shock_df["HADM_ID"].unique().tolist()), 1, 0)
    ami_cohort_df["c-shock?"] = np.where(ami_cohort_df["HADM_ID"].isin(c_shock_df["HADM_ID"].unique().tolist()), 1, 0)
    ami_cohort_df["has-comorbidity?"] = np.where(~pd.isnull(ami_cohort_df["VALUE"]), 1, 0)
    ami_cohort_df["comorbidity"] = ami_cohort_df["VALUE"]
    ami_cohort_df["agegroup"] = ami_cohort_df["age"].apply(assign_age_group)

    features = ["SUBJECT_ID", "HADM_ID", "age", "agegroup", "GENDER", "los-h(days)", "DISCHARGE_LOCATION",
                "INSURANCE", "ETHNICITY", "died-after-d?", "mad(days)", "ICD9_CODE", "SEQ_NUM",
                "n-stemi?", "shock?", "c-shock?", "has-comorbidity?", "comorbidity"]
    ami_cohort_features_df = ami_cohort_df[features]
    return ami_cohort_features_df


def process_mimic_iv_basics():
    """
    Function generates the required features for the MIMIC-IV dataset.
    It reads the unprocessed master file of patients with a diagnosis of AMI, created by **obtain_ami_cohorts**
    function in **create_cohort.py** and adds features such as 'age', 'age-group', 'whether patient has comorbidities',
    and 'whether the patient had diagnoses of shock'
    :return: a dataframe of the patients with only the required features/variables
    """
    ami_cohort_df = pd.read_csv("mimic-iv/data/processed/features-files/mimic-iv-ami-patients-master-unprocessed.csv")
    ami_cohort_df[["admittime-f", "dischtime-f", "dod-f"]] = ami_cohort_df[
        ["admittime", "dischtime", "dod"]].apply(pd.to_datetime)
    ami_cohort_df["los-h(days)"] = (ami_cohort_df["dischtime-f"].dt.to_pydatetime() - ami_cohort_df["admittime-f"].dt.
                                    to_pydatetime()) / dt.timedelta(days=1)
    ami_cohort_df["died-in-h?"] = np.where(ami_cohort_df["discharge_location"].str.contains(
        "death|died|dead", case=False, na=False), 1, 0)
    ami_cohort_df["died-after-d?"] = np.where((~pd.isnull(ami_cohort_df["dod"])) &
                                              (ami_cohort_df["died-in-h?"] == 0), 1, 0)
    # MAD stands for mortality after discharge
    ami_cohort_df["mad(days)"] = (ami_cohort_df["dod-f"].dt.to_pydatetime() - ami_cohort_df["dischtime-f"].dt.
                                  to_pydatetime()) / dt.timedelta(days=1)

    # diagnosis features, i.e., N/STEMI, shock, and cardiogenic shock and prior medical history
    ami_cohort_df["n-stemi?"] = np.where(ami_cohort_df["icd_code"].str.contains("4107", na=False), 1, 0)
    diagnoses_df = pd.read_csv("./mimic-iv/data/raw/hosp/diagnoses_icd.csv.gz")
    diagnoses_icdcode_df = pd.read_csv("./mimic-iv/data/raw/hosp/d_icd_diagnoses.csv.gz")
    shock_icd_codes = diagnoses_icdcode_df[diagnoses_icdcode_df["long_title"].str.contains(
        "shock", case=False, na=False)]["icd_code"].unique().tolist()  # todo: edit, too broad (includes shock therapy)
    shock_df = diagnoses_df[diagnoses_df["icd_code"].isin(shock_icd_codes)]
    c_shock_df = diagnoses_df[((diagnoses_df["icd_version"] == 9) & (diagnoses_df["icd_code"].str.contains(
        "78551", na=False))) | ((diagnoses_df["icd_version"] == 10) & (diagnoses_df["icd_code"].str.contains(
        "R570", na=False)))]
    ami_cohort_df["shock?"] = np.where(ami_cohort_df["hadm_id"].isin(shock_df["hadm_id"].unique().tolist()), 1, 0)
    ami_cohort_df["c-shock?"] = np.where(ami_cohort_df["hadm_id"].isin(c_shock_df["hadm_id"].unique().tolist()), 1, 0)
    ami_cohort_df["has-comorbidity?"] = np.where(~pd.isnull(ami_cohort_df["value"]), 1, 0)
    ami_cohort_df["comorbidity"] = ami_cohort_df["value"]
    ami_cohort_df["agegroup"] = ami_cohort_df["anchor_age"].apply(assign_age_group)

    features = ["subject_id", "hadm_id", "anchor_age", "agegroup", "gender", "los-h(days)", "discharge_location",
                "insurance", "race", "died-after-d?", "mad(days)", "icd_code", "seq_num",
                "n-stemi?", "shock?", "c-shock?", "has-comorbidity?", "comorbidity", "anchor_year",
                "anchor_year_group"]
    ami_cohort_features_df = ami_cohort_df[features]
    return ami_cohort_features_df


def process_eicu_basics():
    """
    Function generates the required features for the eICU dataset.
    It reads the unprocessed master file of patients with a diagnosis of AMI, created by **obtain_ami_cohorts**
    function in **create_cohort.py** and adds features such as 'age', 'age-group', 'whether patient has comorbidities',
    and 'whether the patient had diagnoses of shock'
    :return: a dataframe of the patients with only the required features/variables
    """

    eicu_ami_cohort_df = pd.read_csv("eicu/data/processed/features-files/eicu-ami-patients-master-unprocessed.csv")
    eicu_ami_cohort_df["los-h(days)"] = round((eicu_ami_cohort_df["hospitaldischargeoffset"] -
                                               eicu_ami_cohort_df["hospitaladmitoffset"]) / 1440, 2)
    eicu_ami_cohort_df["los-icu(days)"] = round((eicu_ami_cohort_df["unitdischargeoffset"] -
                                                 eicu_ami_cohort_df["hospitaladmitoffset"]) / 1440, 2)
    eicu_hospitals_df = pd.read_csv(
        "./eicu/data/raw/hospital.csv.gz")[["hospitalid", "teachingstatus", "region"]]
    eicu_ami_cohort_df = pd.merge(left=eicu_ami_cohort_df, right=eicu_hospitals_df, how="left", on="hospitalid")

    # rename "teachingstatus" to "hosp-teachingstatus"
    eicu_ami_cohort_df = eicu_ami_cohort_df.rename(columns={"teachingstatus": "hospitalteachingstatus"})

    eicu_ami_cohort_df["died-in-h?"] = np.where(
        eicu_ami_cohort_df["hospitaldischargelocation"].str.contains("death", case=False, na=False), 1, 0)

    # diagnosis features, i.e., N/STEMI, shock, and cardiogenic shock and prior medical history
    eicu_ami_cohort_df["n-stemi?"] = np.where(eicu_ami_cohort_df["icd9code"].str.contains("410.7", na=False), 1, 0)
    diagnoses_df = pd.read_csv("./eicu/data/raw/diagnosis.csv.gz")
    shock_df = diagnoses_df[diagnoses_df["diagnosisstring"].str.contains("shock", case=False, na=False)]
    c_shock_df = diagnoses_df[diagnoses_df["icd9code"].str.contains("785.51", na=False)]

    eicu_ami_cohort_df["shock?"] = np.where(eicu_ami_cohort_df["patientunitstayid"].isin(
        shock_df["patientunitstayid"].unique().tolist()), 1, 0)
    eicu_ami_cohort_df["c-shock?"] = np.where(eicu_ami_cohort_df["patientunitstayid"].isin(
        c_shock_df["patientunitstayid"].unique().tolist()), 1, 0)
    eicu_ami_cohort_df["has-comorbidity?"] = np.where(eicu_ami_cohort_df["pasthistoryvaluetext"].str.contains(
        "nohealthproblems", case=False, na=False), 0, 1)
    eicu_ami_cohort_df["comorbidity"] = np.where(eicu_ami_cohort_df["pasthistoryvaluetext"].str.contains(
        "nohealthproblems", case=False, na=False), "", eicu_ami_cohort_df["pasthistoryvaluetext"])

    eicu_ami_cohort_df["age-mod"] = np.where(eicu_ami_cohort_df["age"] == "> 89", 90,
                                             eicu_ami_cohort_df["age"])  # is this problematic? possible!
    eicu_ami_cohort_df["agegroup"] = eicu_ami_cohort_df["age-mod"].apply(assign_age_group)  # this helps with the
    # possible problem in the line above, because, everybody 80yo and above is put into 1 group

    features = ["uniquepid", "patientunitstayid", "age", "agegroup", "gender", "ethnicity", "n-stemi?", "shock?",
                "c-shock?", "hospitaldischargelocation", "admissionweight", "dischargeweight", "diagnosispriority",
                "los-h(days)", "los-icu(days)", "region", "hospitalteachingstatus", "died-in-h?", "has-comorbidity?",
                "comorbidity"]

    eicu_ami_cohort_features_df = eicu_ami_cohort_df[features]
    return eicu_ami_cohort_features_df


def process_aumc_basics():
    """
    Function generates the required features for the AmsterdamUMCdb dataset.
    It reads the unprocessed master file of patients with a diagnosis of AMI, created by **obtain_ami_cohorts**
    function in **create_cohort.py** and adds features such as 'whether patient died during hospitalization' and
    'length of ICU stay'. Unlike MIMIC and eICU datasets, this dataset does not have features such as
    'whether the patient had a diagnosis of shock' because the data was unavailable.
    :return: a dataframe of the patients with only the required features/variables

    :return:
    """
    aumc_ami_cohort_df = pd.read_csv("aumc/data/processed/features-files/aumc-ami-patients-master-unprocessed.csv")
    aumc_ami_cohort_df["died-in-hosp?"] = np.where(aumc_ami_cohort_df["destination"] == "Overleden", 1, 0)
    aumc_ami_cohort_df["los-icu(days)"] = round(aumc_ami_cohort_df["lengthofstay"] / 24, 2)

    # STEMI: # ANTERIOR, INFEROLATERAL, NON Q Wave = STEMI, none of the above (probably?) = STEMI as I understand it
    aumc_ami_cohort_df["stemi?"] = np.where(aumc_ami_cohort_df["diagnosis"].str.contains(
        r"ANTERIOR|INFEROLATERAL|NON Q Wave|none of the above", case=False, na=False), 1, 0)

    features = ["patientid", "admissionid", "gender", "agegroup", "weightgroup",
                "heightgroup", "lengthofstay", "los-icu(days)", "died-in-hosp?", "specialty",
                "diagnosis-category", "diagnosis", "stemi?", "location", "urgency"]
    aumc_ami_cohort_features_df = aumc_ami_cohort_df[features]
    return aumc_ami_cohort_features_df


def add_drug_features(basic_features_df, dataset_data_folder, dataset, admission_col_name, process_per_day=False):
    """
    Function combines the basic features created by functions above (process_mimic_III_basics, process_mimic_iv_basics,
    process_eicu_basics, and process_aumc_basics) with the medication order features,
    created by **process_medications** function in **process_medication_orders.py** file
    :param basic_features_df: dataframe containing the 'basic' features for each dataset
    :param dataset_data_folder: path to the folder containing the files of the dataset being processed
    :param dataset: the name of the dataset being processed
    :param admission_col_name: the column in 'basic_features_df' containing the patients' admission ids.
    important when merging dataframes
    :param process_per_day: whether the per-day medication order features should be added or not. Default it NO.
    :return: Nothing. Output files containing 'basic' and 'medication order' features are written to CSV files and
    saved in 'dataset_data_folder'
    """
    read_path_eh = dataset_data_folder + "prescription-orders/entire-hospitalization/"
    read_path_pd = dataset_data_folder + "prescription-orders/per-day/"
    analgesics_df = pd.read_csv(read_path_eh + dataset + "-received-analgesics-during-hospitalization.csv")
    ami_drugs_df = pd.read_csv(read_path_eh + dataset + "-received-ami-drugs-during-hospitalization.csv")

    features_df = pd.merge(left=basic_features_df, right=analgesics_df, how="left", on=admission_col_name)
    features_df = pd.merge(left=features_df, right=ami_drugs_df, how="left", on=admission_col_name)

    if process_per_day:
        treatments_per_day_df = pd.read_csv(read_path_pd + dataset + "-received-drug-treatments-per-day.csv")
        features_df = pd.merge(left=features_df, right=treatments_per_day_df, how="left", on=admission_col_name)

    features_df = features_df.applymap(lambda s: s.lower() if type(s) == str else s)
    features_df.columns = [s.lower() for s in features_df.columns.tolist()]
    Path(dataset_data_folder+"features-files/").mkdir(parents=True, exist_ok=True)
    features_df.to_csv(dataset_data_folder+"features-files/" + dataset + "-ami-patients-features.csv", index=False)


def create_feature_files(dataset="all"):
    """
    This function puts together everything. By calling previous functions (created above) it generates a features file
    for each of the datasets in the study. This can be done for all (default), or for a single dataset by passing the
    dataset name as argument
    :param dataset: the name of the dataset for which a features file should be generated. Default values is 'all',
    meaning features files for MIMIC-III, MIMIC-IV, eICU and AUMCdb datasets will be generated.
    :return: Nothing. Output files containing 'basic' and 'medication order' features are written to CSV files and
    saved in 'dataset_data_folder'
    """
    if dataset == "all":
        datasets = ["aumc", "eicu", "mimic-iii", "mimic-iv"]
        basic_features_dfs = [process_aumc_basics(), process_eicu_basics(),
                              process_mimic_iii_basics(), process_mimic_iv_basics()]
        admission_col_names = ["admissionid", "patientunitstayid", "HADM_ID", "hadm_id"]
        dataset_data_folders = ["./aumc/data/processed/", "./eicu/data/processed/", "./mimic-iii/data/processed/",
                                "./mimic-iv/data/processed/"]
        for i in range(len(datasets)):
            add_drug_features(basic_features_df=basic_features_dfs[i], admission_col_name=admission_col_names[i],
                              dataset=datasets[i], dataset_data_folder=dataset_data_folders[i])
    elif dataset == "aumc":
        add_drug_features(basic_features_df=process_aumc_basics(), admission_col_name="admissionid",
                          dataset=dataset, dataset_data_folder="./aumc/data/processed/")
    elif dataset == "eicu":
        add_drug_features(basic_features_df=process_eicu_basics(), admission_col_name="patientunitstayid",
                          dataset=dataset, dataset_data_folder="./eicu/data/processed/")
    elif dataset == "mimic-iii":
        add_drug_features(basic_features_df=process_mimic_iii_basics(), admission_col_name="HADM_ID",
                          dataset=dataset, dataset_data_folder="./mimic-iii/data/processed/")
    elif dataset == "mimic-iv":
        add_drug_features(basic_features_df=process_mimic_iv_basics(), admission_col_name="hadm_id",
                          dataset=dataset, dataset_data_folder="./mimic-iv/data/processed/")
    else:
        raise ValueError("incorrect dataset name. expected values are: 'aumc', 'eicu', 'mimic-iii', 'mimic-iv', 'all'")


def create_sub_cohorts_feature_files():
    """
    This function generates data-files for 7 ami sub-cohorts, from the default ami cohort.

    Default: cohort1.0 (ami_df) -  patients who have AMI as one of their diagnoses, and all ages and comorbidities
    are included. This is the same cohort created in by **create_feature_files()** function above.
    (i.e., the file <dataset>-ami-patients-features.csv).


    cohort 1.1 (ami_young_df): patients who have AMI as one of their diagnoses, who are aged between 18 and 49 years.
    all comorbidities are included
    cohort 1.2 (ami_healthy_df): patients who have AMI as one of their diagnoses, who have no recorded
    prior medical history
    cohort 1.3 (ami_healthy_and_young_df): patients who have AMI as one of their diagnoses, who have no recorded prior
    medical history, and who are aged between 18 and 49 years
    cohort 2.0 (ami_primary_df): patients for whom AMI is the primary/main diagnosis. (all ages and comorbidities
    included)
    cohort 2.1 (ami_primary_young_df): patients for whom AMI is the primary/main diagnosis, and are aged between
    18 and 49 years.
    cohort 2.2 (ami_primary_healthy_df): patients for whom AMI is the primary/main diagnosis, and have no recorded
    prior medical history
    cohort 2.3 (ami_primary_healthy_and_young_df): patients for whom AMI is the primary/main diagnosis, who have no
    recorded prior medical history, and who are aged between 18 and 49 years
    :return: Nothing. This function simply writes files to dataset/cohorts folder
    """
    dataset_folders = ["./aumc", "./eicu", "./mimic-iii", "./mimic-iv"]
    datasets = ["aumc", "eicu", "mimic-iii", "mimic-iv"]

    for i in range(len(dataset_folders)):
        print("dataset = ", datasets[i])
        ami_df = pd.read_csv(dataset_folders[i] + "/data/processed/features-files/" + datasets[i] +
                             "-ami-patients-features.csv")  # c1.0
        ami_young_df = ami_df[ami_df["agegroup"].isin(["18-39", "40-49"])]  # c1.1
        save_dir = dataset_folders[i] + "/data/processed/cohorts/"
        save_path = save_dir + datasets[i]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ami_young_df.to_csv(save_path + "-cohort1.1-any-ami-diagnosis-and-young.csv", index=False)

        try:
            ami_healthy_df = ami_df[ami_df["has-comorbidity?"] == 0]  # c1.2
            ami_healthy_and_young_df = ami_healthy_df[ami_healthy_df["agegroup"].isin(["18-39", "40-49"])]  # c1.3
            ami_healthy_df.to_csv(save_path + "-cohort1.2-any-ami-diagnosis-and-healthy.csv", index=False)
            ami_healthy_and_young_df.to_csv(save_path + "-cohort1.3-any-ami-diagnosis-and-healthy-and-young.csv",
                                            index=False)
        except KeyError:
            pass
        if datasets[i] != "aumc":
            if datasets[i] == "mimic-iii":
                ami_primary_df = ami_df[ami_df["seq_num"].str.startswith(("1.0", "[1.0"), na=False)]  # c2.0
            elif datasets[i] == "eicu":
                ami_primary_df = ami_df[
                    ami_df["diagnosispriority"].str.contains("primary", case=False, na=False)]  # c2.0
            elif datasets[i] == "mimic-iv":
                ami_primary_df = ami_df[ami_df["seq_num"].str.startswith(("1.0", "[1.0"), na=False)]  # c2.0
            else:
                raise ValueError("wrong dataset name value. expected values are: 'mimic-iii', 'mimic-iv', and 'eicu'")
            ami_primary_young_df = ami_primary_df[ami_primary_df["agegroup"].isin(["18-39", "40-49"])]  # c2.1
            ami_primary_healthy_df = ami_primary_df[ami_primary_df["has-comorbidity?"] == 0]  # c2.2
            ami_primary_healthy_and_young_df = ami_primary_healthy_df[ami_primary_healthy_df["agegroup"].isin(
                ["18-39", "40-49"])]  # c2.3
            ami_primary_df.to_csv(save_path + "-cohort2.0-ami-is-primary-diagnosis.csv", index=False)
            ami_primary_young_df.to_csv(save_path + "-cohort2.1-ami-is-primary-diagnosis-and-young.csv", index=False)
            ami_primary_healthy_df.to_csv(save_path + "-cohort2.2-ami-is-primary-diagnosis-and-healthy.csv",
                                          index=False)
            ami_primary_healthy_and_young_df.to_csv(save_path +
                                                    "-cohort2.3-ami-is-primary-diagnosis-and-healthy-and-young.csv",
                                                    index=False)
        else:
            pass


def create_mimic_iv_unique_cohorts():
    """
    This function separates out patients in mimic-iv that are not in mimic-iii
    :return:
    """
    # first get the len of cohorts in m3 and m4
    m4_c1 = pd.read_csv("./mimic-iv/data/processed/features-files/mimic-iv-ami-patients-features.csv")
    m3_c1 = pd.read_csv("./mimic-iii/data/processed/features-files/mimic-iii-ami-patients-features.csv")

    m4_cohorts_folder = "./mimic-iv/data/processed/cohorts/"

    def get_anchor_year_upper_bound(anchor_yr_range):
        upper_bound_yr = anchor_yr_range.split('-')[0].strip()
        return int(upper_bound_yr)

    Path(m4_cohorts_folder + "/m4-only/").mkdir(parents=True, exist_ok=True)

    filenames = [f for f in listdir(m4_cohorts_folder) if isfile(join(m4_cohorts_folder, f))]
    for filename in filenames:
        df = pd.read_csv(join(m4_cohorts_folder, filename))
        m4_only_df = df[df["anchor_year_group"].apply(lambda x: get_anchor_year_upper_bound(x)) > 2013]
        m4_only_df.to_csv(m4_cohorts_folder + "/m4-only/" + filename, index=False)


def assign_age_group(age):
    """
    Helper function to assign an 'age-group' to a patient when provided with their 'age' value
    :param age: patient's age
    :return: age-group
    """
    age = float(age)

    if 18 <= age <= 39:
        age_group = "18-39"
    elif 40 <= age <= 49:
        age_group = "40-49"
    elif 50 <= age <= 59:
        age_group = "50-59"
    elif 60 <= age <= 69:
        age_group = "60-69"
    elif 70 <= age <= 79:
        age_group = "70-79"
    elif age >= 80:
        age_group = "80+"
    else:
        age_group = "<18"
    return age_group


if __name__ == '__main__':
    create_feature_files()
    create_sub_cohorts_feature_files()
    create_mimic_iv_unique_cohorts()
