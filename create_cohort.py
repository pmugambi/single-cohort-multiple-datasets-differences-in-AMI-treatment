import pandas as pd
from dask import dataframe as dd
from pathlib import Path
import time


def obtain_ami_cohorts(dataset="all"):
    """
    This function generates files containing patients with any AMI diagnosis.
    Specifically, for MIMIC-III and eICU datasets, we look for patients whose diagnoses ICD 9 code starts with 410,
    the ICD9 code for MI.

    To create the files:
    1) the patients with the MI diagnoses are obtained from the diagnoses file,
    2) the patient and admission information is obtained for patients in 1) above, and these two are merged
    3) patient medical history is obtained and merged with file from 2)above
    4) finally, the file is written to dataset folders; named "...ami-patients-master-unprocessed.csv"

    :return: Nothing. Writes created files to output folders
    """

    def mimic_iii():
        print("---creating an AMI cohort from MIMIC-III---")
        print("1. obtaining diagnoses information")
        mimic_diagnoses_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz")
        mimic_ami_diagnoses_df = mimic_diagnoses_df[mimic_diagnoses_df["ICD9_CODE"].str.startswith("410", na=False)]
        mimic_ami_diagnoses_df = mimic_ami_diagnoses_df.drop(columns=["ROW_ID", "SUBJECT_ID"])

        # combine ami diagnoses by hadm_id
        mimic_ami_diagnoses_df = combine_records(pid_col_name="HADM_ID",
                                                 records_mini_df=mimic_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        mimic_admissions_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz")
        mimic_ami_patients_admissions_df = mimic_admissions_df[mimic_admissions_df["HADM_ID"].isin(
            mimic_ami_diagnoses_df["HADM_ID"].unique())]
        mimic_ami_patients_admissions_df = mimic_ami_patients_admissions_df.drop(columns=["ROW_ID"])

        mimic_patients_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/PATIENTS.csv.gz")
        mimic_ami_patients_df = mimic_patients_df[mimic_patients_df["SUBJECT_ID"].isin(
            mimic_ami_patients_admissions_df["SUBJECT_ID"].unique())]
        mimic_ami_patients_df = mimic_ami_patients_df.drop(columns=["ROW_ID"])
        mimic_ami_cohort_df = pd.merge(left=mimic_ami_patients_admissions_df, right=mimic_ami_diagnoses_df,
                                       how="inner", on="HADM_ID")
        mimic_ami_cohort_df = pd.merge(left=mimic_ami_cohort_df, right=mimic_ami_patients_df, how="left",
                                       on="SUBJECT_ID")

        # past histories
        print("3. obtaining past medical information")
        mimic_item_ids_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz")
        mimic_past_history_item_ids_df = mimic_item_ids_df[mimic_item_ids_df["LABEL"].str.contains(
            "past medical history|medical history", case=False,
            na=False)]  # I looked this up on the DB to make sure that
        # just these 2 are relevant to search for
        mimic_past_history_item_ids = mimic_past_history_item_ids_df["ITEMID"].unique().tolist()

        # mimic_chartevents_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz")
        # mimic_ami_patients_chart_events_df = mimic_chartevents_df[mimic_chartevents_df["HADM_ID"].isin(
        #     mimic_ami_cohort_df["HADM_ID"].unique())]
        mimic_chartevents_chunks = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/CHARTEVENTS.csv.gz",
                                               chunksize=1000000)

        def process_chunk(df):
            df = df[df["ITEMID"].isin(mimic_past_history_item_ids)]
            return df

        mimic_past_medical_history_chartevents_chunk_list = []

        for chunk in mimic_chartevents_chunks:
            filtered_chunk = process_chunk(chunk)
            mimic_past_medical_history_chartevents_chunk_list.append(filtered_chunk)

        mimic_past_medical_history_df = pd.concat(mimic_past_medical_history_chartevents_chunk_list)
        mimic_ami_past_medical_history_df = mimic_past_medical_history_df[
            mimic_past_medical_history_df["HADM_ID"].isin(mimic_ami_cohort_df["HADM_ID"].unique())]

        # mimic_ami_past_medical_history_df = mimic_ami_patients_chart_events_df[
        #     mimic_ami_patients_chart_events_df["ITEMID"].isin(mimic_past_history_item_ids)]
        mimic_ami_past_medical_history_df = mimic_ami_past_medical_history_df.drop(columns=[
            "ROW_ID", "SUBJECT_ID", "ICUSTAY_ID", "CHARTTIME", "STORETIME", "CGID",
            "VALUENUM", "VALUEUOM", "WARNING", "ERROR", "RESULTSTATUS", "STOPPED"])

        # combine past histories into one patient admission id
        print("4. generating and writing cohort file")
        mimic_ami_past_medical_history_df = combine_records(records_mini_df=mimic_ami_past_medical_history_df,
                                                            pid_col_name="HADM_ID")

        mimic_ami_cohort_df = pd.merge(left=mimic_ami_cohort_df, right=mimic_ami_past_medical_history_df,
                                       how="left", on="HADM_ID")
        Path("./mimic-iii/data/").mkdir(parents=True, exist_ok=True)
        mimic_ami_cohort_df.to_csv("./mimic-iii/data/mimic-iii-ami-patients-master-unprocessed.csv")

    def eicu():
        print("---creating an AMI cohort from eICU---")
        print("1. obtaining diagnoses information")
        eicu_diagnoses_df = pd.read_csv("./eicu/data/eicu-collaborative-research-database-2.0/diagnosis.csv.gz")
        eicu_ami_diagnoses_df = eicu_diagnoses_df[eicu_diagnoses_df["icd9code"].str.startswith("410", na=False)]
        eicu_ami_diagnoses_df = eicu_ami_diagnoses_df.drop(
            columns=["diagnosisid", "activeupondischarge", "diagnosisoffset"])

        # combine ami diagnoses by patientunitstayid
        eicu_ami_diagnoses_df = combine_records(pid_col_name="patientunitstayid",
                                                records_mini_df=eicu_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        eicu_patients_df = pd.read_csv("./eicu/data/eicu-collaborative-research-database-2.0/patient.csv.gz")
        eicu_ami_patients_df = eicu_patients_df[eicu_patients_df["patientunitstayid"].isin(
            eicu_ami_diagnoses_df["patientunitstayid"].unique())]
        eicu_ami_cohort_df = pd.merge(left=eicu_ami_patients_df, right=eicu_ami_diagnoses_df, how="inner",
                                      on="patientunitstayid")

        # past histories
        print("3. obtaining past medical information")
        eicu_past_histories_df = pd.read_csv("./eicu/data/eicu-collaborative-research-database-2.0/pastHistory.csv.gz")
        eicu_ami_patients_histories_df = eicu_past_histories_df[eicu_past_histories_df["patientunitstayid"].isin(
            eicu_ami_diagnoses_df["patientunitstayid"].unique())]
        eicu_ami_patients_histories_df = eicu_ami_patients_histories_df.drop(
            columns=["pasthistoryid", "pasthistoryoffset", "pasthistoryenteredoffset"])

        # combine past histories into one admission id
        eicu_ami_patients_histories_df = combine_records(records_mini_df=eicu_ami_patients_histories_df,
                                                         pid_col_name="patientunitstayid")
        print("4. generating and writing cohort file")
        eicu_ami_cohort_df = pd.merge(left=eicu_ami_cohort_df, right=eicu_ami_patients_histories_df, how="left",
                                      on="patientunitstayid")
        Path("./eicu/data/").mkdir(parents=True, exist_ok=True)
        eicu_ami_cohort_df.to_csv("./eicu/data/eicu-ami-patients-master-unprocessed.csv")

    def amsterdamUMCdb():
        print("---creating an AMI cohort from AmsterdamUMCdb---")
        print("1. obtaining diagnoses information")
        aumc_listitems_df = dd.read_csv("./aumc/data/AmsterdamUMCdb-v1.0.2/listitems.csv",
                                        encoding="ISO-8859-1",
                                        assume_missing=True)
        aumc_diagnoses_items_df = aumc_listitems_df[
            aumc_listitems_df["item"].str.contains("APACHE|D_|DMC_", case=False, na=True)]
        aumc_ami_diagnoses_df = aumc_diagnoses_items_df[aumc_diagnoses_items_df["value"].str.contains(
            r'myocardial|MI|cardiogene shock|shock, cardiogenic', na=False)]
        aumc_ami_diagnoses_df = aumc_ami_diagnoses_df.rename(columns={"itemid": "diagnosis-category-id",
                                                                      "item": "diagnosis-category",
                                                                      "value": "diagnosis"}).compute()
        print(aumc_ami_diagnoses_df.head(), aumc_ami_diagnoses_df.columns, len(aumc_ami_diagnoses_df))

        aumc_ami_diagnoses_df = aumc_ami_diagnoses_df.drop(columns=["valueid", "measuredat", "registeredat",
                                                                    "registeredby", "updatedat", "updatedby",
                                                                    "islabresult"])

        # combine ami diagnoses by admissionid
        aumc_ami_diagnoses_df = combine_records(pid_col_name="admissionid",
                                                records_mini_df=aumc_ami_diagnoses_df)
        print("2. obtaining admission and patient information")
        aumc_admissions_df = pd.read_csv("aumc/data/AmsterdamUMCdb-v1.0.2/admissions.csv")
        aumc_ami_cohort_df = pd.merge(left=aumc_admissions_df, right=aumc_ami_diagnoses_df, how="right",
                                      on="admissionid")
        print("3. generating and writing cohort file")
        Path("aumc/data/").mkdir(parents=True, exist_ok=True)
        aumc_ami_cohort_df.to_csv("./aumc/data/aumc-ami-patients-master-unprocessed.csv")

    def mimic_iv():
        print("---creating an AMI cohort from MIMIC-III---")
        print("1. obtaining diagnoses information")
        mimic_iv_diagnoses_df = pd.read_csv("./mimic-iv/data/hosp/diagnoses_icd.csv.gz")
        mimic_iv_ami_diagnoses_df = mimic_iv_diagnoses_df[((mimic_iv_diagnoses_df["icd_version"] == 9) & (
            mimic_iv_diagnoses_df["icd_code"].str.startswith("410", na=False))) | (
                (mimic_iv_diagnoses_df["icd_version"] == 10) &
                mimic_iv_diagnoses_df["icd_code"].str.startswith("I21", na=False))]

        print("mimic_iv_ami_diagnoses_df.len = ", len(mimic_iv_ami_diagnoses_df),
              mimic_iv_ami_diagnoses_df.icd_code.unique().tolist())

        mimic_iv_ami_diagnoses_df = mimic_iv_ami_diagnoses_df.drop(columns=["subject_id"])

        # combine ami diagnoses by hadm_id
        mimic_iv_ami_diagnoses_df = combine_records(pid_col_name="hadm_id",
                                                    records_mini_df=mimic_iv_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        mimic_iv_admissions_df = pd.read_csv("./mimic-iv/data/hosp/admissions.csv.gz")
        mimic_iv_ami_patients_admissions_df = mimic_iv_admissions_df[mimic_iv_admissions_df["hadm_id"].isin(
            mimic_iv_ami_diagnoses_df["hadm_id"].unique())]

        mimic_iv_patients_df = pd.read_csv("./mimic-iv/data/hosp/patients.csv.gz")
        mimic_iv_ami_patients_df = mimic_iv_patients_df[mimic_iv_patients_df["subject_id"].isin(
            mimic_iv_ami_patients_admissions_df["subject_id"].unique())]
        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_patients_admissions_df, right=mimic_iv_ami_diagnoses_df,
                                          how="inner", on="hadm_id")
        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_cohort_df, right=mimic_iv_ami_patients_df,
                                          how="left", on="subject_id")

        # past histories
        # print("3. obtaining past medical information")
        mimic_iv_item_ids_df = pd.read_csv("./mimic-iv/data/icu/d_items.csv.gz")
        # print("mimic_iv_item_ids_df.head() = ", mimic_iv_item_ids_df.head(), mimic_iv_item_ids_df.columns.tolist())
        mimic_iv_past_history_item_ids_df = mimic_iv_item_ids_df[mimic_iv_item_ids_df["label"].str.contains(
            "past medical history|medical history", case=False, na=False)]
        mimic_iv_past_history_item_ids = mimic_iv_past_history_item_ids_df["itemid"].unique().tolist()

        mimic_iv_chartevents_chunks = pd.read_csv("./mimic-iv/data/icu/chartevents.csv.gz", chunksize=1000000)

        def process_chunk(df):
            df = df[df["itemid"].isin(mimic_iv_past_history_item_ids)]
            return df

        mimic_iv_past_medical_history_chartevents_chunk_list = []
        start = time.time()
        for chunk in mimic_iv_chartevents_chunks:
            filtered_chunk = process_chunk(chunk)
            mimic_iv_past_medical_history_chartevents_chunk_list.append(filtered_chunk)

        mimic_iv_past_medical_history_df = pd.concat(mimic_iv_past_medical_history_chartevents_chunk_list)
        end = time.time()
        print("it took ", (end - start), " seconds to read and process all chartevent records. start-time = ",
              start, " and end time = ", end)
        mimic_iv_ami_past_medical_history_df = mimic_iv_past_medical_history_df[
            mimic_iv_past_medical_history_df["hadm_id"].isin(mimic_iv_ami_cohort_df["hadm_id"].unique())]

        print("mimic_iv_ami_past_medical_history_df.head() = ", mimic_iv_ami_past_medical_history_df.head(),
              mimic_iv_ami_past_medical_history_df.columns.tolist(), len(mimic_iv_ami_past_medical_history_df))

        mimic_iv_ami_past_medical_history_df = mimic_iv_ami_past_medical_history_df.drop(columns=[
            "subject_id", "charttime", "storetime", "valuenum", "valueuom", "warning", "caregiver_id", "stay_id"])

        # combine past histories into one patient admission id
        # print("4. generating and writing cohort file")
        mimic_iv_ami_past_medical_history_df = combine_records(records_mini_df=mimic_iv_ami_past_medical_history_df,
                                                               pid_col_name="hadm_id")

        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_cohort_df, right=mimic_iv_ami_past_medical_history_df,
                                          how="left", on="hadm_id")
        print("mimic_iv_ami_cohort_df.head() = ", mimic_iv_ami_cohort_df.head(),
              mimic_iv_ami_cohort_df.columns.tolist(), len(mimic_iv_ami_cohort_df))
        Path("./mimic-iv/data/").mkdir(parents=True, exist_ok=True)
        mimic_iv_ami_cohort_df.to_csv("./mimic-iv/data/mimic-iv-ami-patients-master-unprocessed.csv")

    if dataset == "all":
        # 1. MIMIC-III
        mimic_iii()
        # 2. eICU
        eicu()
        # 3. AmsterdamUMCdb
        amsterdamUMCdb()
        # 4. MIMIC-IV
        mimic_iv()
    elif dataset == "mimic-iii":
        mimic_iii()
    elif dataset == "mimic-iv":
        mimic_iv()
    elif dataset == "eicu":
        eicu()
    else:
        raise ValueError("wrong value for dataset provided. "
                         "the expected values are: 'all', 'mimic-iii', 'mimic-iv', 'eicu', 'amsterdamUMCdb'")


def combine_records(records_mini_df, pid_col_name):
    """
    This is a helper function.
    Given a dataframe containing records duplicated by "pid_col_name", it merges the values for each column in the
    dataframe for the specific "pid_col_name" value.

    For instance, suppose a patient has several diagnoses recorded in the diagnoses_df.
    This function would combine the patients' diagnoses into one list, and write just 1 row for the patient,
    instead of the several that existed in diagnoses_df

    :param records_mini_df: dataframe containing duplicated rows
    :param pid_col_name: identifier column, containing values for which every other column values
    should be aggregated by. For instance, patientunitstayid is used for eICU. It is the unique patient identifier.
    :return: a non-duplicated dataframe, by column "pid_col_name". For instance, for all patients,
    this dataframe would have at most 1 row for each patient
    """
    records_cols = records_mini_df.columns.tolist()
    records_cols.remove(pid_col_name)
    pids = records_mini_df[pid_col_name].unique().tolist()
    diagnoses_items = []
    for pid in pids:
        pid_diagnoses_items = []
        pid_df = records_mini_df[records_mini_df[pid_col_name] == pid]
        for col in records_cols:
            pid_col_values = pid_df[col].unique().tolist()
            if len(pid_col_values) > 1:
                pid_diagnoses_items.append(pid_col_values)
            else:
                pid_diagnoses_items.append(pid_col_values[0])
        diagnoses_items.append([pid] + pid_diagnoses_items)
    df = pd.DataFrame(data=diagnoses_items, columns=[pid_col_name] + records_cols)
    return df


if __name__ == '__main__':
    obtain_ami_cohorts(dataset="mimic-iv")
