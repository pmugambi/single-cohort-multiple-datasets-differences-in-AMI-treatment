import pandas as pd
from dask import dataframe as dd
from pathlib import Path


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
        mimic_diagnoses_df = pd.read_csv("./mimic-iii/data/raw/DIAGNOSES_ICD.csv.gz")
        mimic_ami_diagnoses_df = mimic_diagnoses_df[mimic_diagnoses_df["ICD9_CODE"].str.startswith("410", na=False)]
        mimic_ami_diagnoses_df = mimic_ami_diagnoses_df.drop(columns=["ROW_ID", "SUBJECT_ID"])

        # combine ami diagnoses by hadm_id
        mimic_ami_diagnoses_df = combine_records(pid_col_name="HADM_ID",
                                                 records_mini_df=mimic_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        mimic_admissions_df = pd.read_csv("./mimic-iii/data/raw/ADMISSIONS.csv.gz")
        mimic_ami_patients_admissions_df = mimic_admissions_df[mimic_admissions_df["HADM_ID"].isin(
            mimic_ami_diagnoses_df["HADM_ID"].unique())]
        mimic_ami_patients_admissions_df = mimic_ami_patients_admissions_df.drop(columns=["ROW_ID"])

        mimic_patients_df = pd.read_csv("./mimic-iii/data/raw/PATIENTS.csv.gz")
        mimic_ami_patients_df = mimic_patients_df[mimic_patients_df["SUBJECT_ID"].isin(
            mimic_ami_patients_admissions_df["SUBJECT_ID"].unique())]
        mimic_ami_patients_df = mimic_ami_patients_df.drop(columns=["ROW_ID"])
        mimic_ami_cohort_df = pd.merge(left=mimic_ami_patients_admissions_df, right=mimic_ami_diagnoses_df,
                                       how="inner", on="HADM_ID")
        mimic_ami_cohort_df = pd.merge(left=mimic_ami_cohort_df, right=mimic_ami_patients_df, how="left",
                                       on="SUBJECT_ID")

        # past histories
        print("3. obtaining past medical information")
        mimic_item_ids_df = pd.read_csv("./mimic-iii/data/raw/D_ITEMS.csv.gz")
        mimic_past_history_item_ids_df = mimic_item_ids_df[mimic_item_ids_df["LABEL"].str.contains(
            "past medical history|medical history", case=False,
            na=False)]  # NOTE:  I looked this up on the DB to make sure that
        # just these 2 are relevant to search for
        mimic_past_history_item_ids = mimic_past_history_item_ids_df["ITEMID"].unique().tolist()

        # NOTE: chartevents is a large table, reading it all with pandas as a single DF causes memory issues.
        # therefore, it was read in chunks, and the medical history values searched for in each chunk.
        mimic_chartevents_chunks = pd.read_csv("./mimic-iii/data/raw/CHARTEVENTS.csv.gz",
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

        mimic_ami_past_medical_history_df = mimic_ami_past_medical_history_df.drop(columns=[
            "ROW_ID", "SUBJECT_ID", "ICUSTAY_ID", "CHARTTIME", "STORETIME", "CGID",
            "VALUENUM", "VALUEUOM", "WARNING", "ERROR", "RESULTSTATUS", "STOPPED"])

        # combine past histories into one patient admission id
        print("4. generating and writing cohort file")
        mimic_ami_past_medical_history_df = combine_records(records_mini_df=mimic_ami_past_medical_history_df,
                                                            pid_col_name="HADM_ID")

        mimic_ami_cohort_df = pd.merge(left=mimic_ami_cohort_df, right=mimic_ami_past_medical_history_df,
                                       how="left", on="HADM_ID")
        Path("./mimic-iii/data/processed/features-files/").mkdir(parents=True, exist_ok=True)
        mimic_ami_cohort_df.to_csv(
            "./mimic-iii/data/processed/features-files/mimic-iii-ami-patients-master-unprocessed.csv")

    def eicu():
        print("---creating an AMI cohort from eICU---")
        print("1. obtaining diagnoses information")
        eicu_diagnoses_df = pd.read_csv("./eicu/data/raw/diagnosis.csv.gz")
        eicu_ami_diagnoses_df = eicu_diagnoses_df[eicu_diagnoses_df["icd9code"].str.startswith("410", na=False)]
        eicu_ami_diagnoses_df = eicu_ami_diagnoses_df.drop(
            columns=["diagnosisid", "activeupondischarge", "diagnosisoffset"])

        # combine ami diagnoses by patientunitstayid
        eicu_ami_diagnoses_df = combine_records(pid_col_name="patientunitstayid",
                                                records_mini_df=eicu_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        eicu_patients_df = pd.read_csv("./eicu/data/raw/patient.csv.gz")
        eicu_ami_patients_df = eicu_patients_df[eicu_patients_df["patientunitstayid"].isin(
            eicu_ami_diagnoses_df["patientunitstayid"].unique())]
        eicu_ami_cohort_df = pd.merge(left=eicu_ami_patients_df, right=eicu_ami_diagnoses_df, how="inner",
                                      on="patientunitstayid")

        # past histories
        print("3. obtaining past medical information")
        eicu_past_histories_df = pd.read_csv("./eicu/data/raw/pastHistory.csv.gz")
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
        Path("./eicu/data/processed/features-files/").mkdir(parents=True, exist_ok=True)
        eicu_ami_cohort_df.to_csv("./eicu/data/processed/features-files/eicu-ami-patients-master-unprocessed.csv")

    def amsterdamumcdb():
        print("---creating an AMI cohort from AmsterdamUMCdb---")
        print("1. obtaining diagnoses information")
        aumc_listitems_df = dd.read_csv("./aumc/data/raw/listitems.csv",
                                        encoding="ISO-8859-1",
                                        assume_missing=True)
        aumc_diagnoses_items_df = aumc_listitems_df[
            aumc_listitems_df["item"].str.contains("APACHE", case=False, na=True)]

        # # aumc_ami_diagnoses_df = aumc_diagnoses_items_df[aumc_diagnoses_items_df["value"].str.contains(
        # #     r'myocardial|MI|Cardiogene shock|cardiogene shock|shock, cardiogenic|shock, Cardiogenic|Shock,
        # # Cardiogenic', na=False)] NOTE: criteria abandoned for the one below

        # NOTE: diagnoses generated by clinical expert -- after reading through a list of unique APACHE diagnoses items
        # listed in the DB.

        apache_def_MI_diagnoses = [
            "Non-operative cardiovascular - Infarction, acute myocardial (MI), none of the above",
            "Non-operative cardiovascular - Infarction, acute myocardial (MI), ANTERIOR",
            "Non-operative cardiovascular - Angina, stable (asymp or stable pattern of symptoms w/meds)",
            "Non-operative cardiovascular - Infarction, acute myocardial (MI), INFEROLATERAL",
            "Non-operative cardiovascular - Angina, unstable (angina interferes w/quality of "
            "life or meds are tolerated poorly)",
            "Non-operative cardiovascular - MI admitted > 24hrs after onset of ischemia",
            "Non-operative cardiovascular - Infarction, acute myocardial (MI), NON Q Wave"]

        # apache_possible_MI_diagnoses = [
        #     "Non-operative cardiovascular - Papillary muscle rupture",
        #     "Non-operative cardiovascular - Rhythm disturbance (conduction defect)",
        #     "Non-operative cardiovascular - Rhythm disturbance (ventricular)",
        #     "Non-operative cardiovascular - Rhythm disturbance (atrial, supraventricular)"]

        # aumc_ami_diagnoses_df = aumc_diagnoses_items_df[aumc_diagnoses_items_df["value"].isin(
        #     apache_def_MI_diagnoses+apache_possible_MI_diagnoses)]
        aumc_ami_diagnoses_df = aumc_diagnoses_items_df[aumc_diagnoses_items_df["value"].isin(
            apache_def_MI_diagnoses)]  # include def_MI_diagnoses only
        aumc_ami_diagnoses_df = aumc_ami_diagnoses_df.rename(columns={"itemid": "diagnosis-category-id",
                                                                      "item": "diagnosis-category",
                                                                      "value": "diagnosis"}).compute()

        aumc_ami_diagnoses_df = aumc_ami_diagnoses_df.drop(columns=["valueid", "measuredat", "registeredat",
                                                                    "registeredby", "updatedat", "updatedby",
                                                                    "islabresult"])

        # combine ami diagnoses by admissionid
        aumc_ami_diagnoses_df = combine_records(pid_col_name="admissionid",
                                                records_mini_df=aumc_ami_diagnoses_df)
        # print("unique number of admissions = ", len(aumc_ami_diagnoses_df))
        print("2. obtaining admission and patient information")
        aumc_admissions_df = pd.read_csv("aumc/data/raw/admissions.csv")
        aumc_ami_cohort_df = pd.merge(left=aumc_admissions_df, right=aumc_ami_diagnoses_df, how="right",
                                      on="admissionid")
        print("3. generating and writing cohort file")
        Path("aumc/data/processed/features-files/").mkdir(parents=True, exist_ok=True)
        aumc_ami_cohort_df.to_csv("./aumc/data/processed/features-files/aumc-ami-patients-master-unprocessed.csv")

    def mimic_iv():
        # NOTE: pipeline very similar to that of MIMIC-III, except that the tables are organized differently here
        # and diagnoses are recorded both in ICD 9 and 10 codes.
        print("---creating an AMI cohort from MIMIC-IV---")
        print("1. obtaining diagnoses information")
        mimic_iv_diagnoses_df = pd.read_csv("./mimic-iv/data/raw/hosp/diagnoses_icd.csv.gz")
        mimic_iv_ami_diagnoses_df = mimic_iv_diagnoses_df[((mimic_iv_diagnoses_df["icd_version"] == 9) & (
            mimic_iv_diagnoses_df["icd_code"].str.startswith("410", na=False))) | (
                                                                  (mimic_iv_diagnoses_df["icd_version"] == 10) &
                                                                  mimic_iv_diagnoses_df["icd_code"].str.startswith(
                                                                      "I21", na=False))]
        # print("mimic_iv_ami_diagnoses_df.len = ", len(mimic_iv_ami_diagnoses_df),
        #       mimic_iv_ami_diagnoses_df.icd_code.unique().tolist())

        mimic_iv_ami_diagnoses_df = mimic_iv_ami_diagnoses_df.drop(columns=["subject_id"])

        # convert seq_num to floats so that down the line, we can check for primary diagnosis by checking whether
        # diagnoses startwith 1.0 or [1.0
        mimic_iv_ami_diagnoses_df["seq_num"] = mimic_iv_ami_diagnoses_df["seq_num"].astype(float)

        # combine ami diagnoses by hadm_id
        mimic_iv_ami_diagnoses_df = combine_records(pid_col_name="hadm_id",
                                                    records_mini_df=mimic_iv_ami_diagnoses_df)

        print("2. obtaining admission and patient information")
        mimic_iv_admissions_df = pd.read_csv("./mimic-iv/data/raw/hosp/admissions.csv.gz")
        mimic_iv_ami_patients_admissions_df = mimic_iv_admissions_df[mimic_iv_admissions_df["hadm_id"].isin(
            mimic_iv_ami_diagnoses_df["hadm_id"].unique())]

        mimic_iv_patients_df = pd.read_csv("./mimic-iv/data/raw/hosp/patients.csv.gz")
        mimic_iv_ami_patients_df = mimic_iv_patients_df[mimic_iv_patients_df["subject_id"].isin(
            mimic_iv_ami_patients_admissions_df["subject_id"].unique())]
        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_patients_admissions_df, right=mimic_iv_ami_diagnoses_df,
                                          how="inner", on="hadm_id")
        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_cohort_df, right=mimic_iv_ami_patients_df,
                                          how="left", on="subject_id")

        # past histories
        print("3. obtaining past medical information")
        mimic_iv_item_ids_df = pd.read_csv("./mimic-iv/data/raw/icu/d_items.csv.gz")
        mimic_iv_past_history_item_ids_df = mimic_iv_item_ids_df[mimic_iv_item_ids_df["label"].str.contains(
            "past medical history|medical history", case=False, na=False)]
        mimic_iv_past_history_item_ids = mimic_iv_past_history_item_ids_df["itemid"].unique().tolist()

        # NOTE: similar to MIMIC-III, chartevents is a large table, reading it all with pandas as a single
        # DF causes memory issues.
        # therefore, it was read in chunks, and the medical history values searched for in each chunk.

        mimic_iv_chartevents_chunks = pd.read_csv("./mimic-iv/data/raw/icu/chartevents.csv.gz", chunksize=1000000)

        def process_chunk(df):
            df = df[df["itemid"].isin(mimic_iv_past_history_item_ids)]
            return df

        mimic_iv_past_medical_history_chartevents_chunk_list = []
        for chunk in mimic_iv_chartevents_chunks:
            filtered_chunk = process_chunk(chunk)
            mimic_iv_past_medical_history_chartevents_chunk_list.append(filtered_chunk)

        mimic_iv_past_medical_history_df = pd.concat(mimic_iv_past_medical_history_chartevents_chunk_list)
        mimic_iv_ami_past_medical_history_df = mimic_iv_past_medical_history_df[
            mimic_iv_past_medical_history_df["hadm_id"].isin(mimic_iv_ami_cohort_df["hadm_id"].unique())]

        mimic_iv_ami_past_medical_history_df = mimic_iv_ami_past_medical_history_df.drop(columns=[
            "subject_id", "charttime", "storetime", "valuenum", "valueuom", "warning", "caregiver_id", "stay_id"])

        # combine past histories into one patient admission id
        print("4. generating and writing cohort file")
        mimic_iv_ami_past_medical_history_df = combine_records(records_mini_df=mimic_iv_ami_past_medical_history_df,
                                                               pid_col_name="hadm_id")

        mimic_iv_ami_cohort_df = pd.merge(left=mimic_iv_ami_cohort_df, right=mimic_iv_ami_past_medical_history_df,
                                          how="left", on="hadm_id")
        Path("./mimic-iv/data/processed/features-files/").mkdir(parents=True, exist_ok=True)
        mimic_iv_ami_cohort_df.to_csv(
            "./mimic-iv/data/processed/features-files/mimic-iv-ami-patients-master-unprocessed.csv")

    if dataset == "all":
        # 1. MIMIC-III
        mimic_iii()
        # 2. eICU
        eicu()
        # 3. AmsterdamUMCdb
        amsterdamumcdb()
        # 4. MIMIC-IV
        mimic_iv()
    elif dataset == "aumc":
        amsterdamumcdb()
    elif dataset == "mimic-iii":
        mimic_iii()
    elif dataset == "mimic-iv":
        mimic_iv()
    elif dataset == "eicu":
        eicu()
    else:
        raise ValueError("wrong value for dataset provided. "
                         "the expected values are: 'all', 'aumc', 'mimic-iii', 'mimic-iv', 'eicu'")


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
    obtain_ami_cohorts()


