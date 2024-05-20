import math
from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
import numpy as np
import json
from helpers import create_combined_drugs_string as combine_string
from helpers import create_batches as create_batches


def create_ami_treatments_file():
    """

    :return:
    """
    ace_inhibitors = ["Benazepril", "Lotensin", "Captopril", "Capoten", "Enalapril", "Enalaprilat", "Vasotec",
                      "Fosinopril", "Monopril", "Lisinopril", "Zestril", "Prinivil", "Moexipril", "Univasc",
                      "Perindopril", "Aceon", "Quinapril", "Accupril", "Ramipril", "Altace", "Trandolapril", "Mavik",
                      "Renitec"]
    beta_blockers = ["Acebutolol", "Sectral", "Atenolol", "Tenormin", "Betaxolol", "Kerlone", "Bisoprolol", "Zebeta",
                     "Ziac", "Carteolol", "Cartrol", "Carvedilol", "Coreg", "Labetalol", "Normodyne", "Trandate",
                     "Metoprolol", "Lopressor", "Toprol", "Nadolol", "Corgard", "Nebivolol", "Bystolic", "Penbutolol",
                     "Levatol", "Pindolol", "Visken", "Propanolol", "Inderal", "Sotalol", "Betapace", "Timolol",
                     "Blocadren"]
    aspirin = ["Bayer chewable", "Ecotrin", "Asaphen", "Entrophen", "Novasen", "acetylsalicylzuur", "aspegic",
               "aspirin"]
    non_aspirin_anti_platelets = ["Abciximab", "Eptifibatide", "Tirofiban", "Cangrelor", "Cilostazol", "Clopidogrel",
                                  "Plavix", "Dipyridamole", "Prasugrel", "Effient", "Ticlopidine", "Ticagrelor",
                                  "Brilinta", "Vorapaxar"]
    statins = ["Atorvastatin", "Lipitor", "Fluvastatin", "Lescol", "Lovastatin", "Altoprev", "Pitavastatin", "Livalo",
               "Zypitamag", "Pravastatin", "Pravachol", "Rosuvastatin", "Crestor", "Ezallor", "Simvastatin", "Zocor"]

    obj = {"ace-inhibitors": ace_inhibitors, "aspirin": aspirin, "beta-blockers": beta_blockers,
           "non-aspirin-antiplatelets": non_aspirin_anti_platelets, "statins": statins}
    Path("./shared-files/").mkdir(parents=True, exist_ok=True)
    with open("./shared-files/ami-drug-treatments.json", "w") as outfile:
        json.dump(obj, outfile)


def create_analgesics_file():
    """

    :return:
    """

    opioids = ["Tramadol", "Codeine", "Demerol", "meperidine", "Pethidine", "Oxymorphone", "Buprenorphine", "Codeine",
               "Fentanyl", "Actiq", "Abstral", "Duragesic", "Fentora", "sublimaze", "Hydrocodone", "Hysingla",
               "Zohydro", "Lorcet", "Lortab", "Norco", "Vicodin", "Hydromorphone", "Dilaudid", "Exalgo", "Methadone",
               "Dolophine", "Methadose", "Methadon", "Morphine", "Kadian", "MS Contin", "Morphabond", "Oliceridine",
               "Olynvik", "Oxycodone", "Oxaydo", "OxyContin", "roxicodone", "Percocet", "Roxicet"]

    otc_analgesics = ["Acetaminophen", "Paracetamol", "Tylenol", "Excedrin", "Vanquish", "Aspirin", "Bayer", "Bufferin",
                      "Ecotrin", "Excedrin", "Vanquish", "Diclofenac", "Voltaren", "Ibuprofen", "Advil", "Motrin",
                      "Naproxen", "Aleve", "Aspercreme", "BenGay", "Capsaicin cream", "Icy Hot"]

    local_anaesthetics = ["lidocaine", "prilocaine", "Bupivacaine", "Marcaine", "levobupivacaine", "ropivacaine",
                          "mepivacaine", "dibucaine", "Etidocaine", "Benzocaine", "Hurricaine"]

    eicu_drugs_oracle_df = pd.read_csv(
        "./shared-files/oracle-analgesics-files/eicu-unique-medications-list-no-comorbidity-ami-patients.csv")
    aumc_drugs_oracle_df = pd.read_csv(
        "./shared-files/oracle-analgesics-files/umcdb_cardio_patients_analgesics_revised.csv")
    mimic_drugs_oracle_df = pd.read_csv(
        "./shared-files/oracle-analgesics-files/unique-medications-list-ami-patients_SPC.csv")
    # todo: change this back, remove the maybe. Only testing for previous mimic results
    mimic_analgesics = mimic_drugs_oracle_df[
        mimic_drugs_oracle_df["pain_med?"].str.contains("yes|maybe", case=False, na=False)]["drug"].unique().tolist()
    mimic_opioids = mimic_drugs_oracle_df[
        mimic_drugs_oracle_df["narcotic?"].str.contains("yes", case=False, na=False)]["drug"].unique().tolist()

    eicu_analgesics = eicu_drugs_oracle_df[
        eicu_drugs_oracle_df["pain_med?"].str.contains("yes", case=False, na=False)]["drugname"].unique().tolist()
    eicu_opioids = eicu_drugs_oracle_df[
        eicu_drugs_oracle_df["narcotic?"].str.contains("yes", case=False, na=False)]["drugname"].unique().tolist()

    aumc_analgesics = aumc_drugs_oracle_df[
        aumc_drugs_oracle_df["analgesic?"].str.contains("yes", case=False, na=False)]["item"].unique().tolist()
    aumc_opioids = aumc_drugs_oracle_df[
        aumc_drugs_oracle_df["opioid?"].str.contains("yes", case=False, na=False)]["item"].unique().tolist()

    analgesics_lists = [aumc_analgesics, eicu_analgesics, mimic_analgesics]
    opioids_lists = [aumc_opioids, eicu_opioids, mimic_opioids]
    dfs = [aumc_drugs_oracle_df, eicu_drugs_oracle_df, mimic_drugs_oracle_df]
    drug_col_names = ["item", "drugname", "drug"]
    datasets = ["aumc", "eicu", "mimic-iii"]

    analgesics = opioids + otc_analgesics + local_anaesthetics
    print("analgesics = ", analgesics)

    for i in range(len(analgesics_lists)):
        print("dataset = ", datasets[i])
        df = dfs[i]
        pain_meds = df[df[drug_col_names[i]].str.contains(
            combine_string(analgesics), case=False, na=False)][drug_col_names[i]].unique().tolist()
        d_opioids = df[df[drug_col_names[i]].str.contains(
            combine_string(opioids), case=False, na=False)][drug_col_names[i]].unique().tolist()
        print("pain_meds = ", len(pain_meds))
        missing_analgesics = list(set(analgesics_lists[i]) - set(pain_meds))
        print("missing_analgesics = ", missing_analgesics, len(missing_analgesics))
        analgesics = analgesics + list(set(analgesics_lists[i]) - set(pain_meds))
        missing_opioids = list(set(opioids_lists[i]) - set(d_opioids))
        print("missing_opioids = ", missing_opioids, len(missing_opioids))
        opioids = opioids + missing_opioids

    analgesics = list(set([x.lower() for x in analgesics]))
    opioids = list(set([x.lower() for x in opioids]))

    analgesics_obj = {"analgesics": analgesics, "opioids": opioids}
    Path("./shared-files/").mkdir(parents=True, exist_ok=True)
    with open("./shared-files/analgesics.json", "w") as outfile:
        json.dump(analgesics_obj, outfile)


def get_ami_treatments():
    """

    :return:
    """
    try:
        with open("./shared-files/ami-drug-treatments.json", 'r') as openfile:
            ami_drugs = json.load(openfile)
    except FileNotFoundError:
        create_ami_treatments_file()
        with open("./shared-files/ami-drug-treatments.json", 'r') as openfile:
            ami_drugs = json.load(openfile)

    ace_inhibitors = [x.lower() for x in ami_drugs["ace-inhibitors"]]
    aspirin = [x.lower() for x in ami_drugs["aspirin"]]
    beta_blockers = [x.lower() for x in ami_drugs["beta-blockers"]]
    non_aspirin_antiplatelets = [x.lower() for x in ami_drugs["non-aspirin-antiplatelets"]]
    statins = [x.lower() for x in ami_drugs["statins"]]
    return ace_inhibitors, aspirin, beta_blockers, non_aspirin_antiplatelets, statins


def get_analgesics():
    """
    :return:
    """
    try:
        with open("./shared-files/analgesics.json", 'r') as openfile:
            ami_drugs = json.load(openfile)
    except FileNotFoundError:
        create_analgesics_file()
        with open("./shared-files/analgesics.json", 'r') as openfile:
            ami_drugs = json.load(openfile)
    analgesics = [x.lower() for x in ami_drugs["analgesics"]]
    opioids = [x.lower() for x in ami_drugs["opioids"]]
    return analgesics, opioids


def process_analgesics_orders(cohort_df, drugs_df, drug_col_name, admission_col_name, dose_col_name, dataset,
                              save_path):
    """

    :param cohort_df
    :param drugs_df:
    :param drug_col_name:
    :param admission_col_name:
    :param dose_col_name:
    :param dataset:
    :param save_path:
    :return:
    """
    analgesics, opioids = get_analgesics()
    print("opioids = ", opioids)
    non_opioid_analgesics = list(set(analgesics) - set(opioids))
    print("non_opioid_analgesics = ", non_opioid_analgesics)
    analgesics_df = drugs_df[drugs_df[drug_col_name].str.contains(
        combine_string(analgesics), case=False, na=False)]
    opioids_df = drugs_df[drugs_df[drug_col_name].str.contains(
        combine_string(opioids), case=False, na=False)]
    non_opioid_analgesics_df = drugs_df[drugs_df[drug_col_name].str.contains(
        combine_string(non_opioid_analgesics), case=False, na=False)]
    analgesics_ordered_or_administered = []
    pids = cohort_df[admission_col_name].unique().tolist()

    for pid in pids:
        pid_ordered_or_administered = []
        for cat_df in [analgesics_df, opioids_df, non_opioid_analgesics_df]:
            if dataset == "aumc":
                pid_ordered_cat_df = cat_df[(cat_df[admission_col_name] == pid) & (cat_df[dose_col_name] > 0)]
            else:
                pid_ordered_cat_df = cat_df[(cat_df[admission_col_name] == pid) & (~cat_df[dose_col_name].isnull())]

            if len(pid_ordered_cat_df) > 0:
                pid_cat_ordered = 1
            else:
                pid_cat_ordered = 0
            pid_ordered_or_administered.append(pid_cat_ordered)
        analgesics_ordered_or_administered.append([pid] + pid_ordered_or_administered)

    analgesics_binary_df = pd.DataFrame(data=analgesics_ordered_or_administered,
                                        columns=[admission_col_name, "received-analgesic?", "received-opioid?",
                                                 "received-non-opioid?"])
    analgesics_binary_df["received-opioids-only?"] = np.where(
        (analgesics_binary_df["received-opioid?"] == 1) & (analgesics_binary_df["received-non-opioid?"] == 0),
        1, 0)
    analgesics_binary_df["received-non-opioids-only?"] = np.where(
        (analgesics_binary_df["received-analgesic?"] == 1) & (analgesics_binary_df["received-opioid?"] == 0),
        1, 0)
    analgesics_binary_df["received-combined-therapy?"] = np.where(
        (analgesics_binary_df["received-opioid?"] == 1) & (analgesics_binary_df["received-non-opioid?"] == 1),
        1, 0)
    analgesics_binary_df = analgesics_binary_df.drop(analgesics_binary_df.filter(regex='Unnamed').columns, axis=1)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    analgesics_binary_df.to_csv(save_path + dataset + "-received-analgesics-during-hospitalization.csv")


def process_ami_drugs_orders(cohort_df, drugs_df, drug_col_name, admission_col_name, dataset, save_path):
    """
    :param cohort_df:
    :param drugs_df:
    :param drug_col_name:
    :param admission_col_name:
    :param dataset:
    :param save_path:
    :return:
    """
    ace_inhibitors, aspirin, beta_blockers, non_aspirin_antiplatelets, statins = get_ami_treatments()
    pids = cohort_df[admission_col_name].unique().tolist()

    received_ami_drugs = []
    for pid in pids:
        pid_drugs_df = drugs_df[drugs_df[admission_col_name] == pid]
        pid_received_drugs = []
        for drug in [ace_inhibitors, aspirin, beta_blockers, non_aspirin_antiplatelets, statins]:
            pid_drug_df = pid_drugs_df[
                pid_drugs_df[drug_col_name].str.contains(combine_string(drug), case=False, na=False)]
            if len(pid_drug_df) > 0:
                pid_received_drug = 1
            else:
                pid_received_drug = 0
            pid_received_drugs.append(pid_received_drug)
        received_ami_drugs.append([pid] + pid_received_drugs)
    ami_drugs_binary_df = pd.DataFrame(data=received_ami_drugs, columns=[admission_col_name,
                                                                         "received-ace-inhibitor?",
                                                                         "received-aspirin?",
                                                                         "received-beta-blocker?",
                                                                         "received-anti-platelet?",
                                                                         "received-statin?"])

    ami_drugs_binary_df = ami_drugs_binary_df.drop(ami_drugs_binary_df.filter(regex='Unnamed').columns, axis=1)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    ami_drugs_binary_df.to_csv(save_path + dataset + "-received-ami-drugs-during-hospitalization.csv")


def write_treatments_by_day_files(drug_df, admit_time, start_time, stop_time, dataset, save_folder, save_category):
    """
    Distributes the prescription amount to all the days in the prescription, and writes
    a new file with each day of treatment as it's own row.
    For instance, if previous data file (i.e., drug_df) had one row of prescription of aspirin to be
    taken from 1/3 - 1/5, this function will create 3 rows for each day of aspirin, i.e., 1/3, 1/4, and 1/5.
    :param drug_df: The DF of all patients' prescriptions with each prescription amount
    :param admit_time:
    :param start_time:
    :param stop_time:
    :param dataset:
    :param save_folder
    :param save_category:
    :return: A DF of all patients' prescriptions with each day in each prescription # todo: update the docstring
    """
    drug_df = drug_df[pd.notnull(drug_df[start_time])]
    drug_df = drug_df[pd.notnull(drug_df[stop_time])]

    if "mimic" in dataset:
        drug_df[admit_time] = pd.to_datetime(drug_df[admit_time], format='%Y-%m-%d %H:%M:%S').dt.date
        drug_df[start_time] = pd.to_datetime(drug_df[start_time], format='%Y-%m-%d %H:%M:%S').dt.date
        drug_df[stop_time] = pd.to_datetime(drug_df[stop_time], format='%Y-%m-%d %H:%M:%S').dt.date

    batches = create_batches(max_length=len(drug_df), batch_size=5000)
    for batch in batches:
        batch_no = batches.index(batch) + 1
        print("working on batch no: ", batch_no)
        batch_df = pd.DataFrame()
        for i in range(batch[0], batch[1]):
            row = drug_df.iloc[[i]]
            if dataset == "eicu":
                admit_day = np.ceil((row[start_time] - row[admit_time]).values[0] / 1440)
                days = np.ceil((row[stop_time] - row[start_time]).values[0] / 1440)
            elif dataset == "aumc":
                admit_day = np.ceil((row[start_time] - row[admit_time]).values[0] / (1440 * 60000))
                days = np.ceil((row[stop_time] - row[start_time]).values[0] / (1440 * 60000))
            else:
                admit_day = (row[start_time] - row[admit_time]).dt.days + 1
                days = (row[stop_time] - row[start_time]).dt.days + 1
            for j in range(0, int(days)):
                row["day-no"] = j + admit_day
                batch_df = pd.concat([batch_df, row])
        batch_df = batch_df.drop(batch_df.filter(regex='Unnamed').columns, axis=1)
        save_dir = save_folder + "/data/batch-files/" + save_category + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        batch_df.to_csv(save_dir + "batch-no-" + str(batch_no) + "-expanded-medication.csv")

    treatment_directory_path = save_folder + "/data/batch-files/" + save_category
    files = [f for f in listdir(treatment_directory_path) if isfile(join(treatment_directory_path, f))]
    treatment_per_day_df = pd.DataFrame()
    for f in files:
        print("now processing file ", f)
        df = pd.read_csv(treatment_directory_path + "/" + f)
        treatment_per_day_df = pd.concat([treatment_per_day_df, df])
        treatment_per_day_df = treatment_per_day_df.drop(treatment_per_day_df.filter(regex='Unnamed').columns, axis=1)
        save_dir = save_folder + "/data/drug-orders-per-day/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        treatment_per_day_df.to_csv(save_dir + dataset + "-ami-cohort-" + save_category + "-per-day-records.csv")


def expand_drug_prescriptions(treatments, treatment_names, drug_df, drug_col_name, admit_time, start_time,
                              stop_time, dataset, save_folder, check_column=None):
    """
    # todo: write the description of this function
    :param treatments
    :param treatment_names
    :param drug_df:
    :param drug_col_name:
    :param admit_time:
    :param start_time:
    :param stop_time:
    :param dataset:
    :param save_folder:
    :param check_column:
    :return:
    """
    for t_name in treatment_names:
        print("now processing ==== ", t_name)
        treatment = treatments[treatment_names.index(t_name)]
        treatment_df = drug_df[drug_df[drug_col_name].str.contains(
            combine_string(treatment), case=False, na=False)]
        if check_column is not None:
            treatment_df = treatment_df[pd.notnull(treatment_df[check_column])]
        write_treatments_by_day_files(drug_df=treatment_df, admit_time=admit_time, start_time=start_time,
                                      stop_time=stop_time, dataset=dataset,
                                      save_folder=save_folder, save_category=t_name)


def create_per_day_therapy_features(dataset_folder, dataset, cohort_df, admission_col_name, number_of_days=5):
    """
    :param dataset_folder:
    :param dataset
    :param cohort_df
    :param admission_col_name
    :param number_of_days:
    :return:
    """
    opioids_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-opioids-per-day-records.csv")
    non_opioid_analgesics_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-non-opioid-analgesics-per-day-records"
                                                                  ".csv")
    analgesics_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-analgesics-per-day-records.csv")
    aspirin_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-aspirin-per-day-records.csv")
    ace_inhibitors_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-ace-inhibitors-per-day-records.csv")
    beta_blockers_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-beta-blockers-per-day-records.csv")
    statin_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-statins-per-day-records.csv")
    anti_platelets_df = pd.read_csv(
        dataset_folder + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-anti-platelets-per-day-records.csv")
    patients_ids = cohort_df[admission_col_name].unique().tolist()

    treatment_dfs = [analgesics_df, opioids_df, non_opioid_analgesics_df, aspirin_df,
                     ace_inhibitors_df, beta_blockers_df, statin_df, anti_platelets_df]
    treatment_names = ["analgesic", "opioid", "non-opioid-analgesic", "aspirin",
                       "ace-inhibitor", "beta-blocker", "statin", "anti-platelet"]

    received_treatment = []
    for pid in patients_ids:
        pid_treatment_values = [pid]
        for treatment_df in treatment_dfs:
            for day in range(1, number_of_days + 1):
                pid_day_treatment_df = treatment_df[(treatment_df[admission_col_name] == pid) &
                                                    (treatment_df["day-no"] == day)]
                if len(pid_day_treatment_df) > 0:
                    pid_treatment_values.append(1)
                else:
                    pid_treatment_values.append(0)
        received_treatment.append(pid_treatment_values)

    treatment_col_names = []
    for treatment in treatment_names:
        for i in range(1, number_of_days + 1):
            treatment_col_names.append(treatment + "-d" + str(i) + "?")

    df = pd.DataFrame(data=received_treatment, columns=[admission_col_name, *treatment_col_names])

    for day in range(1, number_of_days + 1):
        df["combined-therapy-d" + str(day) + "?"] = np.where(
            (df["opioid-d" + str(day) + "?"] == 1) & (df["non-opioid-analgesic-d" + str(day) + "?"] == 1), 1, 0)
        df["opioid-only-d" + str(day) + "?"] = np.where(
            (df["opioid-d" + str(day) + "?"] == 1) & (df["non-opioid-analgesic-d" + str(day) + "?"] == 0), 1, 0)
        df["non-opioid-only-d" + str(day) + "?"] = np.where(
            (df["analgesic-d" + str(day) + "?"] == 1) & (df["opioid-d" + str(day) + "?"] == 0), 1, 0)

    df = df.drop(df.filter(regex='Unnamed').columns, axis=1)
    save_dir = dataset_folder + "/data/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + dataset + "-received-drug-treatments-per-day.csv")


def process_medications(dataset="all"):
    """

    :return:
    """
    ace_inhibitors, aspirin, beta_blockers, non_aspirin_antiplatelets, statins = get_ami_treatments()
    analgesics, opioids = get_analgesics()
    print("analgesics = ", sorted(analgesics))
    non_opioid_analgesics = list(set(analgesics) - set(opioids))

    treatments = [analgesics, opioids, non_opioid_analgesics, aspirin, beta_blockers, statins,
                  non_aspirin_antiplatelets, ace_inhibitors]
    treatment_names = ["analgesics", "opioids", "non-opioid-analgesics", "aspirin", "beta-blockers",
                       "statins", "anti-platelets", "ace-inhibitors"]

    def aumc():
        aumc_drugs_df = pd.read_csv("aumc/data/AmsterdamUMCdb-v1.0.2/drugitems.csv", engine="python")
        aumc_cohort_df = pd.read_csv("aumc/data/aumc-ami-patients-master-unprocessed.csv")
        aumc_pids = aumc_cohort_df["admissionid"].unique().tolist()
        aumc_ami_cohort_drugs_df = aumc_drugs_df[aumc_drugs_df["admissionid"].isin(aumc_pids)]
        aumc_ami_cohort_drugs_df = aumc_ami_cohort_drugs_df[aumc_ami_cohort_drugs_df["administered"] > 0]
        aumc_ami_cohort_drugs_df = pd.merge(left=aumc_ami_cohort_drugs_df,
                                            right=aumc_cohort_df[["admissionid", "admittedat"]],
                                            how="left", on="admissionid")
        aumc_obj = {"dataset-name": "aumc", "cohort_df": aumc_cohort_df, "drug_df": aumc_drugs_df,
                    "ami_drug_df": aumc_ami_cohort_drugs_df,
                    "admission_col_name": "admissionid", "drug_col_name": "item",
                    "drug_admittime_col_name": "admittedat", "drug_start_col_name": "start",
                    "drug_stop_col_name": "stop", "save_folder": "./aumc", "check_column": "administered"}
        return aumc_obj

    def eicu():
        eicu_drugs_df = pd.read_csv("./eicu/data/eicu-collaborative-research-database-2.0/medication.csv.gz")
        eicu_cohort_df = pd.read_csv("./eicu/data/eicu-ami-patients-master-unprocessed.csv")
        eicu_pids = eicu_cohort_df["patientunitstayid"].unique().tolist()
        eicu_ami_cohort_drugs_df = eicu_drugs_df[eicu_drugs_df["patientunitstayid"].isin(eicu_pids)]
        eicu_ami_cohort_drugs_df = eicu_ami_cohort_drugs_df[pd.notnull(eicu_ami_cohort_drugs_df["drugname"])]
        eicu_ami_cohort_drugs_df = pd.merge(left=eicu_ami_cohort_drugs_df,
                                            right=eicu_cohort_df[["patientunitstayid", "hospitaladmitoffset"]],
                                            how="left", on="patientunitstayid")
        eicu_obj = {"dataset-name": "eicu", "cohort_df": eicu_cohort_df, "drug_df": eicu_drugs_df,
                    "ami_drug_df": eicu_ami_cohort_drugs_df,
                    "admission_col_name": "patientunitstayid", "drug_col_name": "drugname",
                    "drug_admittime_col_name": "hospitaladmitoffset", "drug_start_col_name": "drugstartoffset",
                    "drug_stop_col_name": "drugstopoffset", "save_folder": "./eicu", "check_column": "dosage"}
        return eicu_obj

    def mimic_iii():
        mimic_iii_drugs_df = pd.read_csv("./mimic-iii/data/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv.gz")
        mimic_iii_cohort_df = pd.read_csv("mimic-iii/data/mimic-iii-ami-patients-master-unprocessed.csv")
        mimic_iii_pids = mimic_iii_cohort_df["HADM_ID"].unique().tolist()
        mimic_iii_ami_cohort_drugs_df = mimic_iii_drugs_df[mimic_iii_drugs_df["HADM_ID"].isin(mimic_iii_pids)]
        mimic_iii_ami_cohort_drugs_df = mimic_iii_ami_cohort_drugs_df[pd.notnull(mimic_iii_ami_cohort_drugs_df["DRUG"])]
        mimic_iii_ami_cohort_drugs_df = pd.merge(left=mimic_iii_ami_cohort_drugs_df,
                                                 right=mimic_iii_cohort_df[["HADM_ID", "ADMITTIME"]],
                                                 how="left", on="HADM_ID")
        mimic_iii_obj = {"dataset-name": "mimic-iii", "cohort_df": mimic_iii_cohort_df, "drug_df": mimic_iii_drugs_df,
                         "ami_drug_df": mimic_iii_ami_cohort_drugs_df,
                         "admission_col_name": "HADM_ID", "drug_col_name": "DRUG",
                         "drug_admittime_col_name": "ADMITTIME", "drug_start_col_name": "STARTDATE",
                         "drug_stop_col_name": "ENDDATE", "save_folder": "./mimic-iii", "check_column": "DOSE_VAL_RX"}
        return mimic_iii_obj

    def mimic_iv():
        mimic_iv_drugs_df = pd.read_csv("./mimic-iv/data/hosp/prescriptions.csv.gz")
        print("mimic_iv_drugs_df.head() = ", mimic_iv_drugs_df.head(), mimic_iv_drugs_df.columns.tolist())
        mimic_iv_cohort_df = pd.read_csv("./mimic-iv/data/mimic-iv-ami-patients-master-unprocessed.csv")
        mimic_iv_pids = mimic_iv_cohort_df["hadm_id"].unique().tolist()
        mimic_iv_ami_cohort_drugs_df = mimic_iv_drugs_df[mimic_iv_drugs_df["hadm_id"].isin(mimic_iv_pids)]
        mimic_iv_ami_cohort_drugs_df = mimic_iv_ami_cohort_drugs_df[pd.notnull(mimic_iv_ami_cohort_drugs_df["drug"])]
        mimic_iv_ami_cohort_drugs_df = pd.merge(left=mimic_iv_ami_cohort_drugs_df,
                                                right=mimic_iv_cohort_df[["hadm_id", "admittime"]],
                                                how="left", on="hadm_id")

        mimic_iv_obj = {"dataset-name": "mimic-iv", "cohort_df": mimic_iv_cohort_df, "drug_df": mimic_iv_drugs_df,
                        "ami_drug_df": mimic_iv_ami_cohort_drugs_df,
                        "admission_col_name": "hadm_id", "drug_col_name": "drug",
                        "drug_admittime_col_name": "admittime", "drug_start_col_name": "starttime",
                        "drug_stop_col_name": "stoptime", "save_folder": "./mimic-iv", "check_column": "dose_val_rx"}

        print("mimic_iv_ami_cohort_drugs_df.head() = ", mimic_iv_ami_cohort_drugs_df.head(),
              len(mimic_iv_ami_cohort_drugs_df), mimic_iv_ami_cohort_drugs_df.columns.tolist())
        return mimic_iv_obj

    def process_orders(dataset_obj):
        print(" --- now processing medications for ", dataset_obj["dataset-name"], "---")
        print("1. determining whether cohort received analgesics during entire hospitalization")
        process_analgesics_orders(cohort_df=dataset_obj["cohort_df"], drugs_df=dataset_obj["drug_df"],
                                  drug_col_name=dataset_obj["drug_col_name"],
                                  admission_col_name=dataset_obj["admission_col_name"],
                                  dose_col_name=dataset_obj["check_column"],
                                  dataset=dataset_obj["dataset-name"],
                                  save_path=dataset_obj["save_folder"] + "/data/")
        print("2. determining whether cohort received AMI-related drugs during entire hospitalization")
        process_ami_drugs_orders(cohort_df=dataset_obj["cohort_df"], drugs_df=dataset_obj["drug_df"],
                                 drug_col_name=dataset_obj["drug_col_name"],
                                 admission_col_name=dataset_obj["admission_col_name"],
                                 dataset=dataset_obj["dataset-name"],
                                 save_path=dataset_obj["save_folder"] + "/data/")
        print("3. expanding drug orders across each day of hospitalization")
        expand_drug_prescriptions(treatments=treatments, treatment_names=treatment_names,
                                  drug_df=dataset_obj["ami_drug_df"],
                                  drug_col_name=dataset_obj["drug_col_name"],
                                  dataset=dataset_obj["dataset-name"],
                                  admit_time=dataset_obj["drug_admittime_col_name"],
                                  start_time=dataset_obj["drug_start_col_name"],
                                  stop_time=dataset_obj["drug_stop_col_name"], save_folder=dataset_obj["save_folder"],
                                  check_column=dataset_obj["check_column"])
        print("4. determining whether cohort received all treatments during first 5 days of hospitalization")
        create_per_day_therapy_features(dataset_folder=dataset_obj["save_folder"], dataset=dataset_obj["dataset-name"],
                                        cohort_df=dataset_obj["cohort_df"],
                                        admission_col_name=dataset_obj["admission_col_name"], number_of_days=3)

    if dataset == "all":
        dataset_objs = [aumc(), eicu(), mimic_iii(), mimic_iv()]
        for obj in dataset_objs:
            process_orders(obj)
    elif dataset == "aumc":
        process_orders(dataset_obj=aumc())
    elif dataset == "eicu":
        process_orders(dataset_obj=eicu())
    elif dataset == "mimic-iii":
        process_orders(dataset_obj=mimic_iii())
    elif dataset == "mimic-iv":
        process_orders(dataset_obj=mimic_iv())
    else:
        raise ValueError("incorrect dataset name. expected values are: 'aumc', 'eicu', 'mimic-iii', 'mimic-iv', 'all'")


if __name__ == '__main__':
    process_medications(dataset="mimic-iv")
