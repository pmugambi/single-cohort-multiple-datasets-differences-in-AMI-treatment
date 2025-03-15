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
    This function compiles a list of AMI-related drugs under 5 classes: aspirin, antiplatelets (other than aspirin),
    ACE-inhibitors, beta blockers, and statins. Once compiled, they are written into a json file to be used later
    when looking up which patients received orders for these drugs.
    The drug names were looked these up on Google
    :return: Nothing. The drugs names under each category are written to a file.
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
                                  "Brilinta", "Vorapaxar", "Carbasalaat Calcium", "Ascal"]
    statins = ["Atorvastatin", "Lipitor", "Fluvastatin", "Lescol", "Lovastatin", "Altoprev", "Pitavastatin", "Livalo",
               "Zypitamag", "Pravastatin", "Pravachol", "Rosuvastatin", "Crestor", "Ezallor", "Simvastatin", "Zocor"]

    obj = {"ace-inhibitors": ace_inhibitors, "aspirin": aspirin, "beta-blockers": beta_blockers,
           "non-aspirin-antiplatelets": non_aspirin_anti_platelets, "statins": statins}
    Path("./shared-files/").mkdir(parents=True, exist_ok=True)
    with open("./shared-files/ami-drug-treatments.json", "w") as outfile:
        json.dump(obj, outfile)


def create_analgesics_file():
    """
    This function compiles a list of analgesics and writes them into a json file to be used later
    when looking up which patients received orders analgesia.
    The common list of analgesia under 3 categories; opioids, over the counter analgesia, and local anaesthetics
    were looked up on Google (written in the various lists below). Additionally, unique list of medications placed
    in MIMIC-III, eICU, and AUMC datasets were retrieved and manually labeled. For MIMIC-III, the labeling was done
    by a clinician (domain expert in the study) while for the other two, labeling was done by looking up the names
    on Google.
    These manually labelled "oracle" files are then used to add to the list of the commonly prescribed analgesia
    obtained from Google. The final list is categorized into opioids/narcotics, non-opioid/non-narcotic.
    :return: Nothing. The compiled list of analgesia is written to a file.
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
    # datasets = ["aumc", "eicu", "mimic-iii"]

    analgesics = opioids + otc_analgesics + local_anaesthetics

    for i in range(len(analgesics_lists)):
        df = dfs[i]
        pain_meds = df[df[drug_col_names[i]].str.contains(
            combine_string(analgesics), case=False, na=False)][drug_col_names[i]].unique().tolist()
        d_opioids = df[df[drug_col_names[i]].str.contains(
            combine_string(opioids), case=False, na=False)][drug_col_names[i]].unique().tolist()
        # missing_analgesics = list(set(analgesics_lists[i]) - set(pain_meds))
        analgesics = analgesics + list(set(analgesics_lists[i]) - set(pain_meds))
        missing_opioids = list(set(opioids_lists[i]) - set(d_opioids))
        opioids = opioids + missing_opioids

    analgesics = list(set([x.lower() for x in analgesics]))
    opioids = list(set([x.lower() for x in opioids]))

    analgesics_obj = {"analgesics": analgesics, "opioids": opioids}
    Path("./shared-files/").mkdir(parents=True, exist_ok=True)
    with open("./shared-files/analgesics.json", "w") as outfile:
        json.dump(analgesics_obj, outfile)


def get_ami_treatments():
    """
    Reads the list of AMI drugs from the "ami-drug-treatments" json file and returns them.
    If the file do not exist, it calls the function to return it.
    :return: List of AMI drugs for each of the categories.
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
    Reads the list of analgesics from the "analgesics.json" file and returns them.
    If the file do not exist, it calls the function to return it.
    :return: List of analgesics, categorized into "all" analgesics and "opioid" analgesics
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
    Checks whether patients received analgesia at any point in their hospitalization.
    To make it reusable for all datasets, variables such as 'drug_column_name',
    i.e., which column contains the drug names, are passed as arguments to this function.
    :param cohort_df: a dataframe of the list of patients
    :param drugs_df: a dataframe of the prescriptions for the patient cohort in (cohort_df)
    :param drug_col_name: the column name in 'drugs_df' that contains names of the prescribed drug
    :param admission_col_name: the column name in 'cohort_df' that contains admission_ids of the patients
    :param dose_col_name: the column name in 'drugs_df' that contains dosage of the prescribed drug
    :param dataset: the name of the dataset being processed (to enable with the naming of the output file)
    :param save_path: the path where the output file should be written.
    :return: Nothing. Final output written to a file.
    """
    analgesics, opioids = get_analgesics()
    non_opioid_analgesics = list(set(analgesics) - set(opioids))
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
    Path(save_path).mkdir(parents=True, exist_ok=True)
    analgesics_binary_df.to_csv(save_path + dataset + "-received-analgesics-during-hospitalization.csv", index=False)


def process_ami_drugs_orders(cohort_df, drugs_df, drug_col_name, admission_col_name, dataset, save_path):
    """
    Checks whether patients received AMI-related drugs at any point in their hospitalization.
    To make it reusable for all datasets, variables such as 'drug_column_name',
    i.e., which column contains the drug names, are passed as arguments to this function.
    :param cohort_df: a dataframe of the list of patients
    :param drugs_df: a dataframe of the prescriptions for the patient cohort in (cohort_df)
    :param drug_col_name: the column name in 'drugs_df' that contains names of the prescribed drug
    :param admission_col_name: the column name in 'cohort_df' that contains admission_ids of the patients
    :param dataset: the name of the dataset being processed (to enable with the naming of the output file)
    :param save_path: the path where the output file should be written.
    :return: Nothing. Final output written to a file.
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

    Path(save_path).mkdir(parents=True, exist_ok=True)
    ami_drugs_binary_df.to_csv(save_path + dataset + "-received-ami-drugs-during-hospitalization.csv", index=False)


def write_treatments_by_day_files(drug_df, admit_time, start_time, stop_time, dataset, save_folder, save_category):
    """
    Distributes the prescription amount to all the days in the prescription, and writes
    a new file with each day of treatment as it's own row.
    For instance, if previous data file (i.e., drug_df) had one row of prescription of aspirin to be
    taken from 1/3 - 1/5, this function will create 3 rows for each day of aspirin, i.e., 1/3, 1/4, and 1/5.
    This is important in the analysis of drug orders for specific day(s) of hospitalization.
    :param drug_df: The DF of all patients' prescriptions with each prescription amount
    :param admit_time: Time when the patient was admitted to hospital/ICU
    :param start_time: Time when the patient started taking the drug
    :param stop_time: Time when the patient should stop taking the drug
    :param dataset: The name of the dataset being processed. (To enable with the naming of the output file)
    :param save_folder: The path to the folder where the output file should be written
    :param save_category: The drug category being processed (e.g.s, opioids, aspirin, statins)
    :return: A DF of all patients' prescriptions with each day in each prescription is written and saved in save_folder
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
        # print("working on batch no: ", batch_no)
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
                if (j + admit_day).values.tolist()[0] == 0.0:  # adding this check because, even after all the checks,
                    # there are a few instances where day 1 is recorded as day 0
                    row["day-no"] = j + admit_day + 1.0
                row["day-no"] = j + admit_day
                batch_df = pd.concat([batch_df, row])
        save_dir = save_folder + "expanded-batch-files/" + save_category + "/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        batch_df.to_csv(save_dir + "batch-no-" + str(batch_no) + "-expanded-medication.csv", index=False)

    treatment_directory_path = save_folder + "expanded-batch-files/" + save_category
    files = [f for f in listdir(treatment_directory_path) if isfile(join(treatment_directory_path, f))]
    treatment_per_day_df = pd.DataFrame()
    for f in files:
        # print("now processing file ", f)
        df = pd.read_csv(treatment_directory_path + "/" + f)
        treatment_per_day_df = pd.concat([treatment_per_day_df, df])
        save_dir = save_folder + "drug-orders-per-day/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        treatment_per_day_df.to_csv(save_dir + dataset + "-ami-cohort-" + save_category + "-per-day-records.csv",
                                    index=False)


def expand_drug_prescriptions(treatments, treatment_names, drug_df, drug_col_name, admit_time, start_time,
                              stop_time, dataset, save_folder, check_column=None):
    """
    Calls the **write_treatments_by_day_files** function above and creates dataframes of expanded prescriptions for
    each treatment in this study.
    :param treatments: A list of all pharmacological treatments that have to be analyzed
    :param treatment_names: A list of the names of the treatments
    :param drug_df: A dataframe of the prescriptions from a specific dataset
    :param drug_col_name: The column in drug_df that contains names of the drugs
    :param admit_time: Time when the patient was admitted to hospital/ICU
    :param start_time: Time when the patient started taking the drug
    :param stop_time: Time when the patient should stop taking the drug
    :param dataset: The name of the dataset being processed. (To enable with the naming of the output file)
    :param save_folder: The path to the folder where the output file should be written
    :param check_column: Extra columns (other than those included in this function as arguments) that may need to be
    retrieved and values checked before/during the expansion
    :return: A DF of all patients' prescriptions with each day in each prescription is written and saved in save_folder
    """
    for t_name in treatment_names:
        # print("now processing ==== ", t_name)
        treatment = treatments[treatment_names.index(t_name)]
        treatment_df = drug_df[drug_df[drug_col_name].str.contains(
            combine_string(treatment), case=False, na=False)]
        if check_column is not None:
            treatment_df = treatment_df[pd.notnull(treatment_df[check_column])]
        write_treatments_by_day_files(drug_df=treatment_df, admit_time=admit_time, start_time=start_time,
                                      stop_time=stop_time, dataset=dataset,
                                      save_folder=save_folder, save_category=t_name)


def create_per_day_therapy_features(dataset_folder, dataset, cohort_df, admission_col_name, number_of_days=2):
    """
    Once prescriptions have been expanded and day numbers added by function **expand_drug_prescriptions** above,
    this function reads the output files and creates per-day drug order features for each treatment.
    :param dataset_folder: The path to the folder containing files of the dataset being processed
    :param dataset: The name of the dataset being processed
    :param cohort_df: A dataframe of the cohort under study for the dataset being analyzed
    :param admission_col_name: The column name (in cohort_df) that contains patients' admission ids
    :param number_of_days: The number of days for which the features should be created
    :return: A dataframe containing patients' admission ids and the corresponding 0/1 values for whether they received
    specific treatments for each of the days (in number_of_days) being processed. This dataframe is written to CSV
    and saved in the 'dataset_folder'.
    """
    read_dir_path = dataset_folder + "drug-orders-per-day/"

    opioids_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-opioids-per-day-records.csv")
    non_opioid_analgesics_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-non-opioid-analgesics-per-day-records.csv")
    analgesics_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-analgesics-per-day-records.csv")
    aspirin_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-aspirin-per-day-records.csv")
    ace_inhibitors_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-ace-inhibitors-per-day-records.csv")
    beta_blockers_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-beta-blockers-per-day-records.csv")
    statin_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-statins-per-day-records.csv")
    anti_platelets_df = pd.read_csv(
        read_dir_path + dataset + "-ami-cohort-anti-platelets-per-day-records.csv")
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

    save_dir = dataset_folder + "per-day/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + dataset + "-received-drug-treatments-per-day.csv", index=False)


def process_medications(dataset="all", number_of_days=1, process_per_day=False):
    """
    This function puts it all together. It iterates through each study dataset and determines if each patient in each
    dataset received each of the treatments under study at one point during their hospitalization, and separately, for
    specific days of hospitalization.
    The analysis for specific days of hospitalization is optional and should be specified by setting process_per_day
    argument to True
    :param dataset: The dataset for which drug orders should be processed. Default is 'all' however, each specific
    dataset can be specified as shown by function process_orders() below.
    :param process_per_day: whether to conduct an analysis for the orders for the first 'number_of_days'
    days of hospitalization.
    :param number_of_days: the number of days for which specific per-day analyses of drug orders should be conducted.
    Default value is 1, meaning, first day of hospitalization. Setting the value to 3 (for instance), means that the
    previous functions will check whether each patient received treatment(s) for the first 3 days of hospitalization.
    :return: Nothing. Output files are saved in the datasets' folders
    """
    ace_inhibitors, aspirin, beta_blockers, non_aspirin_antiplatelets, statins = get_ami_treatments()
    analgesics, opioids = get_analgesics()
    non_opioid_analgesics = list(set(analgesics) - set(opioids))

    treatments = [analgesics, opioids, non_opioid_analgesics, aspirin, beta_blockers, statins,
                  non_aspirin_antiplatelets, ace_inhibitors]
    treatment_names = ["analgesics", "opioids", "non-opioid-analgesics", "aspirin", "beta-blockers",
                       "statins", "anti-platelets", "ace-inhibitors"]

    def aumc():
        aumc_drugs_df = pd.read_csv("aumc/data/raw/drugitems.csv", encoding='latin-1', engine="python")
        aumc_cohort_df = pd.read_csv("aumc/data/processed/features-files/aumc-ami-patients-master-unprocessed.csv")
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
                    "drug_stop_col_name": "stop", "save_folder": "./aumc/data/processed/prescription-orders/",
                    "check_column": "administered"}
        return aumc_obj

    def eicu():
        eicu_drugs_df = pd.read_csv("./eicu/data/raw/medication.csv.gz")
        eicu_cohort_df = pd.read_csv("eicu/data/processed/features-files/eicu-ami-patients-master-unprocessed.csv")
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
                    "drug_stop_col_name": "drugstopoffset", "save_folder": "./eicu/data/processed/prescription-orders/",
                    "check_column": "dosage"}
        return eicu_obj

    def mimic_iii():
        mimic_iii_drugs_df = pd.read_csv("./mimic-iii/data/raw/PRESCRIPTIONS.csv.gz")
        mimic_iii_cohort_df = pd.read_csv(
            "mimic-iii/data/processed/features-files/mimic-iii-ami-patients-master-unprocessed.csv")
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
                         "drug_stop_col_name": "ENDDATE",
                         "save_folder": "./mimic-iii/data/processed/prescription-orders/",
                         "check_column": "DOSE_VAL_RX"}
        return mimic_iii_obj

    def mimic_iv():
        mimic_iv_drugs_df = pd.read_csv("./mimic-iv/data/raw/hosp/prescriptions.csv.gz")
        mimic_iv_cohort_df = pd.read_csv(
            "mimic-iv/data/processed/features-files/mimic-iv-ami-patients-master-unprocessed.csv")
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
                        "drug_stop_col_name": "stoptime",
                        "save_folder": "./mimic-iv/data/processed/prescription-orders/",
                        "check_column": "dose_val_rx"}
        return mimic_iv_obj

    def process_orders(dataset_obj):
        print(" --- now processing medications for ", dataset_obj["dataset-name"], "---")
        print("1. determining whether cohort received analgesics during entire hospitalization")
        process_analgesics_orders(cohort_df=dataset_obj["cohort_df"], drugs_df=dataset_obj["drug_df"],
                                  drug_col_name=dataset_obj["drug_col_name"],
                                  admission_col_name=dataset_obj["admission_col_name"],
                                  dose_col_name=dataset_obj["check_column"],
                                  dataset=dataset_obj["dataset-name"],
                                  save_path=dataset_obj["save_folder"] + "entire-hospitalization/")
        print("2. determining whether cohort received AMI-related drugs during entire hospitalization")
        process_ami_drugs_orders(cohort_df=dataset_obj["cohort_df"], drugs_df=dataset_obj["drug_df"],
                                 drug_col_name=dataset_obj["drug_col_name"],
                                 admission_col_name=dataset_obj["admission_col_name"],
                                 dataset=dataset_obj["dataset-name"],
                                 save_path=dataset_obj["save_folder"] + "entire-hospitalization/")
        if process_per_day:
            print("3. expanding drug orders across each day of hospitalization")
            expand_drug_prescriptions(treatments=treatments, treatment_names=treatment_names,
                                      drug_df=dataset_obj["ami_drug_df"],
                                      drug_col_name=dataset_obj["drug_col_name"],
                                      dataset=dataset_obj["dataset-name"],
                                      admit_time=dataset_obj["drug_admittime_col_name"],
                                      start_time=dataset_obj["drug_start_col_name"],
                                      stop_time=dataset_obj["drug_stop_col_name"],
                                      save_folder=dataset_obj["save_folder"],
                                      check_column=dataset_obj["check_column"])

            print("4. determining whether cohort received treatment during their first day of hospitalization")
            create_per_day_therapy_features(dataset_folder=dataset_obj["save_folder"],
                                            dataset=dataset_obj["dataset-name"],
                                            cohort_df=dataset_obj["cohort_df"],
                                            admission_col_name=dataset_obj["admission_col_name"],
                                            number_of_days=number_of_days)

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
    process_medications()
