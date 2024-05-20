import math
from pathlib import Path

import pandas as pd
import numpy as np
from helpers import create_combined_drugs_string as compare_strings


def convert_parental_to_oral(drugname, dose_in_mg):
    """

    :param drugname:
    :param dose_in_mg:
    :return:

    Informed by this chart -
    https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
    """

    if ("morphine" in drugname) | ("morfine" in drugname):
        oral_mg = dose_in_mg * 3
    elif "codeine" in drugname:
        oral_mg = dose_in_mg * 1.7
    elif ("hydromorphone" in drugname) | ("dilaudid" in drugname):
        oral_mg = dose_in_mg * 5
    elif ("meperidine" in drugname) | ("pethidine" in drugname):
        oral_mg = dose_in_mg * 4
    elif ("fentanyl" in drugname) | ("sublimaze" in drugname):
        oral_mg = dose_in_mg * 300  # according to the chart, 0.1mg of fentanyl parental is equivalent to 10mg of
        # parental morphine. 10mg parental morphine is equivalent to 30mg oral morphine. hence the 30*10 multiplier
        # for a 1mg parental fentanyl
    elif "methadon" in drugname:
        # using this paper to obtain this value because it was missing in the chart above -
        # https://pubmed.ncbi.nlm.nih.gov/19042849/#:~:text=The%20described%20ratio%20for%20methadone,
        # methadone%20is%20PR%20to%20PO.
        oral_mg = dose_in_mg * 2
    else:
        # print("drug ", drugname, "not in parental-to-oral list ")
        oral_mg = dose_in_mg  # after running this multiple times, I know the drugs missing in this list are most
        # likely taken po, so I'm treating their dosage as an oral equivalent
    return oral_mg


def convert_to_oral_mme(drugname, route, oral_mg):
    """

    :param drugname:
    :param route:
    :param oral_mg:
    :return:

    using this conversion chart -
    https://www.hhs.gov/guidance/sites/default/files/hhs-guidance-documents/Opioid%20Morphine%20EQ%20Conversion%20Factors%20%28vFeb%202018%29.pdf
    """
    transdermal_routes = ["transdermal", "patch", "top", "tp"]

    if route in transdermal_routes:
        if "fentanyl" in drugname:
            # using the patch conversion value from the chart above
            mme = oral_mg * 7.2
        else:
            print("drug is passed through transdermal route, but is not fentanyl, it is; ", drugname)
            mme = np.nan
    else:
        # even though some of these drugnames e.g., percocet or hydrocode w/ acetaminophen contain non-opioid drugs,
        # here, we can just convert the opioid dosage because we have teased it out e.g., hydrocodone, before this
        if "codeine" in drugname:
            mme = oral_mg * 0.15
        elif drugname == "dihydrocodeine":
            mme = oral_mg * 0.25
        elif ("fentanyl" in drugname) | ("sublimaze" in drugname):
            mme = oral_mg * 0.18  # there were 3 listed values in the chart, I chose the largest. Didn't think I could
            # differentiate the three oral/nasal routes using the provided route admission
        elif ("hydrocodon" in drugname) | ("norco" in drugname):
            mme = oral_mg
        elif ("hydromorphone" in drugname) | ("dilaudid" in drugname):
            mme = oral_mg * 4
        elif "levorphanol" in drugname:
            mme = oral_mg * 11
        elif ("meperidine" in drugname) | ("pethidine" in drugname):
            mme = oral_mg * 0.1
        elif ("morphine" in drugname) | ("morfine" in drugname) | ("ms contin" in drugname):
            mme = oral_mg
        elif "opium" in drugname:
            mme = oral_mg
        elif ("oxycodon" in drugname) | ("percocet" in drugname) | ("roxicodone" in drugname):
            mme = oral_mg * 1.5
        elif "oxymorphone" in drugname:
            mme = oral_mg * 3
        elif "pentazocine" in drugname:
            mme = oral_mg * 0.37
        elif "tapentadol" in drugname:
            mme = oral_mg * 0.4
        elif ("tramadol" in drugname) | ("ultram" in drugname):
            mme = oral_mg * 0.1
        elif "methadon" in drugname:
            if 0 < oral_mg <= 20:
                mme = oral_mg * 4
            elif 20 < oral_mg <= 40:
                mme = oral_mg * 8
            elif 40 < oral_mg <= 60:
                mme = oral_mg * 10
            elif oral_mg > 60:
                mme = oral_mg * 12
            else:
                print("methadone dose not within 0 - >60 range, hence, can't be placed. dose = ", oral_mg)
                mme = np.nan
        else:
            print("drug ", drugname, " with oral_mg ", oral_mg, " was not found in the oral-mg to mme conversion chart")
            mme = np.nan
    return mme


def assign_single_value_dosage(dose):
    if "-" in dose:
        sv_dose = dose.split("-")[0]
    elif "to" in dose:
        sv_dose = dose.split("to")[0]
    elif "," in dose:
        sv_dose = dose.replace(",", "")
    else:
        sv_dose = dose
    return sv_dose


def convert_opioid_dose_to_mg_m3():
    m3_opioid_per_day_df = pd.read_csv(
        "./mimic-iii/data/drug-orders-per-day/mimic-ami-cohort-opioids-per-day-records.csv")
    m3_opioid_per_day_df["minimum-dose"] = m3_opioid_per_day_df["DOSE_VAL_RX"].apply(assign_single_value_dosage)
    m3_opioid_per_day_df["minimum-dose"] = pd.to_numeric(m3_opioid_per_day_df["minimum-dose"])
    m3_opioid_per_day_df = m3_opioid_per_day_df[m3_opioid_per_day_df["minimum-dose"] > 0]
    m3_opioid_per_day_df["DRUG"] = m3_opioid_per_day_df["DRUG"].str.lower()
    m3_opioid_per_day_df["DOSE_UNIT_RX"] = m3_opioid_per_day_df["DOSE_UNIT_RX"].str.lower()

    df = pd.DataFrame()

    for index, row in m3_opioid_per_day_df.iterrows():
        if row["DOSE_UNIT_RX"] != "mg":
            if "mcg" in row["DOSE_UNIT_RX"]:
                row["opioid-dose(mg)"] = row["minimum-dose"] / 1000
            elif row["DOSE_UNIT_RX"] == "udcup":
                # the oral cup solution for codeine/acetaminophen dosage is provided in this link:
                # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=83a536d8-385d-46ed-9cd2-47b7efe96ccf
                if ("oxycodone-acetaminophen" in row["DRUG"]) | ("acetaminophen w/codeine" in row["DRUG"]):
                    row["opioid-dose(mg)"] = row["minimum-dose"] * 12  # udcup doses are recorded in # of cups, of 5ml
            elif row["DOSE_UNIT_RX"] == "ml":  # checked these in data and they seem to map to drugs below
                if ("oxycodone-acetaminophen" in row["DRUG"]) | ("acetaminophen w/codeine" in row["DRUG"]):
                    # the oral cup solution for codeine/acetaminophen dosage is provided in this link:
                    # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=83a536d8-385d-46ed-9cd2-47b7efe96ccf
                    # 5ml: 12mg, 10ml: 24mg, 12.5ml: 30mg, 15ml: 36mg
                    row["opioid-dose(mg)"] = (row["minimum-dose"] / 5) * 12  # from the list in the dco, it appears,
                    # dividing the value by 5ml and multiply it by 12
                elif "guaifenesin-codeine" in row["DRUG"]:
                    # the oral cup solution for guaifenesin-codeine dosage is provided in this link:
                    # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=88d0994c-4270-4408-8837-bd97510b2118
                    row["opioid-dose(mg)"] = (row["minimum-dose"] / 5.0) * 10  # each 5ml has 10mg of codeine and
                    # 100mg of guaifenesin
                else:
                    row["opioid-dose(mg)"] = row["minimum-dose"]  # I'm making an assumption here that ml dose is
                # equivalent to it's mg dose. I haven't seen these cases in data, but just in case
            elif row["DOSE_UNIT_RX"] == "tab":
                if ("oxycodone-acetaminophen" in row["DRUG"]) | (row["DRUG"] == "oxycodone"):
                    row["opioid-dose(mg)"] = row["minimum-dose"] * 5  # using info from this chart, i.e., oxycodone/APAP
                    # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                elif "acetaminophen w/codeine" in row["DRUG"]:
                    row["opioid-dose(mg)"] = row["minimum-dose"] * 30  # using info from this chart, i.e., tylenol
                    # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                elif "hydrocodone-acetaminophen" in row["DRUG"]:
                    row["opioid-dose(mg)"] = row["minimum-dose"] * 5  # using info from this chart, i.e., norco
                    # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                else:
                    print("not sure what to make this be")
                    print(row["DRUG"], row["DOSE_VAL_RX"], row["minimum-dose"], row["DOSE_UNIT_RX"], row["ROUTE"])
            else:
                print("this is a new unit == > ")
                print(row["DRUG"], row["DOSE_VAL_RX"], row["minimum-dose"], row["DOSE_UNIT_RX"], row["ROUTE"])
        else:
            row["opioid-dose(mg)"] = row["minimum-dose"]
        df = df.append(row, sort=False)
    save_dir = "./mimic-iii/data/drug-orders-per-day/dosage/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + "mimic-ami-cohort-opioids-per-day-dose-in-mg.csv")
    print("df.head = ", df.head(), len(df), "df.columns = ", df.columns.tolist())


def convert_opioid_dose_to_mg_eicu():
    oral_routes = ["po", "ng", "oral", "po/ng"]
    parental_routes = ["iv", "drip iv", "im", "sc", "sq"]
    transdermal_routes = ["transdermal", "patch", ]
    eicu_opioid_per_day_df = pd.read_csv("./eicu/data/drug-orders-per-day/eicu-ami-cohort-opioids-per-day-records.csv")
    # eicu_opioid_per_day_df = eicu_opioid_per_day_df[~eicu_opioid_per_day_df["dosage"].isin(["0"])]
    eicu_opioid_per_day_df["drugname"] = eicu_opioid_per_day_df["drugname"].str.lower()
    eicu_opioid_per_day_df["dosage"] = eicu_opioid_per_day_df["dosage"].str.lower()
    print("eicu_opioid_per_day_df.head() = ", eicu_opioid_per_day_df.head(), eicu_opioid_per_day_df.columns.tolist())
    print("unique drug names in eicu_opioid_per_day_df = ",
          sorted(eicu_opioid_per_day_df["drugname"].unique().tolist()))
    # print("unique dosage in eicu_opioid_per_day_df = ", sorted(eicu_opioid_per_day_df["dosage"].unique().tolist()))

    df = pd.DataFrame()

    for index, row in eicu_opioid_per_day_df.iterrows():
        v = row["dosage"].split(" ")
        if len(v) != 2:
            pass
            # print("v = ", v, len(v))
        else:
            dose = v[0]
            unit = v[1]
            try:
                row["minimum-dose"] = float(assign_single_value_dosage(dose))
                if unit != "mg":  # change this to check for parental
                    # print(row["drugname"], row["dosage"], dose, row["minimum-dose"], unit)
                    if "mcg" in unit:
                        row["opioid-dose(mg)"] = row["minimum-dose"] / 1000
                    elif unit == "ml":
                        if "fentanyl 2000 mcg/100 ml ns" in row["drugname"]:
                            row["opioid-dose(mg)"] = 2.0  # I looked over the data, and noticed that this drug made up
                            # the majority of records with a ml unit. The dose is marked as 100ml, which is the
                            # colution volume, but the opioid dose is in the name, 2000mcg, i.e., 2mg.
                            # I'm hardcoding because I can't find an easier way to do this
                        row["opioid-dose(mg)"] = row["minimum-dose"]  # couldn't tell whether the ml dosage is
                        # equivalent to mg, but the values in the data are small, mostly 2ML, so I assumed they are
                    elif "tab" in unit:
                        if ("oxycodone" in row["drugname"]) | ("percocet" in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"] * 5
                            # using info from this chart, i.e., oxycodone/APAP #
                            # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                        elif ("hydrocodon" in row["drugname"]) | ("norco" in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"] * 5  # using info from this chart, i.e., norco
                            # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                        elif ("tramadol hcl 50 mg po tabs" in row["drugname"]) | (
                                'tramadol 50 mg tab' in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"]
                        else:
                            print("not sure what to make this be")
                            print(row["drugname"], row["dosage"], dose, row["minimum-dose"], unit)
                    else:
                        if ("fentanyl" in row["drugname"]) & (row["routeadmin"] not in oral_routes):
                            row["opioid-dose(mg)"] = row["minimum-dose"] * 0.05  # I'm assuming that each injection/iv
                            # is 1ml, which is equivalent to 50mcg/0.05mg of fentanyl according to this doc -
                            # https://www.pfizermedicalinformation.com/en-us/fentanyl-citrate/dosage-admin
                        elif ("oxycodone" in row["drugname"]) | ("percocet" in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"] * 5  # using info from this chart,
                            # i.e., oxycodone/APAP
                            # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                        elif ("hydrocodon" in row["drugname"]) | ("norco" in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"] * 5  # using info from this chart, i.e., norco
                            # https://www.med.unc.edu/aging/wp-content/uploads/sites/753/2018/06/Analgesic-Equivalent-Chart.pdf
                        elif ("morphine" in row["drugname"]) | ("hydromorphone" in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"]  # I'm assuming that the recorded minimum dose
                            # is the opioid equivalent in mgs
                        elif ("tramadol hcl 50 mg po tabs" in row["drugname"]) | (
                                'tramadol 50 mg tab' in row["drugname"]):
                            row["opioid-dose(mg)"] = row["minimum-dose"]
                        elif "meperidine range inj" in row["drugname"]:
                            row["opioid-dose(mg)"] = row["minimum-dose"]
                        else:
                            print("unit it not mg, but not delt with yet")
                            print(row["drugname"], row["dosage"], dose, row["minimum-dose"], unit)

                else:
                    row["opioid-dose(mg)"] = row["minimum-dose"]
            except ValueError:
                pass
        df = df.append(row, sort=False)
    save_dir = "./eicu/data/drug-orders-per-day/dosage/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + "eicu-ami-cohort-opioids-per-day-dose-in-mg.csv")


def convert_opioid_dose_to_mg_aumc():
    """

    :return:
    """
    aumc_opioids_per_day_df = pd.read_csv(
        "aumc/data/drug-orders-per-day/aumc-ami-cohort-opioids-per-day-records.csv")
    aumc_opioids_per_day_df["item"] = aumc_opioids_per_day_df["item"].str.lower()

    df = pd.DataFrame()
    for index, row in aumc_opioids_per_day_df.iterrows():
        if row["administeredunit"] != "mg":
            if row["administeredunit"] == "g":
                row["opioid-dose(mg)"] = row["administered"] * 1000
            elif row["administeredunit"] == "�g":  # I think the ? in-front of g is micro
                row["opioid-dose(mg)"] = row["administered"] / 1000
            else:
                print(row["item"], row["administered"], row["administeredunit"])
        else:
            row["opioid-dose(mg)"] = row["administered"]
        df = df.append(row, sort=False)
    save_dir = "aumc/data/drug-orders-per-day/dosage/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir + "aumc-ami-cohort-opioids-per-day-dose-in-mg.csv")


def convert_prescription_doses_to_mme():
    parental_routes = ["iv", "im", "sc", "sq", "subcutaneous", "ivp", "inj", "intraven", "subq", "cosig iv",
                       "subcut", "intramusc", "ed", "pb", "irr"]
    transdermal_routes = ["transdermal", "patch", "top", "tp"]

    folders = ["./aumc", "./eicu", "./mimic-iii"]
    datasets = ["aumc", "eicu", "mimic"]
    route_columns = ["route", "routeadmin", "ROUTE"]
    drug_columns = ["item", "drugname", "DRUG"]
    # read datasets
    for i in range(len(folders)):
        df = pd.read_csv(folders[i] + "/data/drug-orders-per-day/dosage/" + datasets[i] +
                         "-ami-cohort-opioids-per-day-dose-in-mg.csv")
        print("dataset = ", datasets[i])
        if datasets[i] != "aumc":
            df[route_columns[i]] = df[route_columns[i]].str.lower()
            transdermal_df = df[df[route_columns[i]].str.contains(compare_strings(transdermal_routes),
                                                                  case=False, na=False)]
            oral_et_al_df = df[~df[route_columns[i]].str.contains(
                compare_strings(transdermal_routes + parental_routes), case=False, na=False)]
            oral_et_al_df["opioid-dose-oral(mg)"] = oral_et_al_df["opioid-dose(mg)"]
            transdermal_df["opioid-dose-oral(mg)"] = np.nan
            parental_df = df[df[route_columns[i]].str.contains(compare_strings(
                parental_routes), case=False, na=False)]
            new_df = pd.concat([oral_et_al_df, transdermal_df])
        else:
            parental_df = df
            parental_df["route"] = "unknown"
        # convert parental mg to oral mg
        parental_df["opioid-dose-oral(mg)"] = parental_df.apply(
            lambda x: convert_parental_to_oral(x[drug_columns[i]], x["opioid-dose(mg)"]), axis=1)
        p_df = parental_df
        if datasets[i] != "aumc":
            p_df = pd.concat([new_df, p_df])
        save_dir = folders[i] + "/data/drug-orders-per-day/dosage/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p_df.to_csv(save_dir + datasets[i] + "-opioids-per-day-oral-dose-in-mg.csv")

        # convert_to_oral_mme(drugname, route, oral_mg)
        p_df["opioid-dose-mme"] = p_df.apply(
            lambda x: convert_to_oral_mme(drugname=x[drug_columns[i]], route=x[route_columns[i]],
                                          oral_mg=x["opioid-dose-oral(mg)"]), axis=1)
        print(p_df[[drug_columns[i], "opioid-dose(mg)", "opioid-dose-oral(mg)", "opioid-dose-mme"]].head())
        p_df.to_csv(folders[i] + "/data/drug-orders-per-day/dosage/" + datasets[i] + "-opioids-per-day-dose-in-mme.csv")


def format_eicu_aspirin_doses(dosage):
    """

    :param dosage:
    :return:
    """
    dose_in_mg = math.nan
    try:
        dosage = dosage.split(" ")
        if "mg" in dosage[1]:
            dose_in_mg = float(dosage[0])
        else:
            try:
                ends_with = float(dosage[1])
                dose_in_mg = float(dosage[0])
            except ValueError:
                pass
            else:
                pass
    except IndexError:
        pass
    return dose_in_mg


def consolidate_dosage():
    folders = ["./aumc", "./eicu", "./mimic-iii"]
    datasets = ["aumc", "eicu", "mimic"]
    pid_cols = ["admissionid", "patientunitstayid", "hadm_id"]
    dosage_cols = ["administered", "dosage", "dose_val_rx"]

    treatments = ["opioids", "ace-inhibitors", "aspirin", "anti-platelets", "beta-blockers", "statins"]

    for i in range(len(folders)):
        dataset = datasets[i]
        print("dataset = ", dataset)
        for treatment in treatments:
            if treatment == "opioids":
                df = pd.read_csv(
                    folders[i] + "/data/drug-orders-per-day/dosage/" + dataset + "-opioids-per-day-dose-in-mme.csv")
                dosage_col = "opioid-dose-mme"
                label = "mme"
            elif treatment == "aspirin":
                df = pd.read_csv(folders[i] + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-"
                                 + treatment + "-per-day-records.csv")
                label = "aspirin-dose(mg)"
                if dataset == "eicu":
                    df[dosage_cols[i]] = df[dosage_cols[i]].str.lower()
                    df["aspirin-dosage(mg)"] = df[dosage_cols[i]].apply(format_eicu_aspirin_doses)
                    dosage_col = "aspirin-dosage(mg)"
                else:
                    dosage_col = dosage_cols[i]
            else:
                df = pd.read_csv(folders[i] + "/data/drug-orders-per-day/" + dataset + "-ami-cohort-"
                                 + treatment + "-per-day-records.csv")
                label = treatment+"-orders"
                dosage_col = dosage_cols[i]
            df.columns = [x.lower() for x in df.columns]
            print("before: len of df = ", len(df))
            # df = df[pd.notnull(df["opioid-dose-mme"])]
            df = df[pd.notnull(df[dosage_col])]
            print("after: len of df = ", len(df))
            unique_pids = df[pid_cols[i]].unique().tolist()
            doses_or_orders = []
            for pid in unique_pids:
                p_doses_or_orders = [pid]
                p_df = df[df[pid_cols[i]] == pid]
                for day in range(1, 3):
                    p_day_df = p_df[np.ceil(p_df["day-no"]) == day]
                    if treatment == "opioids":
                        day_dose_or_order = sum(p_day_df[dosage_col])
                    elif treatment == "aspirin":
                        day_dose_or_order = sum(p_day_df[dosage_col])
                    else:
                        # counting number of orders placed, i.e., 1 record for each order
                        # day_dose_or_order = len(p_day_df[dosage_col].dropna().values.tolist())
                        # todo: fix this: try group by sample sizes
                        # df.groupby(['col5','col2']).size()
                        day_dose_or_order = len(p_day_df[dosage_col].dropna().values.tolist())  # this is kind of a
                        # cheat. The better thing to do is to use the records before expanding them by day
                    p_doses_or_orders.append(day_dose_or_order)
                if (treatment == "opioids") | (treatment == "aspirin"):
                    p_sum_doses_or_orders = sum(p_df[dosage_col])
                else:
                    p_sum_doses_or_orders = len(p_df[dosage_col].dropna().values.tolist())
                p_doses_or_orders.append(p_sum_doses_or_orders)
                doses_or_orders.append(p_doses_or_orders)
            new_df = pd.DataFrame(data=doses_or_orders,
                                  columns=[pid_cols[i], "d1-" + label, "d2-" + label, "total-" + label])
            new_df = new_df.drop(new_df.filter(regex='Unnamed').columns, axis=1)
            print("new_df.head() = ", new_df.head())
            new_df.to_csv(folders[i] + "/data/drug-orders-per-day/dosage/" + dataset + "-" + treatment +
                          "-dose-features.csv")
            features_df = pd.read_csv(folders[i] + "/data/" + dataset + "-ami-patients-features.csv")
            print("features_df before: ", len(features_df))
            features_df = pd.merge(left=features_df, right=new_df, on=pid_cols[i], how="left")
            features_df = features_df.drop(features_df.filter(regex='Unnamed').columns, axis=1)
            print("features_df.head() = ", features_df.head(), len(features_df))
            features_df.to_csv(folders[i] + "/data/" + dataset + "-ami-patients-features.csv")


if __name__ == '__main__':
    convert_opioid_dose_to_mg_m3()
    convert_opioid_dose_to_mg_eicu()
    convert_opioid_dose_to_mg_aumc()
    convert_prescription_doses_to_mme()
    consolidate_dosage()
