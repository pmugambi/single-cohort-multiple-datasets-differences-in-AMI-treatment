"""
This file runs the project to obtain differences in treatment of patients with AMI in 4 datasets:
    1. AmsterdamUMCdb
    2. eICU
    3. MIMIC-III
    4. MIMIC-IV
First, the cohort of study id obtained from each dataset, then, the medications for each cohort is processed, then,
feature files for each dataset are created, then analysis are run on each dataset, and the results are visualized
"""
import analysis_of_disparities
import analysis_of_effect_on_outcome
import create_cohort
import create_feature_files
import process_medication_orders
import visualize

if __name__ == '__main__':
    print("----Step 1: extracting AMI cohorts from datasets----")
    create_cohort.obtain_ami_cohorts()
    print("----Step 2: extracting AMI cohorts' prescriptions of analgesia and AMI-related drugs from datasets----")
    process_medication_orders.process_medications()
    print("----Step 3: creating complete features files, with patient and treatment features----")
    create_feature_files.create_feature_files()
    print("----Step 4: dividing up the features files into sub-cohort files----")
    create_feature_files.create_sub_cohorts_feature_files()
    print("----Step 5: extracting cohort only unique to MIMIC=IV, i.e., not in MIMIC-III----")
    create_feature_files.create_mimic_iv_unique_cohorts()
    print("----Step 6: running hypothesis tests to obtain disparities in treatment----")
    analysis_of_disparities.create_analysis_df()
    print("----Step 7: running regression analyses to obtain association between treatment and patient outcomes----")
    analysis_of_effect_on_outcome.run_analyses()
    # visualize.visualize_entire_hospitalization_results()
    # visualize.visualize_per_day_results()


