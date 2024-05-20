"""
This file runs the project to obtain differences in treatment of patients with AMI in 3 datasets:
    1. AmsterdamUMCdb
    2. eICU
    3. MIMIC-III
First, the cohort of study id obtained from each dataset, then, the medications for each cohort is processed, then,
feature files for each dataset are created, then analysis are run on each dataset, and the results are visualized
"""
import analysis
import create_cohort
import create_feature_files
import process_medication_orders
import visualize

if __name__ == '__main__':
    # create_cohort.obtain_ami_cohorts()
    # process_medication_orders.process_medications()
    # create_feature_files.create_feature_files()
    # create_feature_files.create_sub_cohorts_feature_files()
    analysis.create_analysis_df()
    # visualize.visualize_entire_hospitalization_results()
    # visualize.visualize_per_day_results()


