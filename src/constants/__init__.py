import os, sys
from datetime import datetime

# artifact -> pipelien folder -> timestamp -> output

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOt_DIR_KEY = os.getcwd()

## ---Dataset---
DATA_SOURCE= "Data"

# ----------------- Regression ------------------ #

# feature calculated dataset
CALCULATED_FEATURE_DATA = "feature_calculated_data"
FEATURE_CALCULATED_DATA_CSV = "feature_calulated_data.csv"

# clean dataset
CLEAN_DATA = "clean_data"
CLEAN_DATA_CSV = "clean_dataset.csv"

# feature selection dataset
FEATURE_SELECTION = "feature_selection"
FEATURE_SELECTION_DATA_CSV = "feature_selected_data.csv"

#save the descriptors
ARTIFACT_FOLDER = "artifact"
SAVE_DESCRIPTOR_FOLDER = "methods"
DESCRIPTOR_JESON = 'selected_descriptor_methods.json'

##picke the final model
FINAL_MODEL_PICKLE_FOLDER = "models"
PICKLE_FILE_NAME = "final_model.pkl"

###validation dataset
VALIDATION_DATA_FOLDER="validation_data"
FINAL_VALIDATION_FOLDER = "final_validation"
VALIDATION_DATA_CSV="validation_final.csv"

### validation - after droping dulicate smiles save
DUPLICATED_CLEAN_VAL_FOLDER = "Duplicated_clean"
DUPLICATED_CLEAN_VAL_CSV= "duplicated_clean_val.csv"

## validation - save final model
FINAL_MODEL_FOLDER = "Final_report"
TEXT_REPORT = 'text_files'
ZIP_REPORT = 'zip_files'
REPORT_FOLDER = 'Regression_report'
REGRESSION_FINAL_TEXT = 'Regression_final_report.txt'
SAVE_REGRESSION_REPORT = 'Regression_report.zip'


# ----------------- Classification ------------------ #

# feature calculated dataset
CALCULATED_FEATURE_DATA = "feature_calculated_data"
FEATURE_CALCULATED_DATA_CSV_CLF = "feature_calulated_data_classification.csv"

# clean dataset
CLEAN_DATA = "clean_data"
CLEAN_DATA_CSV_CLF = "clean_dataset_classification.csv"

# feature selection dataset
FEATURE_SELECTION = "feature_selection"
FEATURE_SELECTION_DATA_CSV_CLF = "feature_selected_data_classification.csv"

#save the descriptors
ARTIFACT_FOLDER = "artifact"
SAVE_DESCRIPTOR_FOLDER = "methods"
DESCRIPTOR_JESON_CLF = 'selected_descriptor_methods_classification.json'

##picke the final model
FINAL_MODEL_PICKLE_FOLDER = "models"
PICKLE_FILE_NAME_CLF = "final_model_classification.pkl"

###validation dataset
VALIDATION_DATA_FOLDER="validation_data"
FINAL_VALIDATION_FOLDER = "final_validation"
VALIDATION_DATA_CSV_CLF="validation_final_classification.csv"

### validation - after droping dulicate smiles save
DUPLICATED_CLEAN_VAL_FOLDER = "Duplicated_clean"
DUPLICATED_CLEAN_VAL_CSV_CLF= "duplicated_clean_val_classification.csv"

## validation - save final model
FINAL_MODEL_FOLDER = "Final_report"
TEXT_REPORT_CLF = 'text_files_classification'
ZIP_REPORT_CLF = 'zip_files_classification'
REPORT_FOLDER_CLF = 'Classification_report'
REGRESSION_FINAL_TEXT_CLF = 'Classification_final_report.txt'
SAVE_REGRESSION_REPORT_CLF = 'Classification_report.zip'