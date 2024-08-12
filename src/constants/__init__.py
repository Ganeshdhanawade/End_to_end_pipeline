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
DESCRIPTOR_METHODS = 'selected_descriptor_methods.json'
FINGERPRINT_METHODS = 'selected_fingerprint_methods.json'
QM_METHODS = 'selected_qm_methods.json'
DESCRIPTOR_SET_METHODS = 'selected_descriptor_set_methods.json'
DESCRIPTOR_SET_SUB_CATE_METHODS = 'selected_descriptor_set_sub_category.json'

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