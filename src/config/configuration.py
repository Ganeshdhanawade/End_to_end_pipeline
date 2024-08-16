from src.constants import *
import os, sys


ROOT_DIR = ROOt_DIR_KEY

# -------------------------------- Regression -------------------------------

# feature calcuated dataset
FEATURE_CALCULATED_DATA_PATH = os.path.join(ROOT_DIR, DATA_SOURCE, CALCULATED_FEATURE_DATA, FEATURE_CALCULATED_DATA_CSV)

# Data cleaning dataset
DATA_CLEANING_PATH = os.path.join(ROOT_DIR, DATA_SOURCE, CLEAN_DATA, CLEAN_DATA_CSV)

# feature selection dataset
FEATURE_SELECTION_DATA_PATH = os.path.join(ROOT_DIR, DATA_SOURCE, FEATURE_SELECTION, FEATURE_SELECTION_DATA_CSV)

##save the descriptors
DESCRIPTOR_METHODS_JESON_PATH = os.path.join(ROOT_DIR,ARTIFACT_FOLDER, SAVE_DESCRIPTOR_FOLDER, DESCRIPTOR_JESON)

##save the pickle of final model
FINAL_MODEL_PICKLE_PATH = os.path.join(ROOT_DIR, FINAL_MODEL_PICKLE_FOLDER,PICKLE_FILE_NAME)

##save validation dataset
VALIDATION_DATA_PATH = os.path.join(ROOT_DIR, DATA_SOURCE, VALIDATION_DATA_FOLDER,FINAL_VALIDATION_FOLDER, VALIDATION_DATA_CSV)

##save validation dataset
DUPLICATED_CLEAN_VAL_PATH = os.path.join(ROOT_DIR, DATA_SOURCE, VALIDATION_DATA_FOLDER, DUPLICATED_CLEAN_VAL_FOLDER, DUPLICATED_CLEAN_VAL_CSV)

## validation - save final model
REGRESSION_TEXT_PATH = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, TEXT_REPORT, REGRESSION_FINAL_TEXT)
REGRESSION_ZIP_PATH = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, ZIP_REPORT, SAVE_REGRESSION_REPORT)
REGRESSION_MODEL_FOLDER = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, REPORT_FOLDER)


# -------------------------------- Classification -------------------------------

# feature calcuated dataset
FEATURE_CALCULATED_DATA_PATH_CLF = os.path.join(ROOT_DIR, DATA_SOURCE, CALCULATED_FEATURE_DATA, FEATURE_CALCULATED_DATA_CSV_CLF)

# Data cleaning dataset
DATA_CLEANING_PATH_CLF = os.path.join(ROOT_DIR, DATA_SOURCE, CLEAN_DATA, CLEAN_DATA_CSV_CLF)

# feature selection dataset
FEATURE_SELECTION_DATA_PATH_CLF = os.path.join(ROOT_DIR, DATA_SOURCE, FEATURE_SELECTION, FEATURE_SELECTION_DATA_CSV_CLF)

##save the descriptors
DESCRIPTOR_METHODS_JESON_PATH_CLF = os.path.join(ROOT_DIR,ARTIFACT_FOLDER, SAVE_DESCRIPTOR_FOLDER, DESCRIPTOR_JESON_CLF)

##save the pickle of final model
FINAL_MODEL_PICKLE_PATH_CLF = os.path.join(ROOT_DIR, FINAL_MODEL_PICKLE_FOLDER,PICKLE_FILE_NAME_CLF)

##save validation dataset
VALIDATION_DATA_PATH_CLF = os.path.join(ROOT_DIR, DATA_SOURCE, VALIDATION_DATA_FOLDER,FINAL_VALIDATION_FOLDER, VALIDATION_DATA_CSV_CLF)

##save validation dataset
DUPLICATED_CLEAN_VAL_PATH_CLF = os.path.join(ROOT_DIR, DATA_SOURCE, VALIDATION_DATA_FOLDER, DUPLICATED_CLEAN_VAL_FOLDER, DUPLICATED_CLEAN_VAL_CSV_CLF)

## validation - save final model
REGRESSION_TEXT_PATH_CLF = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, TEXT_REPORT_CLF, REGRESSION_FINAL_TEXT_CLF)
REGRESSION_ZIP_PATH_CLF = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, ZIP_REPORT_CLF, SAVE_REGRESSION_REPORT_CLF)
REGRESSION_MODEL_FOLDER_CLF = os.path.join(ROOT_DIR, FINAL_MODEL_FOLDER, REPORT_FOLDER_CLF)