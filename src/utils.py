import os 
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Button to save the cleaned dataset
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_data(path):
    data = pd.read_csv(path)
    return data

#---------------------------------------------------------
#for method selection
# Function to save selected method names to a file
def save_selected_methods(methods, file_path):
    with open(file_path, 'w') as f:
        json.dump(methods, f)

# Function to load selected method names from a file
def load_selected_methods(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}