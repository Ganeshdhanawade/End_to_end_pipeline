import os 
import json
import pickle
import zipfile
import numpy as np
import seaborn as sns
import pandas as pd
from src.logger import logging
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------- Regression ---------------------------------

#-------------------------------------------------------
# Button to save the cleaned dataset
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_data(path):
    data = pd.read_csv(path)
    return data

#---------------------------------------------------------
# Function to save selected method names to a file -- method selection
def save_selected_methods(methods, file_path):
    with open(file_path, 'w') as f:
        json.dump(methods, f)

# Function to load selected method names from a file
def load_selected_methods(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

#--------------------------------------------------------
# Feature selection -- diffrent plots for the feature selection
# Function to plot horizontal bar chart
def plot_horizontal_bar(data, feature_col, value_col, title):
    fig = px.bar(data, x=value_col, y=feature_col, orientation='h', title=title,
                labels={value_col: value_col, feature_col: feature_col},
                color=value_col, color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title=value_col, yaxis_title=feature_col)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot heatmap
def plot_heatmap(data, title):
    fig = go.Figure(data=go.Heatmap(z=data.values,
                                    x=data.columns,
                                    y=data.index,
                                    colorscale='YlGnBu',
                                    colorbar=dict(title='Importance'),
                                    zmin=0, zmax=data.values.max()))
    fig.update_layout(title=title, xaxis_title='Feature', yaxis_title='Method')
    st.plotly_chart(fig, use_container_width=True)

# Function to plot correlation heatmap
def plot_correlation_heatmap(data, title):
    corr = data.corr()

    # Create the text for the heatmap (correlation values)
    text = np.round(corr.values, 2)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',  # Using 'RdBu' colorscale, which is similar to 'coolwarm'
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        colorbar=dict(title='Correlation', titleside='right'),
        text=text,  # Add text to the heatmap
        texttemplate='%{text}',  # Display the text
        textfont={"size":12, "color":"black"},  # Customize text font
        xgap=1,  # Add gap between cells for horizontal borders
        ygap=1   # Add gap between cells for vertical borders
    ))

    # Update the layout of the figure
    fig.update_layout(
        title=title,
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.index),
        xaxis=dict(tickmode='array', tickvals=list(range(len(corr.columns))), ticktext=corr.columns),
        yaxis=dict(tickmode='array', tickvals=list(range(len(corr.index))), ticktext=corr.index),
        plot_bgcolor='rgba(0,0,0,0)'  # Set the background color to transparent
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)


#-----------------------------------------------------------------------
## validation -save text file

def load_methods(file_paths):
    """Load methods from given file paths into lists."""
    methods = {}
    for key, path in file_paths.items():
        try:
            with open(path, 'r') as file:
                methods[key] = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            methods[key] = []
    return methods

def save_to_txt(file_path, data, column_names, model_name):
    """Save dictionary data, column names, and model name to a text file."""
    with open(file_path, 'w') as file:
        # Descriptor Methods Section
        if data.get("descriptor_methods"):
            file.write("Descriptor Methods:\n")
            file.write(f"{data['descriptor_methods']}\n")
            file.write("\n")
        
        # Fingerprint Methods Section
        if data.get("fingerprint_methods"):
            file.write("Fingerprint Methods:\n")
            file.write(f"{data['fingerprint_methods']}\n")
            file.write("\n")

        # Fingerprint Types Section
        if data.get("fingerprint_types"):
            file.write("Fingerprint Types:\n")
            file.write(f"{data['fingerprint_types']}\n")
            file.write("\n")
        
        # QM Methods Section
        if data.get("qm_methods"):
            file.write("QM Methods:\n")
            file.write(f"{data['qm_methods']}\n")
            file.write("\n")
        
        # Descriptor Set Methods Section
        if data.get("descriptor_set_methods"):
            file.write("Descriptor Set Methods:\n")
            file.write(f"{data['descriptor_set_methods']}\n")
            file.write("\n")
        
        # Write column names as a list
        file.write("Dataset Columns:\n")
        file.write(f"{column_names}\n")
        file.write("\n")
        
        # Write selected model name as a list
        file.write("Selected Model:\n")
        file.write(f"[{model_name}]\n")


def zip_folder(folder_path, zip_path):
    """Zip the contents of a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

#-----------------------------------------------------
# this is the code for feature selection

def save_selected_methods_to_json(descriptor_methods, fingerprint_methods, sub_fingerprint_methods, file_path):
    data = {}

    if descriptor_methods:
        data['descriptor'] = descriptor_methods

    if fingerprint_methods:
        data['fingerprint'] = {
            "methods": fingerprint_methods
        }
        if "FingerprintCalculator" in fingerprint_methods:
            data['fingerprint']['sub-category'] = sub_fingerprint_methods

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        logging.info(f"Selected methods saved to {file_path}")



def load_selected_methods_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        logging.warning(f"File {file_path} does not exist.")
        return {}
    

#---------------------------------------------
# model validation - add validation saving descriptors

def load_selected_methods_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    selected_methods = {
        'descriptor_methods': data.get('descriptor', []),
        'fingerprint_methods': data.get('fingerprint', {}).get('methods', []),
        'fingerprint_types': data.get('fingerprint', {}).get('sub-category', []),
        'qm_methods': [],  # Assuming no data for 'qm_methods' in this example
        'descriptor_set_methods': []  # Assuming no data for 'descriptor_set_methods' in this example
    }
    
    return selected_methods


# ----------------------------------- Classification ------------------------------------


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{model_name}_confusion_matrix.png")
    return f"{model_name}_confusion_matrix.png"