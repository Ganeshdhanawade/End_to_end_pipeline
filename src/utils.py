import os 
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

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
# Saving the final model - validation navigation
# Define functions
# Utility Functions
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
        # Write methods data
        for category, methods in data.items():
            file.write(f"{category}:\n")
            for method in methods:
                file.write(f" - {method}\n")
            file.write("\n")

        # Write column names
        file.write("Dataset Columns:\n")
        file.write(f"{list(names for names in column_names)}\n")
        
        # Write selected model name
        file.write(f"\nSelected Model:\n - {model_name}\n")

def zip_folder(folder_path, zip_path):
    """Zip the contents of a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))



# def save_to_txt(file_path, data, column_names=None, model_name=None):
#     """Save dictionary data, column names, and model name to a text file."""
#     with open(file_path, 'w') as file:
#         # Write model name and column names if provided
#         if model_name:
#             file.write(f"Model: {model_name}\n\n")
#         if column_names:
#             file.write(f"Columns: {', '.join(column_names)}\n\n")

#         # Write methods data
#         for category, methods in data.items():
#             # Only write the category if there are methods present
#             if methods:
#                 file.write(f"{category}:\n")
#                 for method in methods:
#                     file.write(f" - {method}\n")
#                 file.write("\n")

# # Example of the data structure you might have
# data = {
#     "descriptors": ["DescriptorMethod1", "DescriptorMethod2"],
#     "fingerprints": ["FingerprintCalculator", "AnotherFingerprintMethod"],
#     "qms": ["QMMethod1", "QMMethod2"],
#     "descriptor_sets": []  # No methods here, so it won't be written
# }

# file_paths = {
#     "descriptors": "descriptors.txt",
#     "fingerprints": "fingerprints.txt",
#     "sub-fingerprint": "sub_fingerprint.txt", 
#     "qms": "qms.txt",
#     "descriptor_sets": "descriptor_sets.txt"
# }

# # Save the main categories to their respective files
# for category, path in file_paths.items():
#     # Skip sub-fingerprint initially
#     if category == "sub-fingerprint":
#         continue
    
#     # Only save if there is data for that category
#     if category in data and data[category]:
#         save_to_txt(path, {category: data[category]})

# # Handle sub-fingerprint only if FingerprintCalculator is present
# if 'FingerprintCalculator' in data.get('fingerprints', []):
#     save_to_txt(file_paths["sub-fingerprint"], {"sub-fingerprint": data.get("sub-fingerprint", [])})