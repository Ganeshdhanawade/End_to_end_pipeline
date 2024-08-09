import sys,os
sys.path.append('/home/ganesh')

import os
import base64
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from src.package.descriptor_names import descriptor_methods, fingerprint_methods, qm_methods, descriptor_set_methods
from src.config.configuration import *
from src.exception import CustomException  # Import the CustomException class
from src.logger import logging
from src.utils import *
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from src.preproces.data_cleaning import DataCleaner
from src.preproces.feature_selection import FeatureSelector
from src.preproces.model_building import Models
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn import neighbors
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error

# Initialize session state variables if they do not exist
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'feature_selection_done' not in st.session_state:
    st.session_state['feature_selection_done'] = False

### page navigations
nav = st.sidebar.radio("End-to-end pipeline",["Feature calculation","Data cleaning","Feature selection","Model building","Data validation"])

if nav == "Feature calculation":

    import streamlit as st
    import logging

    # Assume the methods load_data, descriptor_methods, fingerprint_methods, qm_methods, descriptor_set_methods, and save_data are defined elsewhere

    # Placeholder for the selected methods (these should come from your Streamlit sidebar selections)
    selected_descriptor_methods = st.sidebar.multiselect("Select Descriptor Methods", options=list(descriptor_methods.keys()))
    selected_fingerprint_methods = st.sidebar.multiselect("Select Fingerprint Methods", options=list(fingerprint_methods.keys()))
    selected_qm_methods = st.sidebar.multiselect("Select QM Methods", options=list(qm_methods.keys()))
    selected_descriptor_set_methods = st.sidebar.multiselect("Select Descriptor Set Methods", options=list(descriptor_set_methods.keys()))

    fp_names = [
        'ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'ecfc0', 'ecfc2', 'ecfc4', 'ecfc6',
        'fcfp2', 'fcfp4', 'fcfp6', 'fcfc2', 'fcfc4', 'fcfc6', 'lecfp4', 'lecfp6',
        'lfcfp4', 'lfcfp6', 'maccs', 'ap', 'tt', 'hashap', 'hashtt', 'avalon',
        'laval', 'rdk5', 'rdk6', 'rdk7'
    ]

    # Add a condition to show the list of fingerprints if "FingerprintCalculator" is selected
    if "FingerprintCalculator" in selected_fingerprint_methods:
        selected_fingerprint_types = st.sidebar.multiselect("Select Fingerprint Types", options=fp_names)
    else:
        selected_fingerprint_types = []

    uploaded_file = st.file_uploader("Upload your dataset")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Uploaded DataFrame:")
            st.write(df.head())

            if st.button("Run Selected Calculations"):
                if not selected_descriptor_methods and not selected_fingerprint_methods and not selected_qm_methods and not selected_descriptor_set_methods:
                    st.warning("Please select at least one method.")
                    logging.warning("No methods selected")
                else:
                    # Save selected methods to files
                    save_selected_methods(selected_descriptor_methods, DESCRIPTOR_METHODS_FILE_PATH)
                    save_selected_methods(selected_fingerprint_methods, FINGERPRINT_METHODS_FILE_PATH)
                    save_selected_methods(selected_qm_methods, QM_METHODS_FILE_PATH)
                    save_selected_methods(selected_descriptor_set_methods, DESCRIPTOR_SET_METHODS_FILE_PATH)

                    # Save selected fingerprint types if FingerprintCalculator is selected
                    if "FingerprintCalculator" in selected_fingerprint_methods:
                        st.session_state['selected_fingerprint_types'] = selected_fingerprint_types
                        save_selected_methods(selected_fingerprint_types, FINGERPRINT_TYPES_SUB_CATEGORY_PATH)
                    else:
                        st.session_state['selected_fingerprint_types'] = []

                    # Process selected methods
                    for method_name in selected_descriptor_methods:
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_methods[method_name](df)
                            logging.info(f"Ran descriptor method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running descriptor method {method_name}: {error_message}")

                    for method_name in selected_fingerprint_methods:
                        try:
                            st.write(f"Running {method_name}...")
                            if method_name == "FingerprintCalculator":
                                if selected_fingerprint_types:
                                    df = fingerprint_methods[method_name](df, selected_fingerprint_types)
                                    logging.info(f"Ran FingerprintCalculator with types {selected_fingerprint_types}")
                                else:
                                    st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
                                    logging.warning("No fingerprint types selected for FingerprintCalculator")
                            else:
                                df = fingerprint_methods[method_name](df)
                                logging.info(f"Ran fingerprint method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running fingerprint method {method_name}: {error_message}")

                    for method_name in selected_qm_methods:
                        try:
                            st.write(f"Running {method_name}...")
                            df = qm_methods[method_name](df)
                            logging.info(f"Ran QM method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running QM method {method_name}: {error_message}")

                    for method_name in selected_descriptor_set_methods:
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_set_methods[method_name](df)
                            logging.info(f"Ran descriptor set method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running descriptor set method {method_name}: {error_message}")

                    st.write("Resulting DataFrame:")
                    st.write(df.head())

                    # Create columns for buttons
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        # Display the shape of the resulting DataFrame
                        col1.markdown(f"**DataFrame Shape:** {df.shape}")
                        logging.info(f"Displayed DataFrame shape: {df.shape}")

                    with col3:
                        # Button to download dataset
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Data",
                            data=csv,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )
                        logging.info("Download button created")

                    # Save the dataset
                    save_data(df, FEATURE_CALCULATED_DATA_PATH)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Data saved successfully")


if nav == "Data cleaning":
    st.title('Data Cleaner with Streamlit')

    st.sidebar.title("Settings")
    row_thresh = st.sidebar.slider('Row Null Threshold (%)', 0, 100, 20) / 100
    col_thresh = st.sidebar.slider('Column Null Threshold (%)', 0, 100, 20) / 100
    
    ## import data
    st.write("Are you clean the feature calculated data?")

    if st.button('Import Data'):
        
        df = load_data(FEATURE_CALCULATED_DATA_PATH)
        ##saprate orignal and claen data
        
        #show original dataset
        st.write("### Original Dataset")
        st.write(df.head())
        st.write(f"Original dataset shape: {df.shape}")

        ##plot for null values
        st.write("### Plot for null values")
        null_counts = df.isnull().sum()
        # Filter columns with null values only
        null_counts = null_counts[null_counts > 0]
        # Check if there are any columns with null values
        if null_counts.empty:
            st.write("No null values present in the dataset.")
        else:
            # Plotting the bar graph
            fig, ax = plt.subplots()
            null_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel("Columns")
            ax.set_ylabel("Number of Null Values")
            st.pyplot(fig)

        #show clean dataset
        cleaner = DataCleaner(row_thresh=row_thresh, col_thresh=col_thresh)
        df_cleaned = cleaner.fit_transform(df)
        st.write("### Cleaned Dataset")
        st.write(df_cleaned.head())
        st.write(f"Clean dataset shape: {df_cleaned.shape}")

        # Display row and column count information
        st.write("### Deleted dataset information")
        rows_deleted = df.shape[0] - df_cleaned.shape[0]
        columns_dropped = df.shape[1] - df_cleaned.shape[1]
        st.write(f"Rows Deleted: {rows_deleted}")
        st.write(f"Columns Dropped: {columns_dropped}")
        st.write(f"Drop columns names: {list(set(df.columns) - set(df_cleaned.columns))}")

        # ##save the dataset
        # if st.button("Save Cleaned Dataset"):
        #     save_data(df_cleaned, DATA_CLEANING_PATH)
        #     st.success(f"Model saved successfully to {DATA_CLEANING_PATH}")

        # Option to download cleaned dataset directly
        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download, pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        tmp_download_link = download_link(df_cleaned, "cleaned_data.csv", "Click here to download your cleaned data!")
        st.markdown(tmp_download_link, unsafe_allow_html=True) 
        
        #data saving
        save_data(df_cleaned, DATA_CLEANING_PATH)
        st.success(f"Model saved successfully to {DATA_CLEANING_PATH}")  


if nav == "Feature selection":

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

    st.title("Feature Selection Dashboard")

    # Data uploader
    st.write("Are you select the important feature in clean data?")

    if st.button('Import Data'):
        st.session_state['df'] = load_data(DATA_CLEANING_PATH)
        st.session_state['df'].drop(columns=['SMILES'], inplace=True)

    # Check if dataset is loaded and display it
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Display the dataframe
        st.write("### Data Overview")
        st.write(df.head())
        
        # Select the response variable
        response_var = st.selectbox("Select the response variable", df.columns)
        
        # Select the feature selection methods
        st.sidebar.header("Feature Selection Settings")
        corr_threshold = st.sidebar.slider("Correlation Threshold", 0.0, 1.0, 0.5)
        multicoll_threshold = st.sidebar.slider("Multicollinearity Threshold", 0.0, 1.0, 0.9)
        top_features = st.sidebar.slider("Number of Top Features", 1, 50, 20)

        if st.button("Run Feature Selection"):
            st.session_state['feature_selection_done'] = True
            st.session_state['response_var'] = response_var

            # Feature selection process
            fs = FeatureSelector(corr_threshold=corr_threshold, multicoll_threshold=multicoll_threshold, top_features=top_features)
            response_var = st.session_state['response_var']
            X = df.drop(columns=[response_var])
            y = df[response_var]
            
            # Feature Selection for Continuous Response Variable
            st.write("### Selecting Correlated Features")
            corr_features = fs.select_corr_features(X, y)
            st.write(f"Selected Correlated Features: {corr_features}")
            st.write(f"Shape Correlated Features: {len(corr_features)}")

            st.write("### Selecting Top Features")
            top_features_list = fs.select_top_features(X, y)
            st.write(f"Top Selected Features: {top_features_list}")
            st.write(f"Shape top Selected Features: {len(top_features_list)}")

            # Create DataFrames for top features from each method
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            # ANOVA F-test
            f_scores, p_values = f_regression(X_encoded, y)
            anova_scores_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'F-Score': f_scores
            }).sort_values(by='F-Score', ascending=False).head(top_features)
            
            # Mutual Information
            mi_scores = mutual_info_regression(X_encoded, y)
            mi_scores_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'MI Score': mi_scores
            }).sort_values(by='MI Score', ascending=False).head(top_features)
            
            # ExtraTreesRegressor Feature Importance
            model = ExtraTreesRegressor()
            model.fit(X_encoded, y)
            importances = model.feature_importances_
            extra_trees_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(top_features)
            
            # RandomForestRegressor Feature Importance
            model_rf = RandomForestRegressor()
            model_rf.fit(X_encoded, y)
            importances_rf = model_rf.feature_importances_
            random_forest_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': importances_rf
            }).sort_values(by='Importance', ascending=False).head(top_features)

            # Plot for ANOVA F-test
            plot_horizontal_bar(anova_scores_df, 'Feature', 'F-Score', 'Top Features by ANOVA F-test')
            
            # Plot for Mutual Information
            plot_horizontal_bar(mi_scores_df, 'Feature', 'MI Score', 'Top Features by Mutual Information')
            
            # Plot for Extra Trees
            plot_horizontal_bar(extra_trees_df, 'Feature', 'Importance', 'Top Features by Extra Trees')
            
            # Plot for Random Forest
            plot_horizontal_bar(random_forest_df, 'Feature', 'Importance', 'Top Features by Random Forest')

            # Create heatmap for feature importance
            heatmap_data = pd.DataFrame({
                'ANOVA': anova_scores_df.set_index('Feature')['F-Score'],
                'Mutual Information': mi_scores_df.set_index('Feature')['MI Score'],
                'Extra Trees': extra_trees_df.set_index('Feature')['Importance'],
                'Random Forest': random_forest_df.set_index('Feature')['Importance']
            }).fillna(0)

            st.write("### Feature Importance Heatmap")
            plot_heatmap(heatmap_data, 'Feature Importance Heatmap')

            X_filtered = pd.concat([X[corr_features],y],axis=1)

            # Plot Correlation Heatmap of Selected Features
            st.write("### Correlation Heatmap of Selected Features")
            plot_correlation_heatmap(X_filtered, 'Correlation Heatmap of Selected Features')

            ## Clean dataset
            st.write("### Final Feature Selected Dataset")

            selected_features_df = fs.save_selected_features(X, y)
            st.write(selected_features_df.head())
            st.write(f"Final dataset shape: {selected_features_df.shape}")

            # Save selected features to CSV
            save_path = FEATURE_SELECTION_DATA_PATH
            save_data(selected_features_df, save_path)
            st.write(f"Saved Selected Features to {save_path}")
            st.success("Dataset saved successfully.....")

        if st.session_state['feature_selection_done']:
            # Display results
            st.write("### Feature Selection Completed")


if nav == "Model building":

    st.title('Model building Pipeline')

    dataset_option = st.selectbox('Select dataset for analysis',['clean dataset', 'feature selection dataset'])

    # Load the selected dataset
    if dataset_option == 'clean dataset':
        data1 = load_data(DATA_CLEANING_PATH)
        data=data1.iloc[:,1:]
        st.write(data.head())
    elif dataset_option == 'feature selection dataset':
        data = load_data(FEATURE_SELECTION_DATA_PATH)
        st.write(data.head())
    
    tab1, tab2, tab3 = st.tabs(["Model Building", "Cross-validation", "Save Model"])

    tab1.write("### Build the all models")
    
    # You can also use "with" notation:
    with tab1:
        if st.button("Run Models"):
            results, plots = Models.all_models(data)
            
            st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
            
            for model_name, metrics in results.items():
                with st.expander(f"Model: {model_name}", expanded=True):
                    st.markdown(f"<h3 style='color: #ff6347;'>Model: {model_name}</h3>", unsafe_allow_html=True)
                    st.write(f"**R2 Score (Train):** {metrics['r2_train']:.4f}")
                    st.write(f"**RMSE (Train):** {metrics['rmse_train']:.4f}")
                    st.write(f"**R2 Score (Test):** {metrics['r2_test']:.4f}")
                    st.write(f"**RMSE (Test):** {metrics['rmse_test']:.4f}")
                    
                    st.markdown(f"<h4 style='text-align: center;'>Plot for {model_name}</h4>", unsafe_allow_html=True)
                    st.image(plots[model_name], use_column_width=True)

    with tab2:
        tab2.write("### Cross validations")
        model_name = st.selectbox("Select Model for Cross Validation", ["DecisionTree", "XGBoost", "ExtraTrees", "RandomForest", "LGBM", "KNN"])
        model_dict = {
                'DecisionTree': DecisionTreeRegressor,
                'XGBoost': XGBRegressor,
                'ExtraTrees': ExtraTreesRegressor,
                'RandomForest': RandomForestRegressor,
                'LGBM': LGBMRegressor,
                'KNN': neighbors.KNeighborsRegressor
            }
        model_class = model_dict[model_name]
        n_fold = st.slider("Select Number of Folds for Cross Validation", 2, 10, 5)
        # Option to perform cross-validation
        if st.button("Perform Cross Validation"):
            avg_r2, avg_rmse, r2_scores, rmse_scores, fold_plots = Models.cross_validation(model_class, data, n_fold)
            
            # Display overall average metrics
            st.write(f"**Cross Validation Average R2:** {avg_r2:.4f}")
            st.write(f"**Cross Validation Average RMSE:** {avg_rmse:.4f}")
            
            # Display fold-wise metrics and plots
            for i in range(len(r2_scores)):
                with st.expander(f"Fold {i+1}", expanded=False):
                    st.write(f"**R2 Score:** {r2_scores[i]:.4f}")
                    st.write(f"**RMSE:** {rmse_scores[i]:.4f}")
                    
                    # Display the plot for the current fold
                    st.markdown(f"<h5 style='text-align: center;'>Plot for Fold {i+1}</h5>", unsafe_allow_html=True)
                    st.image(fold_plots[i])                
                    
    with tab3:
        model_name_to_save = st.selectbox("Select Model to Save", ["DecisionTree", "XGBoost", "ExtraTrees", "RandomForest", "LGBM", "KNN"], key='save_model')
        if st.button("Save Model"):
            model_dict_to_save = {
                'DecisionTree': DecisionTreeRegressor(),
                'XGBoost': XGBRegressor(n_jobs=16),
                'ExtraTrees': ExtraTreesRegressor(n_jobs=16),
                'RandomForest': RandomForestRegressor(n_jobs=16),
                'LGBM': LGBMRegressor(n_jobs=16),
                'KNN': neighbors.KNeighborsRegressor(n_jobs=16)
            }
            
            model_to_save = model_dict_to_save[model_name_to_save]
            
            # Fit the model to the entire dataset
            X = data.loc[:, data.columns != 'res']
            y = data.loc[:, 'res']

            model_to_save.fit(X, y)
            # Save the model
            with open(FINAL_MODEL_PICKLE_PATH, 'wb') as f:
                pickle.dump(model_to_save, f)
            
            st.write("Model saved as final_model.pkl")


if nav == "Data validation":
    
    import streamlit as st
    import logging
    import pickle
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Assume the methods load_data, descriptor_methods, fingerprint_methods, qm_methods, descriptor_set_methods, and save_data are defined elsewhere
    # Also assume that CustomException, load_selected_methods, and save_selected_methods are defined elsewhere

    # Initialization of session state variables
    if 'df1' not in st.session_state:
        st.session_state['df1'] = None

    if 'data_saved' not in st.session_state:
        st.session_state['data_saved'] = False

    if 'selected_descriptor_methods' not in st.session_state:
        st.session_state['selected_descriptor_methods'] = load_selected_methods(DESCRIPTOR_METHODS_FILE_PATH)

    if 'selected_fingerprint_methods' not in st.session_state:
        st.session_state['selected_fingerprint_methods'] = load_selected_methods(FINGERPRINT_METHODS_FILE_PATH)

    if 'selected_qm_methods' not in st.session_state:
        st.session_state['selected_qm_methods'] = load_selected_methods(QM_METHODS_FILE_PATH)

    if 'selected_descriptor_set_methods' not in st.session_state:
        st.session_state['selected_descriptor_set_methods'] = load_selected_methods(DESCRIPTOR_SET_METHODS_FILE_PATH)

    if 'selected_fingerprint_types' not in st.session_state:
        st.session_state['selected_fingerprint_types'] = []

    if 'data_final' not in st.session_state:
        st.session_state['data_final'] = False

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset")

    if uploaded_file is not None:
        st.session_state['df1'] = load_data(uploaded_file)
        
        if st.session_state['df1'] is not None:
            st.write("Uploaded DataFrame:")
            st.write(st.session_state['df1'].head())

            # Manage the button state to avoid multiple runs
            if 'run_calculations' not in st.session_state:
                st.session_state['run_calculations'] = False

            if st.button("Run Selected Calculations"):
                if not st.session_state['selected_descriptor_methods'] and not st.session_state['selected_fingerprint_methods'] and not st.session_state['selected_qm_methods'] and not st.session_state['selected_descriptor_set_methods']:
                    st.warning("Please select at least one method.")
                    logging.warning("No methods selected")
                else:
                    df = st.session_state['df1']

                    # Process selected descriptor methods
                    for method_name in st.session_state['selected_descriptor_methods']:
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_methods[method_name](df)
                            logging.info(f"Ran descriptor method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running descriptor method {method_name}: {error_message}")

                    # Process selected fingerprint methods
                    for method_name in st.session_state['selected_fingerprint_methods']:
                        try:
                            st.write(f"Running {method_name}...")
                            if method_name == "FingerprintCalculator":
                                if st.session_state['selected_fingerprint_types']:
                                    df = fingerprint_methods[method_name](df, st.session_state['selected_fingerprint_types'])
                                    logging.info(f"Ran FingerprintCalculator with types {st.session_state['selected_fingerprint_types']}")
                                else:
                                    pass
                                    #st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
                                    #logging.warning("No fingerprint types selected for FingerprintCalculator")
                            else:
                                df = fingerprint_methods[method_name](df)
                                logging.info(f"Ran fingerprint method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running fingerprint method {method_name}: {error_message}")

                    # Process selected QM methods
                    for method_name in st.session_state['selected_qm_methods']:
                        try:
                            st.write(f"Running {method_name}...")
                            df = qm_methods[method_name](df)
                            logging.info(f"Ran QM method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running QM method {method_name}: {error_message}")

                    # Process selected descriptor set methods
                    for method_name in st.session_state['selected_descriptor_set_methods']:
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_set_methods[method_name](df)
                            logging.info(f"Ran descriptor set method {method_name}")
                        except Exception as e:
                            error_message = CustomException(e)
                            st.error(f"Error running {method_name}: {error_message}")
                            logging.error(f"Error running descriptor set method {method_name}: {error_message}")

                    st.write("Resulting DataFrame:")
                    st.write(df.head())
                    st.session_state['data_final'] = df

                    # Create columns for buttons
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        col1.markdown(f"**DataFrame Shape:** {df.shape}")
                        logging.info(f"Displayed DataFrame shape: {df.shape}")

                    with col3:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Data",
                            data=csv,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )
                        logging.info("Download button created")

                    # Save the dataset
                    save_data(df, FEATURE_CALCULATED_DATA_PATH)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Data saved successfully")

            # Tabs for additional actions
            tab21, tab22, tab23 = st.tabs(["Drop Duplicates", "Select Feature", "Validation"])

            with tab21:
                from ganesh_package.Classification import Descriptors_smile
                
                # Load and standardize data
                # def process_data(train_df, val_df):
                #     train_data = train_df.copy()
                #     val_data = val_df.copy()
                #     train_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(train_data['SMILES'])
                #     val_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(val_data['SMILES'])
                #     return train_data, val_data
                def process_data(train_df, val_df):
                    train_data = train_df.copy()
                    val_data = val_df.copy()
                    
                    # Ensure 'SMILES' columns are strings and fill NaN with empty strings
                    train_data['SMILES'] = train_data['SMILES'].astype(str).fillna('')
                    val_data['SMILES'] = val_data['SMILES'].astype(str).fillna('')
                    
                    # Standardize SMILES
                    train_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(train_data['SMILES'])
                    val_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(val_data['SMILES'])
                    
                    return train_data, val_data
                
                st.write("Are you interested in dropping duplicate smiles and saving data?")
                
                if st.button("Submit"):
                    train_file = load_data(DATA_CLEANING_PATH)
                    val_file = st.session_state['data_final']

                    if train_file is not None and val_file is not None:
                        train_data, val_data = process_data(train_file, val_file)
                        
                        # Display original data
                        st.write("Training Data")
                        st.write(train_data.head(2))
                        st.write(f"The shape of dataset is: {train_data.shape}")
                        st.write("Validation Data")
                        st.write(val_data.head(2))
                        st.write(f"The shape of dataset is: {val_data.shape}")
                        
                        # Check for duplicates
                        train_smiles_set = set(train_data['Standardized_SMILES'].dropna())
                        val_data['Is_Duplicate'] = val_data['Standardized_SMILES'].apply(lambda x: x in train_smiles_set if x is not None else False)
                        
                        # Filter out duplicates
                        clean_val_data = val_data[~val_data['Is_Duplicate']]
                        
                        st.subheader("Cleaned Validation Data")
                        st.write(clean_val_data.head(5))
                        st.write(f"The shape of dataset is: {clean_val_data.shape}")
                        st.write(f'Duplicated records: {val_data.shape[0]-clean_val_data.shape[0]}')
                        
                        # Option to download cleaned data
                        st.download_button(
                            label="Download Cleaned Validation Data",
                            data=clean_val_data.to_csv(index=False),
                            file_name="cleaned_validation_data.csv",
                            mime="text/csv"
                        )

                        save_data(clean_val_data, DUPLICATED_CLEAN_VAL_PATH)
                        st.session_state['data_saved'] = True
                        st.success("Data saved successfully!")
                        logging.info("Validation data saved successfully")                

            with tab22:
                dataset_option = st.selectbox('Select dataset for analysis', ['clean dataset', 'feature selection dataset'])

                if st.button("Save Data"):
                    if dataset_option == 'clean dataset':
                        data1 = load_data(DATA_CLEANING_PATH)
                        data =data1.iloc[:,1:]
                    elif dataset_option == 'feature selection dataset':
                        data = load_data(FEATURE_SELECTION_DATA_PATH)

                    data_feat = data.columns
                    df = st.session_state['data_final']
                    data_f = df[data_feat]

                    # Replace null values
                    data_f.fillna(0, inplace=True)

                    st.write(data_f.head())
                    st.write(f"The shape of the dataset is: {data_f.shape}")

                    save_data(data_f, VALIDATION_DATA_PATH)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Validation data saved successfully")

            with tab23:
                def plot_predictions(y_true, y_pred):
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_true, y_pred, alpha=0.6, color='b')
                    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r', linestyle='--', linewidth=2)
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.title('Actual vs Predicted Values')
                    st.pyplot(plt)
                
                def predict_and_evaluate(model, data_clean):
                    X = data_clean.drop(columns=['res'])  # Assuming 'res' is the column to predict
                    y_true = data_clean['res']
                    y_pred = model.predict(X)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    r2 = r2_score(y_true, y_pred)    
                    return y_true, y_pred, rmse, r2
                
                st.write("Are you interested in validating the model?")

                # Load data and model
                model = pickle.load(open(FINAL_MODEL_PICKLE_PATH, 'rb'))
                data_clean = load_data(VALIDATION_DATA_PATH)

                if st.button("Run Models"):
                    y_true, y_pred, rmse, r2 = predict_and_evaluate(model, data_clean)
                    
                    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='color: #ff6347;'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
                    st.write(f"**R2 Score:** {r2:.4f}")
                    st.write(f"**RMSE:** {rmse:.4f}")
                    
                    st.markdown("<h4 style='text-align: center;'>Actual vs Predicted Plot</h4>", unsafe_allow_html=True)
                    plot_predictions(y_true, y_pred)





                # # Create columns for buttons
                # col1, col2, col3 = st.columns([2, 1, 1])

                # with col1:
                #     # Display the shape of the resulting DataFrame
                #     col1.markdown(f"**DataFrame Shape:** {df.shape}")
                #     logging.info(f"Displayed DataFrame shape: {df.shape}")

                # with col3:
                #     # Button to download dataset
                #     csv = df.to_csv(index=False)
                #     st.download_button(
                #         label="Download Data",
                #         data=csv,
                #         file_name="processed_data.csv",
                #         mime="text/csv"
                #     )
                #     logging.info("Download button created")

