import sys,os
sys.path.append('/home/ganesh')

## import the libraries
import os
import base64
import pickle
import shutil
import pandas as pd
import streamlit as st
from src.utils import *
from sklearn import neighbors
from src.logger import logging
import matplotlib.pyplot as plt
from src.config.configuration import *
from src.exception import CustomException  # Import the CustomException class
from src.preproces.model_building import Models_clf
from src.preproces.data_cleaning import DataCleaner
from src.preproces.feature_selection import FeatureSelector
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.package.descriptor_names import descriptor_methods, fingerprint_methods, qm_methods, descriptor_set_methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#------------------------------------------------------------------------
# Initialize session state - feature selection

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'feature_selection_done' not in st.session_state:
    st.session_state['feature_selection_done'] = False

# Initialization of session - data validation
if 'df1' not in st.session_state:
    st.session_state['df1'] = None

if 'selected_descriptor_methods' not in st.session_state:
    st.session_state['selected_descriptor_methods'] = []

if 'selected_fingerprint_methods' not in st.session_state:
    st.session_state['selected_fingerprint_methods'] = []

if 'selected_qm_methods' not in st.session_state:
    st.session_state['selected_qm_methods'] = []

if 'selected_descriptor_set_methods' not in st.session_state:
    st.session_state['selected_descriptor_set_methods'] = []

if 'data_final' not in st.session_state:
    st.session_state['data_final'] = pd.DataFrame()

if 'data_saved' not in st.session_state:
    st.session_state['data_saved'] = False

if 'selected_fingerprint_types' not in st.session_state:
    st.session_state['selected_fingerprint_types'] = []


#-----------------------------------------------------------------------
### page navigations
nav = st.sidebar.radio("End-to-end pipeline",["Feature calculation","Data cleaning","Feature selection","Model building","Data validation"])

#-----------------------------------------------------------------------
if nav == "Feature calculation":

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
                    save_selected_methods_to_json(
                        selected_descriptor_methods,
                        selected_fingerprint_methods,
                        selected_fingerprint_types,
                        DESCRIPTOR_METHODS_JESON_PATH_CLF
                    )

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
                            if method_name == "FingerprintCalculator" and selected_fingerprint_types:
                                df = fingerprint_methods[method_name](df, selected_fingerprint_types)
                                logging.info(f"Ran FingerprintCalculator with types {selected_fingerprint_types}")
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

                    save_data(df, FEATURE_CALCULATED_DATA_PATH_CLF)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Data saved successfully")


#----------------------------------------------------------------------------------
if nav == "Data cleaning":

    st.title('Data Cleaner with Streamlit')

    st.sidebar.title("Settings")
    row_thresh = st.sidebar.slider('Row Null Threshold (%)', 0, 100, 20) / 100
    col_thresh = st.sidebar.slider('Column Null Threshold (%)', 0, 100, 20) / 100
    
    ## import data
    st.write("Are you clean the feature calculated data?")

    if st.button('Import Data'):
        
        df = load_data(FEATURE_CALCULATED_DATA_PATH_CLF)
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

        # Option to download cleaned dataset directly
        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download, pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        tmp_download_link = download_link(df_cleaned, "cleaned_data.csv", "Click here to download your cleaned data!")
        st.markdown(tmp_download_link, unsafe_allow_html=True) 
        
        #data saving
        save_data(df_cleaned, DATA_CLEANING_PATH_CLF)
        st.success(f"Model saved successfully to {DATA_CLEANING_PATH_CLF}")  


#------------------------------------------------------------------------------
if nav == "Feature selection":

    st.title("Feature Selection Dashboard")

    # Data uploader
    st.write("Are you selecting the important features in clean data?")

    if st.button('Import Data'):
        st.session_state['df'] = load_data(DATA_CLEANING_PATH_CLF)
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

            # Feature Selection for Categorical (Binary) Response Variable
            st.write("### Selecting Correlated Features")
            corr_features = fs.select_corr_features(X, y)
            st.write(f"Selected Correlated Features: {corr_features}")
            st.write(f"Shape Correlated Features: {len(corr_features)}")

            # One-Hot Encoding for Categorical Variables
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Chi-Square Test
            chi2_scores, p_values = chi2(X_encoded, y)
            chi2_scores_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Chi2 Score': chi2_scores
            }).sort_values(by='Chi2 Score', ascending=False).head(top_features)
            
            # Mutual Information
            mi_scores = mutual_info_classif(X_encoded, y)
            mi_scores_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'MI Score': mi_scores
            }).sort_values(by='MI Score', ascending=False).head(top_features)
            
            # ExtraTreesClassifier Feature Importance
            model = ExtraTreesClassifier()
            model.fit(X_encoded, y)
            importances = model.feature_importances_
            extra_trees_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(top_features)
            
            # RandomForestClassifier Feature Importance
            model_rf = RandomForestClassifier()
            model_rf.fit(X_encoded, y)
            importances_rf = model_rf.feature_importances_
            random_forest_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': importances_rf
            }).sort_values(by='Importance', ascending=False).head(top_features)

            # Plot for Chi-Square Test
            plot_horizontal_bar(chi2_scores_df, 'Feature', 'Chi2 Score', 'Top Features by Chi-Square Test') 
            # Plot for Mutual Information
            plot_horizontal_bar(mi_scores_df, 'Feature', 'MI Score', 'Top Features by Mutual Information')
            # Plot for Extra Trees
            plot_horizontal_bar(extra_trees_df, 'Feature', 'Importance', 'Top Features by Extra Trees')
            # Plot for Random Forest
            plot_horizontal_bar(random_forest_df, 'Feature', 'Importance', 'Top Features by Random Forest')

            # Create heatmap for feature importance
            heatmap_data = pd.DataFrame({
                'Chi-Square': chi2_scores_df.set_index('Feature')['Chi2 Score'],
                'Mutual Information': mi_scores_df.set_index('Feature')['MI Score'],
                'Extra Trees': extra_trees_df.set_index('Feature')['Importance'],
                'Random Forest': random_forest_df.set_index('Feature')['Importance']
            }).fillna(0)

            st.write("### Feature Importance Heatmap")
            plot_heatmap(heatmap_data, 'Feature Importance Heatmap')

            X_filtered = pd.concat([X[corr_features], y], axis=1)

            # Plot Correlation Heatmap of Selected Features
            st.write("### Correlation Heatmap of Selected Features")
            plot_correlation_heatmap(X_filtered, 'Correlation Heatmap of Selected Features')

            ## Clean dataset
            st.write("### Final Feature Selected Dataset")

            selected_features_df = fs.save_selected_features(X, y)
            st.write(selected_features_df.head())
            st.write(f"Final dataset shape: {selected_features_df.shape}")

            # Save selected features to CSV
            save_path = FEATURE_SELECTION_DATA_PATH_CLF
            save_data(selected_features_df, save_path)
            st.write(f"Saved Selected Features to {save_path}")
            st.success("Dataset saved successfully.....")

        if st.session_state['feature_selection_done']:
            # Display results
            st.write("### Feature Selection Completed")



#-----------------------------------------------------------------------------
if nav == "Model building":

    st.title('Model Building Pipeline')

    dataset_option = st.selectbox('Select dataset for analysis', ['clean dataset', 'feature selection dataset'])

    # Load the selected dataset
    if dataset_option == 'clean dataset':
        data1 = load_data(DATA_CLEANING_PATH_CLF)
        data = data1.iloc[:, 1:]
        st.write(data.head())
    elif dataset_option == 'feature selection dataset':
        data = load_data(FEATURE_SELECTION_DATA_PATH_CLF)
        st.write(data.head())

    tab1, tab2, tab3 = st.tabs(["Model Building", "Cross-validation", "Save Model"])

    with tab1:
        st.write("### Build the Classification Models")

        if st.button("Run Models"):
            results, plots = Models_clf.all_models(data)
            
            st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
            
            for model_name, metrics in results.items():
                with st.expander(f"Model: {model_name}", expanded=True):
                    st.markdown(f"<h3 style='color: #ff6347;'>Model: {model_name}</h3>", unsafe_allow_html=True)
                    st.write(f"**Accuracy (Train):** {metrics['accuracy_train']:.4f}")
                    
                    st.markdown(f"<h4 style='text-align: center;'>Confusion Matrix for {model_name}</h4>", unsafe_allow_html=True)
                    st.image(plots[model_name], use_column_width=True)

    with tab2:
        st.write("### Cross Validation")
        model_name = st.selectbox("Select Model for Cross Validation", ["DecisionTree", "XGBoost", "ExtraTrees", "RandomForest", "LGBM", "KNN"])
        model_dict = {
            'DecisionTree': DecisionTreeClassifier,
            'XGBoost': XGBClassifier,
            'ExtraTrees': ExtraTreesClassifier,
            'RandomForest': RandomForestClassifier,
            'LGBM': LGBMClassifier,
            'KNN': KNeighborsClassifier
        }
        model_class = model_dict[model_name]
        n_folds = st.slider("Select Number of Folds for Cross Validation", 2, 10, 5)
        
        if st.button("Perform Cross Validation"):
            avg_accuracy, accuracies, fold_plots = Models_clf.cross_validation(model_class, data, n_folds)
            
            st.write(f"**Cross Validation Average Accuracy:** {avg_accuracy:.4f}")
            
            for i in range(len(accuracies)):
                with st.expander(f"Fold {i+1}", expanded=False):
                    st.write(f"**Accuracy:** {accuracies[i]:.4f}")
                    
                    st.markdown(f"<h5 style='text-align: center;'>Confusion Matrix for Fold {i+1}</h5>", unsafe_allow_html=True)
                    st.image(fold_plots[i])
                    
    with tab3:
        model_name_to_save = st.selectbox("Select Model to Save", ["DecisionTree", "XGBoost", "ExtraTrees", "RandomForest", "LGBM", "KNN"], key='save_model')
        st.session_state['model_save'] = model_name_to_save
        if st.button("Save Model"):
            model_dict_to_save = {
                'DecisionTree': DecisionTreeClassifier(),
                'XGBoost': XGBClassifier(),
                'ExtraTrees': ExtraTreesClassifier(),
                'RandomForest': RandomForestClassifier(),
                'LGBM': LGBMClassifier(),
                'KNN': KNeighborsClassifier()
            }
            
            model_to_save = model_dict_to_save[model_name_to_save]
            
            # Fit the model to the entire dataset
            X = data.drop(columns='target')
            y = data['target']

            model_to_save.fit(X, y)
            # Save the model
            with open(FINAL_MODEL_PICKLE_PATH_CLF, 'wb') as f:
                pickle.dump(model_to_save, f)
            
            st.write("Model saved as final_model.pkl")

#-----------------------------------------------------------------------------
if nav == "Data validation":
    uploaded_file = st.file_uploader("Upload your dataset")

    if uploaded_file is not None:
        st.session_state['df1'] = load_data(uploaded_file)
        
        if st.session_state['df1'] is not None:
            st.write("Uploaded DataFrame:")
            st.write(st.session_state['df1'].head())

            if 'run_calculations' not in st.session_state:
                st.session_state['run_calculations'] = False

            if st.button("Run Selected Calculations"):
                selected_methods = load_selected_methods_from_json(DESCRIPTOR_METHODS_JESON_PATH_CLF)

                if not (selected_methods.get('descriptor_methods') or selected_methods.get('fingerprint_methods') or selected_methods.get('qm_methods') or selected_methods.get('descriptor_set_methods')):
                    st.warning("Please select at least one method.")
                    logging.warning("No methods selected")
                else:
                    df = st.session_state['df1']

                    # Process selected descriptor methods
                    for method_name in selected_methods.get('descriptor_methods', []):
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_methods[method_name](df)
                            logging.info(f"Ran descriptor method {method_name}")
                        except Exception as e:
                            st.error(f"Error running {method_name}: {str(e)}")
                            logging.error(f"Error running descriptor method {method_name}: {str(e)}")

                    # Process selected fingerprint methods
                    for method_name in selected_methods.get('fingerprint_methods', []):
                        try:
                            st.write(f"Running {method_name}...")
                            if method_name == "FingerprintCalculator":
                                if selected_methods.get('fingerprint_types'):
                                    df = fingerprint_methods[method_name](df, selected_methods['fingerprint_types'])
                                    logging.info(f"Ran FingerprintCalculator with types {selected_methods['fingerprint_types']}")
                                else:
                                    st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
                                    logging.warning("No fingerprint types selected for FingerprintCalculator")
                            else:
                                df = fingerprint_methods[method_name](df)
                                logging.info(f"Ran fingerprint method {method_name}")
                        except Exception as e:
                            st.error(f"Error running {method_name}: {str(e)}")
                            logging.error(f"Error running fingerprint method {method_name}: {str(e)}")

                    # Process selected QM methods
                    for method_name in selected_methods.get('qm_methods', []):
                        try:
                            st.write(f"Running {method_name}...")
                            df = qm_methods[method_name](df)
                            logging.info(f"Ran QM method {method_name}")
                        except Exception as e:
                            st.error(f"Error running {method_name}: {str(e)}")
                            logging.error(f"Error running QM method {method_name}: {str(e)}")

                    # Process selected descriptor set methods
                    for method_name in selected_methods.get('descriptor_set_methods', []):
                        try:
                            st.write(f"Running {method_name}...")
                            df = descriptor_set_methods[method_name](df)
                            logging.info(f"Ran descriptor set method {method_name}")
                        except Exception as e:
                            st.error(f"Error running {method_name}: {str(e)}")
                            logging.error(f"Error running descriptor set method {method_name}: {str(e)}")

                    st.write("Resulting DataFrame:")
                    st.write(df.head())
                    st.session_state['data_final'] = df

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"**DataFrame Shape:** {df.shape}")
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

                    save_data(df, FEATURE_CALCULATED_DATA_PATH_CLF)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Data saved successfully")

            # Tabs for additional actions
            tab21, tab22, tab23, tab24 = st.tabs(["Drop Duplicates", "Select Feature", "Validation", "Model saving"])

            with tab21:
                from ganesh_package.Classification import Descriptors_smile
                
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
                    train_file = load_data(DATA_CLEANING_PATH_CLF)
                    val_file = st.session_state['data_final']

                    if train_file is not None and val_file is not None:
                        train_data, val_data = process_data(train_file, val_file)
                        
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
                        
                        st.download_button(
                            label="Download Cleaned Validation Data",
                            data=clean_val_data.to_csv(index=False),
                            file_name="cleaned_validation_data.csv",
                            mime="text/csv"
                        )

                        save_data(clean_val_data, DUPLICATED_CLEAN_VAL_PATH_CLF)
                        st.session_state['data_saved'] = True
                        st.success("Data saved successfully!")
                        logging.info("Validation data saved successfully")                

            with tab22:
                dataset_option = st.selectbox('Select dataset for analysis', ['clean dataset', 'feature selection dataset'])

                if st.button("Save Data"):
                    if dataset_option == 'clean dataset':
                        data1 = load_data(DATA_CLEANING_PATH_CLF)
                        data = data1.iloc[:,1:]
                    elif dataset_option == 'feature selection dataset':
                        data = load_data(FEATURE_SELECTION_DATA_PATH_CLF)

                    data_feat = data.columns
                    df = st.session_state['data_final']
                    data_f = df[data_feat]

                    # Replace null values
                    data_f.fillna(0, inplace=True)

                    st.write(data_f.head())
                    st.write(f"The shape of the dataset is: {data_f.shape}")

                    save_data(data_f, VALIDATION_DATA_PATH_CLF)
                    st.session_state['data_saved'] = True
                    st.success("Data saved successfully!")
                    logging.info("Validation data saved successfully")

            with tab23:
                st.write("Are you interested in validating the model?")
                
                data_clean = load_data(VALIDATION_DATA_PATH_CLF)
                models = Models_clf.all_models(data_clean)

                st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
                
                for model_name, result in models[0].items():
                    st.markdown(f"### {model_name}")
                    st.write(f"**Accuracy (Train):** {result['accuracy_train']:.4f}")
                    st.write("**Classification Report:**")
                    st.json(result['class_report_train'])
                    st.image(models[1][model_name])

            with tab24:
                file_paths = {
                    "methods": DESCRIPTOR_METHODS_JESON_PATH_CLF  # Updated to use the single JSON file
                }

                all_selected_methods = load_selected_methods_from_json(file_paths['methods'])

                clean_data = load_data(VALIDATION_DATA_PATH_CLF)
                column_names = clean_data.columns.tolist()

                model_name_to_save = st.session_state.get('model_save', 'default_model')

                os.makedirs(os.path.dirname(REGRESSION_TEXT_PATH_CLF), exist_ok=True)
                os.makedirs(os.path.dirname(REGRESSION_ZIP_PATH_CLF), exist_ok=True)
                os.makedirs(REGRESSION_MODEL_FOLDER_CLF, exist_ok=True)

                save_to_txt(REGRESSION_TEXT_PATH_CLF, all_selected_methods, column_names, model_name_to_save)
                
                existing_pickle_target_path = os.path.join(REGRESSION_MODEL_FOLDER_CLF, os.path.basename(FINAL_MODEL_PICKLE_PATH_CLF))
                if os.path.exists(FINAL_MODEL_PICKLE_PATH_CLF):
                    shutil.copy(FINAL_MODEL_PICKLE_PATH_CLF, existing_pickle_target_path)

                text_file_target_path = os.path.join(REGRESSION_MODEL_FOLDER_CLF, os.path.basename(REGRESSION_TEXT_PATH_CLF))
                shutil.copy(REGRESSION_TEXT_PATH, text_file_target_path)

                zip_folder(REGRESSION_MODEL_FOLDER_CLF, REGRESSION_ZIP_PATH_CLF)

                st.write("Are you interested in downloading the final model?")

                if st.button("Yes, I want to download"):
                    with open(REGRESSION_ZIP_PATH_CLF, "rb") as file:
                        st.download_button(
                            label="Download All Files",
                            data=file,
                            file_name="output_files.zip",
                            mime="application/zip"
                        )