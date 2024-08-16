  
###regression-nav=feature selection ##

 # import streamlit as st
    # import logging

    # # Assume the methods `load_data`, `descriptor_methods`, `fingerprint_methods`, `qm_methods`, `descriptor_set_methods`, and `save_data` are defined elsewhere

    # # Placeholder for the selected methods (these should come from your Streamlit sidebar selections)
    # selected_descriptor_methods = st.sidebar.multiselect("Select Descriptor Methods", options=list(descriptor_methods.keys()))
    # selected_fingerprint_methods = st.sidebar.multiselect("Select Fingerprint Methods", options=list(fingerprint_methods.keys()))
    # selected_qm_methods = st.sidebar.multiselect("Select QM Methods", options=list(qm_methods.keys()))
    # selected_descriptor_set_methods = st.sidebar.multiselect("Select Descriptor Set Methods", options=list(descriptor_set_methods.keys()))

    # fp_names = [
    #     'ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'ecfc0', 'ecfc2', 'ecfc4', 'ecfc6',
    #     'fcfp2', 'fcfp4', 'fcfp6', 'fcfc2', 'fcfc4', 'fcfc6', 'lecfp4', 'lecfp6',
    #     'lfcfp4', 'lfcfp6', 'maccs', 'ap', 'tt', 'hashap', 'hashtt', 'avalon',
    #     'laval', 'rdk5', 'rdk6', 'rdk7'
    # ]

    # # Add a condition to show the list of fingerprints if "FingerprintCalculator" is selected
    # if "FingerprintCalculator" in selected_fingerprint_methods:
    #     selected_fingerprint_types = st.sidebar.multiselect("Select Fingerprint Types", options=fp_names)
    # else:
    #     selected_fingerprint_types = []

    # uploaded_file = st.file_uploader("Upload your dataset")

    # if uploaded_file is not None:
    #     df = load_data(uploaded_file)
    #     if df is not None:
    #         st.write("Uploaded DataFrame:")
    #         st.write(df.head())

    #         if st.button("Run Selected Calculations"):
    #             if not selected_descriptor_methods and not selected_fingerprint_methods and not selected_qm_methods and not selected_descriptor_set_methods:
    #                 st.warning("Please select at least one method.")
    #                 logging.warning("No methods selected")
    #             else:
    #                 # Save selected methods to files
    #                 save_selected_methods(selected_descriptor_methods, DESCRIPTOR_METHODS_FILE_PATH)
    #                 save_selected_methods(selected_fingerprint_methods, FINGERPRINT_METHODS_FILE_PATH)
    #                 save_selected_methods(selected_qm_methods, QM_METHODS_FILE_PATH)
    #                 save_selected_methods(selected_descriptor_set_methods, DESCRIPTOR_SET_METHODS_FILE_PATH)

    #                 # Process selected methods
    #                 for method_name in selected_descriptor_methods:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_methods[method_name](df)
    #                         logging.info(f"Ran descriptor method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor method {method_name}: {error_message}")

    #                 for method_name in selected_fingerprint_methods:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         if method_name == "FingerprintCalculator":
    #                             if selected_fingerprint_types:
    #                                 df = fingerprint_methods[method_name](df, selected_fingerprint_types)
    #                                 logging.info(f"Ran FingerprintCalculator with types {selected_fingerprint_types}")
    #                             else:
    #                                 st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
    #                                 logging.warning("No fingerprint types selected for FingerprintCalculator")
    #                         else:
    #                             df = fingerprint_methods[method_name](df)
    #                             logging.info(f"Ran fingerprint method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running fingerprint method {method_name}: {error_message}")

    #                 for method_name in selected_qm_methods:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = qm_methods[method_name](df)
    #                         logging.info(f"Ran QM method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running QM method {method_name}: {error_message}")

    #                 for method_name in selected_descriptor_set_methods:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_set_methods[method_name](df)
    #                         logging.info(f"Ran descriptor set method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor set method {method_name}: {error_message}")

    #                 st.write("Resulting DataFrame:")
    #                 st.write(df.head())

    #                 # Create columns for buttons
    #                 col1, col2, col3 = st.columns([2, 1, 1])

    #                 with col1:
    #                     # Display the shape of the resulting DataFrame
    #                     col1.markdown(f"**DataFrame Shape:** {df.shape}")
    #                     logging.info(f"Displayed DataFrame shape: {df.shape}")

    #                 with col3:
    #                     # Button to download dataset
    #                     csv = df.to_csv(index=False)
    #                     st.download_button(
    #                         label="Download Data",
    #                         data=csv,
    #                         file_name="processed_data.csv",
    #                         mime="text/csv"
    #                     )
    #                     logging.info("Download button created")

    #                 # Save the dataset
    #                 save_data(df, FEATURE_CALCULATED_DATA_PATH)
    #                 st.session_state['data_saved'] = True
    #                 st.success("Data saved successfully!")
    #                 logging.info("Data saved successfully")


## regression-model validation navigation ##

# # Initialize session state variables if they do not exist
    # if 'df1' not in st.session_state:
    #     st.session_state['df1'] = None

    # if 'data_saved' not in st.session_state:
    #     st.session_state['data_saved'] = False

    # if 'selected_descriptor_methods' not in st.session_state:
    #     st.session_state['selected_descriptor_methods'] = load_selected_methods(DESCRIPTOR_METHODS_FILE_PATH)

    # if 'selected_fingerprint_methods' not in st.session_state:
    #     st.session_state['selected_fingerprint_methods'] = load_selected_methods(FINGERPRINT_METHODS_FILE_PATH)

    # if 'selected_qm_methods' not in st.session_state:
    #     st.session_state['selected_qm_methods'] = load_selected_methods(QM_METHODS_FILE_PATH)

    # if 'selected_descriptor_set_methods' not in st.session_state:
    #     st.session_state['selected_descriptor_set_methods'] = load_selected_methods(DESCRIPTOR_SET_METHODS_FILE_PATH)

    # if 'selected_fingerprint_types' not in st.session_state:
    #     st.session_state['selected_fingerprint_types'] = []
        
    # if 'data_final' not in st.session_state:
    #     st.session_state['data_final'] = False

    # # File uploader
    # uploaded_file = st.file_uploader("Upload your dataset for validation")

    # if uploaded_file is not None:
    #     st.session_state['df1'] = load_data(uploaded_file)
        
    #     if st.session_state['df1'] is not None:
    #         st.write("Uploaded DataFrame:")
    #         st.write(st.session_state['df1'].head())

    #         # Manage the button state to avoid multiple runs
    #         if 'run_calculations' not in st.session_state:
    #             st.session_state['run_calculations'] = False

    #         if st.button("Run Selected Calculations for Validation") or st.session_state['run_calculations']:
    #             if not st.session_state['run_calculations']:
    #                 st.session_state['run_calculations'] = True
    #                 df = st.session_state['df1']

    #                 # Process selected methods
    #                 for method_name in st.session_state['selected_descriptor_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_methods[method_name](df)
    #                         logging.info(f"Ran descriptor method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor method {method_name}: {error_message}")

    #                 for method_name in st.session_state['selected_fingerprint_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         if method_name == "FingerprintCalculator":
    #                             if st.session_state['selected_fingerprint_types']:
    #                                 # Pass selected fingerprint types properly
    #                                 df = fingerprint_methods[method_name](df, st.session_state['selected_fingerprint_types'])
    #                                 logging.info(f"Ran FingerprintCalculator with types {st.session_state['selected_fingerprint_types']}")
    #                             else:
    #                                 st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
    #                                 logging.warning("No fingerprint types selected for FingerprintCalculator")
    #                         else:
    #                             df = fingerprint_methods[method_name](df)
    #                             logging.info(f"Ran fingerprint method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running fingerprint method {method_name}: {error_message}")

    #                 for method_name in st.session_state['selected_qm_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = qm_methods[method_name](df)
    #                         logging.info(f"Ran QM method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running QM method {method_name}: {error_message}")

    #                 for method_name in st.session_state['selected_descriptor_set_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_set_methods[method_name](df)
    #                         logging.info(f"Ran descriptor set method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor set method {method_name}: {error_message}")

    #                 st.write("Resulting DataFrame:")
    #                 st.write(df.head())
    #                 st.session_state['data_final'] = df

    #             # Tabs for additional actions
    #             tab21, tab22, tab23 = st.tabs(["Drop Duplicates", "Select Feature", "Validation"])

    #             with tab21:
    #                 from ganesh_package.Classification import Descriptors_smile
                    
    #                 # Load and standardize data
    #                 def process_data(train_df, val_df):
    #                     # Load data
    #                     train_data = train_df.copy()
    #                     val_data = val_df.copy()
                        
    #                     # Standardize SMILES
    #                     train_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(train_data['SMILES'])
    #                     val_data['Standardized_SMILES'] = Descriptors_smile.STANDERDIZE_SMILES(val_data['SMILES'])
                        
    #                     return train_data, val_data

    #                 st.write("Are you interested in dropping duplicate smiles and saving data?")
                    
    #                 if st.button("Submit"):
    #                     # File upload
    #                     train_file = load_data(DATA_CLEANING_PATH)
    #                     val_file = st.session_state['data_final']

    #                     if train_file is not None and val_file is not None:
    #                         train_data, val_data = process_data(train_file, val_file)
                            
    #                         # Display original data
    #                         st.write("Training Data")
    #                         st.write(train_data.head(2))
    #                         st.write(f"The shape of dataset is: {train_data.shape}")
    #                         st.write("Validation Data")
    #                         st.write(val_data.head(2))
    #                         st.write(f"The shape of dataset is: {val_data.shape}")
                            
    #                         # Check for duplicates in training data
    #                         train_smiles_set = set(train_data['Standardized_SMILES'].dropna())
    #                         val_data['Is_Duplicate'] = val_data['Standardized_SMILES'].apply(lambda x: x in train_smiles_set if x is not None else False)
                            
    #                         # Filter out duplicates
    #                         clean_val_data = val_data[~val_data['Is_Duplicate']]
                            
    #                         st.subheader("Cleaned Validation Data")
    #                         st.write(clean_val_data.head(5))
    #                         st.write(f"The shape of dataset is: {clean_val_data.shape}")
    #                         st.write(f'Duplicated records: {val_data.shape[0]-clean_val_data.shape[0]}')
                            
    #                         # Option to download cleaned data
    #                         st.download_button(
    #                             label="Download Cleaned Validation Data",
    #                             data=clean_val_data.to_csv(index=False),
    #                             file_name="cleaned_validation_data.csv",
    #                             mime="text/csv"
    #                         )

    #                         save_data(clean_val_data, DUPLICATED_CLEAN_VAL_PATH)
    #                         st.session_state['data_saved'] = True
    #                         st.success("Data saved successfully!")
    #                         logging.info("Validation data saved successfully")                

    #             with tab22:
    #                 dataset_option = st.selectbox('Select dataset for analysis', ['clean dataset', 'feature selection dataset'])

    #                 if st.button("Save Data"):
    #                     if dataset_option == 'clean dataset':
    #                         data = load_data(DATA_CLEANING_PATH)
    #                         data_feat = data.columns
    #                     elif dataset_option == 'feature selection dataset':
    #                         data = load_data(FEATURE_SELECTION_DATA_PATH)
    #                         data_feat = data.columns

    #                     # Final feature selected or clean dataset
    #                     df = st.session_state['data_final']
    #                     data_f = df[data_feat]

    #                     # Replace null values
    #                     data_f.fillna(0, inplace=True)

    #                     st.write(data_f.head())
    #                     st.write(f"The shape of the dataset is: {data_f.shape}")

    #                     save_data(data_f, VALIDATION_DATA_PATH)
    #                     st.session_state['data_saved'] = True
    #                     st.success("Data saved successfully!")
    #                     logging.info("Validation data saved successfully")

    #             with tab23:
    #                 def plot_predictions(y_true, y_pred):
    #                     plt.figure(figsize=(10, 6))
    #                     plt.scatter(y_true, y_pred, alpha=0.6, color='b')
    #                     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='r', linestyle='--', linewidth=2)
    #                     plt.xlabel('Actual Values')
    #                     plt.ylabel('Predicted Values')
    #                     plt.title('Actual vs Predicted Values')
    #                     st.pyplot(plt)
                    
    #                 def predict_and_evaluate(model, data_clean):
    #                     X = data_clean.drop(columns=['res'])  # Assuming 'target' is the column to predict
    #                     y_true = data_clean['res']
    #                     y_pred = model.predict(X)
    #                     rmse = mean_squared_error(y_true, y_pred, squared=False)
    #                     r2 = r2_score(y_true, y_pred)    
    #                     return y_true, y_pred, rmse, r2
                    
    #                 st.write("Are you interested in validating the model?")

    #                 # Load data and model
    #                 model = pickle.load(open(FINAL_MODEL_PICKLE_PATH, 'rb'))
    #                 data_clean = load_data(VALIDATION_DATA_PATH)

    #                 if st.button("Run Models"):
    #                     y_true, y_pred, rmse, r2 = predict_and_evaluate(model, data_clean)
                        
    #                     st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Model Evaluation Results</h2>", unsafe_allow_html=True)
                        
    #                     st.markdown("<h3 style='color: #ff6347;'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
    #                     st.write(f"**R2 Score:** {r2:.4f}")
    #                     st.write(f"**RMSE:** {rmse:.4f}")
                        
    #                     st.markdown("<h4 style='text-align: center;'>Actual vs Predicted Plot</h4>", unsafe_allow_html=True)
    #                     plot_predictions(y_true, y_pred)



 # # File uploader
    # uploaded_file = st.file_uploader("Upload your dataset")

    # if uploaded_file is not None:
    #     st.session_state['df1'] = load_data(uploaded_file)
        
    #     if st.session_state['df1'] is not None:
    #         st.write("Uploaded DataFrame:")
    #         st.write(st.session_state['df1'].head())

    #         # Manage the button state to avoid multiple runs
    #         if 'run_calculations' not in st.session_state:
    #             st.session_state['run_calculations'] = False

    #         if st.button("Run Selected Calculations"):
    #             if not st.session_state['selected_descriptor_methods'] and not st.session_state['selected_fingerprint_methods'] and not st.session_state['selected_qm_methods'] and not st.session_state['selected_descriptor_set_methods']:
    #                 st.warning("Please select at least one method.")
    #                 logging.warning("No methods selected")
    #             else:
    #                 df = st.session_state['df1']

    #                 # Process selected descriptor methods
    #                 for method_name in st.session_state['selected_descriptor_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_methods[method_name](df)
    #                         logging.info(f"Ran descriptor method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor method {method_name}: {error_message}")

    #                 # Process selected fingerprint methods
    #                 for method_name in st.session_state['selected_fingerprint_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         if method_name == "FingerprintCalculator":
    #                             if st.session_state['selected_fingerprint_types']:
    #                                 df = fingerprint_methods[method_name](df, st.session_state['selected_fingerprint_types'])
    #                                 logging.info(f"Ran FingerprintCalculator with types {st.session_state['selected_fingerprint_types']}")
    #                             else:
    #                                 st.warning("Please select at least one fingerprint type for FingerprintCalculator.")
    #                                 logging.warning("No fingerprint types selected for FingerprintCalculator")
    #                         else:
    #                             df = fingerprint_methods[method_name](df)
    #                             logging.info(f"Ran fingerprint method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running fingerprint method {method_name}: {error_message}")

    #                 # Process selected QM methods
    #                 for method_name in st.session_state['selected_qm_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = qm_methods[method_name](df)
    #                         logging.info(f"Ran QM method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running QM method {method_name}: {error_message}")

    #                 # Process selected descriptor set methods
    #                 for method_name in st.session_state['selected_descriptor_set_methods']:
    #                     try:
    #                         st.write(f"Running {method_name}...")
    #                         df = descriptor_set_methods[method_name](df)
    #                         logging.info(f"Ran descriptor set method {method_name}")
    #                     except Exception as e:
    #                         error_message = CustomException(e)
    #                         st.error(f"Error running {method_name}: {error_message}")
    #                         logging.error(f"Error running descriptor set method {method_name}: {error_message}")

    #                 st.write("Resulting DataFrame:")
    #                 st.write(df.head())
    #                 st.session_state['data_final'] = df

    #                 # Create columns for buttons
    #                 col1, col2, col3 = st.columns([2, 1, 1])

    #                 with col1:
    #                     col1.markdown(f"**DataFrame Shape:** {df.shape}")
    #                     logging.info(f"Displayed DataFrame shape: {df.shape}")

    #                 with col3:
    #                     csv = df.to_csv(index=False)
    #                     st.download_button(
    #                         label="Download Data",
    #                         data=csv,
    #                         file_name="processed_data.csv",
    #                         mime="text/csv"
    #                     )
    #                     logging.info("Download button created")

    #                 # Save the dataset
    #                 save_data(df, FEATURE_CALCULATED_DATA_PATH)
    #                 st.session_state['data_saved'] = True
    #                 st.success("Data saved successfully!")
    #                 logging.info("Data saved successfully")