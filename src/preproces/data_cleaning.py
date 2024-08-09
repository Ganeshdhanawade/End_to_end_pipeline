import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from src.exception import CustomException  # Assuming this is a custom exception class

class DataCleaner:
    def __init__(self, row_thresh=0.20, col_thresh=0.20):
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        self.preprocessor = None
        logging.info("DataCleaner initialized with row_thresh=%s and col_thresh=%s", row_thresh, col_thresh)

    def drop_null_values(self, df):
        try:
            # Calculate the percentage of null values for each row
            row_null_percentage = df.isnull().mean(axis=1)
            # Filter out rows with null percentage greater than or equal to the threshold
            df_filtered_rows = df[row_null_percentage < self.row_thresh]
            # Calculate the percentage of null values for each column in the filtered dataframe
            col_null_percentage = df_filtered_rows.isnull().mean(axis=0)
            # Filter out columns with null percentage greater than the threshold
            df_filtered = df_filtered_rows.loc[:, col_null_percentage < self.col_thresh]
            # Reset index
            df_filtered.reset_index(inplace=True, drop=True)
            logging.info("Dropped null values, resulting shape: %s", df_filtered.shape)
            return df_filtered
        except Exception as e:
            logging.error("Error in drop_null_values: %s", e)
            raise CustomException(e)

    def get_column_types(self, df):
        try:
            float_cols = df.select_dtypes(include=['float64']).columns.tolist()
            int_cols = df.select_dtypes(include=['int64']).columns.tolist()
            logging.info("Float columns: %s, Int columns: %s", float_cols, int_cols)
            return float_cols, int_cols
        except Exception as e:
            logging.error("Error in get_column_types: %s", e)
            raise CustomException(e)

    def fit(self, df):
        try:
            # Drop null values
            df_cleaned = self.drop_null_values(df)
            # Get column types
            float_cols, int_cols = self.get_column_types(df_cleaned)

            # Define pipelines for different column types
            float_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])
            int_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])

            # Combine pipelines
            transformers = [
                ('float', float_pipeline, float_cols),
                ('int', int_pipeline, int_cols)
            ]

            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            # Fit the preprocessor
            self.preprocessor.fit(df_cleaned)
            logging.info("Preprocessor fitted successfully")
        except Exception as e:
            logging.error("Error in fit: %s", e)
            raise CustomException(e)

    def transform(self, df):
        if self.preprocessor is None:
            raise RuntimeError("You must fit the cleaner before transforming data.")
        
        try:
            # Drop null values
            df_cleaned = self.drop_null_values(df)
            # Apply transformations
            df_transformed = self.preprocessor.transform(df_cleaned)

            # Get column names after transformation
            float_cols, int_cols = self.get_column_types(df_cleaned)
            transformed_cols = float_cols + int_cols + [col for col in df_cleaned.columns if col not in float_cols + int_cols]
            # Convert the transformed data back to a DataFrame
            df_transformed = pd.DataFrame(df_transformed, columns=transformed_cols)
            df_final = df_transformed[df_cleaned.columns]
            logging.info("Data transformed successfully")
            return df_final
        except Exception as e:
            logging.error("Error in transform: %s", e)
            raise CustomException(e)

    def fit_transform(self, df):
        try:
            self.fit(df)
            return self.transform(df)
        except Exception as e:
            logging.error("Error in fit_transform: %s", e)
            raise CustomException(e)
