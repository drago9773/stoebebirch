import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import json






def get_median_income_data(df, cbg_geoid_col, median_income_data_source):
    """
    Merges median income data with a DataFrame based on a specified CBG GEOID column.

    Parameters:
    df (pd.DataFrame): The DataFrame that contains the CBG GEOID column.
    cbg_geoid_col (str): The name of the column in `df` that contains the CBG GEOID.
    median_income_data_source (str): The file path to the median income CSV data.

    Returns:
    pd.DataFrame: The original DataFrame with an additional 'median_income' column merged in.
    """
    # Load the median income data
    median_income = pd.read_csv(median_income_data_source, skiprows=[1], na_values='-')
    median_income['GEO_ID'] = median_income['GEO_ID'].astype(str)
    median_income['cbg_geoid'] = median_income['GEO_ID'].str.extract('US(\d+)')
    median_income['median_income'] = median_income['B19013_001E'].str.replace(',', '+').str.replace('+', '').str.replace('-', '').astype(float)
    median_income = median_income[["cbg_geoid", "median_income"]]

    # Ensure the CBG GEOID column in `df` is a string and matches the format
    df[cbg_geoid_col] = df[cbg_geoid_col].astype(str).str.split('.').str[0]
    median_income["cbg_geoid"] = median_income["cbg_geoid"].astype(str)

    # Merge the data based on the specified CBG GEOID column
    merged_df = df.merge(median_income, left_on=cbg_geoid_col, right_on='cbg_geoid', how="inner")

    return merged_df

def one_hot_encode_features(df, columns_to_encode, mode='train', drop_first=True, encoder_filename=r'C:\Users\mattl\OneDrive\Desktop\Projects\stoebebirch\Models\one_hot_encoder.pkl', feature_names_filename='encoded_feature_names.json'):
    """
    One-hot encodes specified columns in the DataFrame. Saves the encoder and encoded feature names during training,
    and uses the saved encoder during prediction.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns_to_encode (list): A list of column names to be one-hot encoded.
    mode (str): 'train' to fit and save the encoder, 'predict' to load and use the saved encoder. Default is 'train'.
    drop_first (bool): Whether to drop the first level to avoid the dummy variable trap. Default is True.
    encoder_filename (str): The filename to save/load the OneHotEncoder. Default is 'one_hot_encoder.pkl'.
    feature_names_filename (str): The filename to save/load the encoded feature names. Default is 'encoded_feature_names.json'.

    Returns:
    pd.DataFrame: The DataFrame with the original columns replaced by their encoded versions.
    list: A list of the names of the new encoded features.
    """
    if mode == 'train':
        # Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)

        # Fit and transform the data
        encoded_features = encoder.fit_transform(df[columns_to_encode])

        # Get feature names for the encoded columns
        encoded_feature_names = encoder.get_feature_names_out(columns_to_encode).tolist()

        # Save the encoder and feature names
        joblib.dump(encoder, encoder_filename)
        with open(feature_names_filename, 'w') as f:
            json.dump(encoded_feature_names, f)

    elif mode == 'predict':
        # Load the saved encoder and feature names
        encoder = joblib.load(encoder_filename)
        with open(feature_names_filename, 'r') as f:
            encoded_feature_names = json.load(f)

        # Transform the data using the loaded encoder
        encoded_features = encoder.transform(df[columns_to_encode]).tolist()


    else:
        raise ValueError("Mode should be either 'train' or 'predict'.")

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Concatenate the original DataFrame (excluding the original columns) with the encoded DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoded_feature_names


def create_knn_benchmark_rent(df, knn_features, target, n_values, save_location, mode='train'):
    """
    Creates benchmark rent predictions using KNeighborsRegressor for various values of n.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    knn_features (list): A list of column names to be used as features (e.g., ['Latitude', 'Longitude']).
    target (str): The name of the target column (e.g., 'Rent'). Used only in 'train' mode.
    n_values (list): A list of n values for KNeighborsRegressor (e.g., [1, 5, 10]).
    save_location (str): The directory path where the KNN models should be saved or loaded from.
    mode (str): 'train' for training and saving models, 'predict' for loading models and predicting. Default is 'train'.

    Returns:
    pd.DataFrame: The DataFrame with new benchmark rent columns added.
    list: A list of the names of the new benchmark features.
    """
    # Convert latitude and longitude from degrees to radians for haversine metric
    df_rad = np.radians(df[knn_features].values)

    benchmark_features = []

    for n in n_values:
        # Create a new column name for the benchmark rent
        column_name = f'Rent_Benchmark_{n}_neighbors'
        benchmark_features.append(column_name)

        if mode == 'train':
            # Drop rows with missing values in the knn_features or target columns
            knn_df = df[knn_features + [target]].dropna()
            
            # Extract feature values and target values
            X = knn_df[knn_features].values
            y = knn_df[target].values
            
            # Convert latitude and longitude from degrees to radians for haversine metric
            X_rad = np.radians(X)
            
            # Initialize the KNeighborsRegressor with the haversine metric
            knn = KNeighborsRegressor(n_neighbors=n, metric='haversine')
            
            # Fit the KNN model
            knn.fit(X_rad, y)
            
            # Save the model to the specified location
            model_filename = os.path.join(save_location, f'knn_model_{n}_neighbors.pkl')
            joblib.dump(knn, model_filename)
            
            # Predict the benchmark rent for each point in the original DataFrame
            df[column_name] = knn.predict(df_rad)

        elif mode == 'predict':
            # Load the KNN model from the specified location
            model_filename = os.path.join(save_location, f'knn_model_{n}_neighbors.pkl')
            knn = joblib.load(model_filename)
            
            # Predict the benchmark rent for each point in the DataFrame
            df[column_name] = knn.predict(df_rad)

        else:
            raise ValueError("Mode should be either 'train' or 'predict'.")

    return df, benchmark_features

def fill_null(df, columns_to_fill, method, groupby=None, percentile=None):
    """
    Fills null values in specified columns using a specified aggregation method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns_to_fill (list): A list of column names to fill.
    method (str): The aggregation method to use. Options include 'min', 'max', 'mean', 'median', 'percentile', '0', etc.
    groupby (str or None): The column to group by before applying the method. If None, applies globally. Default is None.
    percentile (float or None): The percentile to use if the method is 'percentile'. Must be between 0 and 100.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns filled.
    """
    for column in columns_to_fill:
        if method == '0':
            fill_value = 0
        
        elif method == 'min':
            fill_value = df.groupby(groupby)[column].transform('min') if groupby else df[column].min()
        
        elif method == 'max':
            fill_value = df.groupby(groupby)[column].transform('max') if groupby else df[column].max()
        
        elif method == 'mean':
            fill_value = df.groupby(groupby)[column].transform('mean') if groupby else df[column].mean()
        
        elif method == 'median':
            fill_value = df.groupby(groupby)[column].transform('median') if groupby else df[column].median()
        
        elif method == 'percentile':
            if percentile is None or not (0 <= percentile <= 100):
                raise ValueError("Percentile must be specified and must be between 0 and 100.")
            fill_value = df.groupby(groupby)[column].transform(lambda x: np.percentile(x.dropna(), percentile)) if groupby else np.percentile(df[column].dropna(), percentile)
        
        else:
            raise ValueError(f"Method '{method}' is not supported.")
        
        # Fill the null values with the computed fill_value
        df[column] = df[column].fillna(fill_value)
    
    return df

def scale_features(X_train, X_test):
    """
    Scales the training and test features using MinMaxScaler.

    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training feature set.
    X_test (pd.DataFrame or np.ndarray): The test feature set.

    Returns:
    np.ndarray: Scaled training feature set.
    np.ndarray: Scaled test feature set.
    """
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def scale_features(X, mode='train', scaler_filename=r'C:\Users\mattl\OneDrive\Desktop\Projects\stoebebirch\Models\scaler.pkl'):
    """
    Scales the features using MinMaxScaler, with different behavior for training and prediction.

    Parameters:
    X (pd.DataFrame or np.ndarray): The feature set to be scaled.
    mode (str): 'train' for fitting the scaler, 'predict' for applying an existing scaler. Default is 'train'.
    scaler_filename (str): The filename to save/load the scaler. Default is 'scaler.pkl'.

    Returns:
    np.ndarray: Scaled feature set.
    MinMaxScaler: The fitted scaler (only returned if mode is 'train').
    """
    if mode == 'train':
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit the scaler on the data and transform it
        X_scaled = scaler.fit_transform(X)

        # Save the scaler for future use
        joblib.dump(scaler, scaler_filename)

        return X_scaled, scaler

    elif mode == 'predict':
        # Load the scaler
        scaler = joblib.load(scaler_filename)

        # Transform the data using the loaded scaler
        X_scaled = scaler.transform(X)

        return X_scaled
    

def generate_polynomial_features(X_scaled, original_feature_names, mode='train', poly_filename=r'C:\Users\mattl\OneDrive\Desktop\Projects\stoebebirch\Models\poly_transformer.pkl', degree=2, include_bias=False):
    """
    Generates polynomial features with different behavior for training and prediction.

    Parameters:
    X_scaled (np.ndarray): The scaled feature set.
    mode (str): 'train' for fitting the transformer, 'predict' for applying an existing transformer. Default is 'train'.
    poly_filename (str): The filename to save/load the polynomial transformer. Default is 'poly_transformer.pkl'.
    degree (int): The degree of the polynomial features. Default is 2.
    include_bias (bool): Whether to include a bias column (i.e., a column of ones). Default is False.

    Returns:
    np.ndarray: Polynomial feature set.
    PolynomialFeatures: The fitted polynomial transformer (only returned if mode is 'train').
    """
    if mode == 'train':
        # Initialize the PolynomialFeatures transformer
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

        # Fit and transform the data
        X_poly = poly.fit_transform(X_scaled)

        # Save the transformer for future use
        joblib.dump(poly, poly_filename)

        return X_poly, poly

    elif mode == 'predict':
        # Load the polynomial transformer
        poly = joblib.load(poly_filename)

        # Transform the data using the loaded transformer
        X_poly = poly.transform(X_scaled)

        # Get the feature names
        feature_names = poly.get_feature_names_out(original_feature_names)

        return X_poly, feature_names