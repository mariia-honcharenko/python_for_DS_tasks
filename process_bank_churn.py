import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, List, Dict, Any

def drop_na_values(raw_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop rows with missing values in specified columns.
    
    Args:
        raw_df (pd.DataFrame): The raw DataFrame.
        columns (List[str]): List of column names to check for missing values.
        
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values in specified columns dropped.
    """
    return raw_df.dropna(subset=columns)

def split_data(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and validation sets.
    
    Args:
        df (pd.DataFrame): The DataFrame to split.
        split_ratio (float): Ratio of training set size to the total dataset size.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for training and validation sets.
    """
    train_size = int(len(df) * split_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    print(f'Shape of train_df: {train_df.shape}')
    print(f'Shape of val_df: {val_df.shape}')
    return train_df, val_df

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training, validation, and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train, validation, and test dataframes.
        input_cols (List[str]): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train, val, and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'X_{split}'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def scale_numeric_features(data: Dict[str, Any], numeric_cols: List[str]) -> None:
    """
    Scale numeric features using MinMaxScaler.
    
    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        numeric_cols (List[str]): List of numeric columns.
    """
    scaler = MinMaxScaler()
    data['X_train'][numeric_cols] = scaler.fit_transform(data['X_train'][numeric_cols])
    data['X_val'][numeric_cols] = scaler.transform(data['X_val'][numeric_cols])
    data['scaler'] = scaler

def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> None:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['X_train'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    for split in ['train', 'val']:
        encoded = encoder.transform(data[f'X_{split}'][categorical_cols])
        data[f'X_{split}'] = pd.concat([data[f'X_{split}'].drop(columns=categorical_cols), pd.DataFrame(encoded, columns=encoded_cols, index=data[f'X_{split}'].index)], axis=1)
    data['encoder'] = encoder
    data['input_cols'] = [col for col in data['X_train'].columns]

def preprocess_data(raw_df: pd.DataFrame, target_col: str = 'Exited', scaler_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
    """
    Preprocess the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): The name of the target column.
        scaler_numeric (bool): Whether to scale numeric features or not.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]: Processed training inputs, training targets, validation inputs, validation targets, input columns, fitted MinMaxScaler, and fitted OneHotEncoder.
    """
    raw_df = drop_na_values(raw_df, [target_col])
    raw_df = raw_df.drop(columns=['Surname', 'CustomerId'])
    
    train_df, val_df = split_data(raw_df)
    
    input_cols = list(raw_df.columns)
    input_cols.remove(target_col)
    
    df_dict = {'train': train_df, 'val': val_df}
    data = create_inputs_targets(df_dict, input_cols, target_col)
    
    print(f'Shape of X_train before processing: {data["X_train"].shape}')
    print(f'Shape of X_val before processing: {data["X_val"].shape}')
    
    numeric_cols = data['X_train'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['X_train'].select_dtypes(include='object').columns.tolist()
    
    if scaler_numeric:
        scale_numeric_features(data, numeric_cols)
    
    print(f'Shape of X_train before encoding: {data["X_train"].shape}')
    print(f'Shape of X_val before encoding: {data["X_val"].shape}')
    
    encode_categorical_features(data, categorical_cols)
    
    print(f'Shape of X_train after encoding: {data["X_train"].shape}')
    print(f'Shape of X_val after encoding: {data["X_val"].shape}')
    
    return data['X_train'], data['train_targets'], data['X_val'], data['val_targets'], input_cols, data['scaler'], data['encoder']

def preprocess_new_data(new_df: pd.DataFrame, input_cols: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocess new data for prediction or evaluation.

    Args:
        new_df (pd.DataFrame): The new DataFrame to process.
        input_cols (List[str]): List of input columns.
        scaler (MinMaxScaler): Pre-fitted MinMaxScaler.
        encoder (OneHotEncoder): Pre-fitted OneHotEncoder.

    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction or evaluation.
    """
    new_df = new_df.drop(columns=['Surname', 'CustomerId'])
    
    # Identify numeric and categorical columns
    numeric_cols = new_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_df.select_dtypes(include='object').columns.tolist()
    
    print("Numeric columns in new_df:", numeric_cols)
    print("Categorical columns in new_df:", categorical_cols)
    
    if scaler:
        new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    
    if encoder:
        encoded = encoder.transform(new_df[categorical_cols])
        new_df = pd.concat([new_df.drop(columns=categorical_cols), pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=new_df.index)], axis=1)
    
    print("Columns in new_df after processing:", new_df.columns)
    new_df = new_df[input_cols]
    
    return new_df
