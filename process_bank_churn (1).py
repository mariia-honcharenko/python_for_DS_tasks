import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, List

def drop_na(raw_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return raw_df.dropna(subset=columns)

def split_data(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * split_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    print(f'Shape of train_df: {train_df.shape}')
    print(f'Shape of val_df: {val_df.shape}')
    return train_df, val_df

def separate_inputs_targets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    input_cols = list(df.columns)
    input_cols.remove(target_col)
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    print(f'Shape of inputs: {inputs.shape}')
    print(f'Shape of targets: {targets.shape}')
    return inputs, targets

def scale_numeric_features(df: pd.DataFrame, numeric_cols: List[str], scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler

def one_hot_encode_features(df: pd.DataFrame, categorical_cols: List[str], encoder: OneHotEncoder = None) -> Tuple[pd.DataFrame, OneHotEncoder]:
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_df = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df, encoder

def preprocess_data(raw_df: pd.DataFrame, target_col: str = 'Exited', scaler_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
    raw_df = drop_na(raw_df, [target_col])
    raw_df = raw_df.drop(columns=['Surname', 'CustomerId'])
    train_df, val_df = split_data(raw_df)
    train_inputs, train_targets = separate_inputs_targets(train_df, target_col)
    val_inputs, val_targets = separate_inputs_targets(val_df, target_col)
    print(f'Shape of train_inputs before processing: {train_inputs.shape}')
    print(f'Shape of val_inputs before processing: {val_inputs.shape}')
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()
    if scaler_numeric:
        train_inputs, scaler = scale_numeric_features(train_inputs, numeric_cols)
        val_inputs, _ = scale_numeric_features(val_inputs, numeric_cols, scaler)
    else:
        scaler = None
    train_inputs, encoder = one_hot_encode_features(train_inputs, categorical_cols)
    val_inputs, _ = one_hot_encode_features(val_inputs, categorical_cols, encoder)
    print(f'Shape of train_inputs after processing: {train_inputs.shape}')
    print(f'Shape of val_inputs after processing: {val_inputs.shape}')
    input_cols = list(train_inputs.columns)
    return train_inputs, train_targets, val_inputs, val_targets, input_cols, scaler, encoder


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
    
    if scaler:
        new_df, _ = scale_numeric_features(new_df, numeric_cols, scaler)
    
    new_df, _ = one_hot_encode_features(new_df, categorical_cols, encoder)
    
    new_df = new_df[input_cols]
    
    return new_df
