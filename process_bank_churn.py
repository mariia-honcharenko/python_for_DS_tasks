import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, List, Dict, Any

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

def encode_categorical_features(data: Dict[str, Any], categorical_cols: List[str]) -> None:
    """
    One-hot encode categorical features.
    
    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train and val sets.
        categorical_cols (List[str]): List of categorical columns.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train']['inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    for split in ['train', 'val']:
        encoded = encoder.transform(data[split]['inputs'][categorical_cols])
        data[split]['inputs'] = pd.concat([data[split]['inputs'].drop(columns=categorical_cols), pd.DataFrame(encoded, columns=encoded_cols, index=data[split]['inputs'].index)], axis=1)

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
    
    print(f'Shape of train_inputs before encoding: {train_inputs.shape}')
    print(f'Shape of val_inputs before encoding: {val_inputs.shape}')
    
    # Encode the training data and get the feature columns
    train_inputs, encoder = encode_categorical_features(train_inputs, categorical_cols)
    feature_columns = list(train_inputs.columns)
    
    # Encode the validation data using the same columns
    val_inputs, _ = encode_categorical_features(val_inputs, categorical_cols, encoder, columns=feature_columns)
    
    print(f'Shape of train_inputs after encoding: {train_inputs.shape}')
    print(f'Shape of val_inputs after encoding: {val_inputs.shape}')
    
    return train_inputs, train_targets, val_inputs, val_targets, feature_columns, scaler, encoder
    


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
