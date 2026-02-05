#Step four: Create functions for your two pipelines that produces the train and test datasets. 
# The end result should be a series of functions that can be called to produce the train and 
# test datasets for each of your two problems that includes all the data prep steps you took. 
# This is essentially creating a DAG for your data prep steps. Imagine you will need to do this 
# for multiple problems in the future so creating functions that can be reused is important. 
# You don't need to create one full pipeline function that does everything but rather a series 
# of smaller functions that can be called in sequence to produce the final datasets. 
# Use your judgement on how to break up the functions. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# correct variable type/class as needed
def correct_variable_types(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('float64')
    return df

# collapse factor levels as needed
def collapse_factor_levels(df, column, mapping):
    df[column] = df[column].map(mapping)
    return df
# one-hot encoding factor variables 
def one_hot_encode(df, column, mapping):
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True) 
    df = pd.concat([df.drop(columns=[column]), dummies], axis=1) 
    return df
# normalize the continuous variables
def normalize_continuous_variables(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df
# drop unneeded variables
def drop_unneeded_variables(df, columns):
    df = df.drop(columns=columns, errors='ignore')
    return df
# create target variable if needed
def create_target_variable(df, target_column):
    df['target'] = df[target_column]
    return df
# Calculate the prevalence of the target variable 
def calculate_prevalence(df, target_column):
    prevalence = df.groupby(target_column).size()
    return prevalence
# Create the necessary data partitions (Train,Tune,Test)
def create_data_partitions(df, test_size=0.4, random_state=42):
    train, temp = train_test_split(df, test_size=test_size, random_state=random_state)
    tune, test = train_test_split(temp, test_size=0.5, random_state=random_state)
    return train, tune, test