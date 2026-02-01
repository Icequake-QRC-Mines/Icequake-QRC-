import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

 

def preprocess_data(filtered_time, data_orig):
    scaler_X = StandardScaler() # scaler for feature columns
    scaler_y = StandardScaler() # scaler for TTNS
    filter_mask = filtered_time["time_to_next_ev_hr"] != -1 #looking at column 2 of the time filtered data and keeping only the rows that are not -1
    print(filter_mask)
    TTNS = filtered_time[filter_mask.shift(1) & filter_mask][:-1]
    print(TTNS.shape)
    data = data_orig.loc[filter_mask]

#Putting all the events that have a known time until next slip into there own array so that we can train only on known slips 
    #collect_known_slips = (filtered_time.iloc[1:,1] != -1).to_numpy()
    known_next_slips = data_orig[filter_mask.shift(1) & filter_mask][:-1]
    amount_of_known = known_next_slips.shape
    # print(known_next_slips.shape)
    # print(data_orig.shape)

    # Spliting training/testing data

    feature_cols = ["tide_deriv", "form_fac", "slip_size", "high_t_evt", "tide_height", "time_since"]
    known_next_slips["time_since"] *= 60

    X = known_next_slips[feature_cols] # feature columns/variables
    y = TTNS["time_to_next_ev_hr"] *3600  # target column and converted to seconds 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True) # 0.25 x 0.8 = 0.2

    print(y_train)
    print(X_train.head())

    # Make copies to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Scale features - fit only on training data, then transform val/test
    scale_cols = ["tide_deriv", "form_fac", "slip_size", "tide_height", "time_since"]

    # Scale and assign back column by column to preserve DataFrame structure
    scaled_train = scaler_X.fit_transform(X_train[scale_cols])
    scaled_val = scaler_X.transform(X_val[scale_cols])
    scaled_test = scaler_X.transform(X_test[scale_cols])

    for i, col in enumerate(scale_cols):
        X_train.loc[:, col] = scaled_train[:, i]
        X_val.loc[:, col] = scaled_val[:, i]
        X_test.loc[:, col] = scaled_test[:, i]

    # Scale target - fit only on training data, then transform val/test
    y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, amount_of_known, scaler_X, scaler_y
    
