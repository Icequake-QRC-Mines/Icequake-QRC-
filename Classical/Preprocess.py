import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

 

def preprocess_data(filtered_time, data_orig):
    filter_mask = filtered_time["time_to_next_ev_hr"] != -1 #looking at column 2 of the time filtered data and keeping only the rows that are not -1
    print(filter_mask)
    TTNS = filtered_time[filter_mask.shift(1) & filter_mask & filter_mask.shift(-1)][1:-1]
    print(TTNS.shape)
    data = data_orig.loc[filter_mask]

#Putting all the events that have a known time until next slip into there own array so that we can train only on known slips 
    #known_next_slips = data_orig[filter_mask.shift(1) & filter_mask][:-1] 
    known_next_slips = data_orig[filter_mask.shift(1) &  filter_mask & filter_mask.shift(-1)][1:-1]
    amount_of_known = known_next_slips.shape
    print(known_next_slips.shape)
    print(data_orig.shape)


#Target column creation and and converting all times into seconds 


    feature_cols = ["tide_deriv", "form_fac", "time_since", "slip_size", "high_t_evt", "tide_height"]
    known_next_slips["time_since"] *= 60

    X = known_next_slips[feature_cols] 
    y = TTNS["time_to_next_ev_hr"] *3600  

#Splitting for training/validation/testing with random split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

#Normalization Using Standard Scalar 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, amount_of_known