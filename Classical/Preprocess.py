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
    scaler = StandardScaler()
    filter_mask = filtered_time["time_to_next_ev_hr"] != -1 #looking at column 2 of the time filtered data and keeping only the rows that are not -1
    print(filter_mask)
    TTNS = filtered_time[filter_mask.shift(1) & filter_mask][:-1]
    print(TTNS.shape)
    data = data_orig.loc[filter_mask]

#Putting all the events that have a known time until next slip into there own array so that we can train only on known slips 
    #collect_known_slips = (filtered_time.iloc[1:,1] != -1).to_numpy()
    known_next_slips = data_orig[filter_mask.shift(1) & filter_mask][:-1]
    amount_of_known = known_next_slips.shape
    print(known_next_slips.shape)
    print(data_orig.shape)

    '''no_scale_cols = ["high_t_evt", "start_time"]
    scale_cols = [
        "tide_h",
        "tide_deriv",
        "form_fac",
        "time_since",
        "slip_size",
        "tide_height"
    ]
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("passthrough", "passthrough", no_scale_cols)
            ]
    )
    '''



#This is breaking the notebooks and needs to be fixed:
#Should the time to next event and time since both be converted to hours?
#Also will need to update the preprocessing function in the other notebooks 
    #TTNS["time_to_next_ev_hr"] = pd.to_datetime(TTNS["time_to_next_ev_hr"]).dt.total_seconds() # converting to date/time object
    #TTNS["time_to_next_ev_hr"] = TTNS["time_to_next_ev_hr"].dt.total_seconds() #Putting the time interval in seconds
    '''
    # Constructing the target column
    data = data.dropna(subset=["start_time"]) #removing any place holder rows (such as rows of all zeros) before sorting by dropping rows with no time stamp
    data["start_time"] = pd.to_datetime(data["start_time"]) # converting to date/time object
    #data["start_time"] = data["TTNS"].dt.total_seconds() #Putting the time interval in seconds
    data = data.iloc[:-1] #Not including the last row since there isn't a "next event" for it
    '''
    # Sanity checks
    #nan_rows = data[data["TTNS"].isna()]
    #print(nan_rows)
    #print(nan_rows[["ref_time"]].head(20))
    #print("Dups: ", data["ref_time"].duplicated().sum())
    #print(data["TTNS"].isna().sum() == 0)
    #print(data[["ref_time", "TTNS"]].head(10))
    #print(data["TTNS"].describe())

    # Spliting training/testing data

    feature_cols = ["tide_deriv", "form_fac", "slip_size", "high_t_evt", "tide_height", "time_since"]
    known_next_slips["time_since"] *= 60

    X = known_next_slips[feature_cols] # feature columns/variables
    y = TTNS["time_to_next_ev_hr"] *3600  # target column and converted to seconds 

#Still need to find a different way to split it 
    '''n = len(data)
    train_end = int(0.7*n) # 70 % of data for training
    val_end = int(0.85*n) #taking another 15% for validation

    X_train = X.iloc[1:train_end] #to get rid of the NaN value in time since last slip in both x and y 
    y_train = y.iloc[1:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:] #the final 15% is for testing
    y_test = y.iloc[val_end:]'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=False) # 0.25 x 0.8 = 0.2

    print(y_train)
    print(X_train.head())
    y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_val = scaler.fit_transform(y_val.to_numpy().reshape(-1, 1))
    y_test = scaler.fit_transform(y_test.to_numpy().reshape(-1, 1))
    X_train[["tide_deriv", "form_fac", "slip_size","tide_height", "time_since"]] = scaler.fit_transform(X_train[["tide_deriv", "form_fac", "slip_size", "tide_height", "time_since"]].to_numpy()) 
    #data = scaler.fit_transform(data)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, amount_of_known
    
