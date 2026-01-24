import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def preprocess_data(data):
    # Checking the data has loaded as expected. Un-comment to check.
    #print(data.head())
    #print(data.shape)

    # Constructing the target column
    data = data.dropna(subset=["start_time"]) #removing any place holder rows (such as rows of all zeros) before sorting by dropping rows with no time stamp
    data["start_time"] = pd.to_datetime(data["start_time"]) # converting to date/time object
    data = data.sort_values("start_time").reset_index(drop=True) #ordering chronologically and giving new index (for target column and test/training split)
    data["TTNS"] = data["time_since"].shift(-1)  #Shifting the time-since column up by one to make time-to-next-slip 
    #data["start_time"] = data["TTNS"].dt.total_seconds() #Putting the time interval in seconds
    data = data.iloc[:-1] #Not including the last row since there isn't a "next event" for it

    # Sanity checks
    #nan_rows = data[data["TTNS"].isna()]
    #print(nan_rows)
    #print(nan_rows[["ref_time"]].head(20))
    #print("Dups: ", data["ref_time"].duplicated().sum())
    #print(data["TTNS"].isna().sum() == 0)
    #print(data[["ref_time", "TTNS"]].head(10))
    #print(data["TTNS"].describe())

    # Spliting training/testing data

    feature_cols = ["tide_deriv", "form_fac", "slip_size", "high_t_evt", "tide_height"]

    X = data[feature_cols] # feature columns/variables
    y = data["TTNS"] # target column

    #y = np.log10(data["TTNS"]) #use for improved regression models since we're spanning multiple orders of magnitude

    n = len(data)
    train_end = int(0.7*n) # 70 % of data for training
    val_end = int(0.85*n) #taking another 15% for validation

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:] #the final 15% is for testing
    y_test = y.iloc[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    
