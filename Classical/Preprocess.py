import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata




'''def augment_with_sdv(X, y, target_col, n_samples, random_state=42):
    """Generate synthetic training samples using SDV's GaussianCopulaSynthesizer.

    Fits a copula model on the joint distribution of features + target,
    then samples new rows that preserve inter-feature correlations.
    """
    # Build a single DataFrame with features + target
    train_df = X.copy().reset_index(drop=True)
    train_df[target_col] = y.values if hasattr(y, 'values') else y

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_df)

    n_synthetic = n_samples - len(train_df)
    synthetic_df = synthesizer.sample(num_rows=n_synthetic)

    # Combine original + synthetic
    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)

    X_aug = augmented_df.drop(columns=[target_col]).values
    y_aug = augmented_df[target_col].values

    print(np.min(X_aug))
    print(np.min(y_aug))
    print(np.max(X_aug))
    print(np.max(y_aug))

    return X_aug, y_aug, '''


def preprocess_data_window(filtered_time, data_orig, n_previous_events, random_state=42):
    base_mask = filtered_time["time_to_next_ev_hr"] != -1
    filter_mask = base_mask.copy()
    for i in range(1, n_previous_events+1):
        base_mask &= base_mask.shift(i) # this is checking events n-1, n-2, n-3, ... to see if they are valid
    filter_mask &= base_mask.shift(-1) # this is checking the event n+1 to see if it is valid
    filter_mask.fillna(False, inplace=True)
    '''
    base_mask = filtered_time["time_to_next_ev_hr"] != -1

    # Only require the current event and the next event to be valid
    filter_mask = base_mask & base_mask.shift(-1)

    filter_mask.fillna(False, inplace=True)'''

    feature_cols = ["tide_deriv", "form_fac", "time_since", "slip_size", "high_t_evt", "tide_height"]
    X = data_orig[feature_cols].copy()
    X['time_since'] *= 60
    y = filtered_time["time_to_next_ev_hr"] * 3600

    # first make pairs of feature rows with its previous n events
    # we can do this by shifting the feature rows down by 1 and then creating tuples of. If a row is 1xN where N 
    # is the number of features, we want each data point to be a 1x (N*(n_previous_events+1)) array of features, where 
    # the N elements of the  first row is the current event and the next n_previous_events rows are the previous events.
    windows = [X]
    for i in range(1, n_previous_events+1):
        shifted = X.shift(i)
        shifted.columns = [f"{col}-{i}" for col in feature_cols]
        windows.append(shifted)
    
    windows[0].columns = [f"{col}-0" for col in feature_cols]
    
    X_full = pd.concat(windows, axis=1).loc[filter_mask] # combine window of events with original event
    y = filtered_time.loc[filter_mask, "time_to_next_ev_hr"] * 3600

    print("X shape: ", X_full.shape)
    print("y shape: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    '''scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)'''

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

 

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
    print("Next slips shape", known_next_slips.shape)
    print("orig shape", data_orig.shape)


#Target column creation and and converting all times into seconds 


    feature_cols = ["tide_deriv", "form_fac", "time_since", "slip_size", "high_t_evt", "tide_height"]
    known_next_slips["time_since"] *= 60

    X = known_next_slips[feature_cols] 
    y = TTNS["time_to_next_ev_hr"] *3600  

#Training on subsets: First two years 
    #X = known_next_slips[feature_cols][:575]
    #y = TTNS["time_to_next_ev_hr"][:575] * 3600

#Last two years 
    #X = known_next_slips[feature_cols][4497:]
    #y = TTNS["time_to_next_ev_hr"][4497:] *3600

#Without the first two yeas and the last two years
    #X = known_next_slips[feature_cols][575:4497]
    #y = TTNS["time_to_next_ev_hr"][575:4497] *3600

    #Sampling the subset to check if improved results are likely sample size dependent by taking the middle 575 events in the subset w/o the first and last two years 
    '''length = 4497-575
    mid = (575+length) // 2
    half = 575//2
    mid_start = mid-half 
    mid_end= mid_start + 575
    X=known_next_slips[feature_cols][mid_start:mid_end]
    y = TTNS["time_to_next_ev_hr"][mid_start:mid_end] *3600
    print("X Length",len(X))
    print("Y Length", len(y))
    print(X.head())
    print(y.head())'''
#Everything except last two years 
    #X = known_next_slips[feature_cols][:4497]
    #y = TTNS["time_to_next_ev_hr"][:4497] *3600
#Synthetic Data: 
    #X_bs, y_bs = resample(X, y, n_samples=4000, random_state=42)
#Splitting for training/validation/testing with random split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

    #print(np.min(X_train))
    #print(np.min(y_train))
    #print(np.max(X_train))
    #print(np.max(y_train))

#Augment training data with SDV GaussianCopula 
    # 7 Times Original Size 
    #X_train, y_train = augment_with_sdv(X_train, y_train, target_col="TTNS", n_samples=2400, random_state=42)
    #X_val, y_val = augment_with_sdv(X_val, y_val, target_col="TTNS", n_samples=800, random_state=42)
    #X_test, y_test = augment_with_sdv(X_test, y_test, target_col="TTNS", n_samples=800, random_state=42)

    # 13 Times Original Size 
    '''X_train, y_train = augment_with_sdv(X_train, y_train, target_col="TTNS", n_samples=4800, random_state=42)
    X_val, y_val = augment_with_sdv(X_val, y_val, target_col="TTNS", n_samples=1600, random_state=42)
    X_test, y_test = augment_with_sdv(X_test, y_test, target_col="TTNS", n_samples=1600, random_state=42)'''
#Normalization Using Standard Scalar 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(), feature_cols, amount_of_known