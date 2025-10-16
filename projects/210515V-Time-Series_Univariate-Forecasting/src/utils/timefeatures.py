import numpy as np
import pandas as pd
from typing import List

def time_features(dates, freq='h'):
    """
    Creates time features from a pandas DatetimeIndex.
    """
    n_samples = len(dates)
    dates = pd.to_datetime(dates)
    
    # Define features based on frequency
    if freq == 't': # minutely
        features = [
            dates.minute.to_numpy(),
            dates.hour.to_numpy(),
            dates.dayofweek.to_numpy(),
            dates.day.to_numpy(),
            dates.dayofyear.to_numpy(),
            dates.month.to_numpy(),
            dates.weekofyear.to_numpy(),
        ]
    elif freq == 'h': # hourly
        features = [
            dates.hour.to_numpy(),
            dates.dayofweek.to_numpy(),
            dates.day.to_numpy(),
            dates.dayofyear.to_numpy(),
            dates.month.to_numpy(),
            dates.weekofyear.to_numpy(),
        ]
    else: # daily
        features = [
            dates.dayofweek.to_numpy(),
            dates.day.to_numpy(),
            dates.dayofyear.to_numpy(),
            dates.month.to_numpy(),
            dates.weekofyear.to_numpy(),
        ]
    
    # Stack and return
    time_features_arr = np.vstack(features)
    
    return time_features_arr.transpose()
