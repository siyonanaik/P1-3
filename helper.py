import pandas as pd
import numpy as np

def preprocess(df):
    # Ensure all correct data types
    try: 
        df.index = pd.to_datetime(df.index)
    except:
        raise TypeError("df index must be a datetime value!")
    
    try: 
        df['Open'] = df['Open'].astype(float)
    except:
        raise TypeError("Open Column must be a float value!")
    
    try: 
        df['High'] = df['High'].astype(float)
    except:
        raise TypeError("High Column must be a float value!")
    
    try: 
        df['Low'] = df['Low'].astype(float)
    except:
        raise TypeError("Low Column must be a float value!")
    
    try: 
        df['Close'] = df['Close'].astype(float)
    except:
        raise TypeError("Close Column must be a float value!")

    try: 
        df['Volume'] = df['Volume'].astype(int)
    except:
        raise TypeError("Volume Column must be an int value!")
    
    # Check if there are any duplicated dates found
    if df.index.duplicated().any():
        raise ValueError("Duplicated dates found!")
    
    # Check if dates are in chronological order
    if df.index.is_monotonic_increasing == False:
        raise ValueError("Dates are not in chronological order")
    
    # Checks that High>=Low/Open/Close, Low<=Open/Close
    if not((df['High'] >= df['Low']).all() and (df['High'] >= df['Open']).all() and (df['High'] >= df['Close']).all() and (df['Low'] <= df['Open']).all() and (df['Low'] <= df['Close']).all()):
        if not((df['High'] >= df['Low']).all()):
            raise ValueError('Low value is greater than High!')
        if not((df['High'] >= df['Open']).all()):
            raise ValueError('Open value is greater than High!')
        if not((df['High'] >= df['Close']).all()):
            raise ValueError('Close value is greater than High!')
        if not((df['Low'] <= df['Open']).all()):
            raise ValueError('Low value is greater than Open!')
        if not((df['Low'] <= df['Close']).all()):
            raise ValueError('Low value is greater than Close!')
        
    # Since this is a stock price dataset, outlier values will not be handled as they offer critical information about the dataset
    
    # Replace negative values with NaN, will be forward filled using .ffill()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].where(df[numeric_columns] >= 0)

    # Reindexing to fill in weekends
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_index)
    # Forward filling missing data
    df = df.ffill()
    # Back fill in case first index has no value
    df = df.bfill()

    return df