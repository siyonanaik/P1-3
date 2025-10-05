import pandas as pd

# ---------------------------------- YUAN WEI PART ----------------------------------------
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Takes in a dataframe and reindexes it to fill in weekends
    and holidays. Then Forward fill the missing data taken from
    the previous day.

    Parameters:
    data: pd.DataFrame

    Returns: pd.DataFrame
    '''

    # Cleaning data
    # Reindexing to fill in weekends
    data.index = pd.to_datetime(data.index)
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(full_index)
    # Forward filling missing data
    data = data.ffill()
    return data