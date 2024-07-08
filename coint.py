import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

# from helperFunctions import save_matrices_to_file

def calculate_thresholds(data1, data2):
    """
    Calculate upper and lower thresholds for trading signals based on the spread between two time series.

    Parameters:
    data1 (ndarray): Time series data for the first stock.
    data2 (ndarray): Time series data for the second stock.

    Returns:
    tuple: Upper threshold, lower threshold, and exit threshold for the spread.
    """
    spread = data1 - data2
    mu = np.mean(spread)
    sigma = np.std(spread)
    upper_threshold = mu + 2.5 * sigma
    lower_threshold = mu - 2.5 * sigma
    exit_threshold = mu  # Example: Exit when spread reverts to mean
    
    return upper_threshold, lower_threshold, exit_threshold


def find_cointegrated_pairs(prices):
    cointegrated_pairs = []
    n_assets = prices.shape[0]

    for idx1, idx2 in list(combinations(range(n_assets), 2)):
        data1 = prices[idx1, :]
        data2 = prices[idx2, :]
    
        result = coint(data1, data2)

    
        p_value = result[1]
        if (p_value < 0.05):
            upper_threshold, lower_threshold, exit_threshold = calculate_thresholds(data1, data2)
            cointegrated_pairs.append({
                "first": idx1, 
                "second": idx2, 
                "pvalue": p_value, 
                "upper_threshold": upper_threshold, 
                "lower_threshold": lower_threshold,  
                "exit threshold": exit_threshold
            })

    return(cointegrated_pairs)