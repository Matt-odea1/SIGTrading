'''
This files combines the get_matrices_coint and get_matrices_lag functions

Rationale: Matrix handling becomes easy and is consistent - especially with the uncertainty 
matrix because its diagonal nature is not preserved when combining them
'''
import numpy as np
from coint import find_cointegrated_pairs
from lagpairs import find_lag_pairs, calculate_spread, calculate_z_score, determine_thresholds


def get_matrices(prices):
    cointegrated_pairs = find_cointegrated_pairs(prices)  # Find cointegrated pairs
    lag_pairs = find_lag_pairs(prices) #Find lagging and leading pairs
    
    n_assets = prices.shape[0]  # Number of assets
    num_pairs = len(cointegrated_pairs) + len(lag_pairs)  # Number of cointegrated pairs
    
    # Initialize arrays dynamically based on the number of pairs
    #TODO: for a combined matrix of coint and lagpairs - the num_pairs will be the cumulative num of pairs 
    uncertainty_matrix = [np.zeros(num_pairs)] # Initialize array for p-values
    view_matrix = [np.zeros(n_assets)]  # Initialize view matrix
        
    '''
    GETTING TRADING SIGNALS FROM THE COINTEGRATED PAIRS
    '''
    
    view_number = 0
    # Loop through each cointegrated pair
    for pair in cointegrated_pairs:
        # This was not working because it wasn't in the loop rip 
        # view_number = len(view_matrix) - 1

        first_index = pair["first"]
        second_index = pair["second"]
            
        spreads = prices[:, first_index] - prices[:, second_index]  # Calculate the spread between the two assets
        current_spread = spreads[-1]  # Current spread (last value in the spread series)

        upper_threshold = pair["upper_threshold"]
        lower_threshold = pair["lower_threshold"]
            
        if current_spread > upper_threshold:  # If current spread exceeds upper threshold
        
            view_matrix[view_number][first_index] = 1  # Long position on the first asset
            view_matrix[view_number][second_index] = -1  # Short position on the second asset

            uncertainty_matrix[view_number][view_number] = pair["pvalue"]  # Set p-value in the p_values array
            
            uncertainty_matrix = np.vstack([uncertainty_matrix, [np.zeros(num_pairs)]])
            view_matrix = np.vstack([view_matrix, [np.zeros(n_assets)]])
            
            view_number += 1
                
        elif current_spread < lower_threshold:  # If current spread falls below lower threshold
            view_matrix[view_number][first_index] = -1  # Short position on the first asset
            view_matrix[view_number][second_index] = 1  # Long position on the second asset
            uncertainty_matrix[view_number][view_number] = pair["pvalue"]  # Set p-value in the p_values array
            uncertainty_matrix = np.vstack([uncertainty_matrix, [np.zeros(num_pairs)]])
            view_matrix = np.vstack([view_matrix, [np.zeros(n_assets)]])

            view_number += 1
            
            
    '''
    GETTING TRADING SIGNALS FROM THE LAGGING/LEADING PAIRS
    '''
    
    for pair in lag_pairs:
        leading_idx = pair["leading"]
        lagging_idx = pair["lagging"]
        
        stock1 = prices[:, leading_idx]
        stock2 = prices[:, lagging_idx]
        
        spread = calculate_spread(stock1, stock2)
        z_score = calculate_z_score(spread)
        
        entry_threshold, exit_threshold = determine_thresholds(spread)
        
        current_z_score = z_score[-1]
        
        if current_z_score > entry_threshold:
            view_matrix[view_number][lagging_idx] = 1
            # view_matrix[view_number][leading_idx] = -1
            uncertainty_matrix[view_number][view_number] = pair.get("pvalue")
            
            uncertainty_matrix = np.vstack([uncertainty_matrix, [np.zeros(num_pairs, int)]])
            view_matrix = np.vstack([view_matrix, [np.zeros(n_assets, int)]])
            
            view_number += 1
            
        elif current_z_score < -entry_threshold:
            view_matrix[view_number][lagging_idx] = -1
            # view_matrix[view_number][leading_idx] = 1
            uncertainty_matrix[view_number][view_number] = pair.get("pvalue")
            
            uncertainty_matrix = np.vstack([uncertainty_matrix, [np.zeros(num_pairs, int)]])
            view_matrix = np.vstack([view_matrix, [np.zeros(n_assets, int)]])
           
            view_number += 1

    
    #This removes the a last empty row in the matrix
    view_matrix = view_matrix[:-1]
    uncertainty_matrix = uncertainty_matrix[:-1]
    
    #preserving diaagonality of the uncertainty matrix
    non_zero_columns = np.any(uncertainty_matrix != 0, axis=0)
    uncertainty_matrix = uncertainty_matrix[:, non_zero_columns]
    # save_matrices_to_file(uncertainty_matrix, view_matrix)
    
    return uncertainty_matrix, view_matrix