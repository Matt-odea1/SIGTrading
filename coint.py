import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint

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
    upper_threshold = mu + 2 * sigma
    lower_threshold = mu - 2 * sigma
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
            print(f"P-value for pair ({idx1}, {idx2}): {p_value:.2f}")  # Print p_value rounded to 2 decimal places

    return(cointegrated_pairs)


def get_matrices(prices):
    """
    Create P and view matrices for Black-Litterman model based on cointegrated pairs.

    Parameters:
    prices (ndarray): Array containing stock price data.

    Returns:
    None
    """
    cointegrated_pairs = find_cointegrated_pairs(prices)  # Find cointegrated pairs
    
    n_assets = prices.shape[0]  # Number of assets
    num_pairs = len(cointegrated_pairs)  # Number of cointegrated pairs
    
    # Initialize arrays dynamically based on the number of pairs
    #TODO: for a combined matrix of coint and lagpairs - the num_pairs will be the cumulative num of pairs 
    uncertainty_matrix = [np.zeros(num_pairs)] # Initialize array for p-values
    view_matrix = [np.zeros(n_assets)]  # Initialize view matrix
        
    # Loop through each cointegrated pair
    for pair in cointegrated_pairs:
        # This was not working because it wasn't in the loop rip 
        view_number = len(view_matrix) - 1

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
            #print("Signal: Long on", first_index, "and Short on", second_index)
            #print(view_matrix[curr_row])
                
        elif current_spread < lower_threshold:  # If current spread falls below lower threshold
            view_matrix[view_number][first_index] = -1  # Short position on the first asset
            view_matrix[view_number][second_index] = 1  # Long position on the second asset
            uncertainty_matrix[view_number][view_number] = pair["pvalue"]  # Set p-value in the p_values array
            uncertainty_matrix = np.vstack([uncertainty_matrix, [np.zeros(num_pairs)]])
            view_matrix = np.vstack([view_matrix, [np.zeros(n_assets)]])
            #print("Signal: Short on", first_index, "and Long on", second_index)
            #print(view_matrix[curr_row])

    #This removes the a last empty row in the matrix
    view_matrix = view_matrix[:-1]
    uncertainty_matrix = uncertainty_matrix[:-1]
    save_matrices_to_file(uncertainty_matrix, view_matrix)

def replace_zeros_with_space(matrix):
    # Create a string version of the matrix with spaces for zeros and formatted non-zeros
    formatted_matrix = np.where(matrix == 0, " ", np.where(matrix != 0, np.char.mod('%.3f', matrix), matrix))
    return formatted_matrix

def save_matrices_to_file(uncertainty_matrix, view_matrix, output_file='out.txt'):
    # Replace zeros with space in the matrices and format non-zeros
    uncertainty_matrix = replace_zeros_with_space(uncertainty_matrix)
    view_matrix = replace_zeros_with_space(view_matrix)

    with open(output_file, 'w') as file:
        file.write("Uncertainty Matrix:\n")
        for row in uncertainty_matrix:
            file.write('[' + ' '.join(row) + ']\n')
        
        file.write("\n\nView Matrix:\n")
        for row in view_matrix:
            file.write('[' + ' '.join(row) + ']\n')
