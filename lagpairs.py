#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:24:43 2024

@author: anshdeosingh
"""

import numpy as np
import statsmodels.tsa.stattools as ts
# import matplotlib.pyplot as plt
from itertools import combinations
# from helperFunctions import save_matrices_to_file

def calculate_spread(stock1, stock2):
    # Fit a linear regression model
    slope, intercept = np.polyfit(stock2, stock1, 1)
    
    # Calculate spread
    spread = stock1 - (slope * stock2 + intercept)
    
    return spread

def calculate_z_score(spread):
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    z_score = (spread - mean_spread) / std_spread
    return z_score

def determine_thresholds(spread):
    std_spread = np.std(spread)
    entry_threshold = 2 * std_spread
    exit_threshold = 0.5 * std_spread
    return entry_threshold, exit_threshold

def find_lag_pairs(stocks_data):
    granger_causal_pairs = []
    n_stocks = stocks_data.shape[0]

    for pair in combinations(range(n_stocks), 2):
        idx1, idx2 = pair
        stock1 = stocks_data[:, idx1]
        stock2 = stocks_data[:, idx2]
        
        added12 = False
        added21 = False

        # print(f"({idx1}, {idx2})")
        # 1 lag of stock1 on stock2
        granger_result1 = ts.grangercausalitytests(np.vstack((stock1, stock2)).T, maxlag=1, verbose=False)
        pvalue1 = granger_result1[1][0]['params_ftest'][1]
        direction1 = np.sign(granger_result1[1][0]['params_ftest'][0])  # Get direction of causality
        fstat1 = granger_result1[1][0]['params_ftest'][0]
        dict1 = {"leading": idx1, "lagging": idx2, "direction": direction1, "pvalue": pvalue1, "fstat": fstat1}
        
        if pvalue1 < 0.05:
            granger_causal_pairs.append(dict1)
            added12 = True

        # 1 lag of stock2 on stock1
        granger_result2 = ts.grangercausalitytests(np.vstack((stock2, stock1)).T, maxlag=1, verbose=False)
        pvalue2 = granger_result2[1][0]['params_ftest'][1]
        direction2 = np.sign(granger_result2[1][0]['params_ftest'][0])  # Get direction of causality
        fstat2 = granger_result2[1][0]['params_ftest'][0]
        dict2 = {"leading": idx2, "lagging": idx1, "direction": direction2, "pvalue": pvalue2, "fstat": fstat2}
        
        if pvalue2 < 0.05:
            granger_causal_pairs.append(dict2)
            added21 = True

        # Compare and remove duplicates or conflicting directions
        if added12 and added21:
            if pvalue1 < pvalue2:
                granger_causal_pairs.remove(dict2)
            elif fstat1 > fstat2:
                granger_causal_pairs.remove(dict2)
            else:
                granger_causal_pairs.remove(dict1)

    return granger_causal_pairs

'''
-----------------------------------------------------------------------------------------------
!!!! THIS FUNCTION HAS BEEN COMBINED WITH THE 'get_matrices_coint' function in 'getMatrices.py'
Commenting this out just incase
-----------------------------------------------------------------------------------------------
def get_matrices_lag (prices):
    lag_pairs = find_lag_pairs(prices)
    num_pairs = len(lag_pairs)
    n_assets= prices.shape[0]
    
    uncertainty_matrix = [np.zeros(num_pairs)]# Initialize P matrix with zeros
    view_matrix = [np.zeros(n_assets)]
    
    view_number = 0 # iterator
    
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
            
    #removes the last line full of zeros
    view_matrix = view_matrix[:-1]
    uncertainty_matrix = uncertainty_matrix[:-1]
    
    # save_matrices_to_file(uncertainty_matrix, view_matrix)
            
    return uncertainty_matrix, view_matrix
'''         

    
    