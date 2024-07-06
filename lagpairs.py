#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:24:43 2024

@author: anshdeosingh
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from itertools import combinations
import random
from statsmodels.tsa.stattools import adfuller

def calculate_spread(stock1, stock2):
    # Fit a linear regression model
    slope, intercept = polyfit(stock2, stock1, 1)
    
    # Calculate spread
    spread = stock1 - (slope * stock2 + intercept)
    
    return spread

def calculate_z_score(spread):
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    z_score = (spread - mean_spread) / std_spread
    return z_score

def determine_thresholds(spread):
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    entry_threshold = 2 * std_spread
    exit_threshold = 0.5 * std_spread
    return entry_threshold, exit_threshold

def find_lag_pairs(stocks_data):
    granger_causal_pairs = []
    n_stocks = stocks_data.shape[1]

    for pair in combinations(range(n_stocks), 2):
        idx1, idx2 = pair
        stock1 = stocks_data[:, idx1]
        stock2 = stocks_data[:, idx2]

        # 1 lag of stock1 on stock2
        granger_result1 = ts.grangercausalitytests(np.vstack((stock1, stock2)).T, maxlag=1, verbose=False)
        pvalue1 = granger_result1[1][0]['params_ftest'][1]
        direction1 = np.sign(granger_result1[1][0]['params_ftest'][0])  # Get direction of causality
        fstat1 = granger_result1[1][0]['params_ftest'][0]
        dict1 = {"leading": idx1, "lagging": idx2, "direction": direction1, "pvalue": pvalue1, "fstat": fstat1}
        
        if pvalue1 < 0.05:
            granger_causal_pairs.append(dict1)

        # 1 lag of stock2 on stock1
        granger_result2 = ts.grangercausalitytests(np.vstack((stock2, stock1)).T, maxlag=1, verbose=False)
        pvalue2 = granger_result2[1][0]['params_ftest'][1]
        direction2 = np.sign(granger_result2[1][0]['params_ftest'][0])  # Get direction of causality
        fstat2 = granger_result2[1][0]['params_ftest'][0]
        dict2 = {"leading": idx2, "lagging": idx1, "direction": direction2, "pvalue": pvalue2, "fstat": fstat2}
        
        if pvalue2 < 0.05:
            granger_causal_pairs.append(dict2)

        # Compare and remove duplicates or conflicting directions
        if granger_result1 and granger_result2:
            if pvalue1 < pvalue2:
                granger_causal_pairs.remove(dict2)
            elif fstat1 > fstat2:
                granger_causal_pairs.remove(dict2)
            else:
                granger_causal_pairs.remove(dict1)

    return granger_causal_pairs

def get_matrices(stocks_data):
    lag_pairs = find_lag_pairs(stocks_data)
    lag_pairs_count = len(lag_pairs)
    p_matrix = np.zeros((lag_pairs_count, lag_pairs_count))# Initialize P matrix with zeros
    view_matrix = [np.zeros(50)]
    
    lag_pairs = find_lag_pairs(stocks_data)
    
    curr_view_row = view_matrix.shape[0] - 1
    curr_p_row = p_matrix.shape[0] - 1    
    
    for idx, pair in lag_pairs:
        leading_idx = pair["leading"]
        lagging_idx = pair["lagging"]
        
        stock1 = stocks_data[:, leading_idx]
        stock2 = stocks_data[:, lagging_idx]
        
        spread = calculate_spread(stock1, stock2)
        z_score = calculate_z_score(spread)
        
        entry_threshold, exit_threshold = determine_thresholds(spread)
        
        current_spread = spread[-1]
        
        if current_spread > entry_threshold:
            p_matrix = np.vstack([p_matrix, np.zeros(p_matrix, int)])
            view_matrix = np.vstack([view_matrix, np.zeros(50, int)])
            view_matrix[curr_view_row][lagging_idx] = 1
            #view_matrix[iteration][leading_idx] = -1
            p_matrix[curr_p_row][curr_p_row] = pair.get("pvalue")
            curr_view_row += 1
            curr_p_row += 1
        elif current_spread < -entry_threshold:
            p_matrix = np.vstack([p_matrix, np.zeros(p_matrix, int)])
            view_matrix = np.vstack([view_matrix, np.zeros(50, int)])
            view_matrix[curr_view_row][lagging_idx] = -1
            #view_matrix[iteration][leading_idx] = 1
            p_matrix[curr_p_row][curr_p_row] = pair.get("pvalue")
           
            curr_view_row += 1
            curr_p_row += 1
    return p_matrix, view_matrix
            

    
    