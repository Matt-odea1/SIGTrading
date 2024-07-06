#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:51:15 2024

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

# Load data from Excel sheet, ensuring the first row is the header
excel_file = 'stocksdata.xlsx'
df = pd.read_excel(excel_file, sheet_name='Sheet1')

def calculate_thresholds(data1, data2):
    spread = data1 - data2
    # Calculate spreads (difference between prices of data1 and data2)
    mu = np.mean(spread)
    sigma = np.std(spread)
    upper_threshold = mu + 2 * sigma
    lower_threshold = mu - 2 * sigma
    
    # Calculate dynamic thresholds - EDIT
    exit_threshold = mean_spread  # Example: Exit when spread reverts to mean
    
    return upper_threshold, lower_threshold, exit_threshold

def find_cointegrated_pairs(df, stock_names):
    cointegrated_pairs = []
    for pair in combinations(stock_names, 2): 
        stock1 = pair[0]
        stock2 = pair[1]
    
        data1 = np.array(df[stock1])
        data2 = np.array(df[stock2])
    
        result = ts.coint(data1, data2)
    
        p_value = result[1]
        if (p_value < 0.05):
            upper_threshold, lower_threshold, exit_threshold = calculate_thresholds(data1, data2)
            cointegrated_pairs.append({"first": stock1, "second": stock2, "pvalue": p_value, "upper threshold": upper_threshold, "lower_threshold": lower_threshold,  "exit threshold": exit_threshold})
            
        return(cointegrated_pairs)

def get_matrices(df):
    stock_names = list(df.columns)[1::]
    cointegrated_pairs = find_cointegrated_pairs(df, stock_names)
    lag_pairs = find_lag_pairs(df, stock_names)
    
    p_matrix = [np.zeros(50, int)]
    view_matrix = [np.zeros(50, int)]

    #COMPARE TO PREDICTIONS
    for i in cointegrated_pairs:
        iteration = np.size(view_matrix)/50 - 1
        first = df[i.get("first")]
        second = df[i.get("second")]
        
        first_index = int(i.get("first").split(" ")[1])
        second_index = int(i.get("second").split(" ")[1])
        
        spreads = np.array(first) - np.array(second)
        current_spread = spreads[-1]

        upper_threshold = i.get("upper threshold")
        lower_threshold = i.get("lower threshold")
        
        if current_spread > upper_threshold:
            view_matrix[iteration][first_index] = 1
            view_matrix[iteration][second_index] = -1
            p_matrix[iteration][np.size(p_matrix)/50 - 1] = i.get("pvalue")
            
        elif current_spread < lower_threshold:
            view_matrix[iteration][first_index] = -1
            view_matrix[iteration][second_index] = 1
            p_matrix[iteration][np.size(p_matrix)/50 - 1] = i.get("pvalue")