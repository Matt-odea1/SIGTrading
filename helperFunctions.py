#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:06:41 2024

@author: anshdeosingh
"""

"""
Contains the functions to handle files - for testing purposes only  

"""
import numpy as np

def replace_zeros_with_space(matrix):
    # Create a string version of the matrix with spaces for zeros and formatted non-zeros
    formatted_matrix = np.where(matrix == 0, "", np.where(matrix != 0, np.char.mod('%.3f', matrix), matrix))
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