import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp

# Set parameters
NINST = 50  # Number of instruments (stocks) in the portfolio
POSLIMIT = 10000  # $10,000 position limit for each stock
COMMRATE = 0.001  # Commission rate for trading
currentPos = np.zeros(NINST)  # Initial position for all stocks (zero positions)

def optimizePortfolio(prcHist, m_bar, S, P, Q, W, transaction_costs, leverage_ratio, d):
    """
    Perform portfolio optimization using the Black-Litterman model.

    Parameters:
    prcHist (ndarray): Historical price data for the stocks.
    m_bar (ndarray): Posterior mean returns (expected returns based on views).
    S (ndarray): Historical covariance matrix of stock returns.
    P (ndarray): Picking matrix (relates views to the assets).
    Q (ndarray): View vector (expected returns based on specific views).
    W (ndarray): Covariance matrix of views.
    transaction_costs (ndarray): Transaction costs per asset.
    leverage_ratio (float): Maximum leverage ratio allowed.
    d (float): Risk aversion parameter.

    Returns:
    ndarray: Optimal portfolio weights for each stock.
    """
    # Compute the inverse of the covariance matrix S
    S_inv = np.linalg.inv(S)  
    
    # Transpose of the picking matrix P
    P_T = P.T  
    
    # Compute the inverse of the covariance matrix of views W
    W_inv = np.linalg.inv(W)  
    
    # Compute the prior mean estimate of returns (historical mean returns)
    m_hat = np.mean(prcHist, axis=1)  
    
    # Define the objective function for optimization
    def objective(x):
        """
        Objective function to minimize. It consists of:
        - Negative expected returns (maximize returns).
        - Risk penalty term (minimize risk based on covariance matrix S).

        Parameters:
        x (ndarray): Portfolio weights.

        Returns:
        float: Value of the objective function.
        """
        return -np.dot(m_bar, x) + 0.5 * d * np.dot(x, np.dot(S, x))

    # Define the constraint: sum of absolute portfolio weights should equal 1 (fully invested)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}]
    
    # Define the leverage constraint: leverage ratio should not exceed the maximum allowed
    leverage_constraint = {'type': 'ineq', 'fun': lambda x: leverage_ratio - np.sum(np.abs(x))}

    # Initial guess for the optimization (starting with zero weights)
    x0 = np.zeros(NINST)
    
    # Bounds for the weights: allow both long and short positions (-1 to 1)
    bounds = [(-1, 1) for _ in range(NINST)]

    # Perform the optimization using the Sequential Least Squares Quadratic Programming (SLSQP) method
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints + [leverage_constraint])

    # Check if the optimization was successful
    if result.success:
        # Return the optimal weights if the optimization succeeded
        optimal_weights = result.x 
        return optimal_weights
    else:
        # Raise an error if the optimization failed
        raise ValueError(f"Optimization failed: {result.message}")

def getMyPosition(prc_so_far):
    """
    Determine the number of units to trade for each stock based on the optimized portfolio.

    Parameters:
    prc_so_far (ndarray): Historical price data for the stocks.

    Returns:
    ndarray: Number of units to trade for each stock.
    """
    # Determine the number of instruments (stocks) based on the input price data
    NINST = len(prc_so_far)  

    # Example inputs for Black-Litterman model
    m_bar = np.zeros(NINST)  # Posterior mean returns (assumed zero for simplicity)
    S = np.cov(prc_so_far)  # Historical covariance matrix of stock returns
    P = np.eye(NINST)  # Picking matrix (identity matrix for simplicity)
    Q = np.zeros(NINST)  # View vector (assumed zero for simplicity)
    W = np.eye(NINST)  # Covariance matrix of views (identity matrix for simplicity)

    # Example transaction costs and constraints
    transaction_costs = np.ones(NINST) * COMMRATE  # Fixed transaction costs per stock
    leverage_ratio = 1.5  # Maximum leverage ratio allowed

    # Example risk aversion parameter
    d = 1.0  # Risk aversion parameter

    # Optimize the portfolio using the Black-Litterman model
    try:
        optimal_weights = optimizePortfolio(prc_so_far, m_bar, S, P, Q, W, transaction_costs, leverage_ratio, d)
        print("Optimal Weights:", optimal_weights)

        # Calculate the number of units to trade based on optimal weights and current prices
        current_prices = prc_so_far[:, -1]  # Assuming the last column represents the current prices
        num_units = optimal_weights * POSLIMIT / current_prices  # Convert weights to number of units
        print("Number of Units to Trade:", num_units)

        return num_units  # Return the number of units to trade

    except ValueError as e:
        # Handle optimization errors and return zeros (no trading)
        print(e)
        return np.zeros(NINST)  # Return zeros or handle the error case appropriately