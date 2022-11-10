# Data Analysis for Physics 5C Lab
# Written by Ayushmaan Aggarwal 
# Date Created: 9/13/2022

# Currently implemented features:
# Covariance, Variance, Standard Deviation, Correlation Coefficents, more

import numpy as np
import uncertainties as un

def covariance(x, y):
    assert len(x) == len(y)
    u_x, u_y = np.mean(x), np.mean(y)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i]-u_x)*(y[i]-u_y)

    return sum_covar/(len(x)-1)

def variance(x):
    u_x = np.mean(x)
    sum_covar = 0
    for i in range(len(x)):
        sum_covar += (x[i]-u_x)**2

    return sum_covar/(len(x)-1)

def std(x):
    return (variance(x))**.5

def quartrature_sum(x):
    sum_quart = 0
    for val in x:
        sum_quart += val**2
    return sum_quart**.5

def correlation_coefficients(x,y):
    sigma_xy = covariance(x,y)
    sigma_x = np.sqrt(variance(x))
    sigma_y = np.sqrt(variance(y))
    return sigma_xy/(sigma_x*sigma_y)

def linear_fit_error(x, y, m, c, yerr):
    assert len(x)==len(y)
    N = len(x)
    y_pred = np.array(x)*m + c
    alpha_cu = common_uncertainty(y_pred, y, m, c)
    alpha_cu = alpha_cu if alpha_cu > yerr else yerr
    sigma_x = variance(x)
    
    alpha_m = alpha_cu/(N*sigma_x**2)
    
    alpha_c = np.mean(np.array(x)**2)*alpha_m
    return alpha_m, alpha_c

def common_uncertainty(y_pred, y, m, c):
    assert len(y_pred)==len(y)
    summation = sum((np.array(y) - np.array(y_pred))**2)
    return np.sqrt(summation/(len(y_pred)-2))

def weighted_least_squares_linear(x, y, err):
    """
    Returns: [m, c], [m_err, c_err], [y_pred, res], [chi_squared]
    """
    sum_mult2 = lambda x, y: sum(np.multiply(x, y))
    sum_mult3 = lambda x, y, z: sum(np.multiply(np.multiply(x, y), z))
    
    w = np.divide(1, np.power(err, 2))
    x2 = np.power(x, 2)
    
    delta = sum(w)*sum_mult2(w, x2) - np.power(sum_mult2(w, x), 2)
    m = (sum(w)*sum_mult3(w, x, y) - sum_mult2(w, x)*sum_mult2(w, y))/delta
    c = (sum_mult2(w, y) - m*sum_mult2(w, x))/sum(w)
    
    m_err = np.sqrt(sum(w)/delta)
    c_err = np.sqrt(sum_mult2(w, x2)/delta)
    
    y_pred = np.add(np.multiply(m, x), c)
    res = np.subtract(y_pred, y)
    chi_squared = sum_mult2(w, np.power(res, 2))
    print(f"m = {m:.2}±{m_err:.2}, c = {c:.2}±{c_err:.2}, Χ² = {chi_squared:.2}")
    print(f"Equation: y = ({m:.2}±{m_err:.2})*x + ({c:.2}±{c_err:.2})")
    return [m, c], [m_err, c_err], [y_pred, res], [chi_squared]

def get_uncertain_array(x, error):
    return [un.ufloat(val, error) for val in x]

def seperate_uncertainty_array(x):
    return [val.nominal_value for val in x], [val.std_dev for val in x]
    