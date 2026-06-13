#!/usr/bin/env python3
"""Module to standardize data"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn.

    Args:
        X (numpy.ndarray): The tabular data to be standardized.

    Returns:
        numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
