import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

def RV_jup(t, A_0, A_1, P, phi, C):
    """
    Model for the radial velocity of the Sun due to Jupiter's
    gravity as a sinusoid.

    Parameters
    ----------
        t: numpy.array
            Time series.
        A_0: float
            Initial amplitude.
        A_1: float
            Amplitude drift.
        P: float
            Period of the sinusoid.
        phi: float
            Phase offset.
        C: float
            Vertical shift.

    Returns
    -------
        np.array
            Sinusoidal function.
    """
    return (A_0 + A_1 * t) * np.sin(2 * np.pi * t / P + phi) + C

def double_centering(matrix):
    """ 
    Applies double centering to a given matrix.
    
    Parameters
    ----------
        matrix: numpy.array
            Any 2D array.

    Returns
    -------
        X: numpy.array
            Result of double centering of the original matrix.
    """
    matrix_centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    matrix_centered = matrix_centered - np.mean(matrix_centered, axis=0, keepdims=True)
    return matrix_centered

def extract_pattern(sequence, pattern):
    """
    Finds a certain pattern inside a string.
    Used to extract time string from a HARPS file name.

    Parameters
    ----------
        sequence: str
            Any string.
        pattern: re.Pattern
            Pattern to search inside the string.

    Returns
    -------
        str
            String inside sequence that matches the pattern.
    """
    match = pattern.search(sequence)
    return match.group(0) if match else None

def calculate_lambda_corr(wl, rv_list):
    """
    Calculates the corrected wavelength for a measured wavelength and radial velocity.
    
    Parameters
    ----------
        wl: np.array
            Array of size N containing the wavelengths.
        rv_list: array
            Array of size M containing RV measures. Units must be m/s.
    
    Returns
    -------
        np.array
            2D array of M rows and N columns.
            Each row contains a list of corrected wavelengths for a certain RV value.
    """
    lambda_corr  = []
    for rv in rv_list:
        lambda_corr.append(wl * (1 - rv / 3e8))  # corrected wavelength, same units as wl
    return np.array(lambda_corr)

def calculate_rv(wl, S_0, S_k, C_0):
    """
    Calculates the radial velocity of a certain spectrum with respect to a reference spectrum.
    Both spectra are associated to the same wavelength array.

    Parameters
    ----------
        wl: np.array
            Array containing the wavelengths.
        S_0: np.array
            Array containing the reference spectrum.
        S_k: np.array
            Array containing the observed spectrum.
        C_0: float
            Continuum level constant.
    
    Returns
    -------
        float
            Radial velocity of the observed spectrum.
    """

    c = 3e8  # m/s
    dS_0 = np.gradient(S_0, wl)
    num = sum((S_k - S_0) * wl * dS_0 / (S_0 + C_0))
    den = sum(wl**2 * dS_0**2 / (S_0 + C_0))
    return c * num / den