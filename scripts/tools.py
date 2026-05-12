import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA


def RV_jup(t, A_0, A_1, P, phi, C):
    """
    Model for the radial velocity of the Sun due to Jupiter's
    gravity as a sinusoid.

    Parameters
    ----------
        t: numpy.ndarray
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
        matrix: numpy.ndarray
            Any 2D array.

    Returns
    -------
        X: numpy.ndarray
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
        wl: numpy.ndarray
            Array of size N containing the wavelengths.
        rv_list: numpy.ndarray
            Array of size M containing RV measures. Units must be m/s.
    
    Returns
    -------
        numpy.ndarray
            2D array of M rows and N columns.
            Each row contains a list of corrected wavelengths for a certain RV value.
    """
    lambda_corr  = []
    for rv in rv_list:
        lambda_corr.append(wl * (1 - rv / 3e8))  # corrected wavelength, same units as wl
    return np.array(lambda_corr)


def calculate_rv(wl, S_0, S_k, C_0):
    """
    Template matching. Calculates the radial velocity of a certain spectrum with respect to a reference spectrum.
    Both spectra are associated to the same wavelength array.

    Parameters
    ----------
        wl: numpy.ndarray
            Array containing the wavelengths.
        S_0: numpy.ndarray
            Array containing the reference spectrum.
        S_k: numpy.ndarray
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


def calculate_pca_significance(N_r, N_c, spectra, sigma, bar=True):
    """
    Calculates the significance of the principal components.

    Parameters
    ----------
        N_r: int
            Number of realizations.
        N_c: int
            Number of principal components.
        spectra: numpy.ndarray
            2D array where each row is a spectrum.
        bar: bool
            True to show progress bar.

    Returns
    -------
        stats: numpy.ndarray
            Array of tuples. The first element of each tuple is the average maximum value
            of the dot product between the components obtained from the original matrix and the noisy ones.
            The second element of each tuple is the standard deviation
            of the dot product between the components obtained from the original matrix and the noisy ones.
    """
    pca_spec = PCA(n_components=N_c)  # create PCA object
    pca_spec.fit_transform(spectra)  # apply PCA to the original matrix
    comps = pca_spec.components_.astype(np.float32)

    simi_all = np.empty((N_r, N_c, N_c), dtype=np.float32)  # N_r squared matrices of N_c x N_c

    if bar:
        for i in tqdm(range(N_r)):  # iterate over realizations
            spectra_noisy = spectra + np.random.normal(0, sigma, spectra.shape)  # add gaussian noise to the original data

            pca_noisy = PCA(n_components=N_c, svd_solver="randomized")  # apply PCA to the noisy matrix
            pca_noisy.fit(spectra_noisy)

            comps_noisy = pca_noisy.components_  # save the loadings

            simi_all[i] = comps @ comps_noisy.T  # save dot product between original and noisy loadings
    else:
        for i in range(N_r):  # iterate over realizations
            spectra_noisy = spectra + np.random.normal(0, sigma, spectra.shape)  # add gaussian noise to the original data

            pca_noisy = PCA(n_components=N_c, svd_solver="randomized")  # apply PCA to the noisy matrix
            pca_noisy.fit(spectra_noisy)

            comps_noisy = pca_noisy.components_  # save the loadings

            simi_all[i] = comps @ comps_noisy.T  # save dot product between original and noisy loadings
    
    maxes = np.array([np.amax(s, axis=1) for s in simi_all])  # max of each row of each realization
    stats = np.array([(np.average(s), np.std(s)) for s in maxes.T])  # list of max mean and std for each PC: [(max_avg_1, max_std_1), (max_avg_2, max_std_2), ..., (max_avg_n, max_std_n)]
    return stats


def harvey(params, nu):
    """
    Calculates the Harvey function (https://ui.adsabs.harvard.edu/abs/1985ESASP.235..199H/abstract).

    Parameters
    ----------
        params: iterable
            Parameters of the function. These are:
                a: float
                    Total energy.
                b: float
                    Turnover frequency.
                c: float
                    Slope of the power law.
                d: float
                    White noise.
        nu: float or numpy.ndarray
            Frequency.
    Returns
    -------
        float or numpy.ndarray
            Harvey function for the given values.
    """
    a, b, c, d = params
    f = np.pi / c / np.sin(np.pi / c)  # normalization factor
    return a / (f * b) / (1 + (nu / b)**c) + d


def harvey_unpacked(nu, a, b, c, d):
    """
    Calculates the Harvey function (https://ui.adsabs.harvard.edu/abs/1985ESASP.235..199H/abstract).

    Parameters
    ----------
        nu: float or numpy.ndarray
            Frequency.
        a: float
            Total energy.
        b: float
            Turnover frequency.
        c: float
            Slope of the power law.
        d: float
            White noise.

    Returns
    -------
        float or numpy.ndarray
            Harvey function for the given values.
    """
    f = np.pi / c / np.sin(np.pi / c)  # normalization factor
    return a / (f * b) / (1 + (nu / b)**c) + d


def calculate_CCF(S_i, S_0):
    """
    Calculates the cross-correlation function between a measured and a reference spectrum.

    Parameters
    ----------
        S_i: numpy.ndarray
            Measured spectrum.
        S_0: numpy.ndarray
            Reference spectrum.
    
    Returns
    -------
        float:
            Result of the cross-correlation.
    """
    return sum(S_i * S_0)


def gaussian(x, A, mu, sigma, C):
    """
    Gaussian function with independent amplitude and position on the y-axis.
    
    Parameters
    ----------
        x: float or numpy.ndarray
            Input data.
        A: float
            Height of the curve's peak (amplitude).
        mu: float
            Position of the center of the peak.
        sigma: float
            Standard deviation.
        C: float
            Position coefficient.
    
    Returns
    -------
        float or numpy.ndarray
            Gaussian function.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C


def calculate_lambda_vr(wl, v, c=3e8):
    """
    Calculates Doppler-shifted wavelength as explained in Pepe et al. (2002).
    https://ui.adsabs.harvard.edu/abs/2002A%26A...388..632P/abstract

    Parameters
    ----------
        wl: float or numpy.ndarray
            Wavelength.
        v: float or numpy.ndarray
            Velocity. Must have the same shape as wl and the same units as c.
        c: float
            Speed of light. 300000 m/s by default.
    
    Returns
    -------
        float or numpy.ndarray
            Doppler-shifted wavelength, same units as wl.
    """

    return wl * np.sqrt((1 - v / c) / (1 + v / c))


def cross_correlate(S_j, shifted_mask):
    """
    Calculates the Cross-Correlation Function between a spectrum and
    a series of shifted spectra.

    Parameters
    ----------
        S_j: numpy.ndarray
            1-dimensional array that represents an observed spectrum.
        shifted_mask: numpy.ndarray
            1-dimensional that represents the reference spectrum (mask),
            shifted due to the effect of a certain value of radial velocity.
    
    Returns:
    --------
        float
            The cross correlation function between the observed spectrum and the shifted mask.

    """

    return S_j * shifted_mask / (np.linalg.norm(S_j) * np.linalg.norm(shifted_mask))


def calculate_periodogram(x, y):
    """
    Calculates the periodogram of the function y(x).

    Parameters
    ----------
        x: np.ndarray
            Independent variable. Needs to be regularly spaced.
        y: np.ndarray
            Dependent variable (function of x). Needs to be regularly spaced.
    
    Returns
    -------
        ft: np.ndarray
            Fast Fourier Transform of y.
        ps: np.ndarray
            Normalized power spectrum of y.
        freqs: np.ndarray
            Corresponding frequencies to the Fourier Transform.
    """
    N = len(y)
    dx = x[1] - x[0]
    
    ft = np.fft.fft(y)
    ps = np.abs(ft)**2 * dx / N
    freqs = np.fft.fftfreq(N, dx)

    return ft, ps, freqs


def subtract_v(wl, spectra_matrix, c=3e8):
    """
    Take out the projection over the velocity vector from a matrix of spectra,
    assuming that each spectrum is the same as the mean spectrum centered around
    a different wavelength.
    Detailed explanation in the readme: https://github.com/Etienne99/LAM_Internship/blob/master/README.md.

    Parameters
    ----------
        wl: numpy.ndarray
            1-dimensional array containing the wavelength values.
        spectra_matrix: numpy.ndarray
            2-dimensional array where each row is a spectrum.
        c: float
            The speed of light in vacuum. Default is 3e8 m/s.
    
    Returns
    -------
        S_f: numpy.ndarray
            2-dimensional array, where each row is a spectrum after having subtracted
            the projection over the velocity vector.
        D: numpy.ndarray
            Product between the derivative of the mean spectrum and the wavelength array, divided by c.
            Has units of velocity^-1.
    """
    mean_spec = np.mean(spectra_matrix, axis=0)  # calculate mean spectrum (i.e. the mean flux for each wl)
    d_S0      = np.gradient(mean_spec, wl)  # derivative of the mean spectrum

    S_t       = np.array([i - mean_spec for i in spectra_matrix])  # subtract the mean spectrum from each row of the spectra matrix
    D         = d_S0 * wl / c
    S_f       = np.array([(i - np.dot(D, i)) * D / np.linalg.norm(D)**2 for i in S_t])  # take out projection over the velocity vector
    return S_f, D


def move_sn_y(offs=0, dig=0, side='left', omit_last=False):
    """
    Move scientific notation exponent from top to the side.
    Additionally, one can set the number of digits after the comma
    for the y-ticks, hence if it should state 1, 1.0, 1.00 and so forth.
    Source: https://werthmuller.org/blog/2014/move-scientific-notation/

    Parameters
    ----------
        offs : float, optional; <0>
            Horizontal movement additional to default.
        dig : int, optional; <0>
            Number of decimals after the comma.
        side : string, optional; {<'left'>, 'right'}
            To choose the side of the y-axis notation.
        omit_last : bool, optional; <False>
            If True, the top y-axis-label is omitted.

    Returns
    -------
        locs : list
            List of y-tick locations.

    Note
    ----
        This is kind of a non-satisfying hack, which should be handled more
        properly. But it works. Functions to look at for a better implementation:
        ax.ticklabel_format
        ax.yaxis.major.formatter.set_offset_string
    """

    # Get the ticks
    locs, _ = plt.yticks()

    # Put the last entry into a string, ensuring it is in scientific notation
    # E.g: 123456789 => '1.235e+08'
    llocs = '%.3e' % locs[-1]

    # Get the magnitude, hence the number after the 'e'
    # E.g: '1.235e+08' => 8
    yoff = int(str(llocs).split('e')[1])

    # If omit_last, remove last entry
    if omit_last:
        slocs = locs[:-1]
    else:
        slocs = locs

    # Set ticks to the requested precision
    form = r'$%.'+str(dig)+'f$'
    plt.yticks(locs, list(map(lambda x: form % x, slocs/(10**yoff))))

    # Define offset depending on the side
    if side == 'left':
        offs = -.18 - offs # Default left: -0.18
    elif side == 'right':
        offs = 1 + offs    # Default right: 1.0
        
    # Plot the exponent
    plt.text(offs, .98, r'$\times10^{%i}$' % yoff, transform =
            plt.gca().transAxes, verticalalignment='top')

    # Return the locs
    return locs