import numpy as np
from numpy.polynomial import legendre


def basis_edges_poly(nu, nu_ref=300., params=None):
    """
    Shape (Nfreqs, Nmodes)
    """
    x = nu / nu_ref
    return np.array(
                [ x**-2.5,
                  x**-2.5 * np.log(x),
                  x**-2.5 * np.log(x)**2.,
                  x**-4.5,
                  x**-2.]
        ).T


def basis_powerlaw(nu, nu_ref=300., params=[-2.7,]):
    """
    Shape (Nfreqs, Nmodes)
    """
    x = nu / nu_ref
    beta = params[0]
    return np.array([x**beta,]).T


def basis_powerlaw_curved(nu, nu_ref=300., params=[-2.7, 0.]):
    """
    Shape (Nfreqs, Nmodes).
    """
    x = nu / nu_ref
    beta = params[0]
    c = params[1]
    return np.array([x**(beta + c * np.log(x)),]).T


def basis_poly_legendre(nu, nu_ref=300., params=[5,]):
    """
    Shape (Nfreqs, Nmodes)
    """
    # Generate Legendre functions
    poly_order = params[0]
    lf = legendre(poly_order)
    
    # Normalise freqs. on interval [-1, +1]
    pass


def sky_model(amps, proj_matrix):
    """
    Compute a full sky model given a set of parameters.
    
    Parameters:
        amps (array_like):
            Amplitudes for the basis functions in each pixel. 
            Expected shape: `(Npix, Nmodes)`.
        proj_matrix (array_like):
            Projection operator for each pixel, which goes 
            from parameters to temperature values at each 
            frequency. Shape: `(Nfreqs, Nmodes)`.
    
    Returns:
        sky_model (array_like):
            Array of sky maps at each frequency, of shape `(Nfreqs, Npix)`.
    """
    # Initialise sky model array
    Npix = amps.shape[0]
    Nfreqs = proj_matrix.shape[0]
    sky_map = np.zeros((Nfreqs, Npix))
    
    # Multiply amplitudes by projection operator to get per-freq. maps
    sky_map[:,:] = proj_matrix @ amps.T
    return sky_map