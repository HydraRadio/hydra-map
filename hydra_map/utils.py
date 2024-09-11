
import numpy as np


def chi2_map(freqs, data, amps, inv_noise_var, proj_fn, proj_params, nu_ref=300.):
    """
    Output a chi^2 map.
    """
    Nfreqs, Npix = data.shape
    
    chi2 = np.zeros(Npix)
    for i in range(Npix):

        # Get projection operator
        proj = proj_fn(freqs, nu_ref=nu_ref, params=proj_params[i,:])

        # Calculate chi^2
        chi2[i] = np.sum(inv_noise_var[:,i] * (data_maps[:,i] - proj @ amps[i,:])**2.)
    return chi2