import numpy as np

from scipy.linalg import sqrtm, solve
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats import invwishart
from .models import basis_powerlaw

from mpi4py.MPI import SUM as MPI_SUM
import healpy as hp

def inversion_sample_beta(freqs, data, amps, inv_noise_var, beta_range=(-3.2, -2.2),
                          nu_ref=300., grid_points=400, interp_kind='linear', 
                          realisations=1, comm=None):
    """
    Use inversion sampling to draw samples of the power law beta parameter.

    Parameters:
        freqs (array_like):
            xxx
        data (array_like):
            xxx
        amps (array_like):
            xxx
        inv_noise_var (array_like):
            xxx
        beta_range (tuple of float):
            xxx
        nu_ref (float):
            xxx
        grid_points (int):
            xxx
        interp_kind (str):
            xxx
        realisations (int):
            xxx
        comm (MPI Communicator):
            xxx

    Returns:
        beta_samples (array_like):
            xxx
    """
    # Set up MPI if enabled
    myid = 0
    nworkers = 1
    if comm is not None:
        myid = comm.Get_rank()
        nworkers = comm.Get_size()
        
    # Calculate likelihood values as a function of beta
    beta_min, beta_max = beta_range
    beta_vals = np.linspace(beta_min, beta_max, grid_points)
    
    # Empty results array
    Npix = inv_noise_var.shape[1]
    beta_samples = np.zeros((realisations, Npix))
    
    def loglike(beta, amps_p, inv_noise_var_p, data_p):
        proj = basis_powerlaw(freqs, nu_ref=nu_ref, params=[beta,])
        model_p = proj @ amps_p
        return -0.5 * np.sum(inv_noise_var_p * (data_p - model_p)**2.)
    
    # Loop over pixels
    for p in range(Npix):

        if p % nworkers != myid:
            continue

        if p % 5000 == 0:
            print("Pixel %d / %d" % (p, Npix))
    
        # Data and model values in this pixel
        amps_p = amps[p,:]
        inv_noise_var_p = inv_noise_var[:,p]
        data_p = data[:,p]
        
        # Calculate log likelihood vs beta
        logL = np.array([loglike(beta, amps_p, inv_noise_var_p, data_p) 
                         for beta in beta_vals])
        
        # Remove common factor since it will be normalised away anyway
        logL0 = np.max(logL)
        likefn = np.exp(logL - logL0)

        # Calculate CDF
        cdf = cumulative_trapezoid(likefn, beta_vals, initial=0.)
        cdf /= cdf[-1] # normalise to interval [0, 1]
    
        # Build interpolator
        interp_cdf = interp1d(cdf, beta_vals, kind=interp_kind)
    
        # Draw uniform random sample(s) and map to distribution for beta using CDF
        uvals = np.random.uniform(size=realisations)
        beta_samples[:,p] = np.array([interp_cdf(u) for u in uvals])
    
    # Do Allreduce to share beta samples across all workers
    if comm is not None:
        beta_samples_all = np.zeros_like(beta_samples.flatten())
        comm.Allreduce(beta_samples.flatten(), beta_samples_all, op=MPI_SUM)
        beta_samples = beta_samples_all.reshape(beta_samples.shape)

    return beta_samples


def invwishart_sample_covmat(amps):
    """
    Draw samples of covariance matrix for a set of samples, e.g. of spectral 
    basis function coefficients. This draws from an inverse Wishart distribution.
    
    Parameters:
        amps (array_like):
            Amplitudes for the spectral basis functions in each pixel. 
            Expected shape: `(Npix, Nmodes)`.
    
    Returns:
        cov (array_like):
            Sample of the covariance matrix of the foreground coefficients.
            Shape: `(Nmodes, Nmodes)`.
    """
    Npix, Nmodes = amps.shape

    p = Nmodes # Dimension of the scale matrix
    nu = Npix - p - 1 # Degrees of freedom, must be greater than or equal to dimension of the scale matrix
    assert nu >= p, "Degrees of freedom must be greater than or equal to the dimension of the scale matrix."

    # Compute the scale matrix (uncentred, unnormalised covariance)
    Psi = np.zeros((p, p))
    for i in range(Npix):
        Psi += np.outer(amps[i, :], amps[i, :])

    # Draw a sample from the inverse Wishart distribution
    cov_sample = invwishart.rvs(df=nu, scale=Psi)
    return cov_sample


def gcr_sample_pixel(freqs, data, proj_fn, proj_params, delta_gains, inv_noise_var, 
                       Sinv, realisations=1, nu_ref=300., comm=None):
    """
    Solve Gaussian Constrained Realisation system for spectral parameters 
    in each pixel.
    
    Parameters:
        freqs (array_like):
            Frequencies that the data maps are evaluated at.
        data (array_like):
            Healpix maps for each frequency band, of shape `(Nfreqs, Npix)`.
        proj_fn (func):
            Function that returns a projection operator for each pixel.
        proj_params (array_like):
            Array of (non-coefficient) parameters to pass to `proj_fn`. 
            Has shape `(Npix, Nparams)`.
        delta_gains (array_like):
            Fractional gain fluctuation for each band. Has shape `(Nfreqs,)`.
        inv_noise_var (array_like):
            The inverse of the noise variance per frequency channel, per 
            pixel. Expected shape: `(Nfreqs, Npix)`.
        Sinv (array_like):
            Inverse of the prior covariance matrix for the spectral 
            parameters, assumed to be the same for all pixels.
        realisations (int):
            Number of realisations of the GCR solution to return.
        nu_ref (float):
            Reference frequency, in MHz.
        comm (MPI communicator):
            MPI Communicator object.
    
    Returns:
        g (array_like):
            Array of sampled per-frequency gain values. 
            Shape: `(realisations, Nfreqs)`.
    """
    # Set up MPI if enabled
    myid = 0
    nworkers = 1
    if comm is not None:
        myid = comm.Get_rank()
        nworkers = comm.Get_size()
    
    # Get matrix shapes
    proj = proj_fn(freqs, nu_ref=nu_ref, params=proj_params[0,:]) # test run: first pixel only
    Nfreqs, Nmodes = proj.shape
    Npix = inv_noise_var.shape[1]
    
    # Empty results matrix
    s = np.zeros((realisations, Npix, Nmodes))
    
    # Loop over pixels
    for p in range(Npix):
        
        # Basic status report
        if p % 5000 == 0:
            print("%6d / %6d" % (p, Npix))
        
        # MPI worker assignment
        if p % nworkers != myid:
            continue
        
        # Update projection matrix
        proj = proj_fn(freqs, params=proj_params[p,:])
        
        # LHS matrix: (S^-1 + A^T N^-1 A)
        # proj_matrix has shape (Nfreqs, Nmodes)
        # Sinv has shape (Nmodes, Nmodes)
        # Ninv has shape (Nfreqs, Nfreqs)
        Ninv = np.diag(inv_noise_var[:,p])
        lhs_op = Sinv + proj.T @ Ninv @ proj

        for i in range(realisations):
            # Draw unit Gaussian random numbers
            omega_n = np.random.randn(Nfreqs)
            omega_s = np.random.randn(Nmodes)

            # RHS vector
            # data has shape (Nfreqs,)
            rhs = proj.T @ Ninv @ data[:,p] \
                + proj.T @ sqrtm(Ninv) @ omega_n \
                + sqrtm(Sinv) @ omega_s

            # Do linear solve for symmetric matrix
            s[i,p,:] = solve(lhs_op, rhs, assume_a='sym')
    
    # FIXME: Need to do MPI communication
    total_s = np.zeros_like(s.flatten())
    if comm is not None:
        comm.Allreduce(s.flatten(), total_s, op=MPI_SUM)
        total_s = total_s.reshape(s.shape)
    else:
        total_s = s
    return total_s


def gcr_sample_gain(data, amps, proj_matrix, inv_noise_var, Sinv, realisations=1):
    """
    Solve Gaussian Constrained Realisation system for gain parameters 
    that are common to all pixels at a given frequency (i.e. an overall 
    multiplicative factor per band).
    
    Note that this solves for the gain fractional perturbations, delta g, 
    where we have defined gain = gbar (1 + delta g), and gbar = 1.
    
    Parameters:
        data (array_like):
            Healpix maps for each frequency band, of shape `(Nfreqs, Npix)`.
        amps (array_like):
            Amplitudes for the spectral basis functions in each pixel. 
            Expected shape: `(Npix, Nmodes)`.
        proj_matrix (array_like):
            Projection operator to go from parameters to a temperature 
            at each frequency. This operator is assumed to be the same 
            for each pixel.
        inv_noise_var (array_like):
            The inverse of the noise variance per frequency channel, per 
            pixel. Expected shape: `(Nfreqs, Npix)`.
        Sinv (array_like):
            Inverse of the prior covariance matrix for the per-frequency 
            channel gains.
        realisations (int):
            Number of realisations of the GCR solution to return.
    
    Returns:
        g (array_like):
            Array of sampled per-frequency gain values. 
            Shape: `(realisations, Nfreqs)`.
    """
    Nfreqs, Npix = data.shape
    
    # Get sky model
    model = sky_model(amps, proj_matrix) # (Nfreqs, Npix)
    
    # Construct LHS matrix. The gains are per-band and the noise covariance 
    # is assumed to be diagonal, so the A^T N^-1 A term is diagonal in frequency
    lhs_op = Sinv + np.diag( np.sum(inv_noise_var * model**2., axis=1) )
    
    # Empty results matrix
    g = np.zeros((realisations, Nfreqs))
    
    # Loop over requested realisations
    for i in range(realisations):
        # Draw unit Gaussian random numbers
        omega_n = np.random.randn(*model.shape) # same size as data!
        omega_g = np.random.randn(Nfreqs)

        # RHS vector
        # This works with the residual, as we model the data as:
        # data = gbar (1 + delta g) sky_model and gbar = 1
        # So: resid = data - sky_model = delta_g sky_model
        rhs = np.sum(
                     model * (inv_noise_var * (data - model)
                              + np.sqrt(inv_noise_var) * omega_n),
                     axis=1
                    ) \
            + sqrtm(Sinv) @ omega_g

        # Do linear solve for symmetric matrix
        g[i] = solve(lhs_op, rhs, assume_a='sym')
        
    return g