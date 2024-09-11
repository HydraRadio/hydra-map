#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import hydra_map as hmap
from hydra_map.io import load_and_mask_healpix_map
from hydra_map.samplers import gcr_sample_pixel, inversion_sample_beta
import time

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------
nside_out = 32
niter = 10
nu_ref = 300.

proj_fn = hmap.models.basis_powerlaw
Nmodes = 1
beta_initial = -2.7
sigmaS = 100.

root_dir = "../data"

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nworkers = comm.Get_size()

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------


# LWA1 radio maps: https://lambda.gsfc.nasa.gov/product/foreground/fg_lwa1_radio_maps_info.html
lwa1_freqs = [35, 38, 40, 45, 50, 60, 70, 74, 80]
lwa1_maps = [0 for ff in lwa1_freqs]
lwa1_rms = [0 for ff in lwa1_freqs]
lwa1_masks = [0 for ff in lwa1_freqs]

for i, ff in enumerate(lwa1_freqs):

    # Different workers load different datasets
    #if i % nworkers != myid:
    #    continue

    fname = "healpix-all-sky-rav-wsclean-map-%d.fits" % ff
    fname_rms = "healpix-all-sky-rav-wsclean-err-%d.fits" % ff
    print(fname)
    
    _map, _mask = load_and_mask_healpix_map(
                                    fname="%s/%s" % (root_dir, fname), 
                                    nside_out=nside_out, 
                                    mask_val=None, 
                                    smooth_fwhm=5., 
                                    grow_mask=3., 
                                    udgrade_pess=False, 
                                    udgrade_power=None,
                                    rotate=['C', 'G'])
    
    _rms, _mask2 = load_and_mask_healpix_map(
                                    fname="%s/%s" % (root_dir, fname_rms), 
                                    nside_out=nside_out, 
                                    mask_val=None, 
                                    smooth_fwhm=5., 
                                    grow_mask=3., 
                                    udgrade_pess=False, 
                                    udgrade_power=None, # FIXME: Should add in quadrature
                                    rotate=['C', 'G'])
    
    
    lwa1_maps[i] = _map
    lwa1_rms[i] = _rms
    lwa1_masks[i] = _mask
    #hp.mollview(lwa1_maps[i])

# Haslam map (reprocessed)
haslam_408_map, haslam_408_mask = load_and_mask_healpix_map(
                                fname="%s/%s" % (root_dir, "haslam408_ds_Remazeilles2014.fits"), 
                                nside_out=nside_out, 
                                mask_val=None, 
                                smooth_fwhm=5., 
                                grow_mask=3., 
                                udgrade_pess=False, 
                                udgrade_power=None)
haslam_408_rms = 0.8 * np.ones_like(haslam_408_map) # Remazeilles et al. use 800 mK in their fits

comm.barrier()


# LWA1 + HASLAM DATASETS ONLY
data_freqs = np.array(lwa1_freqs + [408.,]) # MHz
data_maps = np.array(lwa1_maps + [haslam_408_map,])

# LWA1 noise estimates
inv_noise_var = []
for i, rms in enumerate(lwa1_rms):
    ivar = 1./rms**2.
    ivar[lwa1_masks[i]] = 0.
    inv_noise_var.append(ivar)

# Haslam noise estimate
ivar = 1./haslam_408_rms**2.
ivar[haslam_408_mask] = 0.
inv_noise_var.append(ivar)

# Combine into array
inv_noise_var = np.array(inv_noise_var)
inv_noise_var.shape, data_maps.shape


#------------------------------------------------------------------------------
# Gibbs sampler loop
#------------------------------------------------------------------------------

# Data shapes
Nfreqs, Npix = data_maps.shape

# Initial guesses
beta_samples = beta_initial * np.ones((Npix, 1))
Sinv = np.eye(Nmodes) / (sigmaS)**2.

# Do Gibbs loop
for n in range(niter):
    print("ITERATION %d" % n)
    
    # (1) AMPLITUDE SAMPLER
    t0 = time.time()
    s = gcr_sample_pixel(freqs=data_freqs,
                           data=data_maps, 
                           proj_fn=proj_fn,
                           proj_params=beta_samples,
                           delta_gains=np.zeros_like(data_freqs),
                           inv_noise_var=inv_noise_var, 
                           Sinv=Sinv, 
                           realisations=1,
                           nu_ref=nu_ref)
    print("\tAmplitude sample took %7.4f sec" % (time.time() - t0))
    print(beta_samples.shape)
    
    # (2) SPECTRAL INDEX SAMPLER
    t0 = time.time()
    beta_samples = inversion_sample_beta(freqs=data_freqs, 
                                         data=data_maps, 
                                         amps=s[0], 
                                         inv_noise_var=inv_noise_var)
    beta_samples = beta_samples[0][:,np.newaxis] # reshape for next iter
    print("\tBeta sample took %7.4f sec" % (time.time() - t0))