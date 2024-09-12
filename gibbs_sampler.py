#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import hydra_map as hmap
from hydra_map.io import load_and_mask_healpix_map
from hydra_map.samplers import gcr_sample_pixel, inversion_sample_beta
import healpy as hp
import pylab as plt
import time, os

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
out_dir = "../output"
prefix = "lwa1_haslam_nside32"
data_path = "%s.npz" % prefix

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nworkers = comm.Get_size()

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# Load data from file produced by prepare_data.py
if myid == 0:
    print("Loading data from", os.path.join(root_dir, data_path))
t0 = time.time()

data_file = np.load(os.path.join(root_dir, data_path))
data_freqs = data_file['data_freqs']
data_maps = data_file['data_maps']
inv_noise_var = data_file['inv_noise_var']

if myid == 0:
    print("-"*50)
    print("DATA SUMMARY")
    print("    Freqs:", data_freqs)
    print("    Nside:", hp.npix2nside(data_maps.shape[1]))
    print("-"*50)

if myid == 0:
    print("Loaded data in %5.3f sec" % (time.time() - t0))

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
                           nu_ref=nu_ref,
                           comm=comm)
    if myid == 0:
        print("\tAmplitude sample took %7.4f sec" % (time.time() - t0))
    
    # (2) SPECTRAL INDEX SAMPLER
    t0 = time.time()
    beta_samples = inversion_sample_beta(freqs=data_freqs, 
                                         data=data_maps, 
                                         amps=s[0], 
                                         inv_noise_var=inv_noise_var,
                                         comm=comm)
    beta_samples = beta_samples[0][:,np.newaxis] # reshape for next iter
    if myid == 0:
        print("\tBeta sample took %7.4f sec" % (time.time() - t0))

    # OUTPUT SAMPLES
    if myid == 0:
        np.save(os.path.join(out_dir, "%s_amps_%05d" % (prefix, n)), s)
        np.save(os.path.join(out_dir, "%s_beta_%05d" % (prefix, n)), beta_samples)