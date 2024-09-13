#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import hydra_map as hmap
from hydra_map.io import load_and_mask_healpix_map
from hydra_map.samplers import gcr_sample_pixel, inversion_sample_beta
import healpy as hp
import pylab as plt
import time, os

import argparse




#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Run a Gibbs sampler for power-law parameters.')
parser.add_argument('--iters', dest='niters', type=int, action='store',
                    default=10, help='Number of Gibbs iterations to perform')
parser.add_argument('--ref-freq', dest='nu_ref', type=float, action='store',
                    default=300., help='Reference frequency (in MHz)')
parser.add_argument('--prior-sigma', dest='sigmaS', type=float, action='store',
                    default=100., help='Std. dev. of prior on amplitude.')
parser.add_argument('--beta-range', dest='beta_range', type=float, nargs=2, action='store',
                    default=(-3.2, -2.2), help='Prior range of beta (spectral index).')
parser.add_argument('--data-dir', dest='data_dir', type=str, action='store',
                    default="./data", help='Directory containing data files.')
parser.add_argument('--output-dir', dest='out_dir', type=str, action='store',
                    default="./data", help='Directory for output files.')
parser.add_argument('--seed', dest='seed', type=int, action='store',
                    default=1, 
                    help='Initial random seed. Each worker will get a different seed derived from this.')
parser.add_argument('--prefix', dest='prefix', type=str, action='store',
                    default="lwa1_haslam_nside32", 
                    help='Prefix of data files (also used as output prefix).')
args = parser.parse_args()

niter = args.niters
nu_ref = args.nu_ref

proj_fn = hmap.models.basis_powerlaw
Nmodes = 1
beta_initial = -2.7
sigmaS = args.sigmaS
base_seed = args.seed
beta_range = (np.min(args.beta_range), np.max(args.beta_range))
root_dir = args.data_dir
out_dir = args.out_dir
prefix = args.prefix
data_path = "%s.npz" % prefix


# Setup MPI
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nworkers = comm.Get_size()

# Set random seed
np.random.seed(base_seed + 100000 * myid)

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# Load data from file produced by prepare_data.py
if myid == 0:
    print("Loading data from", os.path.join(root_dir, data_path))

# Randomly wait to do IO so the filesystem doesn't get overloaded
sleep_time = np.random.uniform(low=0., high=0.02*nworkers)
time.sleep(sleep_time)
t0 = time.time()

# Load data
data_file = np.load(os.path.join(root_dir, data_path))
data_freqs = data_file['data_freqs']
data_maps = data_file['data_maps']
inv_noise_var = data_file['inv_noise_var']

if myid == 0:
    print("-"*50)
    print("DATA + MODEL SUMMARY")
    print("     Freqs:  ", data_freqs)
    print("     Nside:  ", hp.npix2nside(data_maps.shape[1]))
    print("beta range:  ", "(%6.3f, %6.3f)" % beta_range)
    print("    Prefix:  ", prefix)
    print("      Seed:  ", base_seed)
    print("-"*50)

if myid == 0:
    print("Root worker loaded data in %5.3f sec" % (time.time() - t0))
comm.barrier()
if myid == 0:
    print("All workers finished loading data.")

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
    if myid == 0:
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
                                         beta_range=beta_range,
                                         comm=comm)
    beta_samples = beta_samples[0][:,np.newaxis] # reshape for next iter
    if myid == 0:
        print("\tBeta sample took %7.4f sec" % (time.time() - t0))

    # OUTPUT SAMPLES
    if myid == 0:
        np.save(os.path.join(out_dir, "%s_amps_%05d" % (prefix, n)), s)
        np.save(os.path.join(out_dir, "%s_beta_%05d" % (prefix, n)), beta_samples)

# Print message on completion
if myid == 0:
    print("Gibbs sampler run complete.")
comm.barrier()