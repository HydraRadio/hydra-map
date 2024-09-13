#!/usr/bin/env python

import numpy as np
import hydra_map as hmap
from hydra_map.io import load_and_mask_healpix_map
import time, os

# Settings
root_dir = "../data"
prefix = "lwa1_haslam_nside32"
nside_out = 32
fwhm = 5. # degrees

# LWA1 radio maps: https://lambda.gsfc.nasa.gov/product/foreground/fg_lwa1_radio_maps_info.html
lwa1_fwhm = [4.8, 4.5, 4.3, 3.8, 3.4, 2.8, 2.4, 2.3, 2.1] # Beams are not symmetric; this is largest axis
lwa1_freqs = [35, 38, 40, 45, 50, 60, 70, 74, 80]
lwa1_maps = [0 for ff in lwa1_freqs]
lwa1_rms = [0 for ff in lwa1_freqs]
lwa1_masks = [0 for ff in lwa1_freqs]

for i, ff in enumerate(lwa1_freqs):

    fname = "healpix-all-sky-rav-wsclean-map-%d.fits" % ff
    fname_rms = "healpix-all-sky-rav-wsclean-err-%d.fits" % ff
    print(fname)
    
    _map, _mask = load_and_mask_healpix_map(
                                    fname=os.path.join(root_dir, fname), 
                                    nside_out=nside_out, 
                                    mask_val=None, 
                                    smooth_fwhm=np.sqrt(fwhm**2. - lwa1_fwhm[i]**2.), 
                                    grow_mask=3., 
                                    udgrade_pess=False, 
                                    udgrade_power=None,
                                    rotate=['C', 'G'])
    
    _rms, _mask2 = load_and_mask_healpix_map(
                                    fname=os.path.join(root_dir, fname_rms), 
                                    nside_out=nside_out, 
                                    mask_val=None, 
                                    smooth_fwhm=np.sqrt(fwhm**2. - lwa1_fwhm[i]**2.), 
                                    grow_mask=3., 
                                    udgrade_pess=False, 
                                    udgrade_power=None, # FIXME: Should add in quadrature
                                    rotate=['C', 'G'])
    
    lwa1_maps[i] = _map
    lwa1_rms[i] = _rms
    lwa1_masks[i] = _mask
    #hp.mollview(lwa1_maps[i])

# Haslam map (reprocessed)
haslam_fwhm = 56. / 60. # 56 arcmin +/- 1 arcmin
haslam_408_map, haslam_408_mask = load_and_mask_healpix_map(
                                fname=os.path.join(root_dir, "haslam408_ds_Remazeilles2014.fits"), 
                                nside_out=nside_out, 
                                mask_val=None, 
                                smooth_fwhm=np.sqrt(fwhm**2. - haslam_fwhm**2.), 
                                grow_mask=3., 
                                udgrade_pess=False, 
                                udgrade_power=None)
haslam_408_rms = 0.8 * np.ones_like(haslam_408_map) # Remazeilles et al. use 800 mK in their fits


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

# Save data in file
np.savez_compressed(os.path.join(root_dir, prefix),
                    data_freqs=data_freqs,
                    data_maps=data_maps,
                    inv_noise_var=inv_noise_var)
print("Data saved in %s.npz" % os.path.join(root_dir, prefix))