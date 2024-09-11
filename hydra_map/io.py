import numpy as np
import healpy as hp


def load_and_mask_healpix_map(fname, nside_out=64, mask_val=None, 
                              smooth_fwhm=5., grow_mask=None, 
                              udgrade_pess=False, udgrade_power=None,
                              rotate=None):
    """
    Load a Healpix map from a file, apply a smoothing, pad the mask 
    edges, and degrade resolution.
    
    Parameters:
        fname (str):
            Path to the .fits file containing the Healpix map.
        nside_out (int):
            Healpix NSIDE to use for the output map.
        mask_val (float or str):
            The pixel value that should be interpreted as a mask. 
            By default this will use the Healpix UNSEEN special 
            value. If set to 'min', the minimum value of the map 
            will be used. Otherwise, the specified value will be used.
        smooth_fwhm (float):
            FWHM (in degrees) of the Gaussian smoothing to apply. For 
            maps already convolved with a beam of FWHM `theta`, you 
            should set `smooth_fwhm = sqrt(desired_fwhm^2 - theta^2)` 
            to account for the existing smoothing.
        grow_mask (float):
            At each masked pixel, extend the mask around it by a 
            circular mask of radius `grow_mask` (in degrees). This 
            is useful for masking out pixels adjacent to the mask.
        udgrade_pess (bool):
            xx
        udgrade_power (float):
            xx
        rotate (list of str):
            xx
    
    Returns:
        out_map (array_like):
            A Healpix map that has had the various processing steps 
            above applied.
        mask_idxs (array_like):
            Array of integer pixel indices of masked pixels.
    """
    # Read map from file
    input_map = hp.read_map(fname)
    nside_in = hp.npix2nside(input_map.size)
    
    if rotate is not None:
        rot = hp.Rotator(coord=rotate)
        input_map = rot.rotate_map_pixel(input_map)

    # Re-flag masked regions
    if mask_val is None:
        mask_val = hp.UNSEEN
    if mask_val == 'min':
        mask_val = input_map.min()
    
    # Check for infs and NaNs
    #print("NAN:", np.where(np.isnan(input_map)))
    #print("INF:", np.where(np.isinf(input_map)))
    
    # Get mask idxs and update to use standard Healpix mask value
    input_map[np.where(np.isnan(input_map))] = hp.UNSEEN # replace NaNs
    mask_idxs = np.where(input_map == mask_val)[0]
    input_map[mask_idxs] = hp.UNSEEN
    
    # Copy map
    new_map = input_map.copy()
    
    # Convolve with Gaussian beam
    if smooth_fwhm is not None:
        new_map = hp.smoothing(new_map, 
                               fwhm=np.deg2rad(smooth_fwhm), 
                               pol=False, 
                               use_pixel_weights=False) # FIXME
    
    # Expand mask around existing mask
    if grow_mask is not None:
        
        # Loop over existing mask idxs
        for i in mask_idxs:
            pix_idxs = hp.query_disc(nside=nside_in, 
                                     vec=hp.pix2vec(nside=nside_in, ipix=i), 
                                     radius=np.deg2rad(grow_mask), 
                                     inclusive=True)
            new_map[pix_idxs] = hp.UNSEEN
    
    # Degrade map to new resolution and get mask indices
    out_map = hp.ud_grade(map_in=new_map,
                          nside_out=nside_out,
                          pess=udgrade_pess,
                          power=udgrade_power)
    mask_idxs = np.where(out_map == hp.UNSEEN)[0]

    return out_map, mask_idxs
    

    