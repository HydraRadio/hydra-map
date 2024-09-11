#!/bin/bash
# Download LWA1 and Haslam data files from LAMBDA
# https://lambda.gsfc.nasa.gov/product/foreground/fg_diffuse.html

mkdir data/
cd data/

# LWA1 data maps
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-35.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-38.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-40.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-45.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-50.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-60.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-70.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-74.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-map-80.fits

# LWA1 error maps
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-35.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-38.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-40.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-45.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-50.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-60.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-70.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-74.fits
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/lwa1/healpix-all-sky-rav-wsclean-err-80.fits

# Reprocessed Haslam map
wget --no-check-certificate https://lambda.gsfc.nasa.gov/data/foregrounds/haslam_2014/haslam408_ds_Remazeilles2014.fits