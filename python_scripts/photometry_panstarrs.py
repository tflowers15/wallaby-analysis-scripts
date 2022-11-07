# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import numpy as np
import matplotlib
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import matplotlib.patches as mpatches
import astropy.io.fits as pyfits
import os.path
from os import system
from matplotlib import rc
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.table import Table, join, Column
from astropy.visualization import simple_norm
import aplpy
from scipy.interpolate import interp1d

from photutils.background import Background2D, MedianBackground
from photutils.aperture import aperture_photometry
from photutils.segmentation import deblend_sources
from photutils import detect_threshold, detect_sources, source_properties
from photutils import EllipticalAperture, EllipticalAnnulus


#from functions_plotting import *
from functions_calculations import *


# ================================= #
# ===== Make Segmentation Map ===== #
# ================================= #
def make_segmentation_map(fits_dir, plots_dir, galaxy, band, hi_position, do_seg_id, do_deblend):
    '''
    fits_dir:     PanSTARRS galaxy directory
    
    plots_dir:    Plotting directory
    
    galaxy:       Galaxy name
    
    band:         PanSTARRS image band (g or r)
    
    hi_position:  SoFiA HI position
    
                  hi_position[0]: RA
                  
                  hi_position[1]: Dec
    
    do_seg_id:    Specify a specific segment ID
    
    do_deblend:   Flag for if the segmentation map should be deblended
                  
                  0 - do NOT deblend
                  
                  1 - do deblending
    
    Returns:  Segment x, y, radius, axis ratio (b/a), position angle
    '''
    im_fits         = galaxy + '_%s.fits' % band
    immask_fits     = galaxy + '_%s_mask.fits' % band
    os.chdir(fits_dir)
    os.system('rm -rf %s' % immask_fits)
    f1              = pyfits.open(im_fits, memmap = False)
    data, hdr       = f1[0].data, f1[0].header
    wcs             = WCS(hdr)
    exptime         = hdr['EXPTIME']
    
    threshold       = detect_threshold(data, nsigma=5.)
    sigma           = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel          = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    #kernel.normalize()
    segm1           = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    
    mask            = np.ma.masked_where(data > 1000, data).mask
    
    bkg_estimator   = MedianBackground()
    bkg             = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkgdsb     = data - bkg.background
    threshold       = 2. * bkg.background_rms
    #kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data  = convolve(data_bkgdsb, kernel)
    
    if do_deblend == 1:
      segm2         = detect_sources(convolved_data, threshold, npixels=20)
      segm          = deblend_sources(convolved_data, segm2,
                                      npixels=10, nlevels=32, 
                                      contrast=0.05)#, mode='exponential')
    else:
      segm          = detect_sources(convolved_data, threshold, npixels=20)
    
    seg_data        = segm.data
    cat             = source_properties(data, segm)
    tbl             = cat.to_table()
    ra_seg, dec_seg = wcs.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)
    pos_hi          = SkyCoord(hi_position[0]*u.deg, hi_position[1]*u.deg, frame='icrs')
    #if galaxy == 'J103627-255957':
      #ra_tmp, dec_tmp = wcs.all_pix2world(453, 395, 0)
      #pos_hi          = SkyCoord(ra_tmp*u.deg, dec_tmp*u.deg, frame='icrs')
    pos_seg         = SkyCoord(ra_seg*u.deg, dec_seg*u.deg, frame='icrs')
    cc_dist         = pos_hi.separation(pos_seg)
    
    #print(wcs.all_world2pix(hi_position[0]*u.deg, hi_position[1]*u.deg, 0))
    
    cc_dist[np.isnan(cc_dist.deg)] = 1000*u.deg
    
    #print(cc_dist.deg)
    #print(cc_dist[cat.equivalent_radius.value > 5].deg)
    
    #print(cat.label)
    if do_deblend == 1:
      if len(cc_dist[cat.equivalent_radius.value > 50]) > 0:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 50]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 50]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 50])
      elif len(cc_dist[cat.equivalent_radius.value > 10]) > 0:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 10]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 10]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 10])
      else:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 5]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
    else:
      if len(cc_dist[cat.equivalent_radius.value > 10]) > 0:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 10]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 10]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 10])
      else:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 5]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
    
    seg_label   = label_cut[id_cut]
    
    #print(cat.label)
    
    #print(np.sort(cat.equivalent_radius))
    
    #print(np.sort(cc_dist.deg))
    
    #print(np.argsort(cc_dist))
    #print(np.round(tbl[np.argsort(cc_dist)]['equivalent_radius'].value,1))
    #print(label_array[np.argsort(cc_dist)])
    
    #seg_id          = np.argmin(cc_dist)
    #seg_id          = np.argmin(cc_dist[cat.equivalent_radius.value > 4])
    
    seg_id          = (cat.label == seg_label)
    
    #print(seg_label)
    
    #if do_seg_id[0]:
      #seg_id = do_seg_id[1]
    
    seg_x           = tbl[seg_id]['xcentroid'].value
    seg_y           = tbl[seg_id]['ycentroid'].value
    seg_ellip       = tbl[seg_id]['ellipticity'].value
    seg_radius      = tbl[seg_id]['equivalent_radius'].value
    seg_orientation = tbl[seg_id]['orientation'].value
    seg_elongation  = tbl[seg_id]['elongation'].value

    opt_ba          = 1 - seg_ellip
    opt_pa          = seg_orientation
    
    data[(seg_data != seg_label) & (seg_data > 0)] = np.nan
    
    #if do_cut_above[0]:
      #data[data > do_cut_above[1]] = np.nan
      #seg_x, seg_y = do_cut_above[2], do_cut_above[3]
    
    data = np.ma.masked_invalid(data)

    f1.writeto(immask_fits)
    
    #if ~os.path.exists(plots_dir + 'PHOTOMETRY/'):
      #os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/'))
    #if ~os.path.exists(plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION'):
      #os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION'))
    
    hi_x, hi_y = wcs.all_world2pix(hi_position[0], hi_position[1], 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    norm = simple_norm(data, 'sqrt', percent=98)
    aperture         = EllipticalAperture((seg_x[0], seg_y[0]), 
                                            seg_radius[0] * 2., 
                                            seg_radius[0] * 2. * opt_ba[0], 
                                            theta=opt_pa[0] * math.pi / 180.)
    ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)# vmax=np.nanmax(data)/6)
    aperture_ell  = matplotlib.patches.Ellipse(xy=(seg_x[0], seg_y[0]), 
                            width=seg_radius[0] * 10., 
                            height=seg_radius[0] * 10. * opt_ba[0], 
                            angle=opt_pa[0],
                            edgecolor='peru', linewidth=1, fill=False, zorder=2)
    ax1.add_artist(aperture_ell)
    ax1.plot(hi_x, hi_y, marker='x', color='magenta', markersize=10)
    cmap = segm.make_cmap(seed=123)
    ax2.imshow(seg_data, origin='lower', cmap=cmap, interpolation='nearest')
    plot_name = plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION/%s_segmap.pdf' % galaxy
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    plt.close()
    
    f1.close()
    
    #del data
    #del hdr
    #gc.collect()
    
    return(seg_x[0], seg_y[0], seg_radius[0], opt_ba[0], opt_pa[0])


# ================================= #
# ===== Make Segmentation Map ===== #
# ================================= #
def make_segmentation_map_advanced(fits_dir, plots_dir, galaxy, band, hi_position, do_deblend):
    '''
    fits_dir:     PanSTARRS galaxy directory
    
    plots_dir:    Plotting directory
    
    galaxy:       Galaxy name
    
    band:         PanSTARRS image band (g or r)
    
    hi_position:  SoFiA HI position
    
                  hi_position[0]: RA
    
                  hi_position[1]: Dec
    
    do_deblend:   Flag for if the segmentation map should be deblended
    
                  0 - do NOT deblend
    
                  1 - do deblending
    
    Returns:  Segment x, y, radius, axis ratio (b/a), position angle
    '''
    im_fits         = galaxy + '_%s.fits' % band
    immask_fits     = galaxy + '_%s_mask.fits' % band
    os.chdir(fits_dir)
    os.system('rm -rf %s' % immask_fits)
    f1              = pyfits.open(im_fits, memmap = False)
    data, hdr       = f1[0].data, f1[0].header
    wcs             = WCS(hdr)
    exptime         = hdr['EXPTIME']
    naxis1          = hdr['NAXIS1']
    naxis2          = hdr['NAXIS2']
    
    #hi_x, hi_y = wcs.all_world2pix(hi_position[0]*u.deg, hi_position[1]*u.deg, 0)
    hi_x, hi_y = wcs.all_world2pix(hi_position[0], hi_position[1], 0)
    #pos_hi          = SkyCoord(hi_position[0]*u.deg, hi_position[1]*u.deg, frame='icrs')
    
    centre_sum = 0
    pixel_sum  = 0
    for ii in range(10):
      for jj in range(10):
        #centre_sum += data[int(naxis2/2) - 15 + jj][int(naxis1/2) - 15 + ii]
        centre_sum += data[int(hi_y) - 5 + jj][int(hi_x) - 5 + ii]
        pixel_sum  += 1.
        
    centre_avg1      = centre_sum / pixel_sum
    
    centre_sum = 0
    pixel_sum  = 0
    for ii in range(10):
      for jj in range(10):
        centre_sum += data[int(naxis2/2) - 5 + jj][int(naxis1/2) - 5 + ii]
        #centre_sum += data[int(hi_y) - 5 + jj][int(hi_x) - 5 + ii]
        pixel_sum  += 1.
        
    centre_avg2      = centre_sum / pixel_sum
    
    centre_avg       = np.nanmax(np.array([centre_avg1, centre_avg2]))
    
    threshold       = detect_threshold(data, nsigma=5.)
    sigma           = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel          = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    #kernel.normalize()
    #segm1           = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    
    mask            = np.ma.masked_where(data > 1000, data).mask
    
    bkg_estimator   = MedianBackground()
    bkg             = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkgdsb     = data - bkg.background
    if centre_avg > 5. * np.nanmean(bkg.background_rms):
      threshold       = 3.5 * bkg.background_rms
    elif centre_avg > 2. * np.nanmean(bkg.background_rms):
      threshold       = 2. * bkg.background_rms
    else:
      threshold       = 0.75 * bkg.background_rms
    #kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data  = convolve(data_bkgdsb, kernel)
    
    if do_deblend == 1:
      segm2         = detect_sources(convolved_data, threshold, npixels=10)
      segm          = deblend_sources(convolved_data, segm2,
                                      npixels=10, nlevels=32, 
                                      contrast=0.05)#, mode='exponential')
    else:
      segm          = detect_sources(convolved_data, threshold, npixels=10)
    
    seg_data        = segm.data
    cat             = source_properties(data, segm)
    tbl             = cat.to_table()
    #print(tbl.columns)
    ra_seg, dec_seg = wcs.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)
    pos_hi          = SkyCoord(hi_position[0]*u.deg, hi_position[1]*u.deg, frame='icrs')
    #if galaxy == 'J103627-255957':
      #ra_tmp, dec_tmp = wcs.all_pix2world(453, 395, 0)
      #pos_hi          = SkyCoord(ra_tmp*u.deg, dec_tmp*u.deg, frame='icrs')
    pos_seg         = SkyCoord(ra_seg*u.deg, dec_seg*u.deg, frame='icrs')
    cc_dist         = pos_hi.separation(pos_seg)
    
    cc_dist[np.isnan(cc_dist.deg)] = 1000*u.deg
    
    #print(cat.label)
    if hi_position[2] == 3:
      if len(cc_dist[cat.equivalent_radius.value > 30]) > 0:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 30]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 30]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 30])
        min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > 30].arcsecond)
        if min_sep > 60:
          cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
          label_array = np.array(cat.label)
          label_cut   = label_array[cat.equivalent_radius.value > 5]
          id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
          min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > 5].arcsecond)
      else:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 5]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
        min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > 5].arcsecond)
    else:
      min_radius = 10
      if len(cc_dist[cat.equivalent_radius.value > min_radius]) > 0:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > min_radius]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > min_radius]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > min_radius])
        min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > min_radius].arcsecond)
        if min_sep > 30:
          cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
          label_array = np.array(cat.label)
          label_cut   = label_array[cat.equivalent_radius.value > 5]
          id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
          min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > 5].arcsecond)
      else:
        cc_dist_cut = cc_dist[cat.equivalent_radius.value > 5]
        label_array = np.array(cat.label)
        label_cut   = label_array[cat.equivalent_radius.value > 5]
        id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > 5])
        min_sep     = np.nanmin(cc_dist[cat.equivalent_radius.value > 5].arcsecond)
    
    seg_label       = label_cut[id_cut]
    
    seg_id          = (cat.label == seg_label)
    
    seg_x           = tbl[seg_id]['xcentroid'].value
    seg_y           = tbl[seg_id]['ycentroid'].value
    seg_ellip       = tbl[seg_id]['ellipticity'].value
    seg_radius      = tbl[seg_id]['equivalent_radius'].value
    seg_orientation = tbl[seg_id]['orientation'].value
    seg_elongation  = tbl[seg_id]['elongation'].value
    seg_flux        = tbl[seg_id]['source_sum']#.value
    seg_pixelmax    = tbl[seg_id]['max_value']#.value
    
    opt_ba          = 1 - seg_ellip
    opt_pa          = seg_orientation
    
    data[(seg_data != seg_label) & (seg_data > 0)] = np.nan
    data[data < -500] = np.nan
    
    data = np.ma.masked_invalid(data)

    f1.writeto(immask_fits)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    norm = simple_norm(data, 'sqrt', percent=98)
    aperture         = EllipticalAperture((seg_x[0], seg_y[0]), 
                                            seg_radius[0] * 2., 
                                            seg_radius[0] * 2. * opt_ba[0], 
                                            theta=opt_pa[0] * math.pi / 180.)
    ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)# vmax=np.nanmax(data)/6)
    aperture_ell  = matplotlib.patches.Ellipse(xy=(seg_x[0], seg_y[0]), 
                            width=seg_radius[0] * 10., 
                            height=seg_radius[0] * 10. * opt_ba[0], 
                            angle=opt_pa[0],
                            edgecolor='peru', linewidth=1, fill=False, zorder=2)
    ax1.add_artist(aperture_ell)
    ax1.plot(hi_x, hi_y, marker='x', color='magenta', markersize=10)
    cmap = segm.make_cmap(seed=123)
    ax2.imshow(seg_data, origin='lower', cmap=cmap, interpolation='nearest')
    plot_name = plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION/%s_segmap.pdf' % galaxy
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    plt.close()
    
    f1.close()
    
    if min_sep < 20 or hi_position[2] == 3:
      return(seg_x[0], seg_y[0], seg_radius[0], opt_ba[0], opt_pa[0])
    else:
      return(np.nan, np.nan, np.nan, np.nan, np.nan)


# ================================= #
# ==== Bespoke Segmentation Map === #
# ================================= #
def make_segmentation_bespoke(fits_dir, plots_dir, galaxy, band, hi_position, do_seg_par, do_seg_id, do_deblend):
    '''
    fits_dir:     PanSTARRS galaxy directory
    
    plots_dir:    Plotting directory
    
    galaxy:       Galaxy name
    
    band:         PanSTARRS image band (g or r)
    
    hi_position:  SoFiA HI position
    
                  hi_position[0]: RA
    
                  hi_position[1]: Dec
    
    do_seg_par:   Specify parameters for creating segmentation map
    
                  do_seg_par[0]: Flag to set specific values (0/do NOT or 1/do)
    
                  do_seg_par[1]: nsigma to define the detection threshold
    
                  do_seg_par[2]: Minimum number of pixels for a segment
    
                  do_seg_par[3]: Source finding threshold relative to the background
    
                  do_seg_par[4]: Minimum segment radius [pixels]
    
    do_seg_id:    Specify segment ID
                  
                  do_seg_id[0]: Flag to set segment ID (0/do NOT or 1/do)
                  
                  do_seg_id[1]: Segment ID
    
    do_deblend:   Specify deblending parameters (0/do NOT or 1/do)
    
                  do_deblend[0]: Flag to set deblending values (0/do NOT or 1/do)
    
                  do_deblend[1]: Contrast value
    
                  do_deblend[2]: Minimum segment radius (overides do_seg_par[4])
    
    Returns:  Segment x, y, radius, axis ratio (b/a), position angle
    '''
    im_fits         = galaxy + '_%s.fits' % band
    immask_fits     = galaxy + '_%s_mask.fits' % band
    os.chdir(fits_dir)
    os.system('rm -rf %s' % immask_fits)
    f1              = pyfits.open(im_fits, memmap = False)
    data, hdr       = f1[0].data, f1[0].header
    wcs             = WCS(hdr)
    exptime         = hdr['EXPTIME']
    
    if do_seg_par[0] == 1:
      nsigma        = do_seg_par[1]
      npixels       = do_seg_par[2]
      threshold_sig = do_seg_par[3]
      radius_min    = do_seg_par[4]
    else:
      nsigma        = 5.
      npixels       = 10
      threshold_sig = 2.
      radius_min    = 10.
    
    threshold       = detect_threshold(data, nsigma=nsigma)
    sigma           = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel          = Gaussian2DKernel(sigma, x_size=5, y_size=5)
    #kernel.normalize()
    segm1           = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    
    mask            = np.ma.masked_where(data > 1000, data).mask
    
    bkg_estimator   = MedianBackground()
    bkg             = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkgdsb     = data - bkg.background
    threshold       = threshold_sig * bkg.background_rms
    #kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data  = convolve(data_bkgdsb, kernel)
    
    if do_deblend[0] == 1:
      segm2         = detect_sources(convolved_data, threshold, npixels=npixels)
      segm          = deblend_sources(convolved_data, segm2,
                                      npixels=10, nlevels=32, 
                                      contrast=do_deblend[1])#, mode='exponential')
      radius_min = do_deblend[2]
    else:
      segm          = detect_sources(convolved_data, threshold, npixels=npixels)
    
    seg_data        = segm.data
    cat             = source_properties(data, segm)
    tbl             = cat.to_table()
    ra_seg, dec_seg = wcs.all_pix2world(tbl['xcentroid'], tbl['ycentroid'], 0)
    pos_hi          = SkyCoord(hi_position[0]*u.deg, hi_position[1]*u.deg, frame='icrs')
    pos_seg         = SkyCoord(ra_seg*u.deg, dec_seg*u.deg, frame='icrs')
    cc_dist         = pos_hi.separation(pos_seg)
    
    cc_dist[np.isnan(cc_dist.deg)] = 1000*u.deg
    
    cc_dist_cut = cc_dist[cat.equivalent_radius.value > radius_min]
    label_array = np.array(cat.label)
    label_cut   = label_array[cat.equivalent_radius.value > radius_min]
    id_cut      = np.argmin(cc_dist[cat.equivalent_radius.value > radius_min])
    
    seg_label   = label_cut[id_cut]
    
    if do_seg_id[0] == 1:
      seg_label = do_seg_id[1]
    
    seg_id          = (cat.label == seg_label)
    
    seg_x           = tbl[seg_id]['xcentroid'].value
    seg_y           = tbl[seg_id]['ycentroid'].value
    seg_ellip       = tbl[seg_id]['ellipticity'].value
    seg_radius      = tbl[seg_id]['equivalent_radius'].value
    seg_orientation = tbl[seg_id]['orientation'].value
    seg_elongation  = tbl[seg_id]['elongation'].value

    opt_ba          = 1 - seg_ellip
    opt_pa          = seg_orientation
    
    data[(seg_data != seg_label) & (seg_data > 0)] = np.nan
    
    #if do_cut_above[0]:
      #data[data > do_cut_above[1]] = np.nan
      #seg_x, seg_y = do_cut_above[2], do_cut_above[3]
    
    data = np.ma.masked_invalid(data)

    f1.writeto(immask_fits)
    
    hi_x, hi_y = wcs.all_world2pix(hi_position[0], hi_position[1], 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    norm = simple_norm(data, 'sqrt', percent=98)
    aperture         = EllipticalAperture((seg_x[0], seg_y[0]), 
                                            seg_radius[0] * 2., 
                                            seg_radius[0] * 2. * opt_ba[0], 
                                            theta=opt_pa[0] * math.pi / 180.)
    ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)# vmax=np.nanmax(data)/6)
    aperture_ell  = matplotlib.patches.Ellipse(xy=(seg_x[0], seg_y[0]), 
                            width=seg_radius[0] * 10., 
                            height=seg_radius[0] * 10. * opt_ba[0], 
                            angle=opt_pa[0],
                            edgecolor='peru', linewidth=1, fill=False, zorder=2)
    ax1.add_artist(aperture_ell)
    ax1.plot(hi_x, hi_y, marker='x', color='magenta', markersize=10)
    cmap = segm.make_cmap(seed=123)
    ax2.imshow(seg_data, origin='lower', cmap=cmap, interpolation='nearest')
    plot_name = plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION/%s_segmap_bespoke.pdf' % galaxy
    plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    plt.close()
    
    
    f1.close()
    
    return(seg_x[0], seg_y[0], seg_radius[0], opt_ba[0], opt_pa[0])



# ================================= #
# === Measure GALEX Photometry ==== #
# ================================= #
def extract_surface_brightness(survey_dir, galaxy, aperture_params, band, radius_max, fignum):
    '''
    survey_dir:       PanSTARRS galaxy directory
    
    galaxy:           Galaxy name
    
    aperture_params:  Parameters for annuli/apertures
    
                      aperture_params[0]: PanSTARRS segment x position [pixel]
    
                      aperture_params[1]: PanSTARRS segment y position [pixel]
    
                      aperture_params[2]: PanSTARRS segment radius [arcsec]
    
                      aperture_params[3]: PanSTARRS segment axis ratio b/a
    
                      aperture_params[4]: PanSTARRS segment position angle [degrees]
    
    band:             PanSTARRS image band (g or r)
    
    radius_max:       Maximum radius to define an aperture/annulus
    
    fignum:           Figure number
    
    Returns:  Annulus radius, area, flux (ADU), flux error (ADU), surface brightness,
              Aperture radius, area, flux (ADU), flux error (ADU), total magnitude (curve of growth), 
              exposure time, sky magnitude    
    '''
    ps_aper_x      = aperture_params[0]
    ps_aper_y      = aperture_params[1]
    ps_aper_r      = aperture_params[2]
    ps_aper_ba     = aperture_params[3]
    ps_aper_pa     = aperture_params[4] * math.pi / 180.
    
    fits_dir       = survey_dir + galaxy +'/'
    im_fits        = galaxy + '_%s.fits' % band
    immask_fits    = galaxy + '_r_mask.fits'
    #immask_fits2    = galaxy + '_r_mask4.fits'
    
    #print(fits_dir)
    
    os.chdir(fits_dir)
    
    f1              = pyfits.open(im_fits, memmap=False)
    data, hdr       = f1[0].data, f1[0].header
    wcs             = WCS(hdr)
    exptime         = hdr['EXPTIME']
    npix_f1         = hdr['NAXIS1']
    
    f2              = pyfits.open(immask_fits, memmap=False)
    data_r, hdr_r   = f2[0].data, f2[0].header
    
    #data_r[data_r < -750] = np.nan
    
    #data_r = np.ma.masked_invalid(data_r)

    #f2.writeto(immask_fits2)
    
    data[np.isnan(data_r)] = np.nan
    
    data             = np.ma.masked_invalid(data)
    
    aperture_list    = []
    annulus_list     = []
    radius_list      = []
    area_annul_list  = []
    area_aper_list   = []
    
    rmax_pix         = radius_max / 0.25
    
    if rmax_pix > npix_f1 / 2.:
      rmax_pix = npix_f1 / 2.
      
    if rmax_pix / 10 < 80:
      scale_list       = np.arange(0, rmax_pix, 2)
    else:
      scale_list       = np.arange(0, rmax_pix, 10)
    
    radius_aperture    = scale_list[1:-1] #* a_ps * pix_scale
    radius_annulus     = scale_list[2:] #* a_ps * pix_scale
    
    for i in range(1, len(scale_list)-1):
      aperture         = EllipticalAperture((ps_aper_x, ps_aper_y), 
                                            scale_list[i], 
                                            scale_list[i] * ps_aper_ba, 
                                            theta=ps_aper_pa)
      aperture_annulus = EllipticalAnnulus((ps_aper_x, ps_aper_y), 
                                           a_in=scale_list[i], 
                                           a_out=scale_list[i+1], 
                                           b_out=scale_list[i+1] * ps_aper_ba, 
                                           b_in=scale_list[i] * ps_aper_ba, 
                                           theta=ps_aper_pa)
      aperture_list.append(aperture)
      annulus_list.append(aperture_annulus)
      #radius_list.append(scale_list[i])
      area_annul_list.append(aperture_annulus.area)
      area_aper_list.append(aperture.area)
      
    radius_aperture   = np.array(radius_aperture)
    radius_annulus    = np.array(radius_annulus)
    area_annul_list   = np.array(area_annul_list) #/ (0.25*0.25)
    
    phot_table_annulus1      = aperture_photometry(data, annulus_list)
    
    annulus_adu_list1        = []
    
    for i in range(len(annulus_list)):
      string_name = 'aperture_sum_%s' % (i)
      annulus_adu_list1.append(phot_table_annulus1[string_name][0])
    
    annulus_adu_list1 =  np.array(annulus_adu_list1)
    
    #print(np.round(annulus_adu_list1, 2))
    
    sd_profile1 = 25. + 2.5 * np.log10(exptime) - 2.5 * np.log10(annulus_adu_list1 / area_annul_list / 0.0625)
    
    mean_annulus = []
    mean_ids     = []
    for i in range(len(annulus_list)):
        if radius_annulus[i] > rmax_pix / 1.5:
          annulus_string = string_name = 'aperture_sum_%s' % (i)
          mean_annulus.append(phot_table_annulus1[annulus_string][0] / area_annul_list[i])
          mean_ids.append(i)
    
    sd_profile1       = sd_profile1[radius_annulus > rmax_pix / 1.5]
    
    mean_annulus      = np.array(mean_annulus)
    
    if len(mean_annulus[sd_profile1 > 26.5]) > 0:
      mean_annulus_bkgd = mean_annulus[sd_profile1 > 26.5]
    else:
      mean_annulus_bkgd = mean_annulus
    
    error = np.nanmean(mean_annulus_bkgd) * data / data
    
    phot_table_annulus       = aperture_photometry(data - np.nanmean(mean_annulus_bkgd), 
                                                   annulus_list, error = error)
    phot_table_aperture      = aperture_photometry(data - np.nanmean(mean_annulus_bkgd), 
                                                   aperture_list, error = error)
    
    #phot_table_aperture     = aperture_photometry(data, aperture_list)
    
    annulus_adu_list        = []
    aperture_adu_list       = []
    annulus_adu_err_list    = []
    aperture_adu_err_list   = []
    
    for i in range(len(annulus_list)):
      string_name1 = 'aperture_sum_%s' % (i)
      string_name2 = 'aperture_sum_err_%s' % (i)
      annulus_adu_list.append(phot_table_annulus[string_name1][0])
      aperture_adu_list.append(phot_table_aperture[string_name1][0])
      annulus_adu_err_list.append(phot_table_annulus[string_name2][0])
      aperture_adu_err_list.append(phot_table_aperture[string_name2][0])
    
    annulus_adu_list      = np.array(annulus_adu_list)
    aperture_adu_list     = np.array(aperture_adu_list)
    annulus_adu_err_list  = np.array(annulus_adu_err_list)
    aperture_adu_err_list = np.array(aperture_adu_err_list)
    
    #print(np.round(annulus_adu_list, 2))
    
    fignum.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    norm = simple_norm(data, 'sqrt', percent=98)
    if band == 'g':
      ax1 = fig.add_subplot(2, 2, 1, facecolor = 'w')
    if band == 'r':
      ax1 = fig.add_subplot(2, 2, 2, facecolor = 'w')
    ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)
    #ap_patches = aperture_ps.plot(color='red', lw=1.5, label='Photometry aperture')
    for i in range(len(aperture_list)):
      if (i/10).is_integer():
        ap_patches = aperture_list[i].plot(color='peru', lw=0.75, label='Photometry aperture')
    ap_patches = aperture_list[mean_ids[len(mean_ids) - 1]].plot(color='black', 
                                                                 lw=0.75, 
                                                                 label='Photometry aperture')
    ap_patches = aperture_list[mean_ids[0]].plot(color='black', 
                                                 lw=0.75, 
                                                 label='Photometry aperture')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    
    sd_profile = 25. + 2.5 * np.log10(exptime) - 2.5 * np.log10(annulus_adu_list / area_annul_list / 0.0625)
    cg_profile = 25. + 2.5 * np.log10(exptime) - 2.5 * np.log10(aperture_adu_list)
    
    sky_mag    = 25. + 2.5 * np.log10(exptime) - 2.5 * np.log10(np.nanstd(mean_annulus) / 0.0625)
    
    sd_profile[np.isnan(sd_profile)] = sky_mag
    #25. + 2.5 * np.log10(exptime) - 2.5 * np.log10(1.5)
    
    sd_profile[radius_annulus > radius_max / 0.25 / 2.] = np.nan
    cg_profile[radius_aperture > radius_max / 0.25 / 2.] = np.nan
    
    f1.close()
    f2.close()
    
    del data
    del data_r
    
    return(radius_annulus, 
           area_annul_list, 
           annulus_adu_list,
           annulus_adu_err_list,
           sd_profile,
           radius_aperture,
           np.array(area_aper_list), 
           aperture_adu_list,
           aperture_adu_err_list,
           cg_profile,
           exptime,
           sky_mag)


# ================================= #
# ==== Measure Magintude/Size ===== #
# ================================= #
def measure_mag_size(isophote_limit, exptime, aperture_adu, annulus_mag, radii):
    '''
    isophote_limit:   Surface brightness for measuring isophotal size
    
    exptime:          Image exposure time
    
                      exptime[0]: r-band exposure time
    
                      exptime[1]: g-band exposure time
    
    aperture_adu:     Total measured ADU from apertures
    
                      aperture_adu[0]: r-band ADU
    
                      aperture_adu[1]: g-band ADU
    
    annulus_mag:      r-band annulus magnitudes 
    
    radii:            Annulus/aperture radii
    
                      radii[0]: annulus radii
    
                      radii[1]: aperture radii
                      
    Returns:  r-band magnitude, g-band magnitude, isophotal radius    
    '''
    exptime_r    = exptime[0]
    exptime_g    = exptime[1]
    aper_adu_r   = aperture_adu[0]
    aper_adu_g   = aperture_adu[1]
    annu_radius  = radii[0]
    aper_radius  = radii[1]
    radius_iso   = np.nan
    
    for k in range(len(annu_radius) - 1):
      if annulus_mag[k] < isophote_limit and annulus_mag[k + 1] > isophote_limit:
        radius_iso     = interpolate_value(annu_radius, annulus_mag, isophote_limit, k)
        break
    
    radius_iso       = np.sqrt((radius_iso)**2 - (1.25 / 2.)**2)
    
    r_aper_adu_int  = np.interp(radius_iso, aper_radius, aper_adu_r)
    g_aper_adu_int  = np.interp(radius_iso, aper_radius, aper_adu_g)
    
    mag_r           = 25. + 2.5 * np.log10(exptime_r) - 2.5 * np.log10(r_aper_adu_int)
    mag_g           = 25. + 2.5 * np.log10(exptime_g) - 2.5 * np.log10(g_aper_adu_int)
    
    return(mag_r, mag_g, radius_iso)


def save_table_function(table_name, table_data, table_cols):
    table      = Table(table_data, names=table_cols)
    table.write(table_name, format = 'fits')



def sb_profile_radius_plot3(fig_num, sub1, sub2, sub3, flux1, flux2, radius_array, col_array):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\log_{10}[\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}]$')
    ax1.set_xlabel(r'Radius [arcsec]')
    ax1.set_ylim(np.nanmin(flux2)-0.2, np.nanmax(flux2)+0.2)
    for i in range(len(radius_array)):
      plt.axvline(radius_array[i], color = col_array[i], linestyle = '--', linewidth = 1.5, zorder = 0)
    #plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    ax1.plot(flux1, flux2, color='darkblue', linewidth=2, linestyle='-')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
#RHO_CRIT = cosmo.critical_density(0).value*100**3/1000
#OMEGA_M  = cosmo.Om(0)
#OMEGA_DE = cosmo.Ode(0)
HI_REST  = 1420.406



# ================================= #
# ========== Axes Labels ========== #
# ================================= #
lbl_czhi   = r'c$z_{\mathrm{HI}}$ [km\,s$^{-1}$]'
lbl_czopt  = r'c$z_{\mathrm{opt}}$ [km\,s$^{-1}$]'
lbl_sint   = r'$\log(S_{\mathrm{int}}/\mathrm{Jy})$'
lbl_mstar  = r'$\log(M_*/\mathrm{M}_{\odot})$'
lbl_mhi    = r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$'
lbl_hidef  = r'$\mathrm{DEF}_{\mathrm{HI}}$'
lbl_dhi    = r'$d_{\mathrm{HI}}/\mathrm{kpc}$'
lbl_d25    = r'$d_{\mathrm{25}}/\mathrm{kpc}$'
lbl_hifrac = r'$\log(M_{\mathrm{HI}}/M_*)$'
lbl_dlum   = r'$D_{\mathrm{L}}$ [Mpc]'
lbl_aflux  = r'$A_{\mathrm{flux}}$' 
lbl_aspec  = r'$A_{\mathrm{spec}}$'
lbl_dvsys  = r'$\Delta V_{\mathrm{sys}}$ [km\,s$^{-1}$]'
lbl_w20    = r'$w_{20}$ [km\,s$^{-1}$]'
lbl_sep    = r'$R_{\mathrm{proj}}/R_{\mathrm{vir}}$'
lbl_sfr    = r'$\log$(SFR/[M$_{\odot}$\,yr$^{-1}$])'
lbl_nuvr   = r'$\mathrm{NUV}-r$ [Mag]'
lbl_ssfr   = r'$\log$(sSFR/[yr$^{-1}$])'
lbl_sratio = r'$d_{\mathrm{HI}}/d_{25}$'

# ================================= #
# =========== Switches ============ #
# ================================= #
do_get_source_properties = True          # Always True, provides input source parameters
have_segments            = False         # True to open *_panstarrs_segments_deblend.fits if exists/final
#have_optical             = False         # True to open *_panstarrs_photometry.fits if exists


# ++++ ONLY RUN ONE AT A TIME +++++ #
do_segim                 = True          # True to create segmentation maps
do_segim_bespoke         = False         # True to fix segmentation map for single source
do_fit_phot              = False         # True to fit annuli/apertures to g-/r-band images
do_measure               = False         # True to measure magnitudes/radii


# ================================= #
# ==== Specify Phase + Release ==== #
# ================================= #
tr_i                     = 6

survey_phase_list        = ['PHASE1', 'PHASE1', 'PHASE1', 
                            'PHASE2', 'PHASE2', 'PHASE2', 'PHASE2']
team_release_list        = ['Hydra_DR1', 'Hydra_DR2', 'NGC4636_DR1', 
                            'NGC4808_DR1', 'NGC5044_DR1', 'NGC5044_DR2', 'NGC5044_DR3']

survey_phase             = survey_phase_list[tr_i]
team_release             = team_release_list[tr_i]

# ================================= #
# ========= File Strings ========== #
# ================================= #
basedir                  = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase, team_release)
sofia_dir                = basedir + 'SOFIA/'
dataprod_dir             = basedir + 'SOFIA/%s_source_products/' % team_release
panstarrs_dir            = basedir + 'MULTIWAVELENGTH/PANSTARRS/'
wise_dir                 = basedir + 'MULTIWAVELENGTH/WISE/'
galex_dir                = basedir + 'MULTIWAVELENGTH/GALEX/'
parameter_dir            = basedir + 'PARAMETERS/'
hi_products_dir          = basedir + 'HI_DERIVED_PRODUCTS/'              
plots_dir                = basedir + 'PLOTS/'
  

# ================================= #
# ===== Open/Join FITS Tables ===== #
# ================================= #
if ~have_segments:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
  fits_gaussian  = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_flags      = fits.open(fits_flags)
  data_flags     = hdu_flags[1].data
  
  hdu_gaussian   = fits.open(fits_gaussian)
  data_gaussian  = hdu_gaussian[1].data
  
  join1          = join(data_sofia, data_flags, join_type='left')
  data_join      = join(join1, data_gaussian, join_type='left')

if have_segments:
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
  fits_gaussian  = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
  fits_segments  = parameter_dir + '%s_panstarrs_segments_deblend.fits' % team_release
  if not os.path.exists(fits_segments):
    fits_seg_first  = parameter_dir + '%s_panstarrs_segments.fits' % team_release
    os.system('cp %s %s' % (fits_seg_first, fits_segments))
    print('copy %s to %s' % (fits_seg_first, fits_segments))
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_flags      = fits.open(fits_flags)
  data_flags     = hdu_flags[1].data
  
  hdu_gaussian   = fits.open(fits_gaussian)
  data_gaussian  = hdu_gaussian[1].data
  
  hdu_segments   = fits.open(fits_segments)
  data_segments  = hdu_segments[1].data
  
  join1          = join(data_sofia, data_flags, join_type='left')
  join2          = join(join1, data_gaussian, join_type='left')
  data_join      = join(join2, data_segments, join_type='left')


# ================================= #
# ===== Get Source Properties ===== #
# ================================= #
if do_get_source_properties:
  #flag_mask      = ((data_join['flag_class'] == 1) | (data_join['flag_class'] == 4))
  data_mask      = data_join#[flag_mask]
  gal_name = []
  for i in range(len(data_mask['name'])):
    split_name = data_mask['name'][i][8:]
    gal_name.append(split_name)
  
  galaxy_dir = 'WALLABY_' + gal_name[0] + '/'
  fits_file  = dataprod_dir + galaxy_dir + 'WALLABY_' + gal_name[0] + '_cube.fits.gz'
  
  f1         = pyfits.open(fits_file)
  data, hdr  = f1[0].data, f1[0].header
  if hdr['CTYPE3'] == 'FREQ':
    chan_width = np.abs((HI_REST / (hdr['CRVAL3']/1e6)) - (HI_REST / ((hdr['CRVAL3']-hdr['CDELT3'])/1e6))) * C_LIGHT
    chan_width_hz = hdr['CDELT3']
  else:
    chan_width = np.abs(hdr['CDELT3']/1000.)
  beam_maj, beam_min, beam_pa, pix_scale  = hdr['BMAJ'], hdr['BMIN'], hdr['BPA'], np.abs(hdr['CDELT1'])
  f1.close()
  
  BEAM            = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  redshift        = (HI_REST / (data_mask['freq'] / 1.e6)) - 1.
  galaxies        = np.array(gal_name)
  sofia_ra        = data_mask['ra']
  sofia_dec       = data_mask['dec']
  sofia_vsys      = redshift * C_LIGHT
  sofia_rms       = data_mask['rms']
  sofia_sint      = data_mask['f_sum'] * chan_width / chan_width_hz
  sofia_snr       = data_mask['f_sum'] / data_mask['err_f_sum']
  sofia_kinpa     = data_mask['kin_pa']
  sofia_w20       = data_mask['w20'] / chan_width_hz * chan_width
  sofia_w50       = data_mask['w50'] / chan_width_hz * chan_width
  sofia_ell_maj   = data_mask['ell_maj'] * pix_scale * 3600.
  sofia_ell_min   = data_mask['ell_min'] * pix_scale * 3600.
  sofia_ellpa     = data_mask['ell_pa']
  
  source_flag     = data_mask['flag_src_class']
  opt_fit_flag    = data_mask['flag_opt_fit']
  segment_deblend = data_mask['segment_deblend']
  
  do_seg_par1     = data_mask['do_seg_par1']
  do_seg_par2     = data_mask['do_seg_par2']
  do_seg_par3     = data_mask['do_seg_par3']
  do_seg_par4     = data_mask['do_seg_par4']
  do_seg_par5     = data_mask['do_seg_par5']
  
  do_seg_id1      = data_mask['do_seg_id1']
  do_seg_id2      = data_mask['do_seg_id2']
  
  do_deblend1     = data_mask['do_deblend1']
  do_deblend2     = data_mask['do_deblend2']
  do_deblend3     = data_mask['do_deblend3']
  
  ra_gaussian     = data_mask['GAUSSIAN_RA']
  dec_gaussian    = data_mask['GAUSSIAN_DEC']
  
  #Correction for measured WALLABY fluxes
  sint_corr      = np.array(wallaby_flux_scaling(data_mask['f_sum']))
  scale_factor   = np.array(data_mask['f_sum'] / sint_corr)
  
  if have_segments:
    seg_x        = data_mask['SEG_X']
    seg_y        = data_mask['SEG_Y']
    seg_radius   = data_mask['SEG_RADIUS']
    seg_ba       = data_mask['SEG_BA']
    seg_pa       = data_mask['SEG_PA']
  

print(len(galaxies))



test_galaxies = ['J104059-270456', 'J100746-281451', 'J103737-261641']


# ================================= #
# ==== Create Segmentation Maps === #
# ================================= #
if do_segim:
  do_second_pass = False      # True only to remake segmentation maps for all sources after updating
                              # *_catalogue_flags.fits with manual segmentation parameters for tricky 
                              # sources. For first time creating segmentation maps, set False.
  if ~have_segments:
    seg_x        = np.full(len(galaxies), np.nan)
    seg_y        = np.full(len(galaxies), np.nan)
    seg_radius   = np.full(len(galaxies), np.nan)
    seg_ba       = np.full(len(galaxies), np.nan)
    seg_pa       = np.full(len(galaxies), np.nan)
    
  for i in range(len(galaxies)):
    #print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
    #if galaxies[i] == 'J104059-270456':
    print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
    if i > -1: # and opt_fit_flag[i] == 0:# and segment_deblend[i] == 1:
      band            = 'r'
      fits_dir        = panstarrs_dir + galaxies[i] +'/'
      
      # =========== Segmetation/deblending parameters from flags catalogue =========== #
      do_seg_par_array  = [do_seg_par1[i], do_seg_par2[i], do_seg_par3[i], do_seg_par4[i], do_seg_par5[i]] 
      do_seg_id_array   = [do_seg_id1[i], do_seg_id2[i]]
      do_deblend_array  = [do_deblend1[i], do_deblend2[i], do_deblend3[i]]
      
      # ======== Runs bespoke segmentation for flagged galaxies in flag catalogue, otherwise standard  ======== #
      if do_second_pass:
        if do_seg_par_array[0] == 1 or do_deblend_array[0] == 1:
          segment_parameters = make_segmentation_bespoke(fits_dir, plots_dir, 
                                                      galaxies[i], band, 
                                                      [ra_gaussian[i], dec_gaussian[i]], 
                                                      do_seg_par_array,
                                                      do_seg_id_array,
                                                      do_deblend_array)
        else:
          #segment_parameters = make_segmentation_map(fits_dir, plots_dir, 
                                                    #galaxies[i], band, 
                                                    #[ra_gaussian[i], dec_gaussian[i]], 
                                                    #[False, 0], segment_deblend[i])
          segment_parameters = make_segmentation_map_advanced(fits_dir, plots_dir, 
                                                              galaxies[i], band, 
                                                              [ra_gaussian[i], 
                                                               dec_gaussian[i], 
                                                               source_flag[i]], 
                                                              0)
      
      # ======== Runs standard segmentation procedure ======== #
      else:
        segment_parameters = make_segmentation_map_advanced(fits_dir, plots_dir, 
                                                            galaxies[i], band, 
                                                            [ra_gaussian[i], 
                                                             dec_gaussian[i], 
                                                             source_flag[i]], 
                                                            0)
      
      seg_x[i]      = segment_parameters[0]
      seg_y[i]      = segment_parameters[1]
      seg_radius[i] = segment_parameters[2]
      seg_ba[i]     = segment_parameters[3]
      seg_pa[i]     = segment_parameters[4]
      
      print('%.0f\t%.0f\t%.2f\t%.2f\t%.0f' % (seg_x[i], seg_y[i], seg_radius[i], seg_ba[i], seg_pa[i]))
  
  # =========== Save PanSTARRS segment parameters to file =========== #
  table_str  = parameter_dir + '%s_panstarrs_segments.fits' % team_release
  os.system('rm -rf %s' % table_str)
  
  tdata = []
  tcols = []
  for i in range(len(data_join.columns)):
    if i < len(data_sofia.columns.names):
      tdata.append(data_join[data_join.columns[i].name])
      tcols.append(data_join.columns[i].name)
  
  tdata_1 = [seg_x, seg_y, seg_radius, seg_ba, seg_pa]
  
  tcols_1 = ('SEG_X', 'SEG_Y', 'SEG_RADIUS', 'SEG_BA', 'SEG_PA')
  
  for i in range(len(tdata_1)):
    tdata.append(tdata_1[i])
    tcols.append(tcols_1[i])
    
  save_table_function(table_str, tdata, tcols)




# ================================= #
# ==== Create Segmentation Maps === #
# ================================= #
if do_segim_bespoke:
  galaxy_to_find    = 'J103737-261641'
  gal_select        = (galaxies == galaxy_to_find)
  
  print(galaxies[galaxies == galaxy_to_find])
  
  band              = 'r'
  fits_dir          = panstarrs_dir + galaxy_to_find +'/'
  
  do_seg_par_array  = [True, 5., 10, 4.25, 30] 
  do_seg_id_array   = [False, 0]
  do_deblend_array  = [False, 0.1, 40]
  
  
  print('%.0f\t%.0f\t%.2f\t%.2f\t%.0f' % (seg_x[gal_select], 
                                          seg_y[gal_select], 
                                          seg_radius[gal_select], 
                                          seg_ba[gal_select], 
                                          seg_pa[gal_select]))
  
  # =========== Create segmentation map with manually chosen input parameters =========== #
  segment_parameters = make_segmentation_bespoke(fits_dir, plots_dir, 
                                                  galaxy_to_find, band, 
                                                  [ra_gaussian[gal_select], 
                                                  dec_gaussian[gal_select]], 
                                                  do_seg_par_array,
                                                  do_seg_id_array,
                                                  do_deblend_array)
  
  seg_x[gal_select]      = segment_parameters[0]
  seg_y[gal_select]      = segment_parameters[1]
  seg_radius[gal_select] = segment_parameters[2]
  seg_ba[gal_select]     = segment_parameters[3]
  seg_pa[gal_select]     = segment_parameters[4]
  
  print('%.0f\t%.0f\t%.2f\t%.2f\t%.0f' % (seg_x[gal_select], 
                                          seg_y[gal_select], 
                                          seg_radius[gal_select], 
                                          seg_ba[gal_select], 
                                          seg_pa[gal_select]))
  
  # =========== Save updated PanSTARRS segment parameters to file and make backup of old version =========== #
  table_str      = parameter_dir + '%s_panstarrs_segments_deblend.fits' % team_release
  table_str_old  = parameter_dir + '%s_panstarrs_segments_deblend_old.fits' % team_release
  os.system('mv %s %s' % (table_str, table_str_old))
  #os.system('rm -rf %s' % table_str)
  
  for i in range(len(data_join.columns)):
    if i < len(data_sofia.columns.names):
      tdata.append(data_join[data_join.columns[i].name])
      tcols.append(data_join.columns[i].name)
  
  tdata_1 = [seg_x, seg_y, seg_radius, seg_ba, seg_pa]
  
  tcols_1 = ('SEG_X', 'SEG_Y', 'SEG_RADIUS', 'SEG_BA', 'SEG_PA')
  
  for i in range(len(tdata_1)):
    tdata.append(tdata_1[i])
    tcols.append(tcols_1[i])
  
  save_table_function(table_str, tdata, tcols)


#if do_segim_bespoke:
  #galaxy_to_find = 'J100746-281451'
  #for i in range(len(galaxies)):
    #print(i, galaxies[i])
    #if galaxies[i] == galaxy_to_find:
      #band            = 'r'
      #fits_dir        = panstarrs_dir + galaxies[i] +'/'
      
      #do_seg_par_array  = [False, 5., 10, 1.5, 5] 
      #do_seg_id_array   = [False, 0]
      #do_deblend_array  = [True, 0.1, 40]
      
      #print('%.0f\t%.0f\t%.2f\t%.2f\t%.0f' % (seg_x[i], seg_y[i], seg_radius[i], seg_ba[i], seg_pa[i]))
      
      #segment_parameters = make_segmentation_bespoke(fits_dir, plots_dir, 
                                                     #galaxies[i], band, 
                                                     #[ra_gaussian[i], dec_gaussian[i]], 
                                                     #do_seg_par_array,
                                                     #do_seg_id_array,
                                                     #do_deblend_array)
      
      #seg_x[i]      = segment_parameters[0]
      #seg_y[i]      = segment_parameters[1]
      #seg_radius[i] = segment_parameters[2]
      #seg_ba[i]     = segment_parameters[3]
      #seg_pa[i]     = segment_parameters[4]
      
      #print('%.0f\t%.0f\t%.2f\t%.2f\t%.0f' % (seg_x[i], seg_y[i], seg_radius[i], seg_ba[i], seg_pa[i]))
      
  #table_str      = parameter_dir + '%s_panstarrs_segments_deblend.fits' % team_release
  #table_str_old  = parameter_dir + '%s_panstarrs_segments_deblend_old.fits' % team_release
  #os.system('mv %s %s' % (table_str, table_str_old))
  ##os.system('rm -rf %s' % table_str)
  
  #tdata = []
  #tcols = []
  #for i in range(len(data_sofia.columns.names)):
    #tdata.append(data_sofia[data_sofia.columns.names[i]])
    #tcols.append(data_sofia.columns.names[i])
  
  #tdata_1 = [seg_x, seg_y, seg_radius, seg_ba, seg_pa]
  
  #tcols_1 = ('SEG_X', 'SEG_Y', 'SEG_RADIUS', 'SEG_BA', 'SEG_PA')
  
  #for i in range(len(tdata_1)):
    #tdata.append(tdata_1[i])
    #tcols.append(tcols_1[i])
  
  #save_table_function(table_str, tdata, tcols)


# ================================= #
# ===== Fit Annuli/Apertures ====== #
# ================================= #
if do_fit_phot:
  for i in range(len(galaxies)):
    table_str  = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
    print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
    if galaxies[i] == 'J104059-270456':
    #if i > -1 and opt_fit_flag[i] == 0 and ~os.path.isfile(table_str):
      fig = plt.figure(figsize=(5,5))
      
      segment_parameters = [seg_x[i], seg_y[i], seg_radius[i], seg_ba[i], seg_pa[i]]
      
      radius_max = 7. * seg_radius[i]
      if radius_max < 100:
        radius_max = 100
      
      #print(radius_max)
      
      # =========== Fit apertures/annuli to r-band image =========== #
      aperture_r  = extract_surface_brightness(panstarrs_dir, 
                                                galaxies[i], 
                                                segment_parameters, 
                                                'r',
                                                radius_max,
                                                fig)
      
      # =========== Fit apertures/annuli to g-band image =========== #
      aperture_g  = extract_surface_brightness(panstarrs_dir, 
                                                galaxies[i], 
                                                segment_parameters, 
                                                'g',
                                                radius_max,
                                                fig)
      
      annu_rad_r         = aperture_r[0]
      annu_area_r        = aperture_r[1]
      annu_adu_r         = aperture_r[2]
      annu_adu_e_r       = aperture_r[3]
      annu_mag_r         = aperture_r[4]
      aper_rad_r         = aperture_r[5]
      aper_area_r        = aperture_r[6]
      aper_adu_r         = aperture_r[7]
      aper_adu_e_r       = aperture_r[8]
      aper_mag_r         = aperture_r[9]
      exptime_r          = aperture_r[10]
      sky_mag_r          = aperture_r[11]
      
      annu_rad_g         = aperture_g[0]
      annu_area_g        = aperture_g[1]
      annu_adu_g         = aperture_g[2]
      annu_adu_e_g       = aperture_g[3]
      annu_mag_g         = aperture_g[4]
      aper_rad_g         = aperture_g[5]
      aper_area_g        = aperture_g[6]
      aper_adu_g         = aperture_g[7]
      aper_adu_e_g       = aperture_g[8]
      aper_mag_g         = aperture_g[9]
      exptime_g          = aperture_g[10]
      sky_mag_g          = aperture_g[11]
      
      mag_array_r         = np.full(len(annu_rad_r), sky_mag_r)
      mag_array_g         = np.full(len(annu_rad_g), sky_mag_g)
      
      exptime_array_r     = np.full(len(annu_rad_r), exptime_r)
      exptime_array_g     = np.full(len(annu_rad_g), exptime_g)
      
      print('%.2f\t%.2f' % (sky_mag_r, sky_mag_g))
      
      radius_asec     = annu_rad_r * 0.25
      
      plot_name = plots_dir + 'PHOTOMETRY/PANSTARRS/MAP_ELLIPSE/%s_map_ellipse.pdf' % galaxies[i]
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
      plt.close()
      
      # =========== Save aperture/annulus g-/r-band profiles to file (1 per galaxy) =========== #
      table_str  = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
      os.system('rm -rf %s' % table_str)
      
      tdata = [annu_rad_r, annu_area_r, 
               annu_adu_r, annu_adu_e_r, annu_mag_r, 
               aper_rad_r, aper_area_r, 
               aper_adu_r, aper_adu_e_r, aper_mag_r, 
               exptime_array_r, mag_array_r,
               annu_rad_g, annu_area_g, 
               annu_adu_g, annu_adu_e_g, annu_mag_g, 
               aper_rad_g, aper_area_g, 
               aper_adu_g, aper_adu_e_g, aper_mag_g, 
               exptime_array_g, mag_array_g]
      
      tcols = ('RADIUS_ANNULUS_R', 'AREA_ANNULUS_R', 
               'ADU_ANNULUS_R', 'ADU_ANNULU_ERR_R', 'MAG_ANNULUS_R', 
               'RADIUS_APERTURE_R', 'AREA_APERTURE_R', 
               'ADU_APERTURE_R', 'ADU_APERTURE_ERR_R', 'MAG_APERTURE_R', 
               'EXPTIME_R', 'SKY_MAG_R',
               'RADIUS_ANNULUS_G', 'AREA_ANNULUS_G', 
               'ADU_ANNULUS_G', 'ADU_ANNULUS_ERR_G', 'MAG_ANNULUS_G', 
               'RADIUS_APERTURE_G', 'AREA_APERTURE_G', 
               'ADU_APERTURE_G', 'ADU_APERTURE_ERR_G', 'MAG_APERTURE_G', 
               'EXPTIME_G', 'SKY_MAG_G')
      
      t = Table(tdata, names=tcols)
      t.write(table_str, format = 'fits')
      




# ================================= #
# ==== Measure Photometry/Size ==== #
# ================================= #
if do_measure:
  do_save_table    = True
  counter          = 0
  mag_r25          = np.full(len(galaxies), np.nan)
  mag_r26          = np.full(len(galaxies), np.nan)
  mag_r27          = np.full(len(galaxies), np.nan)
  mag_g25          = np.full(len(galaxies), np.nan)
  mag_g26          = np.full(len(galaxies), np.nan)
  mag_g27          = np.full(len(galaxies), np.nan)
  radius25         = np.full(len(galaxies), np.nan)
  radius26         = np.full(len(galaxies), np.nan)
  radius27         = np.full(len(galaxies), np.nan)
  radius50         = np.full(len(galaxies), np.nan)
  radius90         = np.full(len(galaxies), np.nan)
  
  for i in range(len(galaxies)):
    profile_file  = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
    if i > -1 and opt_fit_flag[i] == 0: #and ~os.path.isfile(profile_file):
      print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
      if os.path.isfile(profile_file):
        counter += 1
        hdu_profile      = fits.open(profile_file, memmap=False)
        data_profile     = hdu_profile[1].data
        
        annu_rad_r         = data_profile['RADIUS_ANNULUS_R']
        annu_area_r        = data_profile['AREA_ANNULUS_R']
        annu_adu_r         = data_profile['ADU_ANNULUS_R']
        annu_mag_r         = data_profile['MAG_ANNULUS_R']
        aper_rad_r         = data_profile['RADIUS_APERTURE_R']
        aper_area_r        = data_profile['AREA_APERTURE_R']
        aper_adu_r         = data_profile['ADU_APERTURE_R']
        aper_mag_r         = data_profile['MAG_APERTURE_R']
        exptime_r          = data_profile['EXPTIME_R'][0]
        sky_bkgd_r         = data_profile['SKY_MAG_R'][0]
        
        annu_rad_g         = data_profile['RADIUS_ANNULUS_G']
        annu_area_g        = data_profile['AREA_ANNULUS_G']
        annu_adu_g         = data_profile['ADU_ANNULUS_G']
        annu_mag_g         = data_profile['MAG_ANNULUS_G']
        aper_rad_g         = data_profile['RADIUS_APERTURE_G']
        aper_area_g        = data_profile['AREA_APERTURE_G']
        aper_adu_g         = data_profile['ADU_APERTURE_G']
        aper_mag_g         = data_profile['MAG_APERTURE_G']
        exptime_g          = data_profile['EXPTIME_G'][0]
        sky_bkgd_g         = data_profile['SKY_MAG_G'][0]
        
        annu_rad_r_asec    = annu_rad_r * 0.25
        aper_rad_r_asec    = aper_rad_r * 0.25
        
        annu_mag_r[annu_mag_r > sky_bkgd_r] = sky_bkgd_r
        annu_mag_r[np.isnan(annu_mag_r)]    = sky_bkgd_r
        
        hdu_profile.close()
        # ======= Total r-band Magnitude (sb < 25) ======== #
        mag_r25[i], mag_g25[i], radius25[i] =  measure_mag_size(25., [exptime_r, exptime_g], 
                                                                [aper_adu_r, aper_adu_g], 
                                                                annu_mag_r, 
                                                                [annu_rad_r_asec, aper_rad_r_asec])
        
        # ======= Total r-band Magnitude (sb < 26) ======== #
        mag_r26[i], mag_g26[i], radius26[i] =  measure_mag_size(26., [exptime_r, exptime_g], 
                                                                [aper_adu_r, aper_adu_g], 
                                                                annu_mag_r, 
                                                                [annu_rad_r_asec, aper_rad_r_asec])
        
         # ======= Total r-band Magnitude (sb < 27) ======== #
        mag_r27[i], mag_g27[i], radius27[i] =  measure_mag_size(27., [exptime_r, exptime_g], 
                                                                [aper_adu_r, aper_adu_g], 
                                                                annu_mag_r, 
                                                                [annu_rad_r_asec, aper_rad_r_asec])
        
        # ======= R50 and R90 ======== #
        adu_r26         = np.interp(radius26[i], aper_rad_r_asec, aper_adu_r)
        bratio          = aper_adu_r / adu_r26 
        
        for k in range(len(aper_rad_r_asec) - 1):
          if bratio[k] < 0.5 and bratio[k + 1] > 0.5:
            radius50[i]     = interpolate_value(aper_rad_r_asec, bratio, 0.5, k)
            break
        
        for k in range(len(aper_rad_r_asec) - 1):
          if bratio[k] < 0.9 and bratio[k + 1] > 0.9:
            radius90[i]     = interpolate_value(aper_rad_r_asec, bratio, 0.9, k)
            break
        
        immask_fits     = panstarrs_dir + galaxies[i] +'/' + galaxies[i] + '_r_mask.fits'
        
        f2              = pyfits.open(immask_fits, memmap=False)
        hdr_r           = f2[0].header
        wcs_r           = WCS(hdr_r)
        f2.close()
        
        fig1 = plt.figure(1, figsize=(7, 3))
        radius_array = [radius25[i], radius26[i], radius27[i], radius50[i], radius90[i]]
        col_array    = ['peru', 'mediumvioletred', 'blue', 'cyan', 'magenta']
        lnsty_array  = ['-', '-', '-', ':', ':']
        sb_profile_radius_plot3(fig1, 1, 2, 1, annu_rad_r_asec, annu_mag_r, radius_array, col_array)
        ax1 = aplpy.FITSFigure(immask_fits, figure=fig1, subplot=(1, 2, 2), dimensions=(0,1))
        racen, deccen = wcs_r.all_pix2world(seg_x[i], seg_y[i], 0)
        pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
        if radius25[i] < 25:
          width, height = 10. * radius25[i] / 3600., 10. * radius25[i] / 3600.
          ax1.recenter(pos_ps.ra, pos_ps.dec, width=width, height=height)
        for ii in range(len(radius_array)):
          a = radius_array[ii] * 2.
          b = a * seg_ba[i]
          pa_ellipse = seg_pa[i] #* 180. / math.pi
          ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=a/3600., height=b/3600., 
                            angle=pa_ellipse, facecolor='none', edgecolor=col_array[ii], ls=lnsty_array[ii], 
                            zorder=2, linewidth=1.5, coords_frame='world')
        ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
        ax1.axis_labels.hide_x()
        ax1.axis_labels.hide_y()
        ax1.tick_labels.hide()
        
        plot_name = plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_PROFILES/%s_profiles.pdf' % galaxies[i]
        fig1.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
        plt.close()
        
        print('%.1f\t%.1f\t%.1f\t%.2f\t%.2f\t%.2f' % (radius25[i], 
                                                      radius50[i], 
                                                      radius90[i], 
                                                      mag_r25[i], 
                                                      mag_g25[i], 
                                                      mag_g25[i] - mag_r25[i]))
        
        rad_interp    = np.linspace(0,5,100)
        func_sd_interp = interp1d(np.array(aper_rad_r_asec/radius25[i]), 
                                  np.array(annu_mag_r), fill_value='extrapolate')
        sd_interp_one  = func_sd_interp(rad_interp)
      else:
        sd_interp_one    = np.empty(100)
        sd_interp_one[:] = np.nan
      if counter == 1:
        table_profile = Table([rad_interp, sd_interp_one], names=('RADIUS', galaxies[i]))
      else:
        col_to_add = Column(data=sd_interp_one, name=galaxies[i])
        table_profile.add_column(col_to_add)
  
  #table_str  = panstarrs_dir + 'PROFILES_BKGDSUB/all_normalised_rband25.fits'
  #table_profile.write(table_str, format = 'fits')
  
  # =========== Save PanSTARRS radii/magnitudes to file =========== #
  if do_save_table:
    table_str  = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
    os.system('rm -rf %s' % table_str)
    
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [seg_x, seg_y, seg_ba, seg_pa, 
               radius25, radius26, radius27, 
               radius50, radius90,
               mag_r25, mag_r26, mag_r27, 
               mag_g25, mag_g26, mag_g27]
    
    tcols_1 = ('SEG_X', 'SEG_Y', 'SEG_BA', 'SEG_PA',
               'RADIUS_R_ISO25', 'RADIUS_R_ISO26', 'RADIUS_R_ISO27', 
               'RADIUS_R_50', 'RADIUS_R_90', 
               'R_mag25', 'R_mag26', 'R_mag27', 
               'G_mag25', 'G_mag26', 'G_mag27')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
    
    save_table_function(table_str, tdata, tcols)











