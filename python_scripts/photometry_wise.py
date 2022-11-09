# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import os.path
from os import system
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, join
from astropy.visualization import simple_norm

from photutils.aperture import aperture_photometry
from photutils import EllipticalAperture, EllipticalAnnulus

from reproject import reproject_interp


#from functions_plotting import *
from functions_calculations import *


# ================================= #
# ==== Measure WISE Photometry ==== #
# ================================= #
def wise_magnitude_old(wise_dir, galaxy, aperture_params, allwise_params, jarrett_params, fignum):
    wise_psf_maj     = 1. * np.array([6.08, 6.84, 7.36, 11.99])
    wise_psf_min     = 1. * np.array([5.60, 6.12, 6.08, 11.65])
    
    adu_w1, adu_w1_err, mag_w1, mag_w1_err = np.nan, np.nan, np.nan, np.nan
    adu_w2, adu_w2_err, mag_w2, mag_w2_err = np.nan, np.nan, np.nan, np.nan
    adu_w3, adu_w3_err, mag_w3, mag_w3_err = np.nan, np.nan, np.nan, np.nan
    adu_w4, adu_w4_err, mag_w4, mag_w4_err = np.nan, np.nan, np.nan, np.nan
    
    ps_aper_x      = aperture_params[0]
    ps_aper_y      = aperture_params[1]
    ps_aper_r      = aperture_params[2]
    ps_aper_ba     = aperture_params[3]
    ps_aper_pa     = aperture_params[4]
    
    if len(jarrett_params) > 1:
      w1r_jarrett    = jarrett_params[0]
      ba_jarrett     = jarrett_params[1]
      pa_jarrett     = jarrett_params[2]
    
    radius_aw      = np.array([allwise_params[0], allwise_params[1], allwise_params[2], allwise_params[3]])
    ba_aw          = np.array([allwise_params[4], allwise_params[5], allwise_params[6], allwise_params[7]])
    pa_aw          = np.array([allwise_params[8], allwise_params[9], allwise_params[10], allwise_params[11]])
    
    for band in range(4):
      fits_dir        = wise_dir + galaxy +'/'
      im_fits         = galaxy + '-w%i.fits' % (band + 1)
      if os.path.isfile(fits_dir + im_fits):
        im_mir          = galaxy + '_w%i.mir' % (band + 1)
        mask_fits       = galaxy + '_r_mask_2.fits'
        if os.path.isfile(panstarrs_dir + galaxy +'/' + mask_fits):
          mask_mir        = galaxy + '_r_mask_2.mir'
          regrid_fits     = galaxy + '_r_mask_2.regrid.fits'
          regrid_mir      = galaxy + '_r_mask_2.regrid.mir'
        else:
          mask_fits       = galaxy + '_r_mask.fits'
          mask_mir        = galaxy + '_r_mask.mir'
          regrid_fits     = galaxy + '_r_mask.regrid.fits'
          regrid_mir      = galaxy + '_r_mask.regrid.mir'
        os.chdir(fits_dir)
        if band == 0:
          os.system('cp ../../PANSTARRS/%s/%s .' % (galaxy, mask_fits))
          os.system('fits in=%s op=xyin out=%s' % (mask_fits, mask_mir))
        os.system('fits in=%s op=xyin out=%s' % (im_fits, im_mir))
        os.system('regrid in=%s out=%s tin=%s' % (mask_mir, regrid_mir, im_mir))
        os.system('fits in=%s op=xyout out=%s' % (regrid_mir, regrid_fits))
        os.system('rm -rf %s' % regrid_mir)
        os.system('rm -rf %s' % im_mir)
        f1              = pyfits.open(im_fits)
        data, hdr       = f1[0].data, f1[0].header
        wcs             = WCS(hdr)
        #magzp           = hdr['MAGZP']
        magzp           = hdr['ZEROPT']
        naxis1          = hdr['NAXIS1']
        naxis2          = hdr['NAXIS2']
        #background      = hdr['MEDINT']
        pix_scale       = np.abs(hdr['CD1_1']) * 3600.
        
        f2              = pyfits.open(regrid_fits)
        data_r, hdr_r   = f2[0].data, f2[0].header
        
        f3              = pyfits.open(mask_fits)
        hdr_hres        = f3[0].header
        wcs_hres        = WCS(hdr_hres)
        
        ra_cen, dec_cen = wcs_hres.all_pix2world(ps_aper_x, ps_aper_y, 0)
        position        = SkyCoord(ra_cen*u.deg, dec_cen*u.deg, frame='icrs')
        x_pix, y_pix    = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
        
        data[np.isnan(data_r)] = np.nan
        
        data             = np.ma.masked_invalid(data)
        
        majax_conv       = np.sqrt(ps_aper_r**2 + wise_psf_maj[band]**2)
        minax_conv       = np.sqrt((ps_aper_r * ps_aper_ba)**2 + wise_psf_maj[band]**2)
        
        a_ps             = ps_aper_r / pix_scale
        b_ps             = ps_aper_r * ps_aper_ba / pix_scale
        #aperture_ps      = EllipticalAperture((x_pix, y_pix), a_ps, b_ps, theta=ps_aper_pa)#*math.pi/180)
                
        a_aw             = radius_aw[band] / pix_scale
        b_aw             = radius_aw[band] * ba_aw[band] / pix_scale
        aperture_aw      = EllipticalAperture((x_pix, y_pix), a_aw, b_aw, theta=pa_aw[band]*math.pi/180)
        
        if len(jarrett_params) > 1:
          a_jarrett        = w1r_jarrett / pix_scale
          b_jarrett        = w1r_jarrett * ba_jarrett / pix_scale
          aperture_jarrett = EllipticalAperture((x_pix, y_pix), a_jarrett, b_jarrett, theta=pa_jarrett*math.pi/180)
            
        a                = majax_conv / pix_scale  #r_aper[i] / pix_scale
        b                = minax_conv / pix_scale  #r_aper[i] * ba_aper[i] / pix_scale
        aperture         = EllipticalAperture((x_pix, y_pix), a, b, theta=ps_aper_pa)#*math.pi/180)
        aperture_annulus = EllipticalAnnulus((x_pix, y_pix), a_in=1.5*a, a_out=2.5*a, 
                                             b_out=2.5*b, b_in=1.5*b, theta=ps_aper_pa)
            
        annulus_masks = aperture_annulus.to_mask(method='center')
        annulus_data = annulus_masks.multiply(data)
        mask = annulus_masks.data
        annulus_data_1d = annulus_data[mask > 0]
        mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d)
        
        error = std_sigclip * data / data
        
        phot_table       = aperture_photometry(data, aperture, error = error)
        
        aperture_mask    = aperture.to_mask(method='center')
        aperture_data    = aperture_mask.multiply(data)
        
        
        #fignum.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
        ##norm = simple_norm(data, 'asinh', asinh_a=0.0001)
        #norm = simple_norm(data, 'sqrt', percent=98)
        #ax1 = fignum.add_subplot(1, 4, band+1, facecolor = 'w')
        #ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)
        #ap_patches = aperture_ps.plot(color='red', lw=2, label='Photometry aperture')
        #ap_patches = aperture_aw.plot(color='magenta', lw=2, label='Photometry aperture')
        #if len(jarrett_params) > 1:
          #ap_patches = aperture_jarrett.plot(color='purple', lw=2, label='Photometry aperture')
        #ap_patches = aperture.plot(color='peru', lw=2, label='Photometry aperture')
        #if a_ps > 2:
          #ax1.set_xlim(x_pix - 3.*a_ps, x_pix + 3.*a_ps)
          #ax1.set_ylim(y_pix - 3.*a_ps, y_pix + 3.*a_ps)
        #else:
          #ax1.set_xlim(x_pix - 20.*a_ps, x_pix + 20.*a_ps)
          #ax1.set_ylim(y_pix - 20.*a_ps, y_pix + 20.*a_ps)
        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        adu     = phot_table['aperture_sum'][0]
        adu_err = phot_table['aperture_sum_err'][0]
            
        if band == 0:
          adu_w1         = adu
          adu_w1_err     = adu_err
          mag_w1         = magzp - 2.5*np.log10(adu)
          mag_w1_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 1:
          adu_w2         = adu
          adu_w2_err     = adu_err
          mag_w2         = magzp - 2.5*np.log10(adu)
          mag_w2_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 2:
          adu_w3         = adu
          adu_w3_err     = adu_err
          mag_w3         = magzp - 2.5*np.log10(adu)
          mag_w3_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 3:
          adu_w4         = adu
          adu_w4_err     = adu_err
          mag_w4         = magzp - 2.5*np.log10(adu)
          mag_w4_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
    
    return(adu_w1, adu_w1_err, mag_w1, mag_w1_err, adu_w2, adu_w2_err, mag_w2, mag_w2_err,
           adu_w3, adu_w3_err, mag_w3, mag_w3_err, adu_w4, adu_w4_err, mag_w4, mag_w4_err)


# ================================= #
# ==== Measure WISE Photometry ==== #
# ================================= #
def wise_magnitude(wise_dir, galaxy, aperture_params, fignum):
    '''
    wise_dir:         WISE galaxy directory
    
    galaxy:           Galaxy name
    
    aperture_params:  Parameters for annuli/apertures
    
                      aperture_params[0]: PanSTARRS segment x position [pixel]
    
                      aperture_params[1]: PanSTARRS segment y position [pixel]
    
                      aperture_params[2]: PanSTARRS radius [arcsec]
    
                      aperture_params[3]: PanSTARRS segment axis ratio b/a
    
                      aperture_params[4]: PanSTARRS segment position angle [degrees]
    
    fignum:           Figure number
    
    Returns: W1-band flux (ADU), flux error (ADU), magnitude, magnitude error,
             
             W2-band flux (ADU), flux error (ADU), magnitude, magnitude error,
             
             W3-band flux (ADU), flux error (ADU), magnitude, magnitude error,
             
             W4-band flux (ADU), flux error (ADU), magnitude, magnitude error
    '''
    wise_psf_maj     = 1. * np.array([6.08, 6.84, 7.36, 11.99])
    wise_psf_min     = 1. * np.array([5.60, 6.12, 6.08, 11.65])
    
    adu_w1, adu_w1_err, mag_w1, mag_w1_err = np.nan, np.nan, np.nan, np.nan
    adu_w2, adu_w2_err, mag_w2, mag_w2_err = np.nan, np.nan, np.nan, np.nan
    adu_w3, adu_w3_err, mag_w3, mag_w3_err = np.nan, np.nan, np.nan, np.nan
    adu_w4, adu_w4_err, mag_w4, mag_w4_err = np.nan, np.nan, np.nan, np.nan
    
    for band in range(4):
      fits_dir        = wise_dir + galaxy +'/'
      im_fits         = galaxy + '-w%i.fits' % (band + 1)
      if os.path.isfile(fits_dir + im_fits):
        mask_fits       = galaxy + '_r_mask.fits'
        os.chdir(fits_dir)
        if band == 0:
          os.system('cp ../../PANSTARRS/%s/%s .' % (galaxy, mask_fits))
        f1              = pyfits.open(im_fits)
        data, hdr       = f1[0].data, f1[0].header
        wcs             = WCS(hdr)
        #magzp           = hdr['MAGZP']
        magzp           = hdr['ZEROPT']
        naxis1          = hdr['NAXIS1']
        naxis2          = hdr['NAXIS2']
        #background      = hdr['MEDINT']
        pix_scale       = np.abs(hdr['CD1_1']) * 3600.
        
        ps_aper_x      = aperture_params[0]
        ps_aper_y      = aperture_params[1]
        ps_aper_r      = aperture_params[2] / pix_scale
        ps_aper_ba     = aperture_params[3]
        ps_aper_pa     = aperture_params[4] * math.pi / 180.
        
        if np.isnan(ps_aper_r):
          ps_aper_r = 1
        
        f2              = pyfits.open(mask_fits, memmap=False)
        hdr_hres        = f2[0].header
        wcs_hres        = WCS(hdr_hres)
        
        ra_cen, dec_cen = wcs_hres.all_pix2world(ps_aper_x, ps_aper_y, 0)
        position        = SkyCoord(ra_cen*u.deg, dec_cen*u.deg, frame='icrs')
        x_pix, y_pix    = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
        
        data_r_reproject, footprint = reproject_interp(f2, hdr)
      
        data[np.isnan(data_r_reproject)] = np.nan
        
        data             = np.ma.masked_invalid(data)
        
        majax_conv       = np.sqrt(ps_aper_r**2 + (wise_psf_maj[band] / pix_scale)**2)
        minax_conv       = np.sqrt((ps_aper_r * ps_aper_ba)**2 + (wise_psf_maj[band] / pix_scale)**2)
        
        a_ps             = ps_aper_r #/ pix_scale
        b_ps             = ps_aper_r * ps_aper_ba #/ pix_scale
        aperture_ps      = EllipticalAperture((x_pix, y_pix), a_ps, b_ps, theta=ps_aper_pa)#*math.pi/180)
                
            
        a                = majax_conv #/ pix_scale  #r_aper[i] / pix_scale
        b                = minax_conv #/ pix_scale  #r_aper[i] * ba_aper[i] / pix_scale
        aperture         = EllipticalAperture((x_pix, y_pix), a, b, theta=ps_aper_pa)#*math.pi/180)
        aperture_annulus = EllipticalAnnulus((x_pix, y_pix), a_in=1.5*a, a_out=2.5*a, 
                                             b_out=2.5*b, b_in=1.5*b, theta=ps_aper_pa)
            
        annulus_masks = aperture_annulus.to_mask(method='center')
        annulus_data = annulus_masks.multiply(data)
        mask = annulus_masks.data
        annulus_data_1d = annulus_data[mask > 0]
        mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d)
        
        error = std_sigclip * data / data
        
        phot_table       = aperture_photometry(data, aperture, error = error)
        
        aperture_mask    = aperture.to_mask(method='center')
        aperture_data    = aperture_mask.multiply(data)
        
        
        fignum.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
        norm = simple_norm(data, 'sqrt', percent=98)
        ax1 = fignum.add_subplot(1, 4, band+1, facecolor = 'w')
        ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)
        ap_patches = aperture_ps.plot(color='red', lw=2, label='Photometry aperture')
        ap_patches = aperture.plot(color='peru', lw=2, label='Photometry aperture')
        if a_ps > 2:
          ax1.set_xlim(x_pix - 3.*a_ps, x_pix + 3.*a_ps)
          ax1.set_ylim(y_pix - 3.*a_ps, y_pix + 3.*a_ps)
        else:
          ax1.set_xlim(x_pix - 20.*a_ps, x_pix + 20.*a_ps)
          ax1.set_ylim(y_pix - 20.*a_ps, y_pix + 20.*a_ps)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        adu     = phot_table['aperture_sum'][0]
        adu_err = phot_table['aperture_sum_err'][0]
            
        if band == 0:
          adu_w1         = adu
          adu_w1_err     = adu_err
          mag_w1         = magzp - 2.5*np.log10(adu)
          mag_w1_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 1:
          adu_w2         = adu
          adu_w2_err     = adu_err
          mag_w2         = magzp - 2.5*np.log10(adu)
          mag_w2_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 2:
          adu_w3         = adu
          adu_w3_err     = adu_err
          mag_w3         = magzp - 2.5*np.log10(adu)
          mag_w3_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
        if band == 3:
          adu_w4         = adu
          adu_w4_err     = adu_err
          mag_w4         = magzp - 2.5*np.log10(adu)
          mag_w4_err     = 2.5 * (np.log10(adu + adu_err) - np.log10(adu))
    
    return(adu_w1, adu_w1_err, mag_w1, mag_w1_err, adu_w2, adu_w2_err, mag_w2, mag_w2_err,
           adu_w3, adu_w3_err, mag_w3, mag_w3_err, adu_w4, adu_w4_err, mag_w4, mag_w4_err)



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
# ==== Remove Fits Header Cards === #
# ================================= #
def remove_hdr_cards(fits_file):
    f1     = pyfits.open(fits_file, mode='update')
    data, hdr  = f1[0].data, f1[0].header
    c_hdr_cards = ['CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4', 'PC01_03', 'PC01_04', 'PC02_03', 'PC02_04', 'PC03_01', 'PC03_02', 'PC03_03', 'PC03_04', 'PC04_01', 'PC04_02', 'PC04_03', 'PC04_04', 'PC1_3', 'PC1_4', 'PC2_3', 'PC2_4', 'PC3_1', 'PC3_2', 'PC3_3', 'PC3_4', 'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4']
    for i in range(len(c_hdr_cards)):
      if c_hdr_cards[i] in hdr:
        del hdr[c_hdr_cards[i]]
    f1.flush()
    f1.close()


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
have_optical             = True          # True to open *_panstarrs_photometry.fits if exists

do_measure               = True          # True to measure magnitudes


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
if ~have_optical and ~have_segments:
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
  
  join1          = join(data_sofia, data_flags, join_type='left', keys=['name']) 
  
  for i in range(1, len(join1.columns)):
    if '_1' in join1.columns[i].name:
      join1.rename_column(join1.columns[i].name, data_sofia.columns.names[i])
  
  for i in range(len(join1.columns)-1, 0, -1):
    if '_2' in join1.columns[i].name:
      join1.remove_column(join1.columns[i].name)
  
  data_join      = join(join1, data_gaussian, join_type='left')

if have_segments:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
  fits_gaussian  = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
  fits_segments  = parameter_dir + '%s_panstarrs_segments_deblend.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_flags      = fits.open(fits_flags)
  data_flags     = hdu_flags[1].data
  
  hdu_gaussian   = fits.open(fits_gaussian)
  data_gaussian  = hdu_gaussian[1].data
  
  hdu_segments   = fits.open(fits_segments)
  data_segments  = hdu_segments[1].data
  
  join1          = join(data_sofia, data_flags, join_type='left', keys=['name']) 
  
  for i in range(1, len(join1.columns)):
    if '_1' in join1.columns[i].name:
      join1.rename_column(join1.columns[i].name, data_sofia.columns.names[i])
  
  for i in range(len(join1.columns)-1, 0, -1):
    if '_2' in join1.columns[i].name:
      join1.remove_column(join1.columns[i].name)
  
  join2          = join(join1, data_gaussian, join_type='left')
  data_join      = join(join2, data_segments, join_type='left')

if have_optical:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
  fits_gaussian  = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
  fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_flags      = fits.open(fits_flags)
  data_flags     = hdu_flags[1].data
  
  hdu_gaussian   = fits.open(fits_gaussian)
  data_gaussian  = hdu_gaussian[1].data
  
  hdu_panstarrs  = fits.open(fits_panstarrs)
  data_panstarrs = hdu_panstarrs[1].data
  
  join1          = join(data_sofia, data_flags, join_type='left', keys=['name']) 
  
  for i in range(1, len(join1.columns)):
    if '_1' in join1.columns[i].name:
      join1.rename_column(join1.columns[i].name, data_sofia.columns.names[i])
  
  for i in range(len(join1.columns)-1, 0, -1):
    if '_2' in join1.columns[i].name:
      join1.remove_column(join1.columns[i].name)
  
  join2          = join(join1, data_gaussian, join_type='left')
  data_join      = join(join2, data_panstarrs, join_type='left')


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
  
  if have_optical:
    #fitness      = data_mask['FIT']
    ps_x         = data_mask['SEG_X']
    ps_y         = data_mask['SEG_Y']
    ps_ba        = data_mask['SEG_BA']
    ps_pa        = data_mask['SEG_PA']
    ps_radius25  = data_mask['RADIUS_R_ISO25']
    ps_radius50  = data_mask['RADIUS_R_50']
    #fitness      = data_mask['FIT']

print(len(galaxies))





# ================================= #
# ==== Measure WISE Photometry ==== #
# ================================= #
if do_measure:
  do_save_table    = True
  mag_w1           = np.full(len(galaxies), np.nan)
  mag_w2           = np.full(len(galaxies), np.nan)
  mag_w3           = np.full(len(galaxies), np.nan)
  mag_w4           = np.full(len(galaxies), np.nan)
  mag_w1_err       = np.full(len(galaxies), np.nan)
  mag_w2_err       = np.full(len(galaxies), np.nan)
  mag_w3_err       = np.full(len(galaxies), np.nan)
  mag_w4_err       = np.full(len(galaxies), np.nan)
  adu_w1           = np.full(len(galaxies), np.nan)
  adu_w2           = np.full(len(galaxies), np.nan)
  adu_w3           = np.full(len(galaxies), np.nan)
  adu_w4           = np.full(len(galaxies), np.nan)
  adu_w1_err       = np.full(len(galaxies), np.nan)
  adu_w2_err       = np.full(len(galaxies), np.nan)
  adu_w3_err       = np.full(len(galaxies), np.nan)
  adu_w4_err       = np.full(len(galaxies), np.nan)
  
  for i in range(len(galaxies)):
    #if galaxies[i] == 'J101049-302538' or galaxies[i] == 'J101434-274133':
    if i > -1:
      print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
      profile_file    = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
      if os.path.isfile(profile_file):
        # ======= Total WISE Magnitudes (aperture from r-band) ======== #
        fig = plt.figure(figsize=(8,2))
        
        ps_aperture_params = [ps_x[i], ps_y[i], ps_radius25[i], ps_ba[i], ps_pa[i]]
        
        wise_phot_params   = wise_magnitude(wise_dir, galaxies[i], ps_aperture_params, fig)
        
        adu_w1[i], adu_w1_err[i] = wise_phot_params[0], wise_phot_params[1]
        mag_w1[i], mag_w1_err[i] = wise_phot_params[2], wise_phot_params[3]
        adu_w2[i], adu_w2_err[i] = wise_phot_params[4], wise_phot_params[5]
        mag_w2[i], mag_w2_err[i] = wise_phot_params[6], wise_phot_params[7]
        adu_w3[i], adu_w3_err[i] = wise_phot_params[8], wise_phot_params[9]
        mag_w3[i], mag_w3_err[i] = wise_phot_params[10], wise_phot_params[11]
        adu_w4[i], adu_w4_err[i] = wise_phot_params[12], wise_phot_params[13]
        mag_w4[i], mag_w4_err[i] = wise_phot_params[14], wise_phot_params[15]
                
        plot_name = plots_dir + 'PHOTOMETRY/WISE/%s_aperture.pdf' % galaxies[i]
        plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
        plt.close()
        
        
  # =========== Save WISE magnitudes to file =========== #
  if do_save_table:
    table_str  = parameter_dir + '%s_wise_photometry.fits' % team_release
    os.system('rm -rf %s' % table_str)
    
    snr_w1 = adu_w1 / adu_w1_err
    snr_w2 = adu_w2 / adu_w2_err
    snr_w3 = adu_w3 / adu_w3_err
    snr_w4 = adu_w4 / adu_w4_err
    
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [adu_w1, adu_w1_err, adu_w2, adu_w2_err,
               adu_w3, adu_w3_err, adu_w4, adu_w4_err,
               mag_w1, mag_w1_err, mag_w2, mag_w2_err, 
               mag_w3, mag_w3_err, mag_w4, mag_w4_err,
               snr_w1, snr_w2, snr_w3, snr_w4]
    
    tcols_1 = ('ADU_W1', 'ADU_W1_SIGMA', 'ADU_W2', 'ADU_W2_SIGMA', 
               'ADU_W3', 'ADU_W3_SIGMA', 'ADU_W4', 'ADU_W4_SIGMA', 
               'W1', 'W1_SIGMA', 'W2', 'W2_SIGMA', 
               'W3', 'W3_SIGMA', 'W4', 'W4_SIGMA',
               'SNR_W1', 'SNR_W2', 'SNR_W3', 'SNR_W4')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
  
    t = Table(tdata, names=tcols)
    t.write(table_str, format = 'fits')
  















