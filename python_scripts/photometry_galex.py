# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, join
from astropy.visualization import simple_norm
import aplpy

from photutils.aperture import aperture_photometry
from photutils import EllipticalAperture, EllipticalAnnulus

from reproject import reproject_interp


#from functions_plotting import *
from functions_calculations import *


# ================================= #
# == Plot GALEX Curve of Growth === #
# ================================= #
def curve_of_growth_plot(fig_num, subfig, data, axlbls, radius, txtstr, txtpos):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    ax1.set_xlabel(axlbls[0])
    ax1.set_ylabel(axlbls[1])
    plt.scatter(data[0], data[1], color='darkblue', s=5, edgecolor='darkblue', facecolor='none')
    plt.scatter(data[0], data[2], color='peru', s=5, edgecolor='peru', facecolor='none')
    plt.axvline(radius[0], color = 'black', linestyle = '--', linewidth = 1, zorder = 0)
    plt.axvline(radius[1], color = 'grey', linestyle = ':', linewidth = 1, zorder = 0)
    #if txtpos == 'upper':
      #plt.text(0.25, 0.9, txtstr, transform=ax1.transAxes)
      #plt.axhline(23.5, color = 'grey', linestyle = '--', linewidth=1, zorder = 0)
    #if txtpos == 'lower':
      #plt.text(0.25, 0.1, txtstr, transform=ax1.transAxes)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')


# ================================ #
# ========= Scatter Plot ========= #
# ================================ #
def scat_mag_derivative_plot(fig_num, subfig, data, fit, txtstr, xlbl, ylbl, col, marker):
    matplotlib.rcParams.update({'font.size': 10})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(subfig[0], subfig[1], subfig[2], facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    #ax1.set_xlabel(xlbl)
    plt.text(0.05, 0.9, txtstr, transform=ax1.transAxes, fontsize=9)
    ax1.set_xlabel(xlbl)
    if subfig[2] == 1:
      ax1.set_ylabel(ylbl)
    ax1.set_ylim(np.nanmean(data[1])-0.5, np.nanmean(data[1])+0.5)
    ax1.scatter(data[0], data[1], color=col[0], marker=marker, edgecolor=col[0], facecolor='none', s=5, zorder=2)
    ax1.plot(fit[0], fit[1], color=col[1], linewidth = 1, zorder=1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

# ================================= #
# === Measure GALEX Photometry ==== #
# ================================= #
def galex_surface_brightness_rmask(galex_dir, galaxy, aperture_params, band, rmax_pix, fignum):
    '''
    galex_dir:        GALEX galaxy directory
    
    galaxy:           Galaxy name
    
    aperture_params:  Parameters for annuli/apertures
    
                      aperture_params[0]: PanSTARRS segment x position [pixel]
    
                      aperture_params[1]: PanSTARRS segment y position [pixel]
    
                      aperture_params[2]: PanSTARRS radius [arcsec]
    
                      aperture_params[3]: PanSTARRS segment axis ratio b/a
    
                      aperture_params[4]: PanSTARRS segment position angle [degrees]
    
    band:             GALEX band (NUV, FUV)
    
    rmax_pix:         The maximum radius at which to fit apertures/annuli
    
    fignum:           Figure number
    
    Returns: Annulus radius, area, 
             background subtracted flux [ADU], background subtracted magnitude, 
             flux [ADU], magnitude,
             Aperture radius, area, 
             background subtracted flux [ADU], flux error [ADU], flux difference [ADU]
             background subtracted magnitude, magnitude difference,
             flux [ADU], flux difference [ADU], magnitude, magnitude difference
    '''
    if band == 'nuv':
      galex_psf = 11.99
    if band == 'fuv':
      galex_psf = 11.99
    #galex_psf   = np.array([11.99, 11.99]) # 4.9, 4.2
    #galex_bands = ['nuv', 'fuv']
    adu_nuv, adu_nuv_err, mag_nuv, mag_nuv_err = np.nan, np.nan, np.nan, np.nan
    adu_fuv, adu_fuv_err, mag_fuv, mag_fuv_err = np.nan, np.nan, np.nan, np.nan
    fits_dir        = galex_dir + galaxy +'/'
    im_fits         = galaxy + '-%s.fits' % band
    
    #print(fits_dir)
    #print(im_fits)
    
    if os.path.isfile(fits_dir + im_fits):
      #im_mir          = galaxy + '_%s.mir' % band
      mask_fits       = galaxy + '_r_mask.fits'
      #conv_fits       = galaxy + '_r_mask_convolve.fits'
      
      os.chdir(fits_dir)
      if ~os.path.isfile(mask_fits):
        os.system('cp ../../PANSTARRS/%s/%s .' % (galaxy, mask_fits))
      
      f1              = pyfits.open(im_fits, memmap=False)
      data, hdr       = f1[0].data, f1[0].header
      wcs             = WCS(hdr)
      wcs             = WCS(hdr)
      magzp           = hdr['ZEROPT']
      naxis1          = hdr['NAXIS1']
      naxis2          = hdr['NAXIS2']
      pix_scale       = np.abs(hdr['CD1_1']) * 3600.
      npix_f1         = hdr['NAXIS1']
            
      f2              = pyfits.open(mask_fits, memmap=False)
      hdr_hres        = f2[0].header
      wcs_hres        = WCS(hdr_hres)
      
      ps_aper_x      = aperture_params[0]
      ps_aper_y      = aperture_params[1]
      ps_aper_r      = aperture_params[2] / pix_scale
      ps_aper_ba     = aperture_params[3]
      ps_aper_pa     = aperture_params[4] * math.pi / 180.
      
      ra_cen, dec_cen = wcs_hres.all_pix2world(ps_aper_x, ps_aper_y, 0)
      position        = SkyCoord(ra_cen*u.deg, dec_cen*u.deg, frame='icrs')
      x_pix, y_pix    = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
      
      data_r_reproject, footprint = reproject_interp(f2, hdr)
      
      data[np.isnan(data_r_reproject)] = np.nan
      
      data             = np.ma.masked_invalid(data)
      
      majax_conv       = np.sqrt(ps_aper_r**2 + (galex_psf / pix_scale)**2)
      minax_conv       = np.sqrt((ps_aper_r * ps_aper_ba)**2 + (galex_psf / pix_scale)**2)
      
      aperture_list    = []
      annulus_list     = []
      radius_list      = []
      area_annul_list  = []
      area_aper_list   = []
      
      rmax_pix         = 3. * ps_aper_r #radius_max #/ 1.25
      
      if rmax_pix < 30:
        rmax_pix = rmax_pix * 3.
      
      if np.isnan(rmax_pix):
        rmax_pix = 150
      
      scale_list       = np.arange(0, rmax_pix, 2)
      
      #a_ps             = ps_aper_r / pix_scale
      #b_ps             = ps_aper_r * ps_aper_ba / pix_scale
      
      a_ps             = majax_conv
      b_ps             = minax_conv
      
      radius_aperture    = scale_list[1:-1]
      radius_annulus     = scale_list[2:]
      
      for i in range(1, len(scale_list)-1):
        aperture         = EllipticalAperture((x_pix, y_pix), 
                                              scale_list[i], 
                                              scale_list[i] * ps_aper_ba, 
                                              theta=ps_aper_pa)
        aperture_annulus = EllipticalAnnulus((x_pix, y_pix), 
                                             a_in=scale_list[i], 
                                             a_out=scale_list[i+1], 
                                             b_out=scale_list[i+1] * ps_aper_ba, 
                                             b_in=scale_list[i] * ps_aper_ba, 
                                             theta=ps_aper_pa)
        aperture_list.append(aperture)
        annulus_list.append(aperture_annulus)
        radius_list.append(scale_list[i]*a_ps)
        area_annul_list.append(aperture_annulus.area)
        area_aper_list.append(aperture.area)
      
      phot_table_annulus       = aperture_photometry(data, annulus_list)#, error = error)
      
      mean_annulus = []
      mean_ids     = []
      
      for i in range(len(annulus_list)):
        if radius_annulus[i] > rmax_pix / 2.:
          annulus_string = string_name = 'aperture_sum_%s' % (i)
          mean_annulus.append(phot_table_annulus[annulus_string][0] / area_annul_list[i])
          mean_ids.append(i)
      
      mean_annulus      = np.array(mean_annulus)
      mean_annulus      = mean_annulus[mean_annulus < 0.01]
      
      
      annulus_masks = aperture_annulus.to_mask(method='center')
      annulus_data = annulus_masks.multiply(data)
      mask = annulus_masks.data
      annulus_data_1d = annulus_data[mask > 0]
      mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d)
      
      error = std_sigclip * data / data
      
      #phot_table       = aperture_photometry(data, aperture, error = error)
      
      print(np.round(np.nanmean(mean_annulus), 5), np.round(np.nanmedian(mean_annulus),5))
      
      phot_table_bksub  = aperture_photometry(data - np.nanmedian(mean_annulus), aperture_list, error = error)
      
      phot_table_bk     = aperture_photometry(data, aperture_list, error = error)
      
      #phot_table_annulus  = aperture_photometry(data - np.nanmedian(mean_annulus), annulus_list, error = error)
      
      aperture_adu_bksub_list     = []
      aperture_adu_bk_list        = []
      annulus_adu_list            = []
      aperture_adu_bksub_err_list = []
      #annulus_adu_err_list        = []
      
      for i in range(len(annulus_list)):
        string_name1 = 'aperture_sum_%s' % (i)
        string_name2 = 'aperture_sum_err_%s' % (i)
        aperture_adu_bksub_list.append(phot_table_bksub[string_name1][0])
        aperture_adu_bk_list.append(phot_table_bk[string_name1][0])
        annulus_adu_list.append(phot_table_annulus[string_name1][0])
        aperture_adu_bksub_err_list.append(phot_table_bksub[string_name2][0])
        #annulus_adu_err_list.append(phot_table_annulus[string_name2][0])
        
      fignum.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
      norm = simple_norm(data, 'sqrt', percent=98)
      if band == 'nuv':
        ax1 = fig.add_subplot(1, 2, 1, facecolor = 'w')
      if band == 'fuv':
        ax1 = fig.add_subplot(1, 2, 2, facecolor = 'w')
      ax1.imshow(data, origin='lower', cmap='Blues', norm=norm)
      for i in range(len(aperture_list)):
        if (i/5).is_integer():
          ap_patches = aperture_list[i].plot(color='peru', lw=0.75, label='Photometry aperture')
      ax1.set_xlim(int(x_pix - naxis1 / 2), int(x_pix + naxis1 / 2))
      ax1.set_ylim(int(y_pix - naxis2 / 2), int(y_pix + naxis2 / 2))
      ap_patches = aperture_list[mean_ids[len(mean_ids) - 1]].plot(color='black', 
                                                                 lw=0.75, 
                                                                 label='Photometry aperture')
      ap_patches = aperture_list[mean_ids[0]].plot(color='black', 
                                                  lw=0.75, 
                                                  label='Photometry aperture')
      
      plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    annulus_diff = np.diff(np.array(annulus_adu_list),n=1)
    annulus_diff = np.concatenate((annulus_diff, [np.nan]))
    
    aperture_bksub_diff = np.diff(np.array(aperture_adu_bksub_list),n=1)
    aperture_bksub_diff = np.concatenate((aperture_bksub_diff, [np.nan]))
    
    mag_bksub_diff = np.abs(np.diff(magzp - 2.5*np.log10(np.array(aperture_adu_bksub_list)),n=1))
    mag_bksub_diff = np.concatenate((mag_bksub_diff, [np.nan]))
    
    aperture_bk_diff = np.diff(np.array(aperture_adu_bk_list),n=1)
    aperture_bk_diff = np.concatenate((aperture_bk_diff, [np.nan]))
    
    mag_bk_diff = np.abs(np.diff(magzp - 2.5*np.log10(np.array(aperture_adu_bk_list)),n=1))
    mag_bk_diff = np.concatenate((mag_bk_diff, [np.nan]))
    
    f1.close()
    f2.close()
    
    return(np.array(radius_annulus), 
           np.array(area_annul_list)/(pix_scale*pix_scale), 
           np.array(annulus_adu_list),
           magzp - 2.5*np.log10(np.array(annulus_adu_list)/np.array(area_annul_list)/(pix_scale*pix_scale)),
           np.array(annulus_adu_list - np.nanmean(mean_annulus)),
           magzp - 2.5*np.log10(np.array(annulus_adu_list - np.nanmean(mean_annulus))),
           np.array(radius_aperture), 
           np.array(area_aper_list)/(pix_scale*pix_scale), 
           np.array(aperture_adu_bksub_list), 
           np.array(aperture_adu_bksub_err_list),
           aperture_bksub_diff,
           magzp - 2.5*np.log10(np.array(aperture_adu_bksub_list)),
           mag_bksub_diff,
           np.array(aperture_adu_bk_list), 
           aperture_bk_diff,
           magzp - 2.5*np.log10(np.array(aperture_adu_bk_list)),
           mag_bk_diff)



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
have_optical             = False         # True to open *_panstarrs_photometry.fits if exists

# ++++ ONLY RUN ONE AT A TIME +++++ #
do_fit_phot              = False         # True to fit annuli/apertures to NUV-band image
do_measure               = False         # True to measure magnitudes/radii



# ================================= #
# ==== Specify Phase + Release ==== #
# ================================= #
tr_i                     = 1

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
  
  join1          = join(data_sofia, data_flags, join_type='left')
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
  
  join1          = join(data_sofia, data_flags, join_type='left')
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
  
  join1          = join(data_sofia, data_flags, join_type='left')
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
# === Measure GALEX Photometry ==== #
# ================================= #
if do_fit_phot:
  for i in range(len(galaxies)):
    if galaxies[i] == 'J104059-270456': #'J123742-033505': #
    #if i > -1:
      print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
      profile_file    = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
      if os.path.isfile(profile_file):
        fig            = plt.figure(figsize=(6,6))
        galex_bands    = 'nuv'
        band_fits_file = galex_dir + '%s/%s-%s.fits' % (galaxies[i], galaxies[i], galex_bands)
        
        if os.path.isfile(band_fits_file):
          segment_parameters = [ps_x[i], ps_y[i], ps_radius25[i], ps_ba[i], ps_pa[i]]
    
          radius_max = 3. * ps_radius25[i]
          
          # =========== Fit apertures/annuli to NUV-band image =========== #
          galex_aperture_params  = galex_surface_brightness_rmask(galex_dir, 
                                                                  galaxies[i], 
                                                                  segment_parameters, 
                                                                  galex_bands,
                                                                  radius_max,
                                                                  fig)
          
          annulus_radius_nuv       = galex_aperture_params[0]
          annulus_area_nuv         = galex_aperture_params[1]
          annulus_adu_nuv          = galex_aperture_params[2]
          annulus_mag_nuv          = galex_aperture_params[3]
          annulus_adu_bk_nuv       = galex_aperture_params[4]
          annulus_mag_bk_nuv       = galex_aperture_params[5]
          aperture_radius_nuv      = galex_aperture_params[6]
          aperture_area_nuv        = galex_aperture_params[7]
          aperture_adu_nuv         = galex_aperture_params[8]
          aperture_adu_err_nuv     = galex_aperture_params[9]
          aperture_adu_diff_nuv    = galex_aperture_params[10]
          aperture_mag_nuv         = galex_aperture_params[11]
          aperture_mag_diff_nuv    = galex_aperture_params[12]
          aperture_adu_bk_nuv      = galex_aperture_params[13]
          aperture_adu_bk_diff_nuv = galex_aperture_params[14]
          aperture_mag_bk_nuv      = galex_aperture_params[15]
          aperture_mag_bk_diff_nuv = galex_aperture_params[16]
            
        else:
          nrings = 40
          n_zeros = int(nrings - 2)
          annulus_radius_nuv       = np.zeros(n_zeros)
          annulus_area_nuv         = np.zeros(n_zeros)
          annulus_adu_nuv          = np.zeros(n_zeros)
          annulus_mag_nuv          = np.zeros(n_zeros)
          annulus_adu_bk_nuv       = np.zeros(n_zeros)
          annulus_mag_bk_nuv       = np.zeros(n_zeros)
          aperture_radius_nuv      = np.zeros(n_zeros)
          aperture_area_nuv        = np.zeros(n_zeros)
          aperture_adu_nuv         = np.zeros(n_zeros)
          aperture_adu_err_nuv     = np.zeros(n_zeros)
          aperture_adu_diff_nuv    = np.zeros(n_zeros)
          aperture_mag_nuv         = np.zeros(n_zeros)
          aperture_mag_diff_nuv    = np.zeros(n_zeros)
          aperture_adu_bk_nuv      = np.zeros(n_zeros)
          aperture_adu_bk_diff_nuv = np.zeros(n_zeros)
          aperture_mag_bk_nuv      = np.zeros(n_zeros)
          aperture_mag_bk_diff_nuv = np.zeros(n_zeros)
                
        plot_name = plots_dir + '/PHOTOMETRY/GALEX/APERTURES/%s_aperture.pdf' % galaxies[i]
        plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
        plt.close()
        
        # =========== Save aperture/annulus NUV-band profiles to file (1 per galaxy) =========== #
        table_str  = galex_dir + 'PROFILES/%s_profile.fits' % galaxies[i]
        os.system('rm -rf %s' % table_str)
        
        tdata = [annulus_radius_nuv, annulus_area_nuv, 
                 annulus_adu_nuv, annulus_mag_nuv, 
                 annulus_adu_bk_nuv, annulus_mag_bk_nuv, 
                 aperture_radius_nuv, aperture_area_nuv, 
                 aperture_adu_nuv, aperture_adu_err_nuv, 
                 aperture_adu_diff_nuv,
                 aperture_mag_nuv, aperture_mag_diff_nuv,
                 aperture_adu_bk_nuv, aperture_adu_bk_diff_nuv,
                 aperture_mag_bk_nuv, aperture_mag_bk_diff_nuv]
        
        tcols = ('RADIUS_ANNULUS', 'AREA_ANNULUS', 
                 'ADU_ANNULUS_NUV', 'MAG_ANNULUS_NUV', 
                 'ADU_ANNULUS_BK_NUV', 'MAG_ANNULUS_BK_NUV', 
                 'RADIUS_APERTURE', 'AREA_APERTURE', 
                 'ADU_APERTURE_NUV', 'ADU_APERTURE_ERR_NUV',
                 'ADU_DIFF_APERTURE_NUV',
                 'MAG_APERTURE_NUV', 'MAG_DIFF_APERTURE_NUV', 
                 'ADU_APERTURE_BKGD_NUV', 'ADU_DIFF_APERTURE_BKGD_NUV',
                 'MAG_APERTURE_BKGD_NUV', 'MAG_DIFF_APERTURE_BKGD_NUV')
        
        t = Table(tdata, names=tcols)
        t.write(table_str, format = 'fits')
    



# ================================= #
# ==== Measure Photometry/Size ==== #
# ================================= #
if do_measure:
  do_save_table              = True
  mag_nuv                    = np.zeros(len(galaxies))
  mag_nuv_err                = np.zeros(len(galaxies))
  snr_nuv                    = np.zeros(len(galaxies))
  radius_nuv                 = np.zeros(len(galaxies))
  mag_nuv_aperture           = np.zeros(len(galaxies))
  mag_nuv_aperture_err       = np.zeros(len(galaxies))
  adu_nuv_aperture           = np.zeros(len(galaxies))
  adu_nuv_aperture_err       = np.zeros(len(galaxies))
  
  for i in range(len(galaxies)):
    if i > -1:
      print('%i\t%s\t%.2f' % (i, galaxies[i], (100.*(i + 1.)/len(galaxies))))
      profile_file    = panstarrs_dir + 'PROFILES_BKGDSUB/' + galaxies[i] + '_profile.fits'
      if os.path.isfile(profile_file):
        fits_file        = galex_dir + 'PROFILES/%s_profile.fits' % galaxies[i]
        hdu_list         = fits.open(fits_file)
        data_list        = hdu_list[1].data
        radius_annu      = data_list['RADIUS_ANNULUS']
        mag_annu_nuv     = data_list['MAG_ANNULUS_NUV']
        mag_diff_nuv     = data_list['MAG_DIFF_APERTURE_NUV']
        radius_aper      = data_list['RADIUS_APERTURE']
        adu_aper_nuv     = data_list['ADU_APERTURE_NUV']
        adu_aper_err_nuv = data_list['ADU_APERTURE_ERR_NUV']
        mag_aper_nuv     = data_list['MAG_APERTURE_NUV']
        mag_bkgd_nuv     = data_list['MAG_APERTURE_BKGD_NUV']
        
        hdu_list.close()
        
        if mag_aper_nuv[0] == 0:
          radius_nuv[i]  = np.nan
          mag_nuv[i]     = np.nan
          mag_nuv_err[i] = np.nan
          snr_nuv[i]     = np.nan
        
        else:
          # ======= NUV-band radius (28 mag/arcsec^2 isophote) ======== #
          if np.nanmean(mag_annu_nuv) > 0:
            for k in range(len(mag_annu_nuv)-1):
              if mag_annu_nuv[k] < 28 and mag_annu_nuv[k+1] > 28:
                radius_nuv_uncor = interpolate_value(radius_annu, 
                                                    mag_annu_nuv, 
                                                    28., k)
                break
              else:
                radius_nuv_uncor = np.nan
          
          # ======= Determine SNR from total flux/error in 28 mag/arcsec^2 isophote aperture ======== #
          adu_aper_riso      = np.interp(radius_nuv_uncor, radius_aper, adu_aper_nuv)
          adu_aper_err_riso  = np.interp(radius_nuv_uncor, radius_aper, adu_aper_err_nuv)
          
          snr_nuv[i]         = adu_aper_riso / adu_aper_err_riso
          
          radius_nuv[i]    = np.sqrt(radius_nuv_uncor**2 - 4.9**2)
          
          print(radius_nuv_uncor, radius_nuv[i])
          
          # ======= Determine asymptotic NUV-band magnitude ======== #
          mag_diff = mag_diff_nuv
          mag_aper = mag_aper_nuv
          mag_bkgd = mag_bkgd_nuv
          
          fig1 = plt.figure(figsize=(3,6))
          
          galex_bands = 'NUV'
            
          if radius_nuv_uncor < 20:
            radius_max_cg = radius_nuv_uncor * 3.
          else:
            radius_max_cg = radius_nuv_uncor * 1.5
          
          y_data = mag_aper[(mag_diff < 0.05) & (radius_aper < radius_max_cg)]
          x_data = mag_diff[(mag_diff < 0.05) & (radius_aper < radius_max_cg)]
          x_fit  = np.arange(0,0.05, 0.001)
          
          if len(y_data) == 0:
            y_data = np.array([0, 0])
          
          if len(y_data) < 4:
            y_data = mag_aper[(mag_diff < 0.1) & (radius_aper < radius_max_cg)]
            x_data = mag_diff[(mag_diff < 0.1) & (radius_aper < radius_max_cg)]
            x_fit  = np.arange(0,0.1, 0.001)
            
          if len(y_data) == 0:
            y_data = np.array([0, 0])
          
          if np.isfinite(y_data[0]) and y_data[0] != 0 and len(y_data) > 3:
            
            pfit = np.polyfit(x_data, y_data, 1)
            func = np.poly1d(pfit)
            
            print(func, round(func(0),3), round(np.nanstd(y_data),3))
            
            mag_nuv[i]     = func(0)
            mag_nuv_err[i] = np.nanstd(y_data)
            
            y_fit = func(x_fit)
            
            curve_of_growth_plot(fig1, [2, 1, 1], [radius_aper, mag_aper, mag_bkgd], 
                              ['radius [arcsec]', 'mag'], [radius_nuv[i], radius_max_cg], None, None)
            
            scat_mag_derivative_plot(fig1, [2, 1, 2], [x_data, y_data], [x_fit, y_fit], 
                                  galex_bands, r'$\Delta$mag', r'mag', ['darkblue', 'peru'], 'o')
          else:
            mag_nuv[i]     = np.nan
            mag_nuv_err[i] = np.nan
                    
          plot_name = plots_dir + '/PHOTOMETRY/GALEX/ASYMPTOTIC_MAG/%s_asymptote_fit.pdf' % galaxies[i]
          plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
          plt.close()
      else:
        radius_nuv[i]  = np.nan
        mag_nuv[i]     = np.nan
        mag_nuv_err[i] = np.nan
        snr_nuv[i]     = np.nan
  
  # =========== Save GALEX radius/magnitudes to file =========== #
  if do_save_table:
    table_str  = parameter_dir + '%s_galex_photometry.fits' % team_release
    os.system('rm -rf %s' % table_str)
    
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [radius_nuv, mag_nuv, mag_nuv_err, snr_nuv]
    tcols_1 = ('RADIUS_NUV', 'MAG_NUV', 'ERROR_NUV', 'SNR_NUV')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
  
    t = Table(tdata, names=tcols)
    t.write(table_str, format = 'fits')
  


  








