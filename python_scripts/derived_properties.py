# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import os.path
from matplotlib import rc
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, join, vstack

from astroquery.irsa_dust import IrsaDust


#from functions_plotting import *
from functions_calculations import *



def scatter_outlier_plot(fig_num, sub1, sub2, sub3, x, y, txtstr, xlbl, ylbl, marker, do_legend):
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    #if sub1 == 1 and sub2 == 1:
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(ylbl)
    ax1.scatter(x, y, color=marker[0], marker=marker[1], 
                edgecolor=marker[0], facecolor=marker[2], s=marker[3], zorder=1, label = txtstr)
    if do_legend:
      ax1.legend(fontsize = 10) #loc='upper right', 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

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



def calculate_sfrs(wise_mags, nuv_mag_ext, distance, upperlimits):
  '''
  wise_mags:     WISE band magnitudes
                 wise_mags[0]: W1-band magnitude
                 wise_mags[1]: W2-band magnitude
                 wise_mags[2]: W3-band magnitude
                 wise_mags[3]: W4-band magnitude
  nuv_mag_ext:   NUV-band magnitude
  distance:      Luminosity distance
  upperlimits:   Flags indicating if NUV, W3 or W4 are upper limits
                 upperlimits[0]: W3 upper limit flag
                 upperlimits[1]: W4 upper limit flag
                 upperlimits[2]: NUV upper limit flag
                 
  Returns: See README_derived_galaxy_sfrs.txt for details of returned quantities.
  '''
  w1_mag = wise_mags[0]
  w2_mag = wise_mags[1]
  w3_mag = wise_mags[2]
  w4_mag = wise_mags[3]
  
  w3_uplim  = upperlimits[0]
  w4_uplim  = upperlimits[1]
  nuv_uplim = upperlimits[2]
  
  # ======== CALC WISE SFR Jarrett13 ===== #
  w1_lum                        = wise_luminosity(w1_mag, distance, 'W1')
  w2_lum                        = wise_luminosity(w2_mag, distance, 'W2')
  w3_lum                        = wise_luminosity(w3_mag, distance, 'W3')
  w4_lum                        = wise_luminosity(w4_mag, distance, 'W4')
  
  w3_lum_corr                   = w3_lum - 0.201 * w1_lum
  w4_lum_corr                   = w4_lum - 0.044 * w1_lum
  
  w3_sfr_mir_nocor              = np.log10((w3_lum) * 4.91 * 10**(-10))
  w4_sfr_mir_nocor              = np.log10((w3_lum) * 7.50 * 10**(-10))
  
  w3_sfr_mir_cor                = np.log10((w3_lum_corr) * 4.91 * 10**(-10))
  w4_sfr_mir_cor                = np.log10((w4_lum_corr) * 7.50 * 10**(-10))
  
  w3_mir_uplim                  = np.zeros(len(w3_uplim), dtype=int) #np.full_like(w3_uplim, False, dtype=bool)
  w4_mir_uplim                  = np.zeros(len(w4_uplim), dtype=int)
  
  w3_mir_uplim[w3_lum_corr < 0] = 1
  w4_mir_uplim[w4_lum_corr < 0] = 1
  
  w3_sfr_mir                    = w3_sfr_mir_cor
  w4_sfr_mir                    = w4_sfr_mir_cor
  
  w3_sfr_mir[w3_mir_uplim]      = w3_sfr_mir_nocor[w3_mir_uplim]
  w4_sfr_mir[w4_mir_uplim]      = w4_sfr_mir_nocor[w4_mir_uplim]
  
  
  # ======== CALC WISE SFR Cluver17 ===== #
  w1_flux                        = wise_flux(w1_mag, distance, 'W1')
  w2_flux                        = wise_flux(w2_mag, distance, 'W2')
  w3_flux                        = wise_flux(w3_mag, distance, 'W3')
  w4_flux                        = wise_flux(w4_mag, distance, 'W4')
  
  w3_flux_corr                   = w3_flux - 0.156 * w1_flux
  w4_flux_corr                   = w4_flux - 0.059 * w1_flux
  
  w3_lum_c17                     = wise_flux_to_lum(w3_flux, distance, 'W3')
  w4_lum_c17                     = wise_flux_to_lum(w4_flux, distance, 'W4')
  
  w3_lum_c17_corr                = wise_flux_to_lum(w3_flux_corr, distance, 'W3')
  w4_lum_c17_corr                = wise_flux_to_lum(w4_flux_corr, distance, 'W4')
  
  w3_c17_uplim                   = np.zeros(len(w3_uplim), dtype=int) #np.full_like(w3_uplim, False, dtype=bool)
  w4_c17_uplim                   = np.zeros(len(w4_uplim), dtype=int)
  
  w3_c17_uplim[w3_flux_corr < 0] = 1
  w4_c17_uplim[w4_flux_corr < 0] = 1
  
  w3_sfr_c17_nocor               = 0.889 * np.log10(w3_lum_c17) - 7.76
  w4_sfr_c17_nocor               = 0.915 * np.log10(w4_lum_c17) - 8.01
  
  w3_sfr_c17_corr                = 0.889 * np.log10(w3_lum_c17_corr) - 7.76
  w4_sfr_c17_corr                = 0.915 * np.log10(w4_lum_c17_corr) - 8.01
  
  w3_sfr_c17                     = w3_sfr_c17_corr
  w4_sfr_c17                     = w4_sfr_c17_corr
  
  w3_sfr_c17[w3_c17_uplim]       = w3_sfr_c17_nocor[w3_c17_uplim]
  w4_sfr_c17[w4_c17_uplim]       = w4_sfr_c17_nocor[w4_c17_uplim]
  
  # ======== CALC GALEX SFR ==== #
  nuv_lum                       = galex_luminosity(np.array(nuv_mag_ext), distance, 'NUV')
  nuv_sfr                       = np.log10(nuv_lum * 10**(-28.165))
  
  # ======== CALC SFR TOTAL ============= #
  sfr_tot_w3_uvir               = np.log10(10**w3_sfr_mir + 10**nuv_sfr)
  sfr_tot_w4_uvir               = np.log10(10**w4_sfr_mir + 10**nuv_sfr)
  
  
  no_galex                        = np.zeros(len(w3_uplim), dtype=int) #np.full_like(w3_uplim, False, dtype=bool)
  no_galex[np.isnan(nuv_mag_ext)] = 1
  
  sfr_tot_w3                      = sfr_tot_w3_uvir
  sfr_tot_w4                      = sfr_tot_w4_uvir
  
  sfr_tot_w3[np.isnan(nuv_sfr)]   = w3_sfr_mir[np.isnan(nuv_sfr)]
  sfr_tot_w4[np.isnan(nuv_sfr)]   = w4_sfr_mir[np.isnan(nuv_sfr)]
  
  sfr_tot                         = sfr_tot_w4
  
  sfr_tot[w4_mir_uplim == 1]      = sfr_tot_w3[w4_mir_uplim == 1]
  
  w34_uplim                       = np.zeros(len(w4_mir_uplim), dtype=int) 
  #np.full_like(w4_mir_uplim, False, dtype=bool)
  w34_uplim[w4_mir_uplim == 1]    = 1
  w34_uplim[w4_mir_uplim == 1]    = w3_mir_uplim[w4_mir_uplim == 1]
  
  total_uplim                     = np.zeros(len(w3_uplim), dtype=int) #np.full_like(w3_uplim, False, dtype=bool)
  total_uplim[w4_uplim == 1]      = 1
  total_uplim[w4_mir_uplim == 1]  = 1
  total_uplim[w4_uplim == 1]      = w3_uplim[w4_uplim == 1]
  total_uplim[w4_mir_uplim == 1]  = w3_mir_uplim[w4_mir_uplim == 1]
  total_uplim[nuv_uplim == 1]     = 1
  total_uplim[no_galex == 1]      = 1
  
  
  return (w3_sfr_mir_nocor, w4_sfr_mir_nocor, 
          w3_sfr_mir_cor, w4_sfr_mir_cor, 
          w3_sfr_mir, w4_sfr_mir, nuv_sfr, 
          sfr_tot_w3_uvir, sfr_tot_w4_uvir, 
          sfr_tot_w3, sfr_tot_w4, sfr_tot,
          w3_sfr_c17_nocor, w4_sfr_c17_nocor, 
          w3_sfr_c17_corr, w4_sfr_c17_corr, 
          w3_sfr_c17, w4_sfr_c17,
          w3_uplim, w4_uplim,
          w3_mir_uplim, w4_mir_uplim, 
          nuv_uplim, no_galex, total_uplim, 
          w3_c17_uplim, w4_c17_uplim)



def calculate_jbar(hi_file, opt_file, parameters):
    redshift   = parameters[0]
    extinction = parameters[1]
    w20        = parameters[2]
    ax_ratio   = parameters[3]
    beam       = parameters[4]
    sint       = parameters[5]
    radius_r   = parameters[6]
    
    hdu_hi     = fits.open(hi_file)
    data_hi    = hdu_hi[1].data
    
    hdu_opt    = fits.open(opt_file)
    data_opt   = hdu_opt[1].data
    
    radius_hi  = np.array(data_hi['RADIUS_ANNULUS'])
    radius_hi0 = np.array([data_hi['RADIUS_APERTURE'][0]])
    flux_hi    = np.array(data_hi['FLUX_ANNULUS']) / beam
    flux_hi0   = np.array([data_hi['FLUX_APERTURE'][0] / beam])
    
    flux_hi    = np.concatenate((flux_hi0, flux_hi))
    radius_hi  = np.concatenate((radius_hi0, radius_hi))
    
    if np.log10(sint) < 5:
      flux_hi /= 0.8
    
    radius_opt = np.array(data_opt['RADIUS_R']) * 1.5
    adu_r      = np.array(data_opt['ADU_ANNULUS_R'])
    adu_g      = np.array(data_opt['ADU_ANNULUS_G'])
    exptime_r  = data_opt['EXPTIME_R'][0]
    exptime_g  = data_opt['EXPTIME_G'][0]
    
    adu_r0     = np.array([data_opt['ADU_APERTURE_R'][0]])
    adu_g0     = np.array([data_opt['ADU_APERTURE_G'][0]])
    adu_r      = np.concatenate((adu_r0, adu_r))
    adu_g      = np.concatenate((adu_g0, adu_g))
    radius_opt = np.concatenate((radius_opt, np.array([np.nanmax(radius_opt)+15]),))
    
    rad_cut    = radius_opt > radius_r
    
    mag_r      = 25. + 2.5 * np.log10(exptime_r) - 2.5 * np.log10(adu_r)
    mag_g      = 25. + 2.5 * np.log10(exptime_g) - 2.5 * np.log10(adu_g)
    mag_r      = 0.014 + 0.162 * (mag_g - mag_r) + mag_r
    mag_g      = 0.014 + 0.162 * (mag_g - mag_r) + mag_g
    mag_r      = mag_r - 2.751 * extinction
    mag_g      = mag_g - 3.793 * extinction
    mag_r[rad_cut]  = np.nan
    mag_g[rad_cut]  = np.nan
    if len(mag_r[~np.isnan(mag_r)]) > 0:
      mag_r_int  = np.interp(radius_hi, radius_opt[~np.isnan(mag_r)], mag_r[~np.isnan(mag_r)])
    else:
      mag_r_int  = np.interp(radius_hi, radius_opt, mag_r)
    if len(mag_g[~np.isnan(mag_g)]) > 0:
      mag_g_int  = np.interp(radius_hi, radius_opt[~np.isnan(mag_g)], mag_g[~np.isnan(mag_g)])
    else:
      mag_g_int  = np.interp(radius_hi, radius_opt, mag_g)
    
    mhi_an     = 10**hi_mass_jyhz(flux_hi, redshift)
    mstar_an   = 10**(calc_lgMstar(-0.840, 1.654, mag_r_int, (mag_g_int - mag_r_int), 
                                   4.64, redshift, h=cosmo.h))
    
    mhi_an[np.isnan(mhi_an)]     = 0
    mstar_an[np.isnan(mstar_an)] = 0
    
    distance   = dist_lum(redshift)
    radius_kpc = distance * np.arctan(radius_hi / 3600. * math.pi / 180.) * 1000.
    
    numer      = np.nansum((mstar_an + 1.35 * mhi_an) * (w20 / 2.) * radius_kpc)
    denom      = np.nansum(mstar_an + 1.35 * mhi_an)
    jbar       = np.log10((numer / denom) / np.sin(np.arccos(ax_ratio)))
    mbar       = np.log10(np.nansum(mstar_an + 1.35 * mhi_an))
    
    numer      = np.nansum(mhi_an * (w20 / 2.) * radius_kpc)
    denom      = np.nansum(mhi_an)
    jhi       = np.log10((numer / denom) / np.sin(np.arccos(ax_ratio)))
    
    return(mbar, jbar, jhi)




# ================================= #
# ========== Axes Labels ========== #
# ================================= #
lbl_czhi   = r'c$z_{\mathrm{HI}}$ [km\,s$^{-1}$]'
lbl_czopt  = r'c$z_{\mathrm{opt}}$ [km\,s$^{-1}$]'
lbl_sint   = r'$\log(S_{\mathrm{int}}/\mathrm{Jy})$'
lbl_mstar  = r'$\log(M_*/\mathrm{M}_{\odot})$'
lbl_mhi    = r'$\log(M_{\mathrm{HI}}/\mathrm{M}_{\odot})$'
lbl_hidef  = r'$\mathrm{DEF}_{\mathrm{HI}}$'
lbl_dhi    = r'$\log(d_{\mathrm{HI}}/\mathrm{kpc})$'
lbl_d25    = r'$d_{\mathrm{\mathrm{opt}}}/\mathrm{kpc}$'
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
lbl_sratio = r'$d_{\mathrm{HI}}/d_{\mathrm{opt}}$'
lbl_cindex = r'$R_{90}/R_{50}$'
lbl_tdep   = r'$\tau_{\rm{dep}}$/[Gyr]'
lbl_sfe    = r'$\log$(SFE/[yr$^{-1}$])'


# ================================= #
# =========== Switches ============ #
# ================================= #
do_open_tables           = True          # Always True, opens and joins tables
do_get_source_properties = True          # Always True, provides input source parameters

do_get_dust_extinction   = True          # Only True once to create table with Galactic dust extinctions E(B-V)

do_derive_quantities     = False         # True to derive quantities and output table for single team release

do_pilot_survey_all      = False         # True to derive quantities and output single table for all team releases

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
if do_open_tables:
  if do_get_dust_extinction:
    fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
    fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
    fits_wallaby   = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
    fits_hiopt     = parameter_dir + '%s_hi_optical_disc.fits' % team_release
    fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
    fits_galex     = parameter_dir + '%s_galex_photometry.fits' % team_release
    fits_wise      = parameter_dir + '%s_wise_photometry.fits' % team_release
    
    hdu_sofia      = fits.open(fits_sofia)
    data_sofia     = hdu_sofia[1].data
    
    hdu_flags      = fits.open(fits_flags)
    data_flags     = hdu_flags[1].data
    
    hdu_wallaby    = fits.open(fits_wallaby)
    data_wallaby   = hdu_wallaby[1].data
    
    hdu_hiopt      = fits.open(fits_hiopt)
    data_hiopt     = hdu_hiopt[1].data
    
    hdu_panstarrs  = fits.open(fits_panstarrs)
    data_panstarrs = hdu_panstarrs[1].data
    
    hdu_galex      = fits.open(fits_galex)
    data_galex     = hdu_galex[1].data
    
    hdu_wise       = fits.open(fits_wise)
    data_wise      = hdu_wise[1].data
    
    join1          = join(data_sofia, data_flags, join_type='left')
    join2          = join(join1, data_wallaby, join_type='left')
    join3          = join(join2, data_hiopt, join_type='left')
    join4          = join(join3, data_panstarrs, join_type='left')
    join5          = join(join4, data_galex, join_type='left')
    data_join      = join(join5, data_wise, join_type='left')
  
  else:
    fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
    fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
    fits_wallaby   = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
    fits_hiopt     = parameter_dir + '%s_hi_optical_disc.fits' % team_release
    fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
    fits_galex     = parameter_dir + '%s_galex_photometry.fits' % team_release
    fits_wise      = parameter_dir + '%s_wise_photometry.fits' % team_release
    fits_extinct   = parameter_dir + '%s_galactic_dust_extinction.fits' % team_release
    
    hdu_sofia      = fits.open(fits_sofia)
    data_sofia     = hdu_sofia[1].data
    
    hdu_flags      = fits.open(fits_flags)
    data_flags     = hdu_flags[1].data
    
    hdu_wallaby    = fits.open(fits_wallaby)
    data_wallaby   = hdu_wallaby[1].data
    
    hdu_hiopt      = fits.open(fits_hiopt)
    data_hiopt     = hdu_hiopt[1].data
    
    hdu_panstarrs  = fits.open(fits_panstarrs)
    data_panstarrs = hdu_panstarrs[1].data
    
    hdu_galex      = fits.open(fits_galex)
    data_galex     = hdu_galex[1].data
    
    hdu_wise       = fits.open(fits_wise)
    data_wise      = hdu_wise[1].data
    
    hdu_extinct    = fits.open(fits_extinct)
    data_extinct   = hdu_extinct[1].data

    join1          = join(data_sofia, data_flags, join_type='left')
    join2          = join(join1, data_wallaby, join_type='left')
    join3          = join(join2, data_hiopt, join_type='left')
    join4          = join(join3, data_panstarrs, join_type='left')
    join5          = join(join4, data_galex, join_type='left')
    join6          = join(join5, data_wise, join_type='left')
    data_join      = join(join6, data_extinct, join_type='left')


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
  
  BEAM                  = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  redshift              = (HI_REST / (data_mask['freq'] / 1.e6)) - 1.
  galaxies              = np.array(gal_name)
  sofia_ra              = data_mask['ra']
  sofia_dec             = data_mask['dec']
  sofia_vsys            = redshift * C_LIGHT
  sofia_rms             = data_mask['rms']
  sofia_sint            = data_mask['f_sum'] * chan_width / chan_width_hz
  sofia_snr             = data_mask['f_sum'] / data_mask['err_f_sum']
  sofia_kinpa           = data_mask['kin_pa']
  sofia_w20             = data_mask['w20'] / chan_width_hz * chan_width
  sofia_w50             = data_mask['w50'] / chan_width_hz * chan_width
  sofia_ell_maj         = data_mask['ell_maj'] * pix_scale * 3600.
  sofia_ell_min         = data_mask['ell_min'] * pix_scale * 3600.
  sofia_ellpa           = data_mask['ell_pa']
  
  source_flag           = data_mask['flag_src_class']
  opt_fit_flag          = data_mask['flag_opt_fit']
  segment_deblend       = data_mask['segment_deblend']
  
  do_seg_par1           = data_mask['do_seg_par1']
  do_seg_par2           = data_mask['do_seg_par2']
  do_seg_par3           = data_mask['do_seg_par3']
  do_seg_par4           = data_mask['do_seg_par4']
  do_seg_par5           = data_mask['do_seg_par5']
  
  do_seg_id1            = data_mask['do_seg_id1']
  do_seg_id2            = data_mask['do_seg_id2']
  
  do_deblend1           = data_mask['do_deblend1']
  do_deblend2           = data_mask['do_deblend2']
  do_deblend3           = data_mask['do_deblend3']
  
  ra_gaussian           = data_mask['GAUSSIAN_RA']
  dec_gaussian          = data_mask['GAUSSIAN_DEC']
  
  #Correction for measured WALLABY fluxes
  sint_corr             = np.array(wallaby_flux_scaling(data_mask['f_sum']))
  scale_factor          = np.array(data_mask['f_sum'] / sint_corr)
  
  
  radius_iso_hi         = data_mask['RADIUS_ISO']
  radius_eff_hi         = data_mask['RADIUS_EFF']
  surfden_iso_hi        = data_mask['SURFACE_DENSITY_ISO']
  surfden_eff_hi        = data_mask['SURFACE_DENSITY_EFF']
  radius_hi_err         = data_mask['RADIUS_ISO_ERR']
  minax_iso_hi          = data_mask['DIAMETER_MINOR'] / 2.
  hi_ba                 = data_mask['AXIS_RATIO_BA']
  
  radius_iso_hi_corr    = data_mask['RADIUS_ISO_SCALE']
  radius_eff_hi_corr    = data_mask['RADIUS_EFF_SCALE']
  surfden_iso_hi_corr   = data_mask['SURFACE_DENSITY_ISO_SCALE']
  surfden_eff_hi_corr   = data_mask['SURFACE_DENSITY_EFF_SCALE']
  
  sint_ps_disc_iso      = data_mask['FLUX_INNER_PS25']
  sint_ps_disc_eff      = data_mask['FLUX_INNER_PS50']
  surfden_ps_disc_iso   = data_mask['SURFACE_DENSITY_INNER_PS25']
  surfden_ps_disc_eff   = data_mask['SURFACE_DENSITY_INNER_PS50']
  
  rband_ba              = data_mask['SEG_BA']
  
  r25_asec              = data_mask['RADIUS_R_ISO25']
  r26_asec              = data_mask['RADIUS_R_ISO26']
  rad_50                = data_mask['RADIUS_R_50']
  rad_90                = data_mask['RADIUS_R_90']
  
  gmag_ps25             = data_mask['G_mag25']
  gmag_ps26             = data_mask['G_mag26']
  rmag_ps25             = data_mask['R_mag25']
  rmag_ps26             = data_mask['R_mag26']
  #ax_ratio       = data_mask['BA_aper']
  
  #nuv            = data_mask['NUV Magnitude']
  #extinct               = data_mask['E(B-V)']
  
  w1_mag                = data_mask['W1'] + 0.03
  w2_mag                = data_mask['W2'] + 0.04
  w3_mag                = data_mask['W3'] + 0.03
  w4_mag                = data_mask['W4'] - 0.03
  
  w1_snr                = data_mask['SNR_W1']
  w2_snr                = data_mask['SNR_W2']
  w3_snr                = data_mask['SNR_W3']
  w4_snr                = data_mask['SNR_W4']
  
  nuv_mag               = data_mask['MAG_NUV']
  nuv_mag_err           = data_mask['ERROR_NUV']
  nuv_snr               = data_mask['SNR_NUV']
  radius_nuv            = data_mask['RADIUS_NUV']
  
  w1_uplim              = np.zeros(len(w1_snr), dtype=int)
  w2_uplim              = np.zeros(len(w2_snr), dtype=int)
  w3_uplim              = np.zeros(len(w3_snr), dtype=int)
  w4_uplim              = np.zeros(len(w4_snr), dtype=int)
  
  nuv_uplim             = np.zeros(len(nuv_mag), dtype=int)
  #nuv_uplim             = np.full_like(nuv_mag, False, dtype=bool)
  
  wise_snr_cut          = 5
  galex_snr_cut         = 5
  
  w1_uplim[w1_snr < wise_snr_cut]    = 1
  w2_uplim[w2_snr < wise_snr_cut]    = 1
  w3_uplim[w3_snr < wise_snr_cut]    = 1
  w4_uplim[w4_snr < wise_snr_cut]    = 1
  
  nuv_uplim[nuv_snr < galex_snr_cut] = 1
  
  
  #nuv_mag[nuv_mag > 20]           = 20
  
  incl            = np.arcsin(np.sqrt(1. - rband_ba**2)) * 180. / math.pi
  
  if not do_get_dust_extinction:
    extinct_SAF = data_mask['E(B-V)_SandF']
    extinct_SFD = data_mask['E(B-V)_SFD']
  
  #if have_segments:
    #seg_x        = data_mask['SEG_X']
    #seg_y        = data_mask['SEG_Y']
    #seg_radius   = data_mask['SEG_RADIUS']
    #seg_ba       = data_mask['SEG_BA']
    #seg_pa       = data_mask['SEG_PA']
  
  #if have_optical:
    ##fitness      = data_mask['FIT']
    #ps_x         = data_mask['SEG_X']
    #ps_y         = data_mask['SEG_Y']
    #ps_ba        = data_mask['SEG_BA']
    #ps_pa        = data_mask['SEG_PA']
    #ps_radius25  = data_mask['RADIUS_R_ISO25']
    #ps_radius50  = data_mask['RADIUS_R_50']
    ##fitness      = data_mask['FIT']


if do_get_dust_extinction:
  # ======== Galactic Dust Extinction ========== #
  pos_hi               = SkyCoord(sofia_ra*u.deg, sofia_dec*u.deg, frame='icrs')
  extinct_SAF          = np.full(len(galaxies), np.nan)
  extinct_SFD          = np.full(len(galaxies), np.nan)
  for i in range(len(pos_hi)):
    extinct_table     = IrsaDust.get_query_table(pos_hi[i], section='ebv')
    extinct_SAF[i]    = extinct_table['ext SandF mean'][0]
    extinct_SFD[i]    = extinct_table['ext SFD mean'][0]
    print(galaxies[i], np.round(100*i/len(galaxies),2), extinct_SAF[i], extinct_SFD[i])
    
    
  table_str  = parameter_dir + '%s_galactic_dust_extinction.fits' % team_release
  os.system('rm -rf %s' % table_str)
  
  tdata = []
  tcols = []
  for i in range(len(data_join.columns)):
    if i < len(data_sofia.columns.names):
      tdata.append(data_join[data_join.columns[i].name])
      tcols.append(data_join.columns[i].name)
  
  tdata_1 = [extinct_SAF, extinct_SFD]
  
  tcols_1 = ('E(B-V)_SandF', 'E(B-V)_SFD')
  
  for i in range(len(tdata_1)):
    tdata.append(tdata_1[i])
    tcols.append(tcols_1[i])
  
  #print(type(tdata))
  #print(len(tdata))
  
  t = Table(tdata, names=tcols)
  t.write(table_str, format = 'fits')


# ================================= #
# === Derive Physical Quantities == #
# ================================= #
if do_derive_quantities:
  do_save_table_calc_par = True
  do_save_table_sfr      = True
  
  # ======== PanSTARRS MSTAR ========== #
  gmag_sdss25          = 0.014 + 0.162 * (gmag_ps25 - rmag_ps25) + gmag_ps25
  rmag_sdss25          = 0.014 + 0.162 * (gmag_ps25 - rmag_ps25) + rmag_ps25
  
  gmag_sdss_ext        = gmag_sdss25 - 3.793 * extinct_SFD
  rmag_sdss_ext        = rmag_sdss25 - 2.751 * extinct_SFD
  
  mstar_sdss25         = calc_lgMstar(-0.840, 1.654, rmag_sdss_ext, 
                                      (gmag_sdss_ext - rmag_sdss_ext), 
                                      4.64, redshift, h=cosmo.h)
  
  # ======== NUV - r ========== #
  nuv_mag_ext          = nuv_mag - 8.2 * extinct_SFD
  nuvr                 = nuv_mag_ext - rmag_sdss25
  
  # ======== Distance, HI Mass, HI Fraction ========== #
  distance             = dist_lum(redshift)
  mhi_msol             = hi_mass_jyhz(sofia_sint, redshift)
  hifrac               = mhi_msol - mstar_sdss25
  
  mhi_msol_corr        = hi_mass_jyhz(sint_corr, redshift)
  mhi_ps_iso_corr      = hi_mass_jyhz(sint_ps_disc_iso, redshift)
  mhi_ps_eff_corr      = hi_mass_jyhz(sint_ps_disc_eff, redshift)
  
  hifrac_corr          = mhi_msol_corr - mstar_sdss25
  hifrac_ps_iso_corr   = mhi_ps_iso_corr - mstar_sdss25
  hifrac_ps_eff_corr   = mhi_ps_eff_corr - mstar_sdss25
  
  # ======== Galaxy Disc Size Ratios ========== #
  sratio_hi_r25        = radius_iso_hi_corr / r25_asec
  sratio_nuv_r25       = radius_nuv / r25_asec
  sratio_hi_nuv        = radius_iso_hi_corr / radius_nuv
  
  # ========= Concentration Index ========== #
  cindex               = rad_90 / rad_50
  
  # ======== Calculate SFRs ========= #
  wise_mags            = [w1_mag, w2_mag, w3_mag, w4_mag]
  upperlimits          = [w3_uplim, w4_uplim, nuv_uplim]
  sfr_parameters       = calculate_sfrs(wise_mags, nuv_mag_ext, distance, upperlimits)
  
  w3_sfr_mir_nocor     = sfr_parameters[0]
  w4_sfr_mir_nocor     = sfr_parameters[1]
  w3_sfr_mir_cor       = sfr_parameters[2]
  w4_sfr_mir_cor       = sfr_parameters[3]
  w3_sfr_mir           = sfr_parameters[4]
  w4_sfr_mir           = sfr_parameters[5]
  nuv_sfr              = sfr_parameters[6]
  sfr_tot_w3_uvir      = sfr_parameters[7]
  sfr_tot_w4_uvir      = sfr_parameters[8]
  sfr_tot_w3           = sfr_parameters[9]
  sfr_tot_w4           = sfr_parameters[10]
  sfr_tot              = sfr_parameters[11]
  w3_sfr_c17_nocor     = sfr_parameters[12]
  w4_sfr_c17_nocor     = sfr_parameters[13]
  w3_sfr_c17_corr      = sfr_parameters[14]
  w4_sfr_c17_corr      = sfr_parameters[15]
  w3_sfr_c17           = sfr_parameters[16]
  w4_sfr_c17           = sfr_parameters[17]
  w3_uplim             = sfr_parameters[18]
  w4_uplim             = sfr_parameters[19]
  w3_mir_uplim         = sfr_parameters[20]
  w4_mir_uplim         = sfr_parameters[21]
  nuv_uplim            = sfr_parameters[22]
  no_galex             = sfr_parameters[23]
  total_uplim          = sfr_parameters[24]
  w3_c17_uplim         = sfr_parameters[25]
  w4_c17_uplim         = sfr_parameters[26]

  ssfr_tot             = sfr_tot - mstar_sdss25
  
  #tdep        = (1.33*10**mhi_msol/10**sfr_tot) / 10**9
  #sfe         = sfr_tot - mhi_msol
  
  # ======== SAVE TABLE ======== #
  if do_save_table_calc_par:
    table_str  = parameter_dir + '%s_derived_galaxy_properties.fits' % team_release
    os.system('rm -rf %s' % table_str)
    
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [redshift, sofia_vsys, distance,
               mstar_sdss25, mhi_msol, hifrac, 
               mhi_msol_corr, hifrac_corr,
               mhi_ps_iso_corr, hifrac_ps_iso_corr,
               mhi_ps_eff_corr, hifrac_ps_eff_corr,
               r25_asec, r26_asec, 
               rad_50, rad_90, 
               radius_iso_hi, radius_eff_hi, radius_hi_err,
               radius_nuv, nuvr,
               surfden_iso_hi, surfden_eff_hi, hi_ba,
               radius_iso_hi_corr, radius_eff_hi_corr, 
               surfden_iso_hi_corr, surfden_eff_hi_corr,
               sratio_hi_r25, sratio_nuv_r25, sratio_hi_nuv, 
               sfr_tot, total_uplim, ssfr_tot,
               gmag_ps25, rmag_ps25, 
               gmag_sdss25, rmag_sdss25, 
               nuv_mag, nuv_uplim, 
               w1_mag, w1_uplim, w2_mag, w2_uplim, 
               w3_mag, w3_uplim, w4_mag, w4_uplim, 
               extinct_SAF, extinct_SFD]
    
    tcols_1 = ('REDSHIFT', 'VSYS', 'DISTANCE',
               'lgMSTAR_SDSS_25', 'lgMHI', 'HIFRAC', 
               'lgMHI_CORRECT', 'HIFRAC_CORRECT', 
               'lgMHI_OPTICAL_DISC_CORRECT_ISO', 'HIFRAC_OPTICAL_DISC_CORRECT_ISO',
               'lgMHI_OPTICAL_DISC_CORRECT_EFF', 'HIFRAC_OPTICAL_DISC_CORRECT_EFF', 
               'RADIUS_R_ISO25', 'RADIUS_R_ISO26',
               'RADIUS_R_50', 'RADIUS_R_90', 
               'RADIUS_HI_ISO', 'RADIUS_HI_EFF', 'RADIUS_ISO_ERR',
               'RADIUS_NUV_ISO', 'NUV-R', 
               'SURFACE_DENSITY_HI_ISO', 'SURFACE_DENSITY_HI_EFF', 'AXIS_RATIO_BA',
               'RADIUS_HI_ISO_CORR', 'RADIUS_HI_EFF_CORR', 
               'SURFACE_DENSITY_HI_ISO_CORR', 'SURFACE_DENSITY_HI_EFF_CORR',
               'HI_R25_SIZE_RATIO', 'NUV_R25_SIZE_RATIO', 'HI_NUV_SIZE_RATIO',
               'SFR_NUV+MIR', 'SFR_UPLIM', 'SSFR',
               'G_MAG_PS25', 'R_MAG_PS25',
               'G_MAG_SDSS_25', 'R_MAG_SDSS_25',
               'NUV_MAG', 'NUV_UPPERLIM', 
               'W1_MAG', 'W1_UPPERLIM', 'W2_MAG', 'W2_UPPERLIM', 
               'W3_MAG', 'W3_UPPERLIM', 'W4_MAG', 'W4_UPPERLIM', 
               'E(B-V)_SandF', 'E(B-V)_SFD')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
    
    #print(type(tdata))
    #print(len(tdata))
    
    t = Table(tdata, names=tcols)
    t.write(table_str, format = 'fits')
  
  
  if do_save_table_sfr:
    table_str  = parameter_dir + '%s_derived_galaxy_sfrs.fits' % team_release
    os.system('rm -rf %s' % table_str)
    
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    #table_str  = basedir + 'PARAMETERS2/derived_galaxy_sfrs5.fits'
    #os.system('rm -rf %s' % table_str)
    
    #tdata_sfr = np.concatenate(([galaxies, sofia_id, environ_class, mstar_sdss25], sfr_parameters))
    
    tdata_sfr1 = [w3_sfr_mir_nocor, w4_sfr_mir_nocor, 
                  w3_sfr_mir_cor, w4_sfr_mir_cor, 
                  w3_sfr_mir, w4_sfr_mir, nuv_sfr, 
                  sfr_tot_w3_uvir, sfr_tot_w4_uvir, 
                  sfr_tot_w3, sfr_tot_w4, sfr_tot,
                  w3_sfr_c17_nocor, w4_sfr_c17_nocor, 
                  w3_sfr_c17_corr, w4_sfr_c17_corr, 
                  w3_sfr_c17, w4_sfr_c17,
                  w3_uplim, w4_uplim,
                  w3_mir_uplim, w4_mir_uplim, 
                  nuv_uplim, no_galex, total_uplim, 
                  w3_c17_uplim, w4_c17_uplim]
    
    tcols_sfr1 = ('SFR_W3_MIR_NOCORR', 'SFR_W4_MIR_NOCORR',
                  'SFR_W3_MIR_CORR', 'SFR_W4_MIR_CORR',
                  'SFR_W3_MIR_FINAL', 'SFR_W4_MIR_FINAL', 'NUV_SFR',
                  'SFR_NUV+W3', 'SFR_NUV+W4',
                  'SFR_NUV+W3_UPLIMS', 'SFR_NUV+W4_UPLIMS', 'SFR_NUV+MIR_FINAL',
                  'SFR_W3_C17_NOCORR', 'SFR_W4_C17_NOCORR', 
                  'SFR_W3_C17_CORR', 'SFR_W4_C17_CORR', 
                  'SFR_W3_C17_FINAL', 'SFR_W4_C17_FINAL',
                  'W3_FLUX_UPLIM_FLAG', 'W4_FLUX_UPLIM_FLAG', 
                  'W3_CORR_FLUX_UPLIM_FLAG', 'W4_CORR_FLUX_UPLIM_FLAG', 
                  'NUV_FLUX_UPLIM_FLAG', 'NO_GALEX_COVERAGE_FLAG', 'SFR_NUV+MIR_UPLIM_FLAG',
                  'W3_CORR_FLUX_C17_UPLIM_FLAG', 'W4_CORR_FLUX_C17_UPLIM_FLAG')
    
    for i in range(len(tdata_sfr1)):
      tdata.append(tdata_sfr1[i])
      tcols.append(tcols_sfr1[i])
    
    #print(len(tdata_sfr))
    #print(len(tcols_sfr))
    
    t = Table(tdata, names=tcols)
    t.write(table_str, format = 'fits')

    


# ================================= #
# =========== ALL Pilot =========== #
# ================================= #
if do_pilot_survey_all:
  #tr_i                     = 4
  survey_phase_list        = ['PHASE1', 'PHASE1', 'PHASE1', 'PHASE2', 'PHASE2', 'PHASE2']
  team_release_list        = ['Hydra_DR1', 'Hydra_DR2', 'NGC4636_DR1', 'NGC4808_DR1', 'NGC5044_DR1', 'NGC5044_DR2']
  
  basedir                  = '/Users/tflowers/WALLABY/PHASE2/NGC5044_DR1/'
  fits_dr1                 = basedir + 'SOFIA/' + 'NGC5044_DR1_catalogue.fits'
  basedir                  = '/Users/tflowers/WALLABY/PHASE2/NGC5044_DR2/'
  fits_dr2                 = basedir + 'SOFIA/' + 'NGC5044_DR2_catalogue.fits'
  
  hdu_dr1                  = fits.open(fits_dr1)
  data_dr1                 = hdu_dr1[1].data
  hdu_dr2                  = fits.open(fits_dr2)
  data_dr2                 = hdu_dr2[1].data
  
  join_ngc5044             = join(data_dr1, data_dr2, keys='name', join_type='inner')
  
  galaxies_both5044        = join_ngc5044['name']
  
  print(len(join_ngc5044))
  
  for tr_i in range(1,6):
    survey_phase             = survey_phase_list[tr_i]
    team_release             = team_release_list[tr_i]
    
    # ================================= #
    # ========= File Strings ========== #
    # ================================= #
    basedir        = '/Users/tflowers/WALLABY/%s/%s/' % (survey_phase, team_release)
    sofia_dir      = basedir + 'SOFIA/'
    parameter_dir  = basedir + 'PARAMETERS/'
    
    fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
    fits_flags     = sofia_dir + '%s_catalogue_flags.fits' % team_release
    fits_wallaby   = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
    fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
    fits_hiopt     = parameter_dir + '%s_hi_optical_disc.fits' % team_release
    
    hdu_sofia      = fits.open(fits_sofia)
    data_sofia     = hdu_sofia[1].data
    
    hdu_flags      = fits.open(fits_flags)
    data_flags     = hdu_flags[1].data
    
    hdu_wallaby    = fits.open(fits_wallaby)
    data_wallaby   = hdu_wallaby[1].data
    
    hdu_panstarrs  = fits.open(fits_panstarrs)
    data_panstarrs = hdu_panstarrs[1].data
    
    hdu_hiopt      = fits.open(fits_hiopt)
    data_hiopt     = hdu_hiopt[1].data
    
    join1          = join(data_sofia, data_flags, join_type='left')
    join2          = join(join1, data_wallaby, join_type='left')
    join3          = join(join2, data_panstarrs, join_type='left')
    data_join      = join(join3, data_hiopt, join_type='left')
    
    if tr_i == 1:
      data_all    = data_join
    else:
      data_all    = vstack([data_all, data_join], join_type='outer')
  
  #print(data_all.columns)
  
  data_mask      = data_all[(data_all['flag_src_class'] == 1) & (data_all['flag_opt_fit'] == 0)]
  
  print(len(data_mask))
  
  galaxy_names   = data_mask['name']
  release        = data_mask['team_release']
  counter        = 0
  
  #print(galaxy_names[0], galaxies_both5044[0], release[0])
  
  for i in range(len(galaxy_names)):
    for j in range(len(galaxies_both5044)):
      if galaxy_names[i] == galaxies_both5044[j] and release[i] == 'NGC 5044 TR1':
        #print(galaxy_names[i], galaxies_both5044[j], release[i])
        data_mask.remove_row(counter)
        counter -= 1
    counter += 1
  
  print(len(data_mask))
  
  galaxy_names   = data_mask['name']
  release        = data_mask['team_release']
  
  #fits_file  = '/Users/tflowers/WALLABY/PHASE1/Hydra_DR2/SOFIA/Hydra_DR2_source_products/WALLABY_J104059-270456/WALLABY_J104059-270456_cube.fits.gz'
  #f1         = pyfits.open(fits_file)
  #data, hdr  = f1[0].data, f1[0].header
  #beam_maj, beam_min, pix_scale  = hdr['BMAJ'], hdr['BMIN'], np.abs(hdr['CDELT1'])
  #f1.close()
  
  #BEAM           = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  #print(BEAM)
  
  sofia_sint     = data_mask['f_sum']
  
  redshift       = (HI_REST / (data_mask['freq'] / 1e6)) - 1.
  
  #mhi_msol       = hi_mass_jyhz(sofia_sint, redshift)
  
  gmag_ps25      = data_mask['G_mag25']
  rmag_ps25      = data_mask['R_mag25']
  
  sint_i25       = data_mask['FLUX_INNER_PS25']
  sint_50        = data_mask['FLUX_INNER_PS50']
  
  radius_r_i25   = data_mask['RADIUS_R_ISO25']
  radius_r_50    = data_mask['RADIUS_R_50']
  
  ba_panstarrs   = data_mask['SEG_BA']
  
  

  # ======== PanSTARRS MSTAR ========== #
  gmag_sdss25   = 0.014 + 0.162 * (gmag_ps25 - rmag_ps25) + gmag_ps25
  rmag_sdss25   = 0.014 + 0.162 * (gmag_ps25 - rmag_ps25) + rmag_ps25
  
  #gmag_sdss_ext = gmag_sdss25 - 3.793 * extinct
  #rmag_sdss_ext = rmag_sdss25 - 2.751 * extinct
  
  mstar_sdss25  = calc_lgMstar(-0.840, 1.654, rmag_sdss25, (gmag_sdss25 - rmag_sdss25), 
                              4.64, redshift, h=cosmo.h)
  
  #hifrac               = mhi_msol - mstar_sdss25
  
  sint_corr            = wallaby_flux_scaling(sofia_sint)
  mhi_msol_corr        = hi_mass_jyhz(sint_corr, redshift)
  hifrac_corr          = mhi_msol_corr - mstar_sdss25
  
  mhi_msol_i25         = hi_mass_jyhz(sint_i25, redshift)
  hifrac_i25           = mhi_msol_i25 - mstar_sdss25
  
  
  #print(np.array(release[mstar_sdss25 > 11.5]))
  #print(np.array(galaxy_names[mstar_sdss25 > 11.5]))
  #print(np.round(np.array(mstar_sdss25[mstar_sdss25 > 11.5]),2))
  #print(np.round(np.array(hifrac_corr[mstar_sdss25 > 11.5]),2))
  
  print(np.array(release[hifrac_corr < -1.5]))
  print(np.array(galaxy_names[hifrac_corr < -1.5]))
  print(np.round(np.array(mstar_sdss25[hifrac_corr < -1.5]),2))
  print(np.round(np.array(hifrac_corr[hifrac_corr < -1.5]),2))
  
  
  
  
  
  fits_xgass     = '/Users/tflowers/WALLABY/Hydra_DR2/CATALOGUES/xGASS_representative_sample.fits'
  
  # ======== xGASS PARAMETERS ======== #
  hdu_xgass      = fits.open(fits_xgass)
  data_xgass     = hdu_xgass[1].data
  
  xgass_mstar    = data_xgass['lgMstar']
  xgass_hifrac   = data_xgass['lgGF']
  
  xlbl         = lbl_mstar
  ylbl         = lbl_hifrac
  
  #print(len(xpar[np.isfinite(xpar) & np.isfinite(ypar)]))
  
  fig4 = plt.figure(1, figsize=(6, 5))
  
  scatter_outlier_plot(fig4, 1, 1, 1, xgass_mstar, xgass_hifrac,
                        'xGASS', xlbl, ylbl, ['grey',  'o', 'grey', 2], True)
  
  #xpar    = mstar_sdss25
  #ypar    = hifrac_corr
  
  xpar    = mstar_sdss25[radius_r_i25 * ba_panstarrs > 15]
  ypar    = hifrac_corr[radius_r_i25 * ba_panstarrs > 15]
  
  scatter_outlier_plot(fig4, 1, 1, 1, xpar, ypar,
                        r'$M_{\rm{HI,total}}$', xlbl, ylbl, ['darkblue',  'o', 'darkblue', 12], True)
  
  xpar    = mstar_sdss25[radius_r_i25 * ba_panstarrs > 15]
  ypar    = hifrac_i25[radius_r_i25 * ba_panstarrs > 15]
  
  #print(len(xpar))
  
  scatter_outlier_plot(fig4, 1, 1, 1, xpar, ypar,
                        r'$M_{\rm{HI,inner}}$', xlbl, ylbl, ['peru',  'o', 'peru', 12], True)
  
  #scatter_outlier_plot(fig4, 1, 1, 1, 
                       #xpar[data_all['name'] == 'WALLABY J102608-280840'], 
                       #ypar[data_all['name'] == 'WALLABY J102608-280840'],
                        #'WALLABY', xlbl, ylbl, ['peru',  '*', 'peru', 20], True)
  
  #scatter_outlier_plot(fig4, 1, 1, 1, 
                       #xpar[data_all['name'] == 'WALLABY J102605-280710'], 
                       #ypar[data_all['name'] == 'WALLABY J102605-280710'],
                        #'WALLABY', xlbl, ylbl, ['peru',  '*', 'peru', 20], True)
  
  #plt.show()
  plot_name = '/Users/tflowers/WALLABY/PHASE2/PLOTS/pilot_survey_gas_fraction_sr_inner2.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
  plt.close()

  xlbl         = lbl_dlum
  ylbl         = lbl_mhi
  
  #print(len(xpar[np.isfinite(xpar) & np.isfinite(ypar)]))
  
  fig4 = plt.figure(2, figsize=(6, 4))
  
  xpar    = data_all['dist_h']
  ypar    = data_all['log_m_hi']
  
  scatter_outlier_plot(fig4, 1, 1, 1, xpar, ypar,
                        'WALLABY', xlbl, ylbl, ['darkblue',  'o', 'darkblue', 8], True)
  
  #xpar    = mstar_sdss25[radius_r_i25 * ba_panstarrs > 15]
  #ypar    = hifrac_i25[radius_r_i25 * ba_panstarrs > 15]
  
  #print(len(xpar))
  
  #scatter_outlier_plot(fig4, 1, 1, 1, xpar, ypar,
                        #r'$M_{\rm{HI,inner}}$', xlbl, ylbl, ['peru',  'o', 'peru', 8], True)
  
  #scatter_outlier_plot(fig4, 1, 1, 1, 
                       #xpar[data_all['name'] == 'WALLABY J102608-280840'], 
                       #ypar[data_all['name'] == 'WALLABY J102608-280840'],
                        #'WALLABY', xlbl, ylbl, ['peru',  '*', 'peru', 20], True)
  
  #scatter_outlier_plot(fig4, 1, 1, 1, 
                       #xpar[data_all['name'] == 'WALLABY J102605-280710'], 
                       #ypar[data_all['name'] == 'WALLABY J102605-280710'],
                        #'WALLABY', xlbl, ylbl, ['peru',  '*', 'peru', 20], True)
  
  #plt.show()
  plot_name = '/Users/tflowers/WALLABY/PHASE2/PLOTS/pilot_survey_dist_mhi.pdf'
  plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
  plt.close()


