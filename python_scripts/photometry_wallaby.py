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
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table, join
import aplpy
from scipy.interpolate import interp1d

from photutils.aperture import aperture_photometry
from photutils import EllipticalAperture, EllipticalAnnulus


#from functions_plotting import *
from functions_calculations import *



# ================================= #
# ==== Fit 2D Gaussian to Mom0 ==== #
# ================================= #
def gaussian_fit(input_dir, galaxy):
    '''
    input_dir: WALLABY data products directory
    
    galaxy:    Galaxy name
    
    Returns: Gaussian RA, Dec, major/minor axes, position angle
    '''
    fits_dir      = input_dir + 'WALLABY_' + galaxy + '/'
    mom0_fits     = 'WALLABY_' + galaxy + '_mom0.fits.gz'
    f1            = pyfits.open(fits_dir + mom0_fits)
    data, hdr     = f1[0].data, f1[0].header
    wcs           = WCS(hdr)
    pix_scale     = np.abs(hdr['CDELT1']) * 3600.
    
    # Create Model 2D Gaussian
    yme, xme = np.where(data==data.max())
    gauss = Gaussian2D(amplitude=data.max(), x_mean=xme, 
                        y_mean=yme)
    
    # Fit to Data using the Levenberg-Marquardt Algorithm
    fitter = LevMarLSQFitter() 
    y,x    = np.indices(data.shape)
    fit    = fitter(gauss, x, y, data, maxiter=1000, acc=1e-08)
    
    f1.close()
    
    # Define Output Parameters. Includes a correction for the effect of the 
    # synthesised beam to the major/minor axis lengths and set position angle 
    # to between 0 and 360 degrees.
    gaus_x      = fit.x_mean[0]
    gaus_y      = fit.y_mean[0]
    gaus_x_stdv = fit.x_stddev[0] * pix_scale
    gaus_y_stdv = fit.y_stddev[0] * pix_scale
    if gaus_x_stdv > gaus_y_stdv:
      if gaus_x_stdv > 35 and gaus_y_stdv > 35:
        gaus_maj = np.sqrt(gaus_x_stdv**2 - 30.**2)
        gaus_min = np.sqrt(gaus_y_stdv**2 - 30.**2)
      else:
        gaus_maj = gaus_x_stdv
        gaus_min = gaus_y_stdv
      gaus_maj_pix = fit.x_stddev[0]
      gaus_min_pix = fit.y_stddev[0]
      gaus_pa  = (fit.theta[0] - np.floor(fit.theta[0]/(2.*math.pi))*2.*math.pi)
    else:
      if gaus_x_stdv > 35 and gaus_y_stdv > 35:
        gaus_min = np.sqrt(gaus_x_stdv**2 - 30.**2)
        gaus_maj = np.sqrt(gaus_y_stdv**2 - 30.**2)
      else:
        gaus_min = gaus_x_stdv
        gaus_maj = gaus_y_stdv
      gaus_min_pix = fit.x_stddev[0]
      gaus_maj_pix = fit.y_stddev[0]
      gaus_pa  = (fit.theta[0] - np.floor(fit.theta[0]/(2.*math.pi))*2.*math.pi) + math.pi/2.
    
    gaus_ra, gaus_dec = wcs.all_pix2world(gaus_x, gaus_y, 0)
    
    return(gaus_ra, gaus_dec, gaus_maj, gaus_min, gaus_pa)

# ================================= #
# ======= HI Radial Profile ======= #
# ================================= #
def measure_hi_radial_profile(input_dir, galaxy, aperture_params, fignum):
    '''
    input_dir:        WALLABY data products directory
    
    galaxy:           Galaxy name
    
    aperture_params:  Parameters for annuli/apertures
    
                      aperture_params[0]: RA
    
                      aperture_params[1]: Dec
    
                      aperture_params[2]: Gaussian major axis
    
                      aperture_params[3]: Gaussian minor axis
    
                      aperture_params[4]: Gaussian position angle
    
                      aperture_params[5]: SoFiA RMS
    
                      aperture_params[6]: WALLABY flux correction value
    
    fignum:           Figure number
    
    Returns:  Annulus - radius, area, flux, flux error, 
                        total surface density, average surface density
                        
              Aperture - radius, area, flux, flux error, 
                         total surface density, average surface density
    '''
    fits_dir      = input_dir + 'WALLABY_' + galaxy + '/'
    mom0_fits     = 'WALLABY_' + galaxy + '_mom0.fits.gz'
    
    if os.path.isfile(fits_dir + mom0_fits):
      f1                 = pyfits.open(fits_dir + mom0_fits)
      data, hdr          = f1[0].data, f1[0].header
      wcs                = WCS(hdr)
      naxis1             = hdr['NAXIS1']
      naxis2             = hdr['NAXIS2']
      pix_scale          = np.abs(hdr['CDELT1']) * 3600.
      
      ra                 = aperture_params[0]
      dec                = aperture_params[1]
      ellmaj             = aperture_params[2]
      ellmin             = aperture_params[3]
      ellpa              = aperture_params[4]
      hi_rms             = aperture_params[5]
      flux_corr          = aperture_params[6]
      
      axis_ratio         = ellmin / ellmaj
      
      # Get x,y pixel position from RA and Dec
      position           = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
      x_pix, y_pix       = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
      
      aperture_list      = []
      annulus_list       = []
      area_aperture_list = []
      area_annulus_list  = []
      
      # Set major/minor axis lengths and separations for defining apertures/annuli
      ring_max           = np.ceil(ellmaj*10. / 7.5) / 2
      scale_list         = np.arange(0,ring_max,1)
      a_ps               = 7.5 / pix_scale
      b_ps               = a_ps * axis_ratio
      
      radius_aperture    = scale_list[1:-1] * a_ps * pix_scale
      radius_annulus     = scale_list[2:] * a_ps * pix_scale
      
      # Define apertures and annuli
      for i in range(1, len(scale_list)-1):
        aperture         = EllipticalAperture((x_pix, y_pix), 
                                              scale_list[i]*a_ps, 
                                              scale_list[i]*b_ps, 
                                              theta=ellpa)
        annulus          = EllipticalAnnulus((x_pix, y_pix), 
                                             a_in=scale_list[i]*a_ps, 
                                             a_out=scale_list[i+1]*a_ps, 
                                             b_out=scale_list[i+1]*b_ps, 
                                             b_in=scale_list[i]*b_ps, 
                                             theta=ellpa)
        aperture_list.append(aperture)
        annulus_list.append(annulus)
        area_aperture_list.append(aperture.area)
        area_annulus_list.append(annulus.area)
      
      # Set error in the image as the HI cube rms
      error_ap                = data / data * hi_rms
      
      # Measure aperture and annulus photometry
      phot_table_aperture     = aperture_photometry(data, aperture_list, error=error_ap)
      phot_table_annulus      = aperture_photometry(data, annulus_list, error=error_ap)
      
      aperture_flux_list      = []
      annulus_flux_list       = []
      aperture_err_list       = []
      annulus_err_list        = []
      
      # Extract the aperture/annulus flux and flux error
      for i in range(len(annulus_list)):
        string_name = 'aperture_sum_%s' % (i)
        aperture_flux_list.append(phot_table_aperture[string_name][0])
        annulus_flux_list.append(phot_table_annulus[string_name][0])
        string_name = 'aperture_sum_err_%s' % (i)
        aperture_err_list.append(phot_table_aperture[string_name][0])
        annulus_err_list.append(phot_table_annulus[string_name][0])
      
      f1.close()
      
      # Calculate total uncertainty in each aperture/annulus
      aperture_err_list = np.array(aperture_err_list) * np.array(area_aperture_list) * 1000.
      annulus_err_list  = np.array(annulus_err_list) * np.array(area_annulus_list) * 1000.
      
      # Calculate the annulus average HI surface density - is the radial surface density profile.
      annulus_sd_tot    = np.array(annulus_flux_list) * 8.01 * 10**-21 * 2.33 * 10**20 / 900.
      annulus_surfden   = annulus_sd_tot / np.array(area_annulus_list)
      
      # Calculate the aperture average HI surface density
      aperture_sd_tot   = np.array(aperture_flux_list) * 8.01 * 10**-21 * 2.33 * 10**20 / 900.
      aperture_surfden  = aperture_sd_tot / np.array(area_aperture_list)
      
      return(np.array(radius_annulus), 
            np.array(area_annulus_list), 
            np.array(annulus_flux_list),
            np.array(annulus_err_list),
            annulus_sd_tot,
            annulus_surfden,
            np.array(radius_aperture), 
            np.array(area_aperture_list), 
            np.array(aperture_flux_list), 
            np.array(aperture_err_list),
            aperture_sd_tot,
            aperture_surfden)


# ================================= #
# ==== Measure HI Size/SurfDen ==== #
# ================================= #
def derive_hi_size(aperture_array, annulus_array, gaussian_array):
    '''
    aperture_array: Measured aperture values
                    
                    aperture_array[0]: Aperture radius
    
                    aperture_array[1]: Aperture total surface density

                    aperture_array[2]: Aperture average surface density
    
                    aperture_array[3]: Aperture flux
                    
    annulus_array:  Measured annulus values
                    
                    annulus_array[0]: Annulus radius
    
                    annulus_array[1]: Annulus total surface density
    
                    annulus_array[2]: Annulus average surface density
    
                    annulus_array[3]: Annulus flux
                    
    gaussian_array: Gaussian parameters
    
                    gaussian_array[0]: Gaussian major axis
    
                    gaussian_array[1]: Gaussian minor axis
    
    Returns:  Isodensity radius, surface density, diameter, radius error
              Effective radius, surface density, 
              isodensity radius (uncorrected for beam), effective radius (uncorrected for beam)
    '''
    aperture_radius   = aperture_array[0]
    aperture_total    = aperture_array[1]
    aperture_surfden  = aperture_array[2]
    flux_total        = aperture_array[3]
    annulus_radius    = annulus_array[0]
    annulus_surfden   = annulus_array[1]
    annulus_flux      = annulus_array[2]
    annulus_err       = annulus_array[3]
    gaus_maj          = gaussian_array[0]
    gaus_min          = gaussian_array[1]
    
    # Determine the radius containing 50% of the total 
    # flux (effect, r50) and correct for the effect of
    # the synthesised beam.
    aperture_max       = np.nanmax(aperture_total)
    aperture_eff       = aperture_total / aperture_max
    func_reff          = interp1d(aperture_eff, aperture_radius, fill_value='extrapolate')
        
    for i in range(len(aperture_radius) - 1):
      if aperture_eff[i] < 0.5 and aperture_eff[i+1] > 0.5:
        radius_interp_eff     = interpolate_value(aperture_radius, 
                                            aperture_eff, 
                                            0.5, i)
    
    radius_interp_eff  = np.float(radius_interp_eff)
    
    radius_eff         = np.sqrt(radius_interp_eff**2 - 15.**2)
    
    # Calculate the average HI surface density within r50.
    func_sd_eff        = interp1d(aperture_radius, aperture_surfden, fill_value='extrapolate')
    surfden_eff        = func_sd_eff(radius_eff)
    
    # Determine the 1 Msol/pc^2 isodensity radius, radius 
    # uncertainty and diameter and correct for the effect 
    # of the synthesised beam.
    if np.nanargmax(annulus_surfden) != 0:
      id_max = np.nanargmax(annulus_surfden)
      func_interp       = interp1d(annulus_surfden[id_max:], annulus_radius[id_max:], fill_value='extrapolate')
    else:
      func_interp       = interp1d(annulus_surfden, annulus_radius, fill_value='extrapolate')
    
    if np.nanmax(annulus_surfden) < 1:
      radius_interp_iso = np.nan
    else:
      radius_interp_iso = func_interp(1.0).astype(float)
    
    frac_error        = np.nanmean(annulus_err / annulus_flux)**2
    
    radius_iso_err    = radius_interp_iso * np.sqrt(frac_error)
    
    diameter_iso      = 2. * radius_interp_iso
    diameter_iso      = np.sqrt((diameter_iso)**2 - 30.**2)
    
    radius_iso        = diameter_iso / 2.
    
    # Calculate the average HI surface density within the isodensity radius.
    func_sd_iso       = interp1d(aperture_radius, aperture_surfden, fill_value='extrapolate')
    surfden_iso       = func_sd_iso(radius_iso)
    
    return(radius_iso, surfden_iso, diameter_iso, radius_iso_err, 
           radius_eff, surfden_eff, radius_interp_iso, radius_interp_eff)


# ================================= #
# ======= HI Radial Profile ======= #
# ================================= #
def derive_hi_optdisc(input_dir, ps_dir, galaxy, aperture_params):
    '''
    input_dir:        WALLABY data products directory
    
    ps_dir:           PanSTARRS directory
    
    galaxy:           Galaxy name
    
    aperture_params:  Parameters for annuli/apertures
    
                      aperture_params[0]: PanSTARRS segment x position [pixel]
    
                      aperture_params[1]: PanSTARRS segment y position [pixel]

                      aperture_params[2]: PanSTARRS radius [arcsec]

                      aperture_params[3]: PanSTARRS segment axis ratio b/a

                      aperture_params[4]: PanSTARRS segment position angle [degrees]
    
    Returns:  Aperture flux, average surface density
    '''
    xpos            = aperture_params[0]
    ypos            = aperture_params[1]
    radius          = aperture_params[2]
    ba              = aperture_params[3]
    pa              = aperture_params[4] * math.pi / 180.
    
    fits_dir        = input_dir + 'WALLABY_' + galaxy + '/'
    mom0_fits       = 'WALLABY_' + galaxy + '_mom0.fits.gz'
    
    fits_dir_ps     = ps_dir + galaxy +'/'
    rband_fits      = galaxy + '_r.fits'
    
    f1              = pyfits.open(fits_dir + mom0_fits, memmap=False)
    data, hdr       = f1[0].data, f1[0].header
    wcs             = WCS(hdr)
    asec_p_pix      = np.abs(hdr['CDELT2']) * 3600.
    
    f2              = pyfits.open(fits_dir_ps + rband_fits, memmap=False)
    hdr_ps          = f2[0].header
    wcs_ps          = WCS(hdr_ps)
    
    # Convert optical image pixel position centre to HI image pixel position
    ra_cen, dec_cen = wcs_ps.all_pix2world(xpos, ypos, 0)
    position        = SkyCoord(ra_cen*u.deg, dec_cen*u.deg, frame='icrs')
    x_pix, y_pix    = wcs.all_world2pix(position.ra.deg, position.dec.deg, 0)
    
    # Define and fit optical aperture of HI moment 0 map
    a               = radius / asec_p_pix
    b               = a * ba
    aperture        = EllipticalAperture((x_pix, y_pix), a, b, theta=pa)
    phot_table      = aperture_photometry(data, aperture)
    aperture_flux   = phot_table['aperture_sum'][0]
    
    # Calculate the average HI surface density within the optical aperture.
    aperture_sd_tot   = aperture_flux * 8.01 * 10**-21 * 2.33 * 10**20 / 900.
    aperture_surfden  = aperture_sd_tot / aperture.area * ba
    
    return (aperture_flux, aperture_surfden)


def save_table_function(table_name, table_data, table_cols):
    table      = Table(table_data, names=table_cols)
    table.write(table_name, format = 'fits')


# ================================ #
# === Surface Brightness Plot ==== #
# ================================ #
def surfden_profile_plot(fig_num, sub1, sub2, sub3, flux1, flux2, radius_iso1, radius_iso, radius_eff1, radius_eff):
    '''
    Plot HI radial surface density profile and overlay lines indicating the isodensity and effective radii.
    '''
    matplotlib.rcParams.update({'font.size': 12})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    ax1 = fig_num.add_subplot(sub1, sub2, sub3, facecolor = 'w')
    ax1.set_ylabel(r'$\Sigma/\mathrm{M}_{\odot}\mathrm{pc}^{-2}$')
    ax1.set_xlabel(r'Radius [arcsec]')
    plt.axvline(radius_iso, color = 'peru', linestyle = '--', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_iso1, color = 'sandybrown', linestyle = '--', linewidth = 1, zorder = 0)
    plt.axvline(radius_eff, color = 'mediumvioletred', linestyle = '-.', linewidth = 1.5, zorder = 0)
    plt.axvline(radius_eff1, color = 'violet', linestyle = '-.', linewidth = 1, zorder = 0)
    plt.axhline(1, color = 'black', linestyle = ':', linewidth=1, zorder = 0)
    ax1.plot(flux1, flux2, color='darkblue', linewidth=2, linestyle='-')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax1.get_yaxis().set_tick_params(which='both', direction='in')
    ax1.get_xaxis().set_tick_params(which='both', direction='in')



# ================================= #
# =========== CONTSTANTS ========== #
# ================================= #
C_LIGHT  = const.c.to('km/s').value
H0       = cosmo.H(0).value
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
have_optical             = False         # True to open *_panstarrs_photometry.fits if exists

# ++++ ONLY RUN ONE AT A TIME +++++ #
table_add_flags          = False         # Only set True once to create SoFiA catalogue with flag columns

do_measure_hi            = False          # True to measure HI structural properties
do_hi_opt_disc           = False         # True to measure HI mass/surface density w/in optical disc

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
if ~have_optical:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  #data_join      = hdu_sofia[1].data
  
  data_join      = join(data_sofia, data_sofia, join_type='left')
  
if have_optical:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_panstarrs = parameter_dir + '%s_panstarrs_photometry.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_panstarrs  = fits.open(fits_panstarrs)
  data_panstarrs = hdu_panstarrs[1].data
  
  data_join      = join(data_sofia, data_panstarrs, join_type='left')
  

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
  
  BEAM           = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  print(BEAM)
  print(beam_factor(30., 30., pix_scale*3600.))
  
  redshift       = (HI_REST / (data_mask['freq'] / 1.e6)) - 1.
  galaxies       = np.array(gal_name)
  sofia_ra       = data_mask['ra']
  sofia_dec      = data_mask['dec']
  sofia_vsys     = redshift * C_LIGHT
  sofia_rms      = data_mask['rms']
  sofia_sint     = data_mask['f_sum'] * chan_width / chan_width_hz
  sofia_snr      = data_mask['f_sum'] / data_mask['err_f_sum']
  sofia_kinpa    = data_mask['kin_pa']
  sofia_w20      = data_mask['w20'] / chan_width_hz * chan_width
  sofia_w50      = data_mask['w50'] / chan_width_hz * chan_width
  sofia_ell_maj  = data_mask['ell_maj'] * pix_scale * 3600.
  sofia_ell_min  = data_mask['ell_min'] * pix_scale * 3600.
  sofia_ellpa    = data_mask['ell_pa']
  
  #Correction for measured WALLABY fluxes
  sint_corr      = np.array(wallaby_flux_scaling(data_mask['f_sum']))
  scale_factor   = np.array(data_mask['f_sum'] / sint_corr)
  
  if have_optical:
    ps_x         = data_mask['SEG_X']
    ps_y         = data_mask['SEG_Y']
    ps_ba        = data_mask['SEG_BA']
    ps_pa        = data_mask['SEG_PA']
    ps_radius25  = data_mask['RADIUS_R_ISO25']
    ps_radius50  = data_mask['RADIUS_R_50']
    ps_radius90  = data_mask['RADIUS_R_90']

print(len(galaxies))


# ================================= #
# == Create Catalogue with Flags == #
# ================================= #
if table_add_flags:
  table_str  = sofia_dir + '%s_catalogue_flags.fits' % team_release
  if not os.path.isfile(table_str):
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    flag_src_class     = np.full(len(galaxies), 1)
    flag_opt_fit       = np.full(len(galaxies), 0)
    segment_deblend    = np.full(len(galaxies), 0)
    
    do_seg_par1        = np.full(len(galaxies), 0)
    do_seg_par2        = np.full(len(galaxies), 5.)
    do_seg_par3        = np.full(len(galaxies), 10.)
    do_seg_par4        = np.full(len(galaxies), 2.)
    do_seg_par5        = np.full(len(galaxies), 10)
    
    do_seg_id1         = np.full(len(galaxies), 0)
    do_seg_id2         = np.full(len(galaxies), 0)
    
    do_deblend1        = np.full(len(galaxies), 0)
    do_deblend2        = np.full(len(galaxies), 0.01)
    do_deblend3        = np.full(len(galaxies), 10)
    
    
    col_data           = [flag_src_class, flag_opt_fit, segment_deblend, 
                          do_seg_par1, do_seg_par2, do_seg_par3, do_seg_par4, do_seg_par5,
                          do_seg_id1, do_seg_id2,
                          do_deblend1, do_deblend2, do_deblend3]
    col_name           = ['flag_src_class', 'flag_opt_fit', 'segment_deblend', 
                          'do_seg_par1', 'do_seg_par2', 'do_seg_par3', 'do_seg_par4', 'do_seg_par5',
                          'do_seg_id1', 'do_seg_id2',
                          'do_deblend1', 'do_deblend2', 'do_deblend3']
    
    for i in range(len(col_name)):
      tdata.append(col_data[i])
      tcols.append(col_name[i])
    
    save_table_function(table_str, tdata, tcols)
    
    fits_sofia_old = sofia_dir + '%s_catalogue_original.fits' % team_release
    os.system('mv %s %s' % (fits_sofia, fits_sofia_old))
    
    table_str  = sofia_dir + '%s_catalogue.fits' % team_release
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
        
    save_table_function(table_str, tdata, tcols)
  else:
    print('====== Catalogue File w/ Flags Exists ======')


print(BEAM)

# ================================= #
# ===== Measure HI Structure ====== #
# ================================= #
if do_measure_hi:
  #flux_scale_factor = 0.80
  do_save_table           = True
  counter                 = 0
  radius_iso              = np.full(len(galaxies), np.nan)
  diameter_iso            = np.full(len(galaxies), np.nan)
  gaus_ra                 = np.full(len(galaxies), np.nan)
  gaus_dec                = np.full(len(galaxies), np.nan)
  gaus_maj                = np.full(len(galaxies), np.nan)
  gaus_min                = np.full(len(galaxies), np.nan)
  gaus_pa                 = np.full(len(galaxies), np.nan)
  hi_ba                   = np.full(len(galaxies), np.nan)
  radius_eff              = np.full(len(galaxies), np.nan)
  surfden_eff             = np.full(len(galaxies), np.nan)
  surfden_iso             = np.full(len(galaxies), np.nan)
  radius_iso_err          = np.full(len(galaxies), np.nan)
  flux_integrate          = np.full(len(galaxies), np.nan)
  radius_iso_scale        = np.full(len(galaxies), np.nan)
  radius_iso_err_scale    = np.full(len(galaxies), np.nan)
  diameter_iso_scale      = np.full(len(galaxies), np.nan)
  radius_eff_scale        = np.full(len(galaxies), np.nan)
  surfden_eff_scale       = np.full(len(galaxies), np.nan)
  surfden_iso_scale       = np.full(len(galaxies), np.nan)
  flux_integrate_scale    = np.full(len(galaxies), np.nan)
  sd_interp               = []
  for i in range(len(galaxies)):
    print(i, galaxies[i], np.round(scale_factor[i],3))
    counter += 1
    fig1 = plt.figure(1, figsize=(10, 8))
    fig2 = plt.figure(2, figsize=(6, 4))
    fig3 = plt.figure(3, figsize=(6, 4))
    
    # =========== Fit 2D Gaussian model to moment 0 map =========== #
    gaus_model    = gaussian_fit(dataprod_dir, galaxies[i])
    
    gaus_ra[i]    = gaus_model[0]
    gaus_dec[i]   = gaus_model[1]
    gaus_maj[i]   = gaus_model[2]
    gaus_min[i]   = gaus_model[3]
    gaus_pa[i]    = gaus_model[4]
    hi_ba[i]      = gaus_min[i] / gaus_maj[i]
    
    aperture_params  = [gaus_ra[i], gaus_dec[i], gaus_maj[i], gaus_min[i], 
                        gaus_pa[i], sofia_rms[i], scale_factor[i]]
    
    # =========== Fit annuli/apertures to moment 0 map =========== #
    measured_param   = measure_hi_radial_profile(dataprod_dir, galaxies[i], aperture_params, 'none')
    
    annulus_radius   = np.array(measured_param[0])
    annulus_area     = np.array(measured_param[1])
    annulus_flux     = np.array(measured_param[2])
    annulus_err      = np.array(measured_param[3])
    annulus_surfden  = np.array(measured_param[5]) * hi_ba[i]      # Apply inclination correction
    aperture_radius  = np.array(measured_param[6])
    aperture_area    = np.array(measured_param[7])
    flux_total       = np.array(measured_param[8])
    apeture_err      = np.array(measured_param[9])
    aperture_total   = np.array(measured_param[10])
    aperture_surfden = np.array(measured_param[11]) * hi_ba[i]      # Apply inclination correction
    
    annulus_surfden   = np.array(annulus_surfden, dtype=float)
    aperture_total    = np.array(aperture_total, dtype=float)
    aperture_surfden  = np.array(aperture_surfden, dtype=float)
    
    # =========== Save aperture/annulus HI profiles to file (1 per galaxy) =========== #
    table_str  = hi_products_dir + 'PROFILES/' + galaxies[i] + '_profile.fits'
    os.system('rm -rf %s' % table_str)
    
    tdata = [annulus_radius, annulus_area, 
              annulus_flux, annulus_err, annulus_surfden, 
              aperture_radius, aperture_area, 
              flux_total, apeture_err,
              aperture_total, aperture_surfden]
    
    tcols = ('RADIUS_ANNULUS', 'AREA_ANNULUS', 
              'FLUX_ANNULUS', 'ERROR_ANNULUS', 'SURFACE_DENSITY_ANNULUS', 
              'RADIUS_APERTURE', 'AREA_APERTURE', 
              'FLUX_APERTURE', 'ERROR_APERTURE', 
              'APERTURE_TOTAL', 'SURFACE_DENSITY_APERTURE')
    
    save_table_function(table_str, tdata, tcols)
    
    # =========== Derive HI sizes and surface densities for original (uncorrected) data =========== #
    aperture_array           = [aperture_radius, aperture_total, aperture_surfden, flux_total]
    annulus_array            = [annulus_radius, annulus_surfden, annulus_flux, annulus_err]
    gaussian_array           = [gaus_maj[i], gaus_min[i]]
    
    size_array               = derive_hi_size(aperture_array, annulus_array, gaussian_array)
    
    radius_iso[i]            = size_array[0]
    surfden_iso[i]           = size_array[1]
    diameter_iso[i]          = size_array[2]
    radius_iso_err[i]        = size_array[3]
    radius_eff[i]            = size_array[4]
    surfden_eff[i]           = size_array[5]
    radius_interp_iso        = size_array[4]
    radius_interp_eff        = size_array[5]
    
    flux_integrate[i]        = np.nanmax(flux_total)
    
    # =========== Derive HI sizes and surface densities for flux corrected (scaled) data =========== #
    annulus_surfden_scale    = annulus_surfden / scale_factor[i]
    aperture_surfden_scale   = aperture_surfden / scale_factor[i]
    
    aperture_array           = [aperture_radius, aperture_total, aperture_surfden_scale, flux_total]
    annulus_array            = [annulus_radius, annulus_surfden_scale, annulus_flux, annulus_err]
    gaussian_array           = [gaus_maj[i], gaus_min[i]]
    
    size_array               = derive_hi_size(aperture_array, annulus_array, gaussian_array)
    
    radius_iso_scale[i]      = size_array[0].astype(float)
    surfden_iso_scale[i]     = size_array[1]
    diameter_iso_scale[i]    = size_array[2]
    radius_iso_err_scale[i]  = size_array[3]
    radius_eff_scale[i]      = size_array[4]
    surfden_eff_scale[i]     = size_array[5]
    
    #beam_area                = (math.pi * 30. * 30.) / (4. * np.log(2.)) / 36.
    
    flux_integrate_scale[i]  = np.nanmax(flux_total) / scale_factor[i]
    
    surfden_profile_plot(fig3, 1, 1, 1, 
                          annulus_radius, annulus_surfden, 
                          radius_interp_iso, radius_iso[i], 
                          radius_interp_eff, radius_eff[i])
    plot_name = plots_dir + 'PHOTOMETRY/HI_PROFILES/%s_profile.pdf' % galaxies[i]
    fig3.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
    fig3.clf()
  
  diameter_minor = diameter_iso * hi_ba
  
  # =========== Save HI structural parameters to file =========== #
  if do_save_table:
    table_str  = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
    os.system('rm -rf %s' % table_str)
  
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [galaxies, radius_iso, surfden_iso, radius_iso_err, 
              radius_eff, surfden_eff, 
              diameter_iso, diameter_minor, hi_ba, flux_integrate,
              radius_iso_scale, surfden_iso_scale, radius_iso_err_scale, 
              radius_eff_scale, surfden_eff_scale, 
              diameter_iso_scale, flux_integrate_scale,
              gaus_ra, gaus_dec, gaus_maj, gaus_min, gaus_pa]
    
    tcols_1 = ('OBJECTS', 'RADIUS_ISO', 'SURFACE_DENSITY_ISO', 'RADIUS_ISO_ERR', 
              'RADIUS_EFF', 'SURFACE_DENSITY_EFF', 
              'DIAMETER_MAJOR', 'DIAMETER_MINOR', 'AXIS_RATIO_BA', 'TOTAL_FLUX',
              'RADIUS_ISO_SCALE', 'SURFACE_DENSITY_ISO_SCALE', 'RADIUS_ISO_ERR_SCALE', 
              'RADIUS_EFF_SCALE', 'SURFACE_DENSITY_EFF_SCALE', 
              'DIAMETER_MAJOR_SCALE', 'TOTAL_FLUX_SCALE',
              'GAUSSIAN_RA', 'GAUSSIAN_DEC', 'GAUSSIAN_MAJ', 'GAUSSIAN_MIN', 'GAUSSIAN_PA')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
    
    save_table_function(table_str, tdata, tcols)
      


# ================================= #
# ====== Measure HI Opt Disc ====== #
# ================================= #
if do_hi_opt_disc:
  do_save_table   = True
  counter         = 0
  flux_ps25       = np.full(len(galaxies), np.nan)
  flux_ps50       = np.full(len(galaxies), np.nan)
  flux_ps90       = np.full(len(galaxies), np.nan)
  surfden_ps25    = np.full(len(galaxies), np.nan)
  surfden_ps50    = np.full(len(galaxies), np.nan)
  surfden_ps90    = np.full(len(galaxies), np.nan)
  
  for i in range(len(galaxies)):
    print(galaxies[i])
    # =========== Measure HI flux within isophotal 25 mag/arcsec^2 aperture =========== #
    if np.isfinite(ps_radius25[i]):
      ps_aperture_params = [ps_x[i], ps_y[i], ps_radius25[i], ps_ba[i], ps_pa[i]]
      
      flux_ps25[i], surfden_ps25[i] = derive_hi_optdisc(dataprod_dir, panstarrs_dir, 
                                                        galaxies[i], ps_aperture_params)
      
      flux_ps25[i]       = flux_ps25[i] / BEAM / scale_factor[i]      # Correct flux for beam and scale
      surfden_ps25[i]    = surfden_ps25[i] / scale_factor[i]          # Scale flux
    
    # =========== Measure HI flux within effective (r50) aperture =========== #
    if np.isfinite(ps_radius50[i]):
      ps_aperture_params = [ps_x[i], ps_y[i], ps_radius50[i], ps_ba[i], ps_pa[i]]
      
      flux_ps50[i], surfden_ps50[i] = derive_hi_optdisc(dataprod_dir, panstarrs_dir, 
                                                        galaxies[i], ps_aperture_params)
      
      flux_ps50[i]       = flux_ps50[i] / BEAM / scale_factor[i]      # Correct flux for beam and scale
      surfden_ps50[i]    = surfden_ps50[i] / scale_factor[i]          # Scale flux
      
    # =========== Measure HI flux within r90 aperture =========== #
    if np.isfinite(ps_radius50[i]):
      ps_aperture_params = [ps_x[i], ps_y[i], ps_radius90[i], ps_ba[i], ps_pa[i]]
      
      flux_ps90[i], surfden_ps90[i] = derive_hi_optdisc(dataprod_dir, panstarrs_dir, 
                                                        galaxies[i], ps_aperture_params)
      
      flux_ps90[i]       = flux_ps90[i] / BEAM / scale_factor[i]      # Correct flux for beam and scale
      surfden_ps90[i]    = surfden_ps90[i] / scale_factor[i]          # Scale flux
      
  # =========== Save HI flux in optical disc to file =========== #
  if do_save_table:
    table_str  = parameter_dir + '%s_hi_optical_disc.fits' % team_release
    os.system('rm -rf %s' % table_str)
  
    tdata = []
    tcols = []
    for i in range(len(data_join.columns)):
      if i < len(data_sofia.columns.names):
        tdata.append(data_join[data_join.columns[i].name])
        tcols.append(data_join.columns[i].name)
    
    tdata_1 = [flux_ps25, surfden_ps25, flux_ps50, surfden_ps50, flux_ps90, surfden_ps90]
    
    tcols_1 = ('FLUX_INNER_PS25', 'SURFACE_DENSITY_INNER_PS25', 
               'FLUX_INNER_PS50', 'SURFACE_DENSITY_INNER_PS50', 
               'FLUX_INNER_PS90', 'SURFACE_DENSITY_INNER_PS90')
    
    for i in range(len(tdata_1)):
      tdata.append(tdata_1[i])
      tcols.append(tcols_1[i])
    
    save_table_function(table_str, tdata, tcols)
  
  
















