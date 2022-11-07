# Libraries

import warnings
warnings.simplefilter("ignore")

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import astropy.stats
from matplotlib import rc
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.table import join
import aplpy

#from functions_plotting import *
from functions_calculations import *



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
    c_hdr_cards = ['PC001001', 'PC001002', 'PC002001', 'PC002002']
    for i in range(len(c_hdr_cards)):
      if c_hdr_cards[i] in hdr:
        del hdr[c_hdr_cards[i]]
    f1.flush()
    f1.close()


# ================================ #
# ======= Opt/Contour Plot ======= #
# ================================ #
def wallaby_galaxy_summary_plot(fig_num, subfig1, subfig2, subfig3, image1, image2, colour, txt_array):
    '''
    fig_num:    Figure number
    
    subfig1:    Number of rows
    
    subfig2:    Number of columns
    
    subfig3:    Subfigure (panel) number -- starts from 1
    
    image1:     Primary/background image (FITS file name - string) OR
    
                Integrated spectrum array 
                
                image1[0]: velocity
    
                image1[1]: flux

                image1[2]: error
    
    image2:     Secondary/overlay image used to generate contours (FITS file name)
    
    colour:     Integrated spectrum line colour
    
    txt_array:  Array of parameters/properties to print or use for plotting ellipses
                
                txt_array[0]: galaxy name, 
    
                txt_array[1]: sofia ID 
    
                txt_array[2]: systemic velocity 
    
                txt_array[3]: SoFiA RA [degrees]
    
                txt_array[4]: SoFiA Dec [degrees]
    
                txt_array[5]: log(HI mass) 
    
                txt_array[6]: HI major axis (diameter [arcsec]
    
                txt_array[7]: HI minor axis [arcsec]

                txt_array[8]: 2D Gaussian fit position angle [radians]
    
                txt_array[9]: 2D Gaussian fit RA [degrees]
    
                txt_array[10]: 2D Gaussian fit Dec [degrees]
    '''
    matplotlib.rcParams.update({'font.size': 15})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)
    if type(image1) == str:
      ax1 = aplpy.FITSFigure(image1, figure=fig_num, subplot=(subfig1, subfig2, subfig3), dimensions=(0,1))
      if subfig3 == 1:
        f1        = pyfits.open(image1)
        data1, hdr1 = f1[0].data, f1[0].header
        wcs         = WCS(hdr1)
        ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
        f2        = pyfits.open(image2)
        data, hdr = f2[0].data, f2[0].header
        wcs2         = WCS(hdr)
        pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
        arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
        box_side      = int(np.ceil(arcmin_per_pixel * pixels))
      if subfig3 == 2:
        f2        = pyfits.open(image2)
        data, hdr = f2[0].data, f2[0].header
        pixels        = np.max((hdr['NAXIS1'], hdr['NAXIS2'])) + 5
        arcmin_per_pixel = np.abs(hdr['CDELT2']) * 60.
        box_side      = int(np.ceil(arcmin_per_pixel * pixels))
        try:
          if txt_array[0] == 'J103729-261901':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103726-261843':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103406-270617':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103542-284604':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103545-284609':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103537-284607':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          else:
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
        except KeyError:
          beam_maj = 0
          for i in range(len(hdr['HISTORY'])):
            if beam_maj == 0:
              beam_hist = str(hdr['HISTORY'][i:i+1]).split()
              for beam_i in range(len(beam_hist)):
                if beam_hist[beam_i] == 'BMAJ=':
                  beam_maj = float(beam_hist[beam_i + 1])
                elif beam_hist[beam_i] == 'BMIN=':
                  beam_min = float(beam_hist[beam_i + 1])
                elif beam_hist[beam_i] == 'BPA=':
                  beam_pa = float(beam_hist[beam_i + 1])
        bmaj, bmin = beam_maj*60.*60., beam_min*60.*60.
        f1        = pyfits.open(image1)
        data1, hdr1 = f1[0].data, f1[0].header
        ax1.show_colorscale(vmin=5, pmax=99.75, stretch='arcsinh', cmap='Greys')
        ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
      if subfig3 == 3:
        f1               = pyfits.open(image1)
        data, hdr        = f1[0].data, f1[0].header
        f2               = pyfits.open(image2)
        data2, hdr2      = f2[0].data, f2[0].header
        pixels           = np.max((hdr2['NAXIS1'], hdr2['NAXIS2'])) + 5
        arcmin_per_pixel = np.abs(hdr2['CDELT2']) * 60.
        box_side         = int(np.ceil(arcmin_per_pixel * pixels))
        ax1.show_colorscale(pmin=5, pmax=95, cmap='RdBu_r')
        try:
          if txt_array[0] == 'J103729-261901':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103726-261843':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103406-270617':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103542-284604':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103545-284609':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          elif txt_array[0] == 'J103537-284607':
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], 0.0
          else:
            beam_maj, beam_min, beam_pa = hdr['BMAJ'], hdr['BMIN'], hdr['BPA']
        except KeyError:
          beam_maj = 0
          for i in range(len(hdr['HISTORY'])):
            if beam_maj == 0:
              beam_hist = str(hdr['HISTORY'][i:i+1]).split()
              for beam_i in range(len(beam_hist)):
                if beam_hist[beam_i] == 'BMAJ=':
                  beam_maj = float(beam_hist[beam_i + 1])
                elif beam_hist[beam_i] == 'BMIN=':
                  beam_min = float(beam_hist[beam_i + 1])
                elif beam_hist[beam_i] == 'BPA=':
                  beam_pa = float(beam_hist[beam_i + 1])
        bmaj, bmin = beam_maj*60.*60., beam_min*60.*60.
        ax1.add_beam(major=beam_maj, minor=beam_min, angle=beam_pa, fill=True, color='peru')
      #position      = SkyCoord(txt_array[3]*u.deg, txt_array[4]*u.deg, frame='icrs')
      position      = SkyCoord(txt_array[9]*u.deg, txt_array[10]*u.deg, frame='icrs')
      width, height = box_side/60. - box_side/3./60., box_side/60. - box_side/3./60.
      #print(txt_array[3], txt_array[4], txt_array[9], txt_array[10])
      ax1.recenter(position.ra, position.dec, width=width, height=height)
      if subfig3 == 3:
        #size_ratio = txt_array[6]/txt_array[9]
        #size_ratio_r27 = txt_array[6]/txt_array[12]
        ax1.add_label(3.1, 0.95, '%s (%i)' % (txt_array[0], txt_array[1]), relative=True)
        ax1.add_label(3.1, 0.83, r'$v_{\mathrm{sys}}=%.0f$\,km/s' % txt_array[2], relative=True)
        ax1.add_label(3.1, 0.71, r'$\log(M_{\mathrm{HI}})=%.1f$' % txt_array[5], relative=True)
        #ax1.add_label(3.1, 0.59, r'$\log(M_{\mathrm{*}})=%.1f$' % txt_array[11], relative=True)
        #ax1.add_label(3.3, 0.47, r'$\log(\mathrm{SFR})=%.2f$' % txt_array[3], relative=True)
        ##ax1.add_label(1.4, 0.15, r'$d_{\mathrm{HI,3\sigma}}=%.0f$' % txt_array[4], relative=True)
        #ax1.add_label(3.1, 0.47, r'$d_{\mathrm{HI}}/d_{b25}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[6], txt_array[9], size_ratio), relative=True)
        #ax1.add_label(3.1, 0.35, r'$d_{\mathrm{HI}}/d_{r23}=%.0f/%.0f\,\,(%.1f)$' % (txt_array[6], txt_array[12], size_ratio_r27), relative=True)
        ##ax1.add_label(1.45, 0.23, r'$d_{25}=%.0f$' % txt_array[11], relative=True)
        ##ax1.add_label(1.45, 0.23, r'$\mathrm{HI}/25=%.1f$' % size_ratio, relative=True)
        #ax1.add_label(3.1, 0.23, r'$\mathrm{DEF}_{\mathrm{HI}}=%.2f$' % txt_array[10], relative=True)
      if subfig3 == 2 or subfig3 == 6 or subfig3 == 10 or subfig3 == 14:
        min_flux = 1. * 10**19 * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.33 * 10**20) #1.36*21*21*1.823*1000*(10**18))
        #lvls = np.array([5, 10, 20, 50, 70, 100, 130])
        lvls = np.array([5, 20, 50, 100])
        lvls = lvls*min_flux
        ax1.show_contour(image2, colors='darkred', levels=lvls, slices=(0,1))
        min_flux = 1. * (bmaj * bmin) * math.pi / 4 / np.log(2) / (2.12) / np.cos(np.arcsin(np.sqrt(1-(txt_array[7]/txt_array[6])**2)))
        ax1.show_contour(image2, colors='red', linewidths=4, levels=[min_flux], slices=(0,1))
        ax1.show_ellipses(position.ra, position.dec, width=txt_array[6]/3600., height=txt_array[7]/3600., 
                          angle=txt_array[8], facecolor='none', edgecolor='blue', ls='--', 
                          zorder=2, linewidth=3, coords_frame='world')
      #if subfig3 == 1:
        #racen, deccen = wcs.all_pix2world(txt_array[14], txt_array[15], 0)
        #pos_ps        = SkyCoord(racen*u.deg, deccen*u.deg, frame='icrs')
        #ax1.show_ellipses(pos_ps.ra, pos_ps.dec, width=txt_array[12]/3600., height=txt_array[13]/3600., 
                          #angle=txt_array[16], facecolor='none', edgecolor='magenta', ls='-', 
                          #zorder=2, linewidth=3, coords_frame='world')
      if subfig3 == 1:
        dummy=0
      else:
        ax1.tick_labels.hide_y()
      ax1.axis_labels.hide_y()
      ax1.tick_labels.set_xformat('hh:mm:ss')
      ax1.tick_labels.set_yformat('dd:mm')
      ax1.ticks.show()
      ax1.ticks.set_xspacing(0.05)
      ax1.ticks.set_length(5)
      ax1.ticks.set_color('black')
      ax1.ticks.set_minor_frequency(1)
      plt.subplots_adjust(wspace=0.02, hspace=0.15)
    else:
      ax2 = fig_num.add_subplot(subfig1, subfig2, subfig3, facecolor = 'w')
      ax2.set_xlabel(r'Velocity [km\,s$^{-1}$]')
      ax2.set_ylabel(r'Flux [mJy]')
      velocity = image1[0]
      flux     = image1[1] * 1000.0
      error    = np.array(image1[2]) * 1000.0
      ax2.set_ylim(-1, np.nanmax(flux))
      ax2.set_xlim(np.nanmin(velocity)-10, np.nanmax(velocity)+10)
      ax2.axvline(txt_array[2], linewidth=0.75, linestyle = '--', color = 'darkgrey')
      ax2.plot(velocity, flux, linestyle = '-', color = colour, linewidth = 1.0)
      ax2.fill_between(velocity, flux-error, flux+error, alpha=0.5, edgecolor='none', facecolor='lightblue')
      ax2.tick_params(axis='both', direction='in')
      ax2.tick_params(axis='y', right=True, left=False, labelright=True, labelleft=False)
      ax2.yaxis.set_label_position('right')
      plt.subplots_adjust(wspace=0.02, hspace=0.15)



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


# ================================= #
# =========== Switches ============ #
# ================================= #
open_catalogue            = True          # Always True, provides input source parameters

do_plot_maps              = True          # True to plot summary plots

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
wise_dir                 = basedir + 'MULTIWAVELENGTH/unWISE/'
galex_dir                = basedir + 'MULTIWAVELENGTH/GALEX/'
parameter_dir            = basedir + 'PARAMETERS/'
hi_products_dir          = basedir + 'HI_DERIVED_PRODUCTS/'
plots_dir                = basedir + 'PLOTS/'
  

if open_catalogue:
  print('============ %s ============' % team_release)
  fits_sofia     = sofia_dir + '%s_catalogue.fits' % team_release
  fits_hi        = parameter_dir + '%s_hi_structural_parameters.fits' % team_release
  
  hdu_sofia      = fits.open(fits_sofia)
  data_sofia     = hdu_sofia[1].data
  
  hdu_hi         = fits.open(fits_hi)
  data_hi        = hdu_hi[1].data
  
  data_join      = join(data_sofia, data_hi, join_type='left')
  
  gal_name = []
  for i in range(len(data_join['name'])):
    split_name = data_join['name'][i][8:]
    gal_name.append(split_name)
  
  wallaby_gal_dir = 'WALLABY_' + gal_name[0] + '/'
  fits_file       = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[0] + '_cube.fits.gz'
  f1              = pyfits.open(fits_file)
  data, hdr       = f1[0].data, f1[0].header
  
  if hdr['CTYPE3'] == 'FREQ':
    chan_width = np.abs((HI_REST / (hdr['CRVAL3']/1e6)) - (HI_REST / ((hdr['CRVAL3']-hdr['CDELT3'])/1e6))) * C_LIGHT
    chan_width_hz = hdr['CDELT3']
  else:
    chan_width = np.abs(hdr['CDELT3']/1000.)
    
  beam_maj, beam_min, pix_scale  = hdr['BMAJ'], hdr['BMIN'], np.abs(hdr['CDELT1'])
  f1.close()
  
  BEAM           = beam_factor(beam_maj*3600., beam_min*3600., pix_scale*3600.)
  
  redshift       = (HI_REST / (data_join['freq'] / 1e6)) - 1.
  galaxies       = np.array(gal_name)
  sofia_ra       = data_join['ra']
  sofia_dec      = data_join['dec']
  sofia_vsys     = redshift * C_LIGHT
  sofia_sint     = data_join['f_sum'] * chan_width / chan_width_hz
  sofia_snr      = data_join['f_sum'] / data_join['err_f_sum']
  gaus_maj       = data_join['GAUSSIAN_MAJ']
  gaus_min       = data_join['GAUSSIAN_MIN']
  gaus_pa        = data_join['GAUSSIAN_PA'] * 180. / math.pi
  
  gaus_ra        = data_join['GAUSSIAN_RA']
  gaus_dec       = data_join['GAUSSIAN_DEC']
  
  hi_ba          = gaus_min / gaus_maj
  
  hi_dmaj        = data_join['DIAMETER_MAJOR']
  hi_dmin        = hi_dmaj * hi_ba
  
  sint_corr      = np.array(wallaby_flux_scaling(data_join['f_sum']))
  scale_factor   = np.array(data_join['f_sum'] / sint_corr)
  
  distance       = dist_lum(redshift)
  mhi            = hi_mass_jyhz(sint_corr, redshift)


# ================================= #
# ========== Plot Maps ============ #
# ================================= #
if do_plot_maps:
  rows           = 1
  cols           = 4
  
  for i in range(len(galaxies)):
    #if galaxies[i] == 'J104059-270456':
    if i > -1:
      txt_array = [galaxies[i], 0, sofia_vsys[i], 
                  sofia_ra[i], sofia_dec[i], mhi[i], 
                  hi_dmaj[i], hi_dmin[i], gaus_pa[i],
                  gaus_ra[i], gaus_dec[i]]
      print('%s -- %.1f' % (galaxies[i], sofia_vsys[i]))
      
      # =========== Image/Spectra Files to Open =========== #
      wallaby_gal_dir = 'WALLABY_' + gal_name[i] + '/'
      
      optical_file = panstarrs_dir + galaxies[i] + '/' + galaxies[i] + '_r.fits'
      
      remove_hdr_cards(optical_file)
      
      mom0_file    = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_mom0.fits.gz'
      mom1_file    = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_mom1.fits.gz'
      spec_file    = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_spec.txt'
      cube_file    = dataprod_dir + wallaby_gal_dir + 'WALLABY_' + gal_name[i] + '_cube.fits.gz'
      
      fig1 = plt.figure(2, figsize=(12, 4))
      
      # =========== Plot r-band/moment map galaxy images =========== #
      wallaby_galaxy_summary_plot(fig1, rows, cols, 1, optical_file, mom0_file, 'darkblue', txt_array)
      wallaby_galaxy_summary_plot(fig1, rows, cols, 2, optical_file, mom0_file, 'darkblue', txt_array)
      wallaby_galaxy_summary_plot(fig1, rows, cols, 3, mom1_file, mom0_file, 'darkblue', txt_array)
      
      # =========== Plot HI Integrated Spectrum =========== #
      velocity, flux, npix = np.genfromtxt(spec_file, usecols=(1,2,3), unpack=True)
      #flux  = 1000. * np.array(flux)
      if velocity[0] < 1e8:
        velocity = velocity / 1000.
      else:
        velocity = (HI_REST / (velocity / 1e6) - 1.) * C_LIGHT
      
      # =========== Determine noise in HI cubelet =========== #
      f1_cube  = pyfits.open(cube_file)
      data_cube, hdr_cube  = f1_cube[0].data, f1_cube[0].header
      cube_mean, cube_median, cube_std = astropy.stats.sigma_clipped_stats(data_cube)
      f1_cube.close()
      
      # =========== Determine channel RMS =========== #
      spec_rms       = []
      pix_per_beam   = (math.pi * beam_maj * beam_min) / (pix_scale * pix_scale)
      for rms_chan in range(len(npix)):
        if npix[rms_chan] < pix_per_beam:
          spec_rms.append(cube_std)
        else:
          spec_rms.append(cube_std * (np.sqrt(npix[rms_chan] / pix_per_beam)))
      spec_rms            = np.array(spec_rms)
      spec_rms[flux == 0] = np.nan
      spec_input          = [velocity, flux, spec_rms]
      wallaby_galaxy_summary_plot(fig1, rows, cols, 4, spec_input, False, 'darkblue', txt_array)
      
      plot_name = plots_dir + 'MAPS/ALL/%s_map.pdf' % galaxies[i]
      plt.savefig(plot_name, bbox_inches = 'tight', dpi = 100)
      plt.close()
      plt.clf()
      



  


  


