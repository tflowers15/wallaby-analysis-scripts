# Libraries

import warnings
warnings.simplefilter("ignore")


import os.path
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo



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
create_directories       = True


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
# ====== Create Directories ======= #
# ================================= #
if create_directories:
  print('============ Make Directories ============')
  #os.system('mkdir %s' % (basedir))
  os.system('mkdir %s' % ('sofia_dir'))
  os.system('mkdir %s' % (basedir + 'MULTIWAVELENGTH/'))
  os.system('mkdir %s' % panstarrs_dir)
  os.system('mkdir %s' % (panstarrs_dir + 'PROFILES_BKGDSUB/'))
  os.system('mkdir %s' % galex_dir)
  os.system('mkdir %s' % (galex_dir + 'PROFILES/'))
  os.system('mkdir %s' % wise_dir)
  os.system('mkdir %s' % (wise_dir + 'PROFILES/'))
  os.system('mkdir %s' % parameter_dir)
  os.system('mkdir %s' % hi_products_dir)
  os.system('mkdir %s' % (hi_products_dir + 'PROFILES/'))
  os.system('mkdir %s' % plots_dir)
  os.system('mkdir %s' % (plots_dir + 'MAPS/'))
  os.system('mkdir %s' % (plots_dir + 'MAPS/ALL/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/HI_PROFILES/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/PANSTARRS/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/PANSTARRS/MAP_ELLIPSE/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_PROFILES/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/PANSTARRS/OPTICAL_SEGMENTATION'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/GALEX/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/GALEX/APERTURES/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/GALEX/ASYMPTOTIC_MAG/'))
  os.system('mkdir %s' % (plots_dir + 'PHOTOMETRY/WISE/'))
  









