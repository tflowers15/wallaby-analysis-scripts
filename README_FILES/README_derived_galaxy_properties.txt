Description of *derived_galaxy_properties.fits files.

####### *derived_galaxy_properties.fits #######

Columns 1-37            	- Default SoFiA catalogue columns

REDSHIFT	 	        - Redshift derived from systemic frequency
VSYS		 	        - Systemic velocity, cz [km/s]
DISTANCE		 	- Luminosity distance [Mpc]

lgMSTAR_SDSS_25	 	        - log stellar mass from 25 mag / arcsec^2 aperture 
				  photometry [Msol]
lgMHI		 	        - log HI mass [Msol]
HIFRAC			 	- HI gas fraction (lgMHI / lgMSTAR_SDSS_25)

lgMHI_CORRECT	 	        - log HI mass (flux corrected) [Msol]
HIFRAC_CORRECT	 	        - HI gas fraction (flux corrected) 
				  (lgMHI_CORRECT / lgMSTAR_SDSS_25)

lgMHI_OPTICAL_DISC_CORRECT_ISO	- log HI mass in 25 mag /arcsec^2 optical disc 
				  (flux corrected) [Msol]
HIFRAC_OPTICAL_DISC_CORRECT_ISO	- HI gas fraction in 25 mag /arcsec^2 optical disc 
				  (flux corrected) 
				  (HIFRAC_OPTICAL_DISC_CORRECT_ISO / lgMSTAR_SDSS_25)

lgMHI_OPTICAL_DISC_CORRECT_EFF	- log HI mass in effective optical disc (flux 
				  corrected) [Msol]
HIFRAC_OPTICAL_DISC_CORRECT_EFF	- HI gas fraction in effective optical disc (flux 
				  corrected) 
				  (lgMHI_OPTICAL_DISC_CORRECT_EFF / lgMSTAR_SDSS_25)

RADIUS_R_ISO25	 		- Isophotal 25 mag /arcsec^2 r-band radius [arcsec]
RADIUS_R_ISO26	 		- Isophotal 26 mag /arcsec^2 r-band radius [arcsec]
RADIUS_R_50			- Effective (r50) r-band radius [arcsec]
RADIUS_R_90	 		- r90 r-band radius [arcsec]

RADIUS_HI_ISO	 		- Isodensity HI 1 Msol / pc^2 radius [arcsec]
RADIUS_HI_EFF	 	        - Effective HI radius [arcsec]
RADIUS_ISO_ERR	 	        - Isodensity HI 1 Msol / pc^2 radius error [arcsec]

RADIUS_NUV_ISO	 	        - Isophotal 28 mag / arcsec^2 NUV-band radius [arcsec]
NUV-R	 	        	- NUV - R (25 mag /arcsec^2) colour [mag]

SURFACE_DENSITY_HI_ISO	 	- Average HI surface density w/in isodensity radius 
				  aperture (RADIUS_HI_ISO) [Msol / pc^2]
SURFACE_DENSITY_HI_EFF	 	- Average HI surface density w/in effective radius 
				  aperture (RADIUS_HI_EFF) [Msol / pc^2]

AXIS_RATIO_BA	 	        - HI moment 0 map axis ratio (b/a)

RADIUS_HI_ISO_CORR	 	- Isodensity HI 1 Msol / pc^2 radius (flux 
				  corrected) [arcsec]
RADIUS_HI_EFF_CORR	 	- Effective HI radius (flux corrected) [arcsec]

SURFACE_DENSITY_HI_ISO_CORR	- Average HI surface density w/in isodensity radius 
				  aperture (RADIUS_HI_ISO) (flux corrected) 
				  [Msol / pc^2]
SURFACE_DENSITY_HI_EFF_CORR	- Average HI surface density w/in effective radius 
				  aperture (RADIUS_HI_EFF) (flux corrected) 
				  [Msol / pc^2]

HI_R25_SIZE_RATIO	 	- HI to r-band size ratio 
				  (RADIUS_HI_ISO_CORR / RADIUS_R_ISO25)
NUV_R25_SIZE_RATIO	 	- NUV to r-band size ratio 
				  (RADIUS_NUV_ISO / RADIUS_R_ISO25)
HI_NUV_SIZE_RATIO	 	- HI to NUV size ratio 
				  (RADIUS_HI_ISO_CORR / RADIUS_NUV_ISO)

SFR_NUV+MIR	 	        - log combined NUV + mid-IR star formation rate 
				  [Msol / yr]
SFR_UPLIM	 	        - Flag for SFR_NUV+MIR upper limits 
				  (0 - measured, 1 - upper limit) 
SSFR	 	        	- Specific SFR (SFR_NUV+MIR - lgMSTAR_SDSS_25) [1/yr]

G_MAG_PS25	 	        - PanSTARRS g-band magnitude in RADIUS_R_ISO25 
				  aperture [mag]
R_MAG_PS25	 	        - PanSTARRS r-band magnitude in RADIUS_R_ISO25 
				  aperture [mag]

G_MAG_SDSS_25	 	        - SDSS photometric system g-band magnitude in 						  RADIUS_R_ISO25 aperture [mag]
R_MAG_SDSS_25	 	        - SDSS photometric system r-band magnitude in 
				  RADIUS_R_ISO25 aperture [mag]

NUV_MAG	 	        	- Asymptotic NUV-band magnitude [mag]
NUV_UPPERLIM	 	        - Flag for NUV_MAG upper limits 
				  (0 - measured, 1 - upper limit) 

W1_MAG	 	        	- W1-band magnitude in RADIUS_R_ISO25 [mag]
W1_UPPERLIM	 	        - Flag for W1_MAG upper limits 
				  (0 - measured, 1 - upper limit)
W2_MAG	 	        	- W2-band magnitude in RADIUS_R_ISO25 [mag]
W2_UPPERLIM	 	        - Flag for W2_MAG upper limits 
				  (0 - measured, 1 - upper limit)
W3_MAG	 	        	- W3-band magnitude in RADIUS_R_ISO25 [mag]
W3_UPPERLIM	 	        - Flag for W3_MAG upper limits 
				  (0 - measured, 1 - upper limit)
W4_MAG	 	        	- W4-band magnitude in RADIUS_R_ISO25 [mag]
W4_UPPERLIM	 	        - Flag for W4_MAG upper limits 
				  (0 - measured, 1 - upper limit)

E(B-V)_SandF	 	       	- Galactic dust extinction correction (Schlafly and
				  Finkbeiner 2011 (ApJ 737, 103))
E(B-V)_SFD	 	       	- Galactic dust extinction correction (Schlegel et 
				  al. 1998 (ApJ 500, 525))





