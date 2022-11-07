Description of *derived_galaxy_properties.fits files.

####### *derived_galaxy_properties.fits #######

Columns 1-37            	- Default SoFiA catalogue columns

++++++++ Janowiecki+2017 (NUV+MIR SFRs) ++++++++

SFR_W3_MIR_NOCORR	 	- log SFR derived from W3-band without W1-band stellar 
				  correction [Msol / yr]
SFR_W4_MIR_NOCORR	 	- log SFR derived from W4-band without W1-band stellar 
				  correction [Msol / yr]

SFR_W3_MIR_CORR	 		- log SFR derived from W3-band with W1-band stellar 
				  correction [Msol / yr]
SFR_W4_MIR_CORR	 	        - log SFR derived from W4-band with W1-band stellar 
				  correction [Msol / yr]

SFR_W3_MIR_FINAL	 	- log SFR derived from W3-band. For galaxies with 
				  negative corrected W3 luminosities, the 
				  SFR_W3_MIR_NOCORR is used. [Msol / yr]
SFR_W4_MIR_FINAL	 	- log SFR derived from W4-band. For galaxies with 
				  negative corrected W4 luminosities, the 
				  SFR_W4_MIR_NOCORR is used. [Msol / yr]

NUV_SFR		 	        - log SFR derived from NUV-band [Msol / yr]

SFR_NUV+W3	 	        - log SFR combined NUV and W3 [Msol / yr]
SFR_NUV+W4	 	        - log SFR combined NUV and W4 [Msol / yr]

SFR_NUV+W3_UPLIMS	 	- log SFR combined NUV and W3. Galaxies with no NUV 
				  SFR are replaced with W3 MIR SFR [Msol / yr]
SFR_NUV+W4_UPLIMS	 	- log SFR combined NUV and W4. Galaxies with no NUV 
				  SFR are replaced with W4 MIR SFR [Msol / yr]

SFR_NUV+MIR_FINAL		- log SFR combined NUV and MIR. For galaxies with W4 
				  upper limits (not detections) the NUV+W4 SFRs are 
				  replaced with NUV+W3 SFRs [Msol / yr]

++++++++ Cluver+2017 (WISE SFRs) ++++++++

SFR_W3_C17_NOCORR	 	- log SFR derived from W3-band without W1-band stellar 
				  correction [Msol / yr]
SFR_W4_C17_NOCORR	 	- log SFR derived from W4-band without W1-band stellar 
				  correction [Msol / yr]]

SFR_W3_C17_CORR			- log SFR derived from W3-band with W1-band stellar 
				  correction [Msol / yr]
SFR_W4_C17_CORR	 		- log SFR derived from W4-band with W1-band stellar 
				  correction [Msol / yr]

SFR_W3_C17_FINAL	 	- log SFR derived from W3-band. For galaxies with 
				  negative corrected W3 luminosities, the 
				  SFR_W3_C17_NOCORR is used. [Msol / yr]
SFR_W4_C17_FINAL	 	- log SFR derived from W4-band. For galaxies with 
				  negative corrected W4 luminosities, the 
				  SFR_W4_C17_NOCORR is used. [Msol / yr]

++++++++ Flags (0 - measured, 1 - upper limit/no coverage) ++++++++

W3_FLUX_UPLIM_FLAG	 	- Flag for W3 flux upper limit
W4_FLUX_UPLIM_FLAG	 	- Flag for W4 flux upper limit

W3_CORR_FLUX_UPLIM_FLAG	 	- Flag for negative corrected W3 luminosity
W4_CORR_FLUX_UPLIM_FLAG	 	- Flag for negative corrected W4 luminosity

NUV_FLUX_UPLIM_FLAG	 	- Flag for NUV flux upper limit
NO_GALEX_COVERAGE_FLAG	 	- Flag for no GALEX coverage

SFR_NUV+MIR_UPLIM_FLAG	 	- Flag for final combined SFR containing an 
				  upper limit

W3_CORR_FLUX_C17_UPLIM_FLAG	- Flag for negative corrected Cluver+2017 W3 flux
W4_CORR_FLUX_C17_UPLIM_FLAG	- Flag for negative corrected Cluver+2017 W4 flux




