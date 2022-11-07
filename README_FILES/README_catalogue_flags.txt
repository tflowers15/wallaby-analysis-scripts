Description of *catalogue_flag.fits files.

####### *catalogue_flag.fits #######

Columns 1-37            - Default SoFiA catalogue columns

flag_src_class          - HI source flag (column 38)
                          0 - No optical counterpart in r-band image
                          1 - Single optical counterpart
                          2 - Multiple optical counterparts (i.e. interacting system)
                          3 - Shredded HI source (i.e. only half of galaxy detected)
                          5 - Artefacts present in HI detection (i.e. continuum residual)

flag_opt_fit            - Optical source flag (column 39)
                          0 - Successfully found optical counterpart for photometry
			  1 - Segmentation map to be fix (change to 0 once done)
                          3 - Foreground star/s coincident with optical counterpart
                          5 - Artefacts present in r-band image

segment_deblend         - Flag to run a default deblending (Depreciated due to the 
			  do_deblend* flags, which should be used instead).

do_seg_par1             - Flag to set specific segmentation values (0/do NOT or 1/do) 
do_seg_par2             - nsigma to define the detection threshold (not in use)
do_seg_par3             - Minimum number of pixels for a segment (not in use)
do_seg_par4             - Source finding threshold relative to the background
do_seg_par5             - Minimum segment radius [pixels]

do_seg_id1              - Flag to set specific segmentation ID (0/do NOT or 1/do) 
do_seg_id2              - Segment ID

do_deblend1             - Flag to set deblending values (0/do NOT or 1/do) 
do_deblend2             - The fraction of the total source flux that a local peak 
			  must have (at any one of the multi-thresholds) to be 
			  deblended as a separate object. Between 0 and 1. The 
			  default is 0.001, which will deblend sources with a 7.5 
			  magnitude difference.
do_deblend3             - Minimum segment radius







