import numpy as np
from astropy.wcs import WCS


def make_wcs(ra, dec, size, pixsize=1.0):
    scale = pixsize / 3600
    w = WCS(naxis=2)
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = np.array([-scale, scale])
    w.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = [size, size]
    return w


