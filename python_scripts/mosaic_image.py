import os
import logging
import subprocess
from astropy.io import fits


def set_zeropoint(flist, target, key='ZEROPT', force=False):
    for f in flist:
        with fits.open(f, mode='update') as img:
            # check original information
            if not (key in img[0].header):
                if force:
                    img[0].header[key] = target
                    img[0].header['ZEROPT'] = target
                    logging.debug('{}: force zero point {}'.format(f, target))
                    continue
                else:
                    logging.exception('{}: No zero point '
                            'information in header'.format(f))
            if 'ZEROPT' in img[0].header:
                if abs(img[0].header['ZEROPT'] - target) > 1e-7:
                    logging.exception('{}: conflict ZEROPT '
                            'values in header'.format(f))

            # multiplication factor
            factor = 10 ** (0.4 * (target - img[0].header[key]))
            logging.debug('{}: zero point {} --> {}, factor {}'.format(f,
                img[0].header[key], target, factor))

            # keep original information
            img[0].header['MZP_ORIG'] = img[0].header[key]
            img[0].header['MZP_FACT'] = factor

            # update zeropoint information
            img[0].data = img[0].data * factor
            img[0].header[key] = target
            img[0].header['ZEROPT'] = target


def do_mosaic(ra, dec, size, flist, outname, prog='swarp',
        backvalue=None, backsize=128, weight=False):
    cmd = '{} {}'.format(prog, ' '.join(flist))
    # output file name
    wname = outname.replace('.fits', '.weight.fits')
    cmd += ' -IMAGEOUT_NAME {}'.format(outname)
    cmd += ' -WEIGHTOUT_NAME {}'.format(wname)
    # astrometry
    cmd += ' -CENTER_TYPE MANUAL'
    cmd += ' -CENTER {},{}'.format(ra, dec)
    cmd += ' -IMAGE_SIZE {}'.format(size)  # number of pixels
    # background
    if backvalue is None:
        backtype = 'AUTO'
        backvalue = 0.0
    else:
        backtype = 'MANUAL'
    cmd += ' -BACK_TYPE {}'.format(backtype)
    cmd += ' -BACK_DEFAULT {}'.format(backvalue)
    cmd += ' -BACK_SIZE {}'.format(backsize)
    # weight
    if weight:
        cmd += ' -WEIGHT_TYPE MAP_WEIGHT'
    # others
    cmd += ' -COMBINE_TYPE MEDIAN'
    cmd += ' -COPY_KEYWORDS TELNAME,FILTER,ZEROPT'
    cmd += ' -WRITE_XML N'
    cmd += ' -VERBOSE_TYPE QUIET'

    # run
    logging.debug(cmd)
    x = subprocess.run(cmd.split())
    if x.returncode != 0:
        logging.warning('SWarp does not run successfully')
    # clean
    if os.path.exists(wname):
        os.remove(wname)


