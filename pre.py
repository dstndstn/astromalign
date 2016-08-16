from __future__ import print_function
import sys
import os

import numpy as np

import pyfits
#import fitsio
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Sip

if __name__ == '__main__':
    args = sys.argv[1:]
    print('Args', args)
    if len(args) != 3:
        print('Usage: chip1.gst.fits chip2.gst.fits out.gst.fits')
        sys.exit(-1)
        
    gst1, gst2, gstout = args

    flt1 = gst1.replace('.gst.fits', '.fits').replace('.st.fits', '.fits')
    flt2 = gst2.replace('.gst.fits', '.fits').replace('.st.fits', '.fits')
    fltout = gstout.replace('.gst.fits', '.fits').replace('.st.fits', '.fits')

    print('Reading', gst1)
    T1 = fits_table(gst1)
    print('Read', len(T1))
    T1.about()
    print('Reading', gst2)
    T2 = fits_table(gst2)
    print('Read', len(T2))
    T2.about()

    print('Reading WCS from', flt1)
    wcs1 = Sip(flt1, 0)
    wcs1.ensure_inverse_polynomials()
    # print('Reading WCS from', flt2)
    # wcs2 = Sip(flt2, 0)
    # wcs2.ensure_inverse_polynomials()

    print('T1 X,Y ranges', T1.x.min(), T1.x.max(), T1.y.min(), T1.y.max())
    print('T2 X,Y ranges', T2.x.min(), T2.x.max(), T2.y.min(), T2.y.max())

    # ~ 1e-6, 0.0006
    # ok,x,y = wcs2.radec2pixelxy(T2.ra, T2.dec)
    # print('Scatter wcs x vs catalog x:', np.mean(x - T2.x), np.std(x - T2.x))
    # print('Scatter wcs y vs catalog y:', np.mean(y - T2.y), np.std(y - T2.y))
    
    ok,x,y = wcs1.radec2pixelxy(T2.ra, T2.dec)
    print('Converted X,Y ranges:', x.min(), x.max(), y.min(), y.max())
    T2.x = x
    T2.y = y
    
    TT = merge_tables([T1,T2])
    TT.writeto(gstout)
    print('Wrote', gstout)

    hdr = pyfits.open(flt1)[0].header
    hdr['IMAGEW'] = 4096
    hdr['IMAGEH'] = 4096
    pyfits.writeto(fltout, None, header=hdr, clobber=True)
    print('Wrote', fltout)

    #cmd = 'cp "%s" "%s"' % (flt1, fltout)
    #print(cmd)
    #os.system(cmd)
    

    
