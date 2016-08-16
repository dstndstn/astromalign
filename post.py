from __future__ import print_function
import sys
import os
from tempfile import NamedTemporaryFile
import numpy as np
import pyfits
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Sip
from astrom_common import Affine, write_update_script
from singles import parse_flt_filename

if __name__ == '__main__':
    args = sys.argv[1:]
    print('Args', args)
    if len(args) != 4:
        print('Usage: affines-x.fits merged-flt.fits orig-chip2.fits script.sh')
        sys.exit(-1)

    afffn, fltmerged, flt2, scriptout = args

    print('Reading affine transformations from', afffn)
    T = fits_table(afffn)
    print('Read', len(T), 'affines')

    # strip trailing spaces
    T.flt = np.array([f.strip() for f in T.flt])

    I = np.flatnonzero(T.flt == fltmerged)
    if len(I) != 1:
        print('Found %i match to merged FLT filename "%s"; expected 1' % (len(I), fltmerged))
        sys.exit(-1)
    i = I[0]
    #aff = T[i]

    affs = Affine.fromTable(T[I])
    aff = affs[0]

    print('Will write to', scriptout)
    fscript = open(scriptout, 'w')

    print('Reading WCS from', flt2)
    wcsext = 0
    wcs2 = Sip(flt2, wcsext)
    wcs2.ensure_inverse_polynomials()

    tan = wcs2.wcstan
    aff.applyToWcsObject(tan)
    tt = NamedTemporaryFile()
    outfn = tt.name
    wcs2.write_to(outfn)

    info = parse_flt_filename(flt2)
    name = info['name']
    chip = info['chip']

    extmap = {1:1, 2:4, 0:1}
    scriptfn = '%s_flt.fits[%i]' % (name, extmap[chip])
    write_update_script(flt2, outfn, scriptfn, fscript, inext=wcsext)

    fscript.close()
    print('Wrote', scriptout)
    
