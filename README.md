# astromalign
Astrometric alignment of image stacks


This code depends on the Astrometry.net python libraries, which
can be installed from the latest release:
http://astrometry.net/downloads/astrometry.net-latest.tar.gz
or from github:
https://github.com/dstndstn/astrometry.net

You can either install it:
- make
- make py
- make install INSTALL_DIR=/place/to/install
- # add /place/to/install/lib/python to your PYTHONPATH

or build a local copy by downloading or cloning into a directory
(renamed to `astrometry`) within the `astromalign` directory.  You
need to `make && make py` in the `astrometry` directory to build the
python libraries.


Using
-----

Super steep learning curve, roughly zero documentation.

People doing HST work using Dolphot will want to use the *singles.py*
script, which takes a set of data directories containing Dolphot star catalogs
(in FITS format) and aligns them with each other and with a reference catalog.

Others may want to (try) using the code with generic FITS catalogs.
For example, let's align subsets of the 2MASS and NOMAD catalogs:

{{{
from astrometry.util.fits import fits_table

# Read catalogs, cut to small subregion
T1 = fits_table('2mass_hp000.fits')
T2 = fits_table('nomad_000.fits')
print(len(T1), '2MASS and', len(T2), 'NOMAD stars')

from astrom_common import Alignment
from astrom_inter import findAffine

# Matching radius in arcsec
radius = 1.
align = Alignment(T2, T1, radius)
# this actually does the work...
align.shift()

# Plot the dRA,dDec cloud
import pylab as plt
from astrometry.util.plotutils import loghist
plt.clf()
loghist(align.match.dra_arcsec, align.match.ddec_arcsec)
plt.savefig('before.png')

# Now compute an affine transformation from T2 to T1, given the "alignment"
refradec = [np.mean(T1.ra), np.mean(T1.dec)]
aff = findAffine(T2, T1, align, refradec)

# Apply the affine transformation to the HD stars to bring them onto the
# Tycho-2 system.
T2.ra, T2.dec = aff.apply(T2.ra, T2.dec)

# Plot the dRA,dDec cloud after
align.match.recompute_dradec(T2, T1)
plt.clf()
loghist(align.match.dra_arcsec, align.match.ddec_arcsec)
plt.savefig('after.png')

}}}
