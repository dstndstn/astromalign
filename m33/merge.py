import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astrometry.libkd.spherematch import match_radec
from astrometry.util.fits import fits_table, merge_tables
#from astrometry.util.plotutils import plothist, PlotSequence
from glob import glob

def main():
    # Read all tables
    FF = [fits_table(fn) for fn in sorted(glob('14610_M33-B*-F*.gst.fits.gz'))]

    def addnew(F):
        F.nmatched = np.ones(len(F), np.uint8)
        F.avgra = F.ra.copy()
        F.avgdec = F.dec.copy()

    F = FF[0].copy()
    addnew(F)
    merged = F
    
    #ps = PlotSequence('merge')
    
    avgcols = ['avgra', 'avgdec',
        'f475w_rate', 'f475w_raterr', 'f475w_vega', 'f475w_std', 'f475w_err',
        'f475w_chi', 'f475w_snr', 'f475w_sharp', 'f475w_round', 'f475w_crowd',
        'f814w_rate', 'f814w_raterr', 'f814w_vega', 'f814w_std', 'f814w_err',
        'f814w_chi', 'f814w_snr', 'f814w_sharp', 'f814w_round', 'f814w_crowd',]
    
    for F in FF[1:]:
        addnew(F)
        I,J,d = match_radec(merged.ra, merged.dec, F.ra, F.dec, 0.06/3600., nearest=True)
        print('Matched', len(I), 'of', len(merged), 'old and', len(F), 'new')
        
        # plt.clf()
        # plt.hist(d*3600.*1000., bins=50)
        # plt.xlabel('Match distance (mas)')
        # plt.xlim(0, 100)
        # plt.show()
        # ps.savefig()
    
        # unmatched
        Uf = np.ones(len(F), bool)
        Uf[J] = False
        U = F[Uf]
    
        # matched --
        for col in avgcols:
            m = merged.get(col)
            f = F.get(col)
            m[I] += f[J]
        merged.nmatched[I] += 1
        
        merged = merge_tables([merged, U])
    
    for col in avgcols:
        m = merged.get(col)
        m /= merged.nmatched.astype(float)
    
    merged.writeto('merged.fits')
