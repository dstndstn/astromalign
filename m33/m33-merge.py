import matplotlib
matplotlib.rcParams['figure.figsize'] = (12,8)
import pylab as plt
from astrometry.libkd.spherematch import *
from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.plotutils import *
from glob import glob
from collections import Counter
import os

ps = PlotSequence('merge')

TT = []
for brick in [2,3]:
    for field in range(1,18+1):
        for chip in [1,2]:
            fn = ('/astro/store/phat2/mdurbin/pipelines/brick%i_gst/split_catalogs/14610_M33-B%02i-F%02i.chip%i.gst.fits' %
                  (brick, brick, field, chip))
            T = fits_table(fn)
            print(len(T), 'from', fn)
            TT.append(T)
Tall = merge_tables(TT)
print('Total', len(Tall))

print('Matching...')
I,J,d = match_radec(Tall.ra, Tall.dec, Tall.ra, Tall.dec, 0.1/3600., notself=True)
K, = np.nonzero(I < J)
I = I[K]
J = J[K]
print('Matched', len(I))

plt.clf()
plothist(Tall.ra, Tall.dec)
plt.title('All sources: %i' % len(Tall))
ps.savefig()

plt.clf()
plothist(Tall.ra[I], Tall.dec[I])
plt.title('Matches: %i' % len(I))
ps.savefig()

cosdec = np.cos(np.deg2rad(30.))
dra = (Tall.ra[I]-Tall.ra[J])*cosdec * 3600.*1000.
ddec = (Tall.dec[I]-Tall.dec[J]) * 3600.*1000.
plt.clf()
plothist(dra, ddec, range=((-100,100),(-100,100)))
plt.xlabel('dRA (mas)')
plt.ylabel('dDec (mas)');
plt.title('Before alignment: all bricks')
ps.savefig()

plt.clf()
plt.hist(d[K]*3600.*1000., bins=50, range=(0,100))
plt.xlabel('Match distance (mas)')
plt.title('Before alignment: all bricks')
plt.xlim(0,100)
ps.savefig()

del Tall

from astromalign import astrom_intra

print('Aligning...')
align = astrom_intra.intrabrickshift(TT, matchradius=0.1)

print('Applying alignment...')
align.applyTo(TT)
print('Merging...')
Tall2 = merge_tables(TT)

I2,J2,d2 = match_radec(Tall2.ra, Tall2.dec, Tall2.ra, Tall2.dec, 0.1/3600., notself=True)
K2, = np.nonzero(I2 < J2)
I2 = I2[K2]
J2 = J2[K2]
d2 = d2[K2]
print('Matched', len(I2))

dra = (Tall2.ra[I2]-Tall2.ra[J2])*cosdec * 3600.*1000.
ddec = (Tall2.dec[I2]-Tall2.dec[J2]) * 3600.*1000.
plt.clf()
plothist(dra, ddec)
plt.xlabel('dRA (mas)')
plt.ylabel('dDec (mas)');
plt.title('After alignment')
ps.savefig()

plt.clf()
plt.hist(d2*3600.*1000., bins=50)
plt.xlabel('Match distance (mas)')
plt.title('After alignment')
plt.xlim(0,100)
ps.savefig()

avgcols = ['avgra', 'avgdec',
    'f110w_rate', 'f110w_raterr', 'f110w_vega', 'f110w_std', 'f110w_err',
    'f110w_chi', 'f110w_snr', 'f110w_sharp', 'f110w_round', 'f110w_crowd',
    'f160w_rate', 'f160w_raterr', 'f160w_vega', 'f160w_std', 'f160w_err',
    'f160w_chi', 'f160w_snr', 'f160w_sharp', 'f160w_round', 'f160w_crowd',
    'f275w_rate', 'f275w_raterr', 'f275w_vega', 'f275w_std', 'f275w_err',
    'f275w_chi', 'f275w_snr', 'f275w_sharp', 'f275w_round', 'f275w_crowd',
    'f336w_rate', 'f336w_raterr', 'f336w_vega', 'f336w_std', 'f336w_err',
    'f336w_chi', 'f336w_snr', 'f336w_sharp', 'f336w_round', 'f336w_crowd',
    'f475w_rate', 'f475w_raterr', 'f475w_vega', 'f475w_std', 'f475w_err',
    'f475w_chi', 'f475w_snr', 'f475w_sharp', 'f475w_round', 'f475w_crowd',
    'f814w_rate', 'f814w_raterr', 'f814w_vega', 'f814w_std', 'f814w_err',
    'f814w_chi', 'f814w_snr', 'f814w_sharp', 'f814w_round', 'f814w_crowd',]

def addnew(F, avgcols):
    F.nmatched = np.ones(len(F), np.uint8)
    F.avgra = F.ra.copy()
    F.avgdec = F.dec.copy()
    for c in avgcols:
        v = F.get(c)
        I = np.flatnonzero(v != 99)
        n = np.zeros(len(F), np.uint8)
        n[I] += 1
        F.set(c + '_sum') = v[I]
        F.set(c + '_n') = n

def merge(FF, matchdist=0.06):
    F = FF[0].copy()
    addnew(F, avgcols)
    merged = F

    for F in FF[1:]:
        addnew(F, avgcols)
        I,J,d = match_radec(merged.ra, merged.dec, F.ra, F.dec, matchdist/3600., nearest=True)
        print('Matched', len(I), 'of', len(merged), 'old and', len(F), 'new')

        # unmatched
        Uf = np.ones(len(F), bool)
        Uf[J] = False
        U = F[Uf]

        # matched --
        for col in avgcols:
            m = merged.get(col + '_sum')
            n = merged.get(col + '_n')
            f = F.get(col)
            K = (f != 99)
            m[I[K]] += f[J[K]]
            n[I[K]] += 1
        merged.nmatched[I] += 1
        
        merged = merge_tables([merged, U])

    for col in avgcols:
        m = merged.get(col + '_sum')
        m = merged.get(col + '_n')
        avg = m / n.astype(float)
        avg[n == 0] = 99.
        merged.set(col, avg)

        merged.delete_column(col + '_sum')
        merged.delete_column(col + '_n')
    return merged

print('Merging...')
merged = merge(TT)
print('Merged', len(merged))

print('Nmatched:', Counter(merged.nmatched))

merged.writeto('merged-6filt-bricks23.fits')