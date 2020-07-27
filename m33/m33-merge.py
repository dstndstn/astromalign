from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (12,8)
import pylab as plt
from astrometry.libkd.spherematch import *
from astrometry.util.fits import *
from astrometry.util.multiproc import multiproc
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.plotutils import *
from glob import glob
from collections import Counter
import os

ps = PlotSequence('merge')

badval = 99.999

TT = []
BFC = []
for brick in [2,3]:
    for field in range(1,18+1):
        for chip in [1,2]:
            fn = ('/astro/store/phat2/mdurbin/pipelines/brick%i_gst/split_catalogs/14610_M33-B%02i-F%02i.chip%i.gst.fits' %
                  (brick, brick, field, chip))
            T = fits_table(fn)
            print(len(T), 'from', fn)
            TT.append(T)
            BFC.append((brick, field, chip))

gaia = fits_table('gaia.fits')
print('Gaia:', len(gaia))



for i,(T,(brick,field,chip)) in enumerate(zip(TT, BFC)):
    print('Brick', brick, 'field', field, 'chip', chip)
    I,J,d = match_radec(gaia.ra, gaia.dec, T.ra, T.dec, 0.1/3600., nearest=True)
    print('Matched', len(I), 'Gaia')

    cosdec = np.cos(np.deg2rad(30.))
    dra = (gaia.ra[I]-T.ra[J])*cosdec * 3600.*1000.
    ddec = (gaia.dec[I]-T.dec[J]) * 3600.*1000.

    plt.clf()
    plothist(dra, ddec, range=((-100,100),(-100,100)))
    plt.xlabel('dRA (mas)')
    plt.ylabel('dDec (mas)');
    plt.title('Brick %i, field %i, chip %i: to Gaia' % (brick, field, chip))
    ps.savefig()

    ramin = T.ra.min()
    ramax = T.ra.max()
    decmin = T.dec.min()
    decmax = T.dec.max()

    plt.clf()

    plt.subplot(2,2,1)
    plothist(gaia.ra[I], dra, range=((ramin,ramax),(-100,100)),
             doclf=False)
    plt.xlabel('RA (deg)')
    plt.ylabel('dRA (mas)')

    plt.subplot(2,2,2)
    plothist(gaia.ra[I], ddec, range=((ramin,ramax),(-100,100)),
             doclf=False)
    plt.xlabel('RA (deg)')
    plt.ylabel('dDec (mas)')

    plt.subplot(2,2,3)
    plothist(gaia.dec[I], dra, range=((decmin,decmax),(-100,100)),
             doclf=False)
    plt.xlabel('Dec (deg)')
    plt.ylabel('dRA (mas)')

    plt.subplot(2,2,4)
    plothist(gaia.dec[I], ddec, range=((decmin,decmax),(-100,100)),
             doclf=False)
    plt.xlabel('Dec (deg)')
    plt.ylabel('dDec (mas)')

    plt.suptitle('Brick %i, field %i, chip %i: to Gaia' % (brick, field, chip))
    ps.savefig()

    if i == 4:
        break


Tall = merge_tables(TT)
print('Total', len(Tall))

print('Matching to Gaia...')
I,J,d = match_radec(gaia.ra, gaia.dec, Tall.ra, Tall.dec, 0.1/3600., nearest=True)
print('Matched', len(I))

cosdec = np.cos(np.deg2rad(30.))
dra = (gaia.ra[I]-Tall.ra[J])*cosdec * 3600.*1000.
ddec = (gaia.dec[I]-Tall.dec[J]) * 3600.*1000.

plt.clf()
plothist(dra, ddec, range=((-100,100),(-100,100)))
plt.xlabel('dRA (mas)')
plt.ylabel('dDec (mas)');
plt.title('Before alignment: to Gaia')
ps.savefig()

plt.clf()
plt.hist(d*3600.*1000., bins=50, range=(0,100))
plt.xlabel('Match distance (mas)')
plt.title('Before alignment: to Gaia')
plt.xlim(0,100)
ps.savefig()


ramin = Tall.ra.min()
ramax = Tall.ra.max()
decmin = Tall.dec.min()
decmax = Tall.dec.max()

plt.clf()
plt.subplot(2,2,1)
plothist(gaia.ra[I], dra, range=((ramin,ramax),(-100,100)),
         doclf=False)
plt.xlabel('RA (deg)')
plt.ylabel('dRA (mas)')
plt.subplot(2,2,2)
plothist(gaia.ra[I], ddec, range=((ramin,ramax),(-100,100)),
         doclf=False)
plt.xlabel('RA (deg)')
plt.ylabel('dDec (mas)')
plt.subplot(2,2,3)
plothist(gaia.dec[I], dra, range=((decmin,decmax),(-100,100)),
         doclf=False)
plt.xlabel('Dec (deg)')
plt.ylabel('dRA (mas)')
plt.subplot(2,2,4)
plothist(gaia.dec[I], ddec, range=((decmin,decmax),(-100,100)),
         doclf=False)
plt.xlabel('Dec (deg)')
plt.ylabel('dDec (mas)')
plt.suptitle('Before alignment: to Gaia')
ps.savefig()



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

plt.clf()
m = Tall.f814w_vega[I]
plt.hist(m[m<99], bins=50)
plt.xlabel('F814W mag')
ps.savefig()


del Tall

from astromalign import astrom_intra

print('Cutting to bright stars...')
Tbright = []
for T in TT:
    b = T[T.f814w_vega < 24]
    print(len(b), 'of', len(T), 'bright')
    Tbright.append(b)

print('Aligning...')
#mp = multiproc(8)
#align = astrom_intra.intrabrickshift(TT, matchradius=0.1,
align = astrom_intra.intrabrickshift(Tbright, matchradius=0.1,
                                     ref=gaia, refrad=0.1,
                                     do_affine=True)
                                     #do_rotation=True)

print('Applying alignment...')
align.applyTo(TT)


for i,(T,(brick,field,chip)) in enumerate(zip(TT, BFC)):
    print('Brick', brick, 'field', field, 'chip', chip)
    I,J,d = match_radec(gaia.ra, gaia.dec, T.ra, T.dec, 0.1/3600., nearest=True)
    print('Matched', len(I), 'Gaia')

    cosdec = np.cos(np.deg2rad(30.))
    dra = (gaia.ra[I]-T.ra[J])*cosdec * 3600.*1000.
    ddec = (gaia.dec[I]-T.dec[J]) * 3600.*1000.

    plt.clf()
    plothist(dra, ddec, range=((-100,100),(-100,100)))
    plt.xlabel('dRA (mas)')
    plt.ylabel('dDec (mas)');
    plt.title('Brick %i, field %i, chip %i: to Gaia' % (brick, field, chip))
    ps.savefig()

    ramin = T.ra.min()
    ramax = T.ra.max()
    decmin = T.dec.min()
    decmax = T.dec.max()

    plt.clf()

    plt.subplot(2,2,1)
    plothist(gaia.ra[I], dra, range=((ramin,ramax),(-100,100)),
             doclf=False)
    plt.xlabel('RA (deg)')
    plt.ylabel('dRA (mas)')

    plt.subplot(2,2,2)
    plothist(gaia.ra[I], ddec, range=((ramin,ramax),(-100,100)),
             doclf=False)
    plt.xlabel('RA (deg)')
    plt.ylabel('dDec (mas)')

    plt.subplot(2,2,3)
    plothist(gaia.dec[I], dra, range=((decmin,decmax),(-100,100)),
             doclf=False)
    plt.xlabel('Dec (deg)')
    plt.ylabel('dRA (mas)')

    plt.subplot(2,2,4)
    plothist(gaia.dec[I], ddec, range=((decmin,decmax),(-100,100)),
             doclf=False)
    plt.xlabel('Dec (deg)')
    plt.ylabel('dDec (mas)')

    plt.suptitle('Brick %i, field %i, chip %i: to Gaia' % (brick, field, chip))
    ps.savefig()

    if i == 4:
        break



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


print('Matching to Gaia...')
I,J,d = match_radec(gaia.ra, gaia.dec, Tall2.ra, Tall2.dec, 0.1/3600.,
                    nearest=True)
print('Matched', len(I))

cosdec = np.cos(np.deg2rad(30.))
dra = (gaia.ra[I]-Tall2.ra[J])*cosdec * 3600.*1000.
ddec = (gaia.dec[I]-Tall2.dec[J]) * 3600.*1000.
plt.clf()
plothist(dra, ddec, range=((-100,100),(-100,100)))
plt.xlabel('dRA (mas)')
plt.ylabel('dDec (mas)');
plt.title('After alignment: to Gaia')
ps.savefig()

plt.clf()
plt.hist(d*3600.*1000., bins=50, range=(0,100))
plt.xlabel('Match distance (mas)')
plt.title('After alignment: to Gaia')
plt.xlim(0,100)
ps.savefig()

ramin = Tall2.ra.min()
ramax = Tall2.ra.max()
decmin = Tall2.dec.min()
decmax = Tall2.dec.max()

plt.clf()
plt.subplot(2,2,1)
plothist(gaia.ra[I], dra, range=((ramin,ramax),(-100,100)),
         doclf=False)
plt.xlabel('RA (deg)')
plt.ylabel('dRA (mas)')
plt.subplot(2,2,2)
plothist(gaia.ra[I], ddec, range=((ramin,ramax),(-100,100)),
         doclf=False)
plt.xlabel('RA (deg)')
plt.ylabel('dDec (mas)')
plt.subplot(2,2,3)
plothist(gaia.dec[I], dra, range=((decmin,decmax),(-100,100)),
         doclf=False)
plt.xlabel('Dec (deg)')
plt.ylabel('dRA (mas)')
plt.subplot(2,2,4)
plothist(gaia.dec[I], ddec, range=((decmin,decmax),(-100,100)),
         doclf=False)
plt.xlabel('Dec (deg)')
plt.ylabel('dDec (mas)')
plt.suptitle('After alignment: to Gaia')
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
    print('addnew():', len(F))
    F.nmatched = np.ones(len(F), np.uint8)
    F.avgra = F.ra.copy()
    F.avgdec = F.dec.copy()
    for c in avgcols:
        v = F.get(c)
        I = np.flatnonzero(v != badval)
        n = np.zeros(len(F), np.uint8)
        s = np.zeros_like(v)
        s[I] += v[I]
        n[I] += 1
        F.set(c + '_sum', s)
        F.set(c + '_n', n)

        m = F.get(c + '_sum')
        n = F.get(c + '_n')
        print('  ', c, len(m), len(n), m.shape, n.shape,
              m.dtype, n.dtype)

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
            K = (f[J] != badval)
            m[I[K]] += f[J[K]]
            n[I[K]] += 1

        merged.nmatched[I] += 1

        print('Merging:')
        merged.about()
        U.about()

        merged = merge_tables([merged, U])

        print('Merged:')
        merged.about()

    for col in avgcols:
        m = merged.get(col + '_sum')
        n = merged.get(col + '_n')
        avg = m / n.astype(float)
        avg[n == 0] = badval
        merged.set(col, avg)
        merged.delete_column(col + '_sum')
        merged.delete_column(col + '_n')

    return merged

print('Merging...')
merged = merge(TT)
print('Merged', len(merged))

print('Nmatched:', Counter(merged.nmatched))

merged.writeto('merged-6filt-bricks23.fits')
