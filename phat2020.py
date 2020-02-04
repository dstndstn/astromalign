import matplotlib
matplotlib.use('Agg') # check this
import matplotlib.pyplot as plt
import numpy as np
import vaex
import pandas as pd
import os
from glob import glob

from astrom_common import Alignment, plotalignment

from astrometry.util.fits import fits_table

os.environ['NUMEXPR_MAX_THREADS'] = '8'


def apply_alignments():
    from astrom_common import Affine
    T = fits_table('affines.fits')
    affs = Affine.fromTable(T)
    print('Read affines:', affs)

    ibright = dict([(fn.strip(),i) for fn in affs.filenames])

    corners = {}
    for line in open('corners.txt').readlines():
        line = line.strip()
        words = line.split()
        ras = np.array([float(words[i]) for i in [1,3,5,7]])
        decs = np.array([float(words[i]) for i in [2,4,6,8]])
        corners[words[0]] = (ras,decs)
    from astrometry.util.miscutils import point_in_poly

    fns = (glob('data/M31-*ST/proc_default/M31-*ST.phot.hdf5') +
           glob('data/M31-*ST/M31-*ST.phot.hdf5'))
    fns.sort()
    print('Files:', fns)

    veto_polys = []

    for photfile in fns:
        basename = os.path.basename(photfile)
        basename = basename.replace('.phot.hdf5', '')
        print('Base name:', basename)

        brightfn = basename + '-bright.fits'
        ii = ibright[brightfn]
        aff = affs[ii]

        print('Reading', photfile)
        df = pd.read_hdf(photfile, key='data')
        ds = vaex.from_pandas(df)
        print(len(ds), 'rows')
        ra = ds.evaluate(ds['ra'])
        dec = ds.evaluate(ds['dec'])
        ra,dec = aff.apply(ra, dec)
        
        corner = corners[basename]
        Tleft = fits_table()
        Tleft.ra = ra
        Tleft.dec = dec
        Tleft.index = np.arange(len(Tleft))
        ras,decs = corner
        poly = np.vstack((ras, decs)).T
        inside = point_in_poly(Tleft.ra, Tleft.dec, poly)
        print(np.sum(inside), 'of', len(Tleft), 'inside corners of this half-brick')

        inside_veto = np.zeros(len(Tleft), bool)
        for vp in veto_polys:
            inveto = point_in_poly(Tleft.ra, Tleft.dec, vp)
            inside_veto[inveto] = True
        print(np.sum(inside_veto), 'stars are inside the corners of previous half-bricks')
        print('inside:', type(inside))
        inside[inveto] = False
        print(np.sum(inside), 'stars are uniquely in this half-brick')
        
        veto_polys.append(poly)
        
        outfn = 'out-%s.hdf5' % basename
        df[inside].to_hdf5(outfn, key='data', mode='w',
                           format='table', complevel=9, complib='zlib')
        print('Wrote', outfn)
    

def to_fits():
    fns = (glob('data/M31-*ST/proc_default/M31-*ST.phot.hdf5') +
           glob('data/M31-*ST/M31-*ST.phot.hdf5'))
    fns.sort()
    print('Files:', fns)

    plt.clf()

    for photfile in fns:
        #photfile = 'data/M31-B23-WEST/M31-B23-WEST.phot.hdf5'
    
        basename = os.path.basename(photfile)
        basename = basename.replace('.phot.hdf5', '')
        print('Base name:', basename)
    
        outfn = basename + '-bright.fits'
        if os.path.exists(outfn):
            print('Exists:', outfn)
            continue
    
        words = basename.split('-')
        assert(len(words) == 3)
        brick = words[1]
        assert(brick[0] == 'B')
        brick = int(brick[1:], 10)
        print('Brick number:', brick)
        ew = words[2]
        assert(ew in ['EAST', 'WEST'])
        east = (ew == 'EAST')
    
        df = pd.read_hdf(photfile, key='data')
        ds = vaex.from_pandas(df)
        print('Read', photfile)
        #print(ds)
    
        good = ds['f814w_gst']
        print(len(ds), 'rows')
        ds = ds[good]
        print(len(ds), 'gst on F814W')
    
        # good = ds.evaluate(ds['f475w_gst'])
        # print(good)
        # print(len(good))
        # print(type(good))
        # print(good.dtype)
        # print('Of those,', np.sum(ds.evaluate(ds['f475w_gst'])), 'are F475W_GST')
        # print('Of those,', np.sum(ds.evaluate(ds['f336w_gst'])), 'are F336W_GST')
        # print('Of those,', np.sum(ds.evaluate(ds['f275w_gst'])), 'are F275W_GST')
        # print('Of those,', np.sum(ds.evaluate(ds['f110w_gst'])), 'are F110W_GST')
        # print('Of those,', np.sum(ds.evaluate(ds['f160w_gst'])), 'are F160W_GST')
    
        mag = ds.evaluate(ds['f814w_vega'])
        print('Of', len(mag), 'mags,', np.sum(np.isfinite(mag)), 'are finite')
        print('range:', np.nanmin(mag), np.nanmax(mag))
    
        plt.hist(mag, range=(20, 28), bins=50, label=basename)
    
        ds = ds[ds['f814w_vega'] < 24]
        print(len(ds), 'with F814W < 24')
    
        mag = ds.evaluate(ds['f814w_vega'])
        xx = ds.evaluate(ds['x'])
        yy = ds.evaluate(ds['y'])

        xlo = xx.min()
        xhi = xx.max()
        ylo = yy.min()
        yhi = yy.max()
        nx = int(np.round((xhi - xlo) / 1000.)) + 1
        xbins = np.linspace(xlo, xhi, nx)
        ny = int(np.round((yhi - ylo) / 1000.)) + 1
        ybins = np.linspace(ylo, yhi, ny)
        print('x bins', xbins)
        print('y bins', ybins)
        xbin = np.digitize(xx, xbins)
        ybin = np.digitize(yy, ybins)
        xybin = ybin * nx + xbin
        nbins = nx * ny
        print('N bins:', nbins)
        nperbin = int(np.ceil(100000. / nbins))
        II = []
        for ibin in range(nbins):
            I = np.flatnonzero(xybin == ibin)
            if len(I) == 0:
                continue
            Ibright = np.argsort(mag[I])[:nperbin]
            II.append(I[Ibright])
        II = np.hstack(II)

        #I = np.argsort(mag)
        #I = I[:100000]
        #print('100k-th star: mag', mag[I[-1]])
        ds = ds.take(II)

        T = fits_table()
        for col in ['ra','dec','x', 'y', 'index']:
            T.set(col, ds.evaluate(ds[col]))
        for filt in [814, 475, 336, 275, 110, 160]:
            for col in ['f%iw_vega']:
                colname = col % filt
                T.set(colname, ds.evaluate(ds[colname]))
        T.brick = np.zeros(len(T), np.uint8) + brick
        T.east = np.zeros(len(T), bool)
        T.east[:] = east
        T.writeto(outfn)
    
    plt.legend()
    plt.xlabel('F814W mag')
    plt.savefig('mags.png')


if __name__ == '__main__':
    import sys
    #to_fits()
    apply_alignments()
    sys.exit(0)
    
    from astrometry.libkd.spherematch import tree_build_radec, trees_match
    from astrometry.libkd.spherematch import match_radec
    from astrometry.util.plotutils import *
    from astrometry.util.util import Tan
    import fitsio

    from astrom_common import getwcsoutline
    from singles import find_overlaps

    fns = glob('M31-*-bright.fits')
    fns.sort()

    keepfns = []
    if True:
        #data/M31-B23-WEST/M31-B23-WEST_F475W_drz.chip1.fits
        WCS = []
        for fn in fns:
            print()
            print(fn)
            base = fn.replace('-bright.fits', '')
            pat1 = 'data/' + base + '/proc_default/' + base + '*drz.chip1.fits'
            pat2 = 'data/' + base + '/' + base + '*drz.chip1.fits'
            #pat3 = 'data2/' + base + '/proc_default/' + base + '*drz.chip1.fits'
            #pat4 = 'data/' + base + '/' + base + '*_drz_head.fits'
            print(pat1, pat2)#, pat4)
            ff = glob(pat1) + glob(pat2)# + glob(pat4) #+ glob(pat4)
            print('WCS files:', ff)
            #assert(len(ff) == 1)
            if len(ff) < 1:
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                continue
            keepfns.append(fn)
            fn = ff[0]
            wcs = Tan(fn)
            F = fitsio.FITS(fn)
            info = F[0].get_info()
            H,W = info['dims']
            wcs.imagew = W
            wcs.imageh = H
            WCS.append(wcs)

            wcs.write_to(base + '-wcs.fits')

        #bomb()
        fns = keepfns

    

    names = [fn.replace('-bright.fits', '') for fn in fns]

    outlines = [getwcsoutline(wcs) for wcs in WCS]

    overlaps,areas = find_overlaps(outlines)

    print('Reading tables...')
    TT = [fits_table(fn) for fn in fns]
    print('Building trees...')
    kds = [tree_build_radec(T.ra, T.dec) for T in TT]

    for T,name in zip(TT, names):
        T.name = np.array([name]*len(T))

    allra  = np.hstack([T.ra  for T in TT])
    alldec = np.hstack([T.dec for T in TT])
    minra = np.min(allra)
    maxra = np.max(allra)
    mindec = np.min(alldec)
    maxdec = np.max(alldec)

    print('RA,Dec range:', minra, maxra, mindec, maxdec)


    plothist(allra, alldec)
    plt.axis([maxra, minra, mindec, maxdec])
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.savefig('match-all.png')

    Tref = fits_table('gaia.fits')
    r_arcsec = 0.2
    I,J,d = match_radec(Tref.ra, Tref.dec, allra, alldec, r_arcsec/3600.)
    dec = alldec[J]
    cosdec = np.cos(np.deg2rad(dec))
    dr = (Tref.ra[I]  - allra[J]) * cosdec * 3600.
    dd = (Tref.dec[I] - alldec[J]) * 3600.
    plt.clf()
    rr = (-r_arcsec*1000, +r_arcsec*1000)
    plothist(dr*1000., dd*1000., nbins=100, range=(rr, rr))
    plt.xlabel('dRA (milli-arcsec)')
    plt.ylabel('dDec (milli-arcsec)')
    plt.savefig('match-all-ref-before.png')

    # Initial matching of all stars
    r_arcsec = 0.2
    I,J,d = match_radec(allra, alldec, allra, alldec, r_arcsec/3600., notself=True)
    dec = alldec[I]
    cosdec = np.cos(np.deg2rad(dec))
    dr = (allra[I]  - allra[J]) * cosdec * 3600.
    dd = (alldec[I] - alldec[J]) * 3600.

    plt.clf()
    rr = (-r_arcsec*1000, +r_arcsec*1000)
    plothist(dr*1000., dd*1000., nbins=100, range=(rr, rr))
    plt.xlabel('dRA (milli-arcsec)')
    plt.ylabel('dDec (milli-arcsec)')
    plt.savefig('match-all-before.png')

    hulls = []
    from scipy.spatial import ConvexHull
    for T in TT:
        hull = ConvexHull(np.vstack((T.ra, T.dec)).T)
        ra = T.ra[hull.vertices]
        ra = np.append(ra, ra[0])
        dec = T.dec[hull.vertices]
        dec = np.append(dec, dec[0])
        hulls.append((ra, dec))

    aligns = {}

    #for i in []:
    for i in range(len(kds)):
        for j in range(i+1, len(kds)):
            print('Matching trees', i, 'and', j)

            r_arcsec = 0.2
            radius = np.deg2rad(r_arcsec / 3600)

            I,J,d2 = trees_match(kds[i], kds[j], radius)
            print(len(I), 'matches')
            if len(I) == 0:
                continue

            Ti = TT[i]
            Tj = TT[j]
            dec = Ti[I].dec
            cosdec = np.cos(np.deg2rad(dec))
            dr = (Ti[I].ra  - Tj[J].ra) * cosdec * 3600.
            dd = (Ti[I].dec - Tj[J].dec) * 3600.

            if False:
                al = Alignment(Ti, Tj, searchradius=r_arcsec)
                print('Aligning...')
                if not al.shift():
                    print('Failed to find Alignment between fields')
                    continue
                aligns[(i,j)] = al

                plt.clf()
                plotalignment(al)
                plt.savefig('match-align-%02i-%02i.png' % (i,j))

            plt.clf()
            #plothist(np.append(Ti.ra, Tj.ra), np.append(Ti.dec, Tj.dec), docolorbar=False, doclf=False, dohot=False,
            #         imshowargs=dict(cmap=antigray))
            plothist(Ti.ra[I], Ti.dec[I], docolorbar=False, doclf=False)
            r,d = hulls[i]
            plt.plot(r, d, 'r-')
            r,d = hulls[j]
            plt.plot(r, d, 'b-')
            mra = Ti.ra[I]
            mdec = Ti.dec[I]
            mnra = np.min(mra)
            mxra = np.max(mra)
            mndec = np.min(mdec)
            mxdec = np.max(mdec)
            plt.plot([mnra,mnra,mxra,mxra,mnra], [mndec,mxdec,mxdec,mndec,mndec], 'g-')

            plt.axis([maxra, minra, mindec, maxdec])
            plt.xlabel('RA (deg)')
            plt.ylabel('Dec (deg)')
            plt.savefig('match-radec-%02i-%02i.png' % (i, j))

            plt.clf()
            rr = (-r_arcsec, +r_arcsec)
            plothist(dr, dd, nbins=100, range=(rr, rr))
            plt.xlabel('dRA (arcsec)')
            plt.ylabel('dDec (arcsec)')
            plt.savefig('match-dradec-%02i-%02i.png' % (i, j))

    #for roundi,(Nk,R) in enumerate(NkeepRads):

    refrad = 0.15
    targetrad = 0.005

    ps = PlotSequence('shift')

    from astrom_intra import intrabrickshift
    from singles import plot_all_alignments

    #Rads = [0.25, 0.1]
    Rads = [0.2, 0.050, 0.020]
    #Rads = [0.1]
    affs = None
    # this is the reference point around which rotations take place, NOT reference catalog stars.
    refrd = None
    for roundi, R in enumerate(Rads):

        if roundi > 0:
            refrad = 0.050

        TT1 = TT

        nb = int(np.ceil(R / targetrad))
        nb = max(nb, 5)
        if nb % 2 == 0:
            nb += 1
        print('Round', roundi+1, ': matching with radius', R)
        print('Nbins:', nb)

        # kwargs to pass to intrabrickshift
        ikwargs = {}
        minoverlap = 0.01
        tryoverlaps = (overlaps > minoverlap)
        ikwargs.update(do_affine=True, #mp=mp,
                   #alignplotargs=dict(bins=25),
                   alignplotargs=dict(bins=50),
                   overlaps=tryoverlaps)

        ikwargs.update(ref=Tref, refrad=refrad)

        # kwargs to pass to Alignment
        akwargs={}

        i1 = intrabrickshift(TT1, matchradius=R, refradecs=refrd,
                             align_kwargs=dict(histbins=nb, **akwargs),
                             **ikwargs)

        refrd = i1.get_reference_radecs()

        filts = ['' for n in names]
        ap = i1.alplotgrid
        Nk = 100000
        plot_all_alignments(ap, R*1000, refrad*1000, roundi+1, names, filts, ps,
                            overlaps, outlines, Nk)
        for T,aff in zip(TT,i1.affines):
            T.ra,T.dec = aff.apply(T.ra, T.dec)

        if affs is None:
            affs = i1.affines
        else:
            for a,a2 in zip(affs, i1.affines):
                a.add(a2)

    from astrom_common import Affine
    T = Affine.toTable(affs)
    T.filenames = fns
    #T.flt = fltfns
    #T.gst = gstfns
    #T.chip = chips

    # FAKE -- used as a name in alignment_plots
    T.gst = np.array([n + '.gst.fits' for n in names])

    afffn = 'affines.fits'
    T.writeto(afffn)



    # Final matching of all stars
    allra  = np.hstack([T.ra  for T in TT])
    alldec = np.hstack([T.dec for T in TT])

    r_arcsec = 0.2
    I,J,d = match_radec(allra, alldec, allra, alldec, r_arcsec/3600., notself=True)
    dec = alldec[I]
    cosdec = np.cos(np.deg2rad(dec))
    dr = (allra[I]  - allra[J]) * cosdec * 3600.
    dd = (alldec[I] - alldec[J]) * 3600.

    plt.clf()
    rr = (-r_arcsec*1000, +r_arcsec*1000)
    plothist(dr*1000., dd*1000., nbins=100, range=(rr, rr))
    plt.xlabel('dRA (milli-arcsec)')
    plt.ylabel('dDec (milli-arcsec)')
    plt.savefig('match-all-after.png')

    I,J,d = match_radec(Tref.ra, Tref.dec, allra, alldec, r_arcsec/3600.)
    dec = alldec[J]
    cosdec = np.cos(np.deg2rad(dec))
    dr = (Tref.ra[I]  - allra[J]) * cosdec * 3600.
    dd = (Tref.dec[I] - alldec[J]) * 3600.
    plt.clf()
    rr = (-r_arcsec*1000, +r_arcsec*1000)
    plothist(dr*1000., dd*1000., nbins=100, range=(rr, rr))
    plt.xlabel('dRA (milli-arcsec)')
    plt.ylabel('dDec (milli-arcsec)')
    plt.savefig('match-all-ref-after.png')

    r_arcsec = 0.02
    I,J,d = match_radec(allra, alldec, allra, alldec, r_arcsec/3600., notself=True)
    dec = alldec[I]
    cosdec = np.cos(np.deg2rad(dec))
    dr = (allra[I]  - allra[J]) * cosdec * 3600.
    dd = (alldec[I] - alldec[J]) * 3600.
    plt.clf()
    rr = (-r_arcsec*1000, +r_arcsec*1000)
    plothist(dr*1000., dd*1000., nbins=100, range=(rr, rr))
    plt.xlabel('dRA (milli-arcsec)')
    plt.ylabel('dDec (milli-arcsec)')
    plt.savefig('match-all-after2.png')

    T = fits_table()
    T.ra = allra
    T.dec = alldec
    for col in ['f814w_vega', 'f475w_vega', 'f336w_vega',
                'f275w_vega', 'f110w_vega', 'f160w_vega',
                'name']:
        T.set(col, np.hstack([t.get(col) for t in TT]))
    T.writeto('aligned.fits')

    if False:
        from singles import alignment_plots

        dataset = 'M31'
        Nkeep = 100000
        R = 0.1
        minoverlap = 0.01
        perfield=False
        nocache=True
        
        from astrometry.util.multiproc import multiproc
        mp = multiproc()
        
        filts = ['F475W' for n in names]
        chips = [-1]*len(names)
        exptimes = [1]*len(names)
        Nall = [0]*len(names)
        rd = (minra,maxra,mindec,maxdec)
        cnames = names
        meta = (chips, names, cnames, filts, exptimes, Nall, rd)
        
        alignment_plots(afffn, dataset, Nkeep, 0, R, minoverlap, perfield, nocache, mp, 0,
                        tables=(TT, outlines, meta), lexsort=False)

