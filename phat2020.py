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
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import match_radec

os.environ['NUMEXPR_MAX_THREADS'] = '8'

def check_results(fns, tag):

    def get_field(ds, col):
        return ds.evaluate(ds[col.upper()])

    rr = []
    dd = []
    for fn in fns:
        df = pd.read_hdf(fn, key='data')
        ds = vaex.from_pandas(df)
        print(len(ds), 'rows')
        ra = get_field(ds, 'ra')
        dec = get_field(ds, 'dec')
        rr.append(ra)
        dd.append(dec)
    rr = np.hstack(rr)
    dd = np.hstack(dd)
    print('Total of', len(rr), 'stars')

    T = fits_table()
    T.ra = rr
    T.dec = dd
    T.writeto('all-rd-%s.fits' % tag)

    plothist(rr, dd, 500)
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.savefig('all-radec-%s.png' % tag)

    I,J,d = match_radec(rr, dd, rr, dd, 0.2/3600, notself=True)
    plt.clf()
    plt.hist(d * 3600. * 1000., bins=50)
    plt.xlabel('Distance between stars (milli-arcsec)')
    plt.savefig('all-dists-%s.png' % tag)

def check_results_2(tag):
    T = fits_table('all-rd-%s.fits' % tag)
    I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, 0.2/3600, notself=True)
    plt.clf()
    plt.hist(d * 3600. * 1000., bins=50)
    plt.xlabel('Distance between stars (milli-arcsec)')
    plt.savefig('all-dists-%s.png' % tag)

    plt.clf()
    plt.hist(d * 3600. * 1000., bins=50, log=True)
    plt.xlabel('Distance between stars (milli-arcsec)')
    plt.savefig('all-dists-log-%s.png' % tag)

def apply_alignments(aff_fn, corners_fn, infns, pandas=True):
    from astrom_common import Affine
    T = fits_table(aff_fn)
    affs = Affine.fromTable(T)
    print('Read affines:', affs)

    ibright = dict([(fn.strip(),i) for i,fn in enumerate(T.filenames)])

    corners = {}
    for line in open(corners_fn).readlines():
        line = line.strip()
        words = line.split()
        ras = np.array([float(words[i]) for i in [1,3,5,7]])
        decs = np.array([float(words[i]) for i in [2,4,6,8]])
        corners[words[0]] = (ras,decs)
    from astrometry.util.miscutils import point_in_poly

    #fns1 = glob('data/M31-*ST/proc_default/M31-*ST.phot.hdf5')
    #fns2 = glob('data/M31-*ST/M31-*ST.phot.hdf5')
    #fns1.sort()
    #fns2.sort()
    #fns = fns1 + fns2
    fns = infns
    print('Files:', fns)

    veto_polys = []

    for photfile in fns:
        basename = os.path.basename(photfile)
        basename = basename.replace('.phot.hdf5', '')
        print('Base name:', basename)

        corner = corners[basename]
        ras,decs = corner
        poly = np.vstack((ras, decs)).T

        outfn2 = 'cut-%s.hdf5' % basename
        if os.path.exists(outfn2):
            print('File', outfn2, 'exists; skipping')
            veto_polys.append(poly)
            continue

        brightfn = basename + '-bright.fits'
        ii = ibright[brightfn]
        aff = affs[ii]

        print('Reading', photfile)
        if pandas:
            df = pd.read_hdf(photfile, key='data')
            ds = vaex.from_pandas(df)
        else:
            ds = vaex.open(photfile)

        def get_field(ds, col):
            if pandas:
                return ds.evaluate(ds[col])
            else:
                return ds.evaluate(ds[col.upper()])

        print(len(ds), 'rows')
        ra  = get_field(ds, 'ra')
        dec = get_field(ds, 'dec')
        ra,dec = aff.apply(ra, dec)

        Tleft = fits_table()
        Tleft.ra = ra
        Tleft.dec = dec
        Tleft.index = np.arange(len(Tleft))
        inside = point_in_poly(Tleft.ra, Tleft.dec, poly)
        print(np.sum(inside), 'of', len(Tleft), 'inside corners of this half-brick')

        inside_veto = np.zeros(len(Tleft), bool)
        for vp in veto_polys:
            inveto = point_in_poly(Tleft.ra, Tleft.dec, vp)
            inside_veto[inveto] = True
        print(np.sum(inside_veto), 'stars are inside the corners of previous half-bricks')
        print('inside:', type(inside), inside.dtype)
        inside[inside_veto] = False
        print(np.sum(inside), 'stars are uniquely in this half-brick')
        
        veto_polys.append(poly)

        outfn = 'out/out-%s.hdf5' % basename
        if pandas:
            df[inside].to_hdf(outfn, key='data', mode='w',
                          format='table', complevel=9, complib='zlib')
        else:
            df = ds.take(np.flatnonzero(inside)).to_pandas_df()
            df.to_hdf(outfn, key='data', mode='w',
                      format='table', complevel=9, complib='zlib')
        print('Wrote', outfn)

        outfn = 'cut/cut-%s.hdf5' % basename
        if pandas:
            df[np.logical_not(inside)].to_hdf(outfn, key='data', mode='w',
                                              format='table', complevel=9, complib='zlib')
        else:
            df = ds.take(np.flatnonzero(np.logical_not(inside))).to_pandas_df()
            df.to_hdf(outfn, key='data', mode='w',
                      format='table', complevel=9, complib='zlib')
        print('Wrote', outfn)
    

def to_fits(fns, pandas=True):
    print('Files:', fns)

    plt.clf()

    outfns = []

    for photfile in fns:
        #photfile like 'data/M31-B23-WEST/M31-B23-WEST.phot.hdf5'
        print()
        print(photfile)
        basename = os.path.basename(photfile)
        basename = basename.replace('.phot.hdf5', '')
        print('Base name:', basename)
    
        outfn = basename + '-bright.fits'
        outfns.append(outfn)
        if os.path.exists(outfn):
            print('Exists:', outfn)

            st_in  = os.stat(photfile)
            st_out = os.stat(outfn)
            print('Timestamps: in', st_in.st_mtime, 'out', st_out.st_mtime)
            if st_out.st_mtime > st_in.st_mtime:
                continue
            print('Input file is newer!')

        basename = basename.replace('_', '-')
        words = basename.split('-')
        assert(len(words) == 3)
        galaxy = words[0]
        assert(galaxy.startswith('M'))
        brick = words[1]
        assert(brick[0] == 'B')
        brick = int(brick[1:], 10)
        print('Brick number:', brick)
        dirn = words[2]
        #ew = words[2]
        assert(dirn in ['EAST', 'WEST', 'NW','NN','NE','SW','SS','SE'])
        #east = (ew == 'EAST')
    
        if pandas:
            df = pd.read_hdf(photfile, key='data')
            ds = vaex.from_pandas(df)
        else:
            ds = vaex.open(photfile)
        print('Read', photfile)
        #print(ds)

        def get_field(ds, col):
            if pandas:
                return ds.evaluate(ds[col])
            else:
                return ds.evaluate(ds[col.upper()])

        print(len(ds), 'rows')
        if 'f814w_gst' in ds:
            good = get_field(ds, 'f814w_gst')
            print(len(ds), 'rows')
            #print('good:', good.dtype)
            from collections import Counter
            print('good:', Counter(good))
            #print('ds:', ds.dtype)
            #ds = ds[good]
            #ds = ds[np.flatnonzero(good)]
            ds = ds.take(np.flatnonzero(good))
            #print('ds:', ds)
            print(len(ds), 'gst on F814W')
        else:
            ds.select('(F814W_SNR > 4) & (F814W_SHARP**2 < 0.2)', name='F814W_ST')
            ds.select('F814W_ST & (F814W_CROWD < 2.25)', name='F814W_GST')
            ds = ds[ds['F814W_GST']]
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
    
        mag = get_field(ds, 'f814w_vega')
        print('Of', len(mag), 'mags,', np.sum(np.isfinite(mag)), 'are finite')
        print('range:', np.nanmin(mag), np.nanmax(mag))
    
        plt.hist(mag[np.isfinite(mag)], range=(20, 28), bins=50, label=basename)

        with np.errstate(invalid='ignore'):
            print('ds', ds)
            if pandas:
                #ds = ds[mag < 24]
                ds = ds.take(np.flatnonzero(mag < 24))
            else:
                ds = ds[ds['F814W_VEGA'] < 24]
                #ds = ds.take(np.flatnonzero(ds['F814W_VEGA'] < 24))
            print('ds cut', ds)
        print(len(ds), 'with F814W < 24')

        mag = get_field(ds, 'f814w_vega')
        xx  = get_field(ds, 'x')
        yy  = get_field(ds, 'y')

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

        cols = ['ra','dec','x', 'y']
        if pandas:
            cols.append('index')

        T = fits_table()
        for col in cols:
            T.set(col, get_field(ds, col))
        for filt in [814, 475, 336, 275, 110, 160]:
            for col in ['f%iw_vega']:
                colname = col % filt
                T.set(colname, get_field(ds, colname))
        T.galaxy = np.array([galaxy] * len(T))
        T.brick = np.zeros(len(T), np.uint8) + brick
        #T.east = np.zeros(len(T), bool)
        #T.east[:] = east
        T.dirn = np.array([dirn] * len(T))
        T.writeto(outfn)
    
    plt.legend()
    plt.xlabel('F814W mag')
    plt.savefig('mags.png')
    return outfns

def find_alignments(fns, wcsfns, gaia_fn, aff_fn, aligned_fn):
    from astrometry.libkd.spherematch import tree_build_radec, trees_match
    from astrometry.libkd.spherematch import match_radec
    from astrometry.util.plotutils import plothist
    from astrometry.util.util import Tan
    import fitsio

    from astrom_common import getwcsoutline
    from singles import find_overlaps

    if True:
        WCS = []
        for fn in wcsfns:
            wcs = Tan(fn)
            WCS.append(wcs)

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

    Tref = fits_table(gaia_fn)
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

    T.writeto(aff_fn)


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
    T.writeto(aligned_fn)

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


if __name__ == '__main__':
    import sys
    import fitsio
    from astrometry.util.util import Tan

    #check_results_2()
    #aligned_fns = glob('out-M31-B*.hdf5')
    #check_results(aligned_fns, 'M31')
    #sys.exit(0)

    phat = True

    if phat:
        # NOTE -- there *are* duplicates in these sets.  (B23).
        # Take the "proc_default" ones first, if both exist.
        fns1 = glob('data/M31-*ST/proc_default/M31-*ST.phot.hdf5')
        fns1.sort()
        fns2 = glob('data/M31-*ST/M31-*ST.phot.hdf5')
        fns2.sort()
        infns = fns1 + fns2
        basenames = set()
        keepfns = []
        for fn in infns:
            basename = os.path.basename(fn)
            if basename in basenames:
                continue
            keepfns.append(fn)
            basenames.add(basename)
        infns = keepfns

        to_fits(infns, pandas=True)

        outfns = glob('M31-*-bright.fits')
        outfns.sort()

        gaia_fn = 'gaia.fits'
        aff_fn = 'affines.fits'
        aligned_fn = 'aligned.fits'

        keepfns = []
        wcsfns = []
        for fn in outfns:
            base = fn.replace('-bright.fits', '')

            wfn = base + '-wcs.fits'
            if os.path.exists(wfn):
                print('Exists:', wfn)
                keepfns.append(fn)
                wcsfns.append(wfn)
                continue

            #data/M31-B23-WEST/M31-B23-WEST_F475W_drz.chip1.fits
            pat1 = 'data/' + base + '/proc_default/' + base + '*drz.chip1.fits'
            pat2 = 'data/' + base + '/' + base + '*drz.chip1.fits'
            print(pat1, pat2)
            ff = glob(pat1) + glob(pat2)
            print('WCS files:', ff)
            if len(ff) < 1:
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                continue
            if len(ff) > 1:
                print('Keeping', ff[0])
            wcsfn = ff[0]


            F = fitsio.FITS(wcsfn)
            info = F[0].get_info()
            H,W = info['dims']
            wcs = Tan(wcsfn)
            wcs.imagew = W
            wcs.imageh = H
            wfn = base + '-wcs.fits'
            wcs.write_to(wfn)

        corners_fn = 'corners.txt'
        kwargs = dict()

        aligned_fns = glob('out-M31-B*.hdf5')
        tag = 'M31'

    else:
        infns = glob('m33-data/legacy_phot/M33_*.phot.hdf5')
        infns.sort()
        kwargs = dict(pandas=False)
    
        gaia_fn = 'gaia-m33.fits'
        aff_fn = 'affines-m33.fits'
        aligned_fn = 'aligned-m33.fits'
    
        corners_fn = 'corners-m33.txt'

        outfns = to_fits(infns, **kwargs)
    
        wcsfns = []
        for fn in outfns:
            base = fn.replace('-bright.fits', '')
            wcsfn = base + '.wcs'
            if not os.path.exists(wcsfn):
                #fn = 'm33-data/' + base + '/' + base + '_F475W_drc_sci.chip1.fits.gz'
                wfn = 'm33-data/legacy_wcs/' + base + '_F475W_drc_wcs.txt'
                import astropy.io.fits
                hdr = astropy.io.fits.Header.fromtextfile(wfn)
                W = hdr['NAXIS1']
                H = hdr['NAXIS2']
                hdr['IMAGEW'] = W
                hdr['IMAGEH'] = H
                astropy.io.fits.writeto(wcsfn, None, header=hdr)
            wcsfns.append(wcsfn)
            
        for fn,wcsfn in zip(infns, wcsfns):
            # corners
            ds = vaex.open(fn)
            x0, x1 = ds.minmax('X')
            y0, y1 = ds.minmax('Y')
            print(fn, 'x', x0, x1, 'y', y0, y1)
            pix_coords = np.c_[[x0, x0, x1, x1], [y0, y1, y1, y0]]
            from astropy.wcs import WCS
            wcs_coords = WCS(wcsfn).all_pix2world(pix_coords, 0.5) # 0.5 is the dolphot pixel "origin"
            print('wcs', wcs_coords)

        aligned_fns = glob('out-M33_B*.hdf5')
        tag = 'M33'

    find_alignments(outfns, wcsfns, gaia_fn, aff_fn, aligned_fn)
    apply_alignments(aff_fn, corners_fn, infns, **kwargs)
    #check_results(aligned_fns, tag)
    #check_results_2(tag)
    sys.exit(0)
