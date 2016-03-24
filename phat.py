def chipgap_geometry(brick, TT, fields):

    fcorners = []
    fchipgaptop = []
    fchipgapbot = []
    fchipgap = []
    for field in fields:
        fcorners.append(np.array(get_ir_footprint(brick, field)))
        X,Y,Z = get_chip_gap(brick, field)
        fchipgapbot.append(np.array(X))
        fchipgaptop.append(np.array(Y))
        fchipgap.append(np.array(Z))
    
    # Compute in-ir geometry
    for T,F in zip(TT, fcorners):
        T.inside_ir = point_in_poly(T.ra, T.dec, F)

    # Define inside_brick to be the union of inside_field.
    for T in TT:
        T.inside_brick = np.zeros(len(T), bool)
        for F in fcorners:
            T.inside_brick[point_in_poly(T.ra, T.dec, F)] = True

    # Compute in-own-chipgap
    for T,F in zip(TT, fchipgap):
        T.inside_chipgap = point_in_poly(T.ra, T.dec, F)

    # field-to-index map
    fimap = dict([(f,i) for i,f in enumerate(fields)])

    # Compute in-neighbor-chipgap
    for T, field in zip(TT, fields):
        # The position of this field in x,y field coords; field = 1 + fy*6 + fx
        fx = (field - 1) % 6
        fy = (field - 1) / 6

        T.inside_other_chipgap = np.zeros(len(T), bool)

        # Fields with fx != 5 (ie, not the right edge) fill in
        # the bottom half of the gap of the chip to the right.
        if fx < 5 and (field + 1) in fimap:
            F = fchipgapbot[fimap[ field+1 ]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        # Fields with fx != 0 (ie, not the left edge) fill in
        # the top half of the gap of the chip to the left.
        if fx > 0 and (field - 1) in fimap:
            F = fchipgaptop[fimap[ field-1 ]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        # The edges are weird:
        
        if field == 7 and 1 in fimap:
            # Field 7 fills in the bottom half of Field 1
            F = fchipgapbot[fimap[1]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        if field == 13 and 7 in fimap:
            # Field 13 fills in the bottom half of Field 7
            F = fchipgapbot[fimap[7]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        if field == 6 and 12 in fimap:
            # Field 6 fills in the top half of Field 12
            F = fchipgaptop[fimap[12]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        if field == 12 and 18 in fimap:
            # Field 12 fills in the top half of Field 18
            F = fchipgaptop[fimap[18]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

        # Hack -- nobody fills in the bottom of Field 13, so take any of its
        # own sources in the region
        if field == 13:
            F = fchipgapbot[fimap[13]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)
        # Similarly, nobody fills in the top of Field 6
        if field == 6:
            F = fchipgaptop[fimap[6]]
            T.inside_other_chipgap |= point_in_poly(T.ra, T.dec, F)

    return fcorners, fchipgaptop, fchipgapbot, fchipgap


def get_brick_corners(brick):
    '''
    Returns A (field 1 top-left), B (field 6 top-right),
            D (field 18 bottom-right), C (field 13 bottom-left)
    '''
    T = fits_table('corners.fits')
    I = np.flatnonzero(T.brick == brick)
    assert(len(I) == 1)
    corners = T[I[0]].corners
    A = corners[0:2]
    B = corners[2:4]
    C = corners[4:6]
    D = corners[6:8]
    return A,B,D,C

def get_ir_footprint(brick, field):
    T = fits_table('field-corners.fits')
    T.cut((T.field == field) * (T.brick == brick))
    assert(len(T) == 1)
    A,B,D,C = zip(T.ra[0] ,T.dec[0])
    return A,B,D,C

def get_ir_footprint_1(brick, field):
    A,B,D,C = get_brick_corners(brick)

    # The position of this field in x,y field coords; field = 1 + fy*6 + fx
    fx = (field - 1) % 6
    fy = (field - 1) / 6

    #    A-------topl-----topr-------------------------B
    #    |        |         |                          |
    #   lefttop---FA--------FB---------------------righttop
    #    |        |         |                          |
    #   leftbot---FC--------FD---------------------rightbot
    #    |        |         |                          |
    #    C-------botl-----botr-------------------------D

    top0 = A
    dtop = (B - A) / 6.
    topl = top0 + (dtop * fx)
    topr = top0 + (dtop * (fx+1))

    bot0 = C
    dbot = (D - C) / 6.
    botl = bot0 + (dbot * fx)
    botr = bot0 + (dbot * (fx+1))

    left0 = A
    dleft = (C - A) / 3.
    lefttop = left0 + (dleft * fy)
    leftbot = left0 + (dleft * (fy+1))

    right0 = B
    dright = (D - B) / 3.
    righttop = right0 + (dright * fy)
    rightbot = right0 + (dright * (fy+1))

    FA = line_intersection(topl, botl, lefttop, righttop)
    FB = line_intersection(topr, botr, lefttop, righttop)
    FC = line_intersection(topl, botl, leftbot, rightbot)
    FD = line_intersection(topr, botr, leftbot, rightbot)

    return FA,FB,FD,FC

def get_chip_gap(brick, field):
    '''
    Returns (A,B,D,C) points for the bottom half, top half, and full
    chip gap.
    '''
    FA,FB,FD,FC = get_ir_footprint(brick, field)
    # chip gap box
    FB = np.array(FB)
    FC = np.array(FC)
    dc = FB - FC
    dperp = np.array([dc[1], -dc[0]])
    dperp /= np.sqrt(np.sum(dperp**2))
    # size of chip gap in pseudo-degrees
    lp = 2e-3
    GA = FC - lp*dperp
    GB = FC + lp*dperp
    GC = FC - lp*dperp + 0.5*dc
    GD = FC + lp*dperp + 0.5*dc
    bottom = (GA,GB,GD,GC)

    GE = FB - lp*dperp
    GF = FB + lp*dperp
    top = (GC,GD,GF,GE)

    full = (GA,GB,GF,GE)

    return bottom, top, full


def get_flt_objs(cam, brick, field, filt=None):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'phat.settings'
    import dj.settings
    from phat.web.models import PhatImage

    ims = PhatImage.objects.all()
    ims = ims.filter(imagetype='flt')
    ims = ims.filter(detector=camtodet(cam))
    ims = ims.filter(exptime__gt=0, best=True)
    if filt is not None:
        ims = ims.filter(filter=filt)
    flts = ims.filter(brick=brick, field=field)
    return flts

def get_flt_filenames_exts(cam, brick, field, filt=None, firstonly=False):
    from astrom_flt import get_flt_fn_ext

    flts = get_flt_objs(cam, brick, field, filt)
    hdus = camtohdus(cam)
    fltfns = []
    for flt in flts:
        fns = []
        for hdu in hdus:
            fn,ext = get_flt_fn_ext('data/flt-wcs', flt, hdu)
            if os.path.exists(fn):
                fns.append((fn,ext))
            else:
                print 'FLT WCS filename', fn, 'not found!'
        if len(fns):
            fltfns.append(fns)
            if firstonly:
                break
    return fltfns

def filter_to_cam(filt):
    return {475: 'WFC',
            814: 'WFC',
            110: 'IR',
            160: 'IR',
            275: 'UVIS',
            336: 'UVIS',
            }[filt]

def camtohdus(cam):
    # which FITS HDUs have the WCS headers we want?
    camtohdumap = { 'ACS': [1,4],
                    'UV': [1,4],
                    'IR': [1] }
    return camtohdumap[cam]

def camtodet(cam):
    camtodetmap = { 'ACS': 'WFC',
                    'UV': 'UVIS',
                    'IR': 'IR' }
    return camtodetmap[cam]

def phat_file(filetype, brick, field, cam, filter=None, version=None):
    '''
    filter: integer
    version: integer
    '''
    
    prog = bricktoprogram(brick)
    # cam map?

    # filetypes that need cam...
    needcam = ['wcs2', 'drz', 'drz_mask', 'gst', 'st', 'dir', 'fulldir', 'photinfo',
               #'aff'
               ]
    # filetypes that need filter...
    needfilter = ['wcs2', 'drz', 'drz_mask', 'drzwcs']
    # filetypes that need version...
    needversion = ['wcs2'] #, 'aff']

    if filetype in needcam:
        if not cam in ['WFC', 'UVIS', 'IR', 'ACS', 'UV']:
            raise RuntimeError('unknown camera: "%s"' % cam)
    if filetype in needfilter:
        if filter is None:
            raise RuntimeError('need filter for filetype "%s"' % filetype)
    if filetype in needversion:
        if version is None:
            raise RuntimeError('need version for filetype "%s"' % filetype)

    if cam == 'ACS':
        cam = 'WFC'
    if cam == 'UV':
        cam = 'UVIS'

    idstr = '%i_M31-B%02i-F%02i-%s' % (prog, brick, field, cam)

    if filetype == 'dir':
        return idstr
    if filetype == 'fulldir':
        return 'data/pipe/%s' % idstr

    if filetype == 'wcs2':
        return 'wcs-v%i/%s_F%03iW.wcs' % (version, idstr, filter)
    # if filetype == 'aff':
    #   # ugh
    #   cammap = {'WFC':'acs',
    #             'UVIS':'uv',
    #             'IR':'ir'}
    #   return 'affines-b%02i-v%i-%s.fits' % (brick, version, cammap[cam])
    if filetype == 'drzwcs':
        return 'data/pipe/drzwcs/%s_F%03iW_drz.fits.wcs' % (idstr, filter)

    if filetype in ['drz', 'drz_mask']:
        # 'data/pipe/12056_M31-B15-F01-IR/proc/12056_M31-B15-F01-IR_F110W_drz.fits'
        return 'data/pipe/%s/proc/%s_F%03iW_%s.fits' % (idstr, idstr, filter, filetype)
    if filetype in ['gst', 'st']:
        fmap = dict(IR='F110W_F160W', WFC='F475W_F814W', UVIS='F275W_F336W')
        return 'data/pipe/%s/proc/%s_%s.%s.fits' % (idstr, idstr, fmap[cam], filetype)
    if filetype == 'photinfo':
        return 'data/pipe/%s/proc/%s.phot.info' % (idstr, idstr)

    raise RuntimeError('unknown filetype: "%s"' % filetype)


def get_reference_filter(cam):
    if cam in ['ACS']:
        return 475
    if cam in ['IR']:
        return 160
    if cam in ['UV']:
        return 336
    raise RuntimeError('unknown cam' + str(cam))

def bricktoprogram(brick):
    pidmap = {
        1: 12058,
        2: 12073,
        3: 12109,
        4: 12107,
        5: 12074,
        6: 12105,
        7: 12113,
        8: 12075,
        9: 12057,
        10: 12111,
        11: 12115,
        12: 12071,
        13: 12114,
        14: 12072,
        15: 12056,
        16: 12106,
        17: 12059,
        18: 12108,
        19: 12110,
        20: 12112,
        21: 12055,
        22: 12076,
        23: 12070,
        }
    return pidmap[brick]

def splitFieldId(fieldid):
    '''
    Returns (cam, brick, field), eg ('ACS', 21, 4)
    '''
    camid = int(fieldid / 10000)
    camidmap = { 1: 'ACS', 2: 'IR', 3: 'UV'}
    if not camid in camidmap:
        raise RuntimeError('Unknown camera ID: "%i"' % camid)
    brick = int(fieldid / 100) % 100
    field = int(fieldid % 100)
    assert(brick >= 1)
    assert(brick <= 23)
    assert(field >= 1)
    assert(field <= 18)
    return (camidmap[camid], brick, field)

def getFieldId(cam, brick, field):
    camidmap = { 'ACS': 1, 'IR': 2, 'UV': 3 }
    if not cam in camidmap:
        raise RuntimeError('Unknown camera: "%s"' % cam)
    assert(brick >= 1)
    assert(brick <= 23)
    assert(field >= 1)
    assert(field <= 18)
    return camidmap[cam] * 10000 + brick * 100 + field

