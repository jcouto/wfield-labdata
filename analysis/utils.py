import numpy as np

__all__ = [
    'build_alignment_transform',
    'build_atlas_transform',
    'warp_image',
    'transform_coordinates',
    'inverse_transform_coordinates',
    'transform_atlas_regions',
    'similarity_matrix',
    'decompose_similarity',
    'circle_mask',
    'auto_brain_mask',
    'register_similarity',
]


def build_alignment_transform(fw, fh, rotation, scale, ratio, origin_x, origin_y):
    """Build a 3x3 forward affine matrix mapping 2P (col,row) → reference (col,row).

    Parameters
    ----------
    fw, fh : float
        Width and height of the 2P image (after any transpose).
    rotation : float
        Rotation in degrees (counter-clockwise).
    scale : float
        Isotropic scale factor.
    ratio : float
        X/Y aspect ratio; sx = scale * ratio, sy = scale.
    origin_x, origin_y : float
        Translation: col and row of the 2P centre in reference-image coordinates.

    Returns
    -------
    M_fwd : ndarray, shape (3, 3)
        Forward transform matrix.  Pass inv(M_fwd) to skimage AffineTransform.
    """
    cx, cy = fw / 2.0, fh / 2.0
    theta = np.deg2rad(float(rotation))
    cs, sn = np.cos(theta), np.sin(theta)
    sx, sy = float(scale) * float(ratio), float(scale)

    M1 = np.array([[1, 0, -cx],   [0, 1, -cy],   [0, 0, 1]], dtype=float)
    M2 = np.array([[sx*cs, -sy*sn, 0], [sx*sn, sy*cs, 0], [0, 0, 1]], dtype=float)
    M3 = np.array([[1, 0, origin_x], [0, 1, origin_y], [0, 0, 1]], dtype=float)
    return M3 @ M2 @ M1


def warp_image(image, M_fwd, output_shape, **kwargs):
    """Warp *image* from 2P space into reference space using *M_fwd*.

    Parameters
    ----------
    image : ndarray, 2-D
        Source image (2P space).
    M_fwd : ndarray, shape (3, 3)
        Forward transform from `build_alignment_transform`.
    output_shape : tuple (H, W)
        Shape of the output (reference-space) image.
    **kwargs
        Extra keyword arguments forwarded to `skimage.transform.warp`
        (e.g. ``mode='constant'``, ``cval=0.0``).

    Returns
    -------
    warped : ndarray, shape output_shape
    """
    from skimage.transform import AffineTransform, warp as sk_warp
    tf_inv = AffineTransform(matrix=np.linalg.inv(M_fwd))
    kwargs.setdefault('mode', 'constant')
    kwargs.setdefault('cval', 0.0)
    kwargs.setdefault('preserve_range', True)
    return sk_warp(image.astype(float), tf_inv, output_shape=output_shape, **kwargs)


def transform_coordinates(xy, M_fwd):
    """Apply the forward transform to (col, row) coordinate pairs.

    Maps points from 2P space to reference-image space.

    Parameters
    ----------
    xy : array-like, shape (N, 2) or (2,)
        Points as ``(col, row)`` pairs.
    M_fwd : ndarray, shape (3, 3)
        Forward transform from `build_alignment_transform`.

    Returns
    -------
    xy_out : ndarray, shape (N, 2)
        Transformed ``(col, row)`` points in reference space.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))  # (N, 2)
    ones = np.ones((len(xy), 1), dtype=float)
    hom = np.hstack([xy, ones])          # (N, 3)
    out = (M_fwd @ hom.T).T             # (N, 3)
    return out[:, :2]                   # (N, 2)  col, row


def inverse_transform_coordinates(xy, M_fwd):
    """Apply the inverse transform to (col, row) coordinate pairs.

    Maps points from reference-image space back to 2P space.

    Parameters
    ----------
    xy : array-like, shape (N, 2) or (2,)
        Points as ``(col, row)`` pairs in reference space.
    M_fwd : ndarray, shape (3, 3)
        Forward transform from `build_alignment_transform`.

    Returns
    -------
    xy_out : ndarray, shape (N, 2)
        Corresponding ``(col, row)`` points in 2P space.
    """
    return transform_coordinates(xy, np.linalg.inv(M_fwd))


def build_atlas_transform(bregma_xy, resolution, rotation=0.0, scale=1.0, ratio=1.0,
                          mirror=False):
    """Build a 3x3 forward matrix mapping atlas mm coordinates to widefield pixel coordinates.

    Atlas regions are expressed in mm relative to bregma (bregma = origin).
    The resulting matrix places bregma at ``bregma_xy`` in the widefield image.

    Parameters
    ----------
    bregma_xy : (col, row)
        Location of bregma in the widefield image, in pixels.
    resolution : float
        mm per pixel — base scale converting atlas mm to pixel distances.
    rotation : float
        Rotation in degrees (counter-clockwise).
    scale : float
        Additional isotropic scale applied on top of ``1/resolution``.
    ratio : float
        X/Y aspect ratio correction; effective sx = (scale * ratio) / resolution.
    mirror : bool
        Flip the atlas x-axis about bregma (for imaging setups where the x-axis is reversed).

    Returns
    -------
    M_fwd : ndarray, shape (3, 3)
    """
    bregma_col, bregma_row = float(bregma_xy[0]), float(bregma_xy[1])
    theta = np.deg2rad(float(rotation))
    cs, sn = np.cos(theta), np.sin(theta)
    sx = float(scale) * float(ratio) / float(resolution)
    sy = float(scale) / float(resolution)
    if mirror:
        sx = -sx
    return np.array([[sx * cs, -sy * sn, bregma_col],
                     [sx * sn,  sy * cs, bregma_row],
                     [0,        0,       1          ]], dtype=float)


def transform_atlas_regions(ccf_regions, M_fwd):
    """Transform atlas contour DataFrame from mm space to image pixel space.

    Parameters
    ----------
    ccf_regions : DataFrame
        From ``allen_load_reference()``; columns ``*_x`` / ``*_y`` hold mm coordinates
        relative to bregma, and ``*_center`` holds ``[x_mm, y_mm]`` centroids.
    M_fwd : ndarray, shape (3, 3)
        Forward transform mapping atlas mm coordinates to widefield pixel coordinates.
        Build with ``build_atlas_transform()`` (operations path) or compose
        ``M_lm.params @ T_res`` from ``allen_transform_from_landmarks`` (landmarks path).

    Returns
    -------
    DataFrame with the same structure but all coordinates in widefield pixel space.
    """
    result = ccf_regions.copy()
    for i, row in result.iterrows():
        for side in ('left', 'right'):
            x = np.asarray(row[f'{side}_x'], dtype=float)
            y = np.asarray(row[f'{side}_y'], dtype=float)
            xy_px = transform_coordinates(np.column_stack([x, y]), M_fwd)
            result.at[i, f'{side}_x'] = xy_px[:, 0].tolist()
            result.at[i, f'{side}_y'] = xy_px[:, 1].tolist()
            center = np.asarray(row[f'{side}_center'], dtype=float).reshape(1, 2)
            center_px = transform_coordinates(center, M_fwd)
            result.at[i, f'{side}_center'] = center_px[0].tolist()
    return result


# ---------------------------------------------------------------------------
# Image-to-image similarity registration (used to carry a WidefieldAtlasTransform
# from one widefield session to another session of the same animal)
# ---------------------------------------------------------------------------

def similarity_matrix(scale, rotation, tx, ty):
    """3x3 similarity (isotropic scale, rotation in degrees, translation) in the
    ``(col, row, 1)`` convention used throughout this module."""
    th = np.deg2rad(float(rotation))
    c, s = np.cos(th), np.sin(th)
    return np.array([[scale * c, -scale * s, tx],
                     [scale * s,  scale * c, ty],
                     [0., 0., 1.]], dtype=float)


def decompose_similarity(M):
    """Return ``(scale, rotation_deg, tx, ty)`` of a 3x3 similarity matrix.

    For a general affine the scale is ``sqrt(|det|)`` and the rotation is taken
    from the first column, i.e. the closest similarity in the least-squares sense
    only when the matrix has little shear.
    """
    M = np.asarray(M, dtype=float)
    scale = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return scale, rotation, float(M[0, 2]), float(M[1, 2])


def circle_mask(shape, circle):
    """Boolean mask of a circle given as ``(xc, yc, r)`` (ImagingWindow convention)."""
    xc, yc, r = np.asarray(circle, dtype=float).ravel()[:3]
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - xc) ** 2 + (yy - yc) ** 2 <= r ** 2


def auto_brain_mask(image, smooth_frac=0.01, erode_frac=0.03):
    """Rough mask of the illuminated window: Otsu threshold on a smoothed image,
    largest connected component, eroded so the window edge is left out."""
    from scipy.ndimage import gaussian_filter, binary_erosion, label
    from skimage.filters import threshold_otsu
    img = np.nan_to_num(np.asarray(image, dtype=float))
    size = max(img.shape)
    sm = gaussian_filter(img, max(1., smooth_frac * size))
    mask = sm > threshold_otsu(sm)
    lab, n = label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == sizes.argmax()
    it = max(1, int(erode_frac * size))
    return binary_erosion(mask, iterations=it)


def _prepare_for_registration(image, mask=None, sigma_frac=0.03):
    """High-pass filter (keeps the vessel pattern, drops illumination), restrict
    to ``mask`` with normalised convolution so the mask edge does not bleed in,
    and standardise. Returns ``(prepared, mask)``."""
    from scipy.ndimage import gaussian_filter
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    img = np.where(finite, img, 0.)
    mask = finite if mask is None else (np.asarray(mask, bool) & finite)
    m = mask.astype(float)
    sigma = sigma_frac * max(img.shape)
    background = gaussian_filter(img * m, sigma) / (gaussian_filter(m, sigma) + 1e-9)
    hp = (img - background) * m
    sd = hp[mask].std() if mask.any() else 1.
    hp = np.clip(hp / (sd + 1e-12), -4, 4)
    return hp * m, mask


def _downsample_for_registration(img, mask, f):
    """Block-average by an integer factor. Returns ``(img, mask, D)`` where ``D``
    maps full-resolution ``(col, row, 1)`` to the downsampled grid."""
    from skimage.transform import downscale_local_mean
    if f == 1:
        return img, mask, np.eye(3)
    im = downscale_local_mean(img, (f, f))
    mk = downscale_local_mean(mask.astype(float), (f, f)) > 0.5
    D = np.array([[1. / f, 0., 0.5 / f - 0.5],
                  [0., 1. / f, 0.5 / f - 0.5],
                  [0., 0., 1.]])
    return im, mk, D


def _masked_ncc(a, b, mask):
    n = int(mask.sum())
    if n < 50:
        return -1.
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else -1.


def _registration_score(M, src, src_mask, dst, dst_mask):
    """Masked normalised cross-correlation of ``src`` warped by ``M`` against ``dst``."""
    w = warp_image(src, M, dst.shape, order=1, cval=0.)
    wm = warp_image(src_mask.astype(float), M, dst.shape, order=1, cval=0.) > 0.5
    return _masked_ncc(w, dst, wm & dst_mask)


def register_similarity(src, dst, src_mask=None, dst_mask=None, init=None,
                        scales=None, angles=None, coarse_size=128,
                        refine_levels=(4, 2, 1), n_starts=3, verbose=False):
    """Estimate the similarity transform mapping pixels of ``src`` onto ``dst``.

    Intended for two widefield images of the same cranial window taken on
    different days, possibly at a different magnification or binning. Both
    images are high-pass filtered so the vessel pattern drives the fit, and
    compared with a masked normalised cross-correlation (NCC). A coarse grid over
    scale x rotation at low resolution, with the translation found by masked
    phase correlation, seeds a Nelder-Mead refinement over an image pyramid; the
    ``n_starts`` best coarse candidates are refined and the best final NCC wins.

    Parameters
    ----------
    src, dst : 2-D arrays
        Source (has the known transform) and destination images.
    src_mask, dst_mask : bool arrays, optional
        Pixels to use (e.g. the ImagingWindow circle). ``None`` uses all finite
        pixels; see `auto_brain_mask` for a data-driven mask.
    init : 3x3 array, optional
        Initial guess (src px -> dst px); the scale grid is then centred on it.
    scales, angles : sequences, optional
        Coarse grid. Defaults: scale 0.4-3 (41 log steps), rotation ±25° in 2.5°.
    coarse_size : int
        Approximate image size (px) used for the coarse search.
    refine_levels : sequence of int
        Downsampling factors for the pyramid refinement, coarse to fine.
    n_starts : int
        Number of distinct coarse candidates to refine.

    Returns
    -------
    M : (3, 3) ndarray
        Maps ``src`` ``(col, row, 1)`` to ``dst`` ``(col, row, 1)``.
    info : dict
        ``ncc`` (final score at full resolution), ``scale``, ``rotation`` (deg),
        ``translation`` ``(tx, ty)``, and ``candidates`` (the refined starts).
    """
    from skimage.registration import phase_cross_correlation
    from scipy.optimize import minimize

    src_p, src_m = _prepare_for_registration(src, src_mask)
    dst_p, dst_m = _prepare_for_registration(dst, dst_mask)

    if scales is None:
        if init is not None:
            scales = decompose_similarity(init)[0] * np.geomspace(0.8, 1.25, 13)
        else:
            scales = np.geomspace(0.4, 3.0, 41)
    if angles is None:
        angles = np.arange(-25, 25.1, 2.5)

    # ---- coarse search at low resolution
    f_src = max(1, int(round(max(src_p.shape) / coarse_size)))
    f_dst = max(1, int(round(max(dst_p.shape) / coarse_size)))
    s_lo, sm_lo, Ds = _downsample_for_registration(src_p, src_m, f_src)
    d_lo, dm_lo, Dd = _downsample_for_registration(dst_p, dst_m, f_dst)
    Ds_inv, Dd_inv = np.linalg.inv(Ds), np.linalg.inv(Dd)
    src_c = np.array([(src_p.shape[1] - 1) / 2, (src_p.shape[0] - 1) / 2])
    dst_c = np.array([(dst_p.shape[1] - 1) / 2, (dst_p.shape[0] - 1) / 2])
    candidates = []     # (score, M_fullres)
    for s in scales:
        for a in angles:
            M0 = similarity_matrix(s, a, 0., 0.)
            if init is None:
                M0[:2, 2] = dst_c - M0[:2, :2] @ src_c          # centre on centre
            else:
                M0[:2, 2] = init[:2, 2] + (init[:2, :2] - M0[:2, :2]) @ src_c
            M_lo = Dd @ M0 @ Ds_inv
            w = warp_image(s_lo, M_lo, d_lo.shape, order=1, cval=0.)
            wm = warp_image(sm_lo.astype(float), M_lo, d_lo.shape, order=1, cval=0.) > 0.5
            if (wm & dm_lo).sum() < 0.05 * dm_lo.sum():
                continue
            try:
                shift = phase_cross_correlation(d_lo, w, reference_mask=dm_lo,
                                                moving_mask=wm)[0]
            except Exception:
                shift = np.zeros(2)
            M_lo2 = M_lo.copy()
            M_lo2[0, 2] += shift[1]       # shift is (row, col)
            M_lo2[1, 2] += shift[0]
            for Mc in (M_lo, M_lo2):
                sc = _registration_score(Mc, s_lo, sm_lo, d_lo, dm_lo)
                candidates.append((sc, Dd_inv @ Mc @ Ds))
    if not candidates:
        raise RuntimeError('Coarse registration found no overlapping candidate; '
                           'check the masks or pass an initial guess.')
    candidates.sort(key=lambda c: -c[0])
    # keep the best few that are not near-duplicates of a better one
    starts = []
    for sc, Mc in candidates:
        p = np.array(decompose_similarity(Mc))
        if all(abs(np.log(p[0] / q[0])) > 0.05 or abs(p[1] - q[1]) > 3
               or np.hypot(*(p[2:] - q[2:])) > 0.05 * max(dst_p.shape)
               for q in starts):
            starts.append(p)
        if len(starts) >= n_starts:
            break
    if verbose:
        print('coarse candidates:', [(round(c[0], 3),) for c in candidates[:n_starts]])

    # ---- refine over the pyramid: every start at the coarse levels, only the
    # best survivor at the finest (most expensive) level
    pyramid = [(_downsample_for_registration(src_p, src_m, f),
                _downsample_for_registration(dst_p, dst_m, f), f) for f in refine_levels]

    def refine(M, level):
        (s_f, sm_f, Ds_f), (d_f, dm_f, Dd_f), f = level
        Ds_f_inv = np.linalg.inv(Ds_f)
        s0, a0, tx0, ty0 = decompose_similarity(M)

        def cost(p):
            Mp = similarity_matrix(s0 * np.exp(p[0]), a0 + p[1], tx0 + p[2], ty0 + p[3])
            return -_registration_score(Dd_f @ Mp @ Ds_f_inv, s_f, sm_f, d_f, dm_f)
        step = np.array([0.03, 1.0, 2.0 * f, 2.0 * f])
        res = minimize(cost, np.zeros(4), method='Nelder-Mead',
                       options=dict(initial_simplex=np.vstack([np.zeros(4), np.diag(step)]),
                                    xatol=1e-3, fatol=1e-5, maxiter=300))
        p = res.x
        return (similarity_matrix(s0 * np.exp(p[0]), a0 + p[1], tx0 + p[2], ty0 + p[3]),
                -res.fun)

    results = []
    for p0 in starts:
        M = similarity_matrix(*p0)
        score = -1.
        for level in pyramid[:-1]:
            M, score = refine(M, level)
        results.append([score, M])
        if verbose:
            print('refined (coarse levels)', round(score, 3), np.round(decompose_similarity(M), 3))
    results.sort(key=lambda r: -r[0])
    M, _ = refine(results[0][1], pyramid[-1])
    ncc = _registration_score(M, src_p, src_m, dst_p, dst_m)
    results[0] = [ncc, M]
    if verbose:
        print('final', round(ncc, 3), np.round(decompose_similarity(M), 3))
    s, a, tx, ty = decompose_similarity(M)
    info = dict(ncc=ncc, scale=s, rotation=a, translation=(tx, ty),
                candidates=[dict(ncc=r[0], params=decompose_similarity(r[1])) for r in results])
    return M, info


def svd_variance(U, SVT, block=65536):
    '''Pixelwise variance over time of ``U @ SVT``, from the (k, k) covariance
    of SVT, without rebuilding the frames.

    Parameters
    ----------
    U : (height, width, k) array
    SVT : (k, frames) array

    Returns
    -------
    var : (height, width) float32 (np.var over frames, ddof=0)
    '''
    S = np.asarray(SVT, dtype=np.float64)
    S = S - S.mean(axis=1, keepdims=True)
    cov = (S @ S.T / S.shape[1]).astype(np.float32)
    del S
    flat = np.asarray(U, dtype=np.float32).reshape(-1, cov.shape[0])
    var = np.empty(flat.shape[0], dtype=np.float32)
    for a in range(0, flat.shape[0], block):
        u = flat[a:a + block]
        var[a:a + block] = np.einsum('ij,ij->i', u @ cov, u)
    return var.reshape(np.shape(U)[:-1])


HEMO_BANDS = (1.0, 4.0, 8.0)   # Hz, edges of the bands of cids' hemodynamic correction


def cids_widefield(movie, frame_rate, output, functional=0, k=200, register=True,
                   decomposition='svd', recompress_tolerance=1e-3,
                   exclude_mask=None, mask_threshold=None,
                   highpass=0.1, lowpass=14.0, hemo_bands=HEMO_BANDS, hemo_delay='auto',
                   chunk_frames=512, device='auto', verbose=True):
    '''Run ``cids.widefield.run_widefield`` and return its results in the
    layout WfieldStack stores for wfield.

    Parameters
    ----------
    movie : (frames, channels, height, width) array-like (e.g. a zarr array)
    frame_rate : frames per second of each channel
    output : path of the cids results group (a .zarr folder, e.g. in scratch)
    functional : channel of the indicator; with more than one channel, the
        next one ((functional + 1) % channels) is the isosbestic one used for
        the hemodynamic correction
    k : number of SVD components (decomposition 'svd')
    register : rigid registration with rotation (False: the frames are used
        as they are)
    decomposition : 'svd' (k global components) or 'local' (cids' local
        low-rank denoising, as PMD). The local components are then
        re-compressed (``cids.widefield.recompress``) over the whole frame,
        the channels and the corrected components together, keeping all but
        ``recompress_tolerance`` of each pixel's energy, so that SVT and
        SVTcorr share one dense U.
    exclude_mask : (height, width) array, pixels != 0 are left out, or None
    mask_threshold : threshold of cids' ``activity_mask`` (None: no activity
        mask). With ``exclude_mask`` too, the activity mask is computed on the
        frames before registration and combined with it.
    highpass, lowpass, hemo_bands, hemo_delay : cids' hemodynamic correction:
        a regression per pixel in each frequency band (split at ``hemo_bands``
        Hz; edges too close to Nyquist are dropped), after delaying the
        isosbestic channel (``'auto'``: ``channel_delay``); both channels
        high-passed at ``highpass`` Hz, the isosbestic low-passed at
        ``lowpass`` Hz
    chunk_frames : frames read at a time
    device : 'auto' (an NVIDIA card with cupy, else an Apple GPU with MLX, else
        the CPU), 'cpu', 'cuda', 'cuda:N' or 'metal'

    Returns
    -------
    res : dict of arrays for the .npz
        U (height, width, k); SVT (k, frames * channels), channels
        interleaved as in wfield (``SVT[:, c::channels]`` is channel c);
        SVTcorr (k, frames), with more than one channel; motion (3, channels,
        frames): x and y shift of the frame centre and rotation (degrees,
        ``cids.registration.field_rotation``), the same for every channel;
        mean_proj (channels, height, width) uint16, the baseline F0. Then
        every other array cids saved (shifts, baseline, mask, channel_gain,
        hemo_coefficients, singular_values, qc_*, ...; sparse ones are left
        out) and its attributes as JSON in ``cids_attrs``.
    explained_variance : median over the mask of the R2 of the raw dF/F of
        the functional channel by the SVD or the local denoising (cids'
        ``qc_compression_r2``), or None
    '''
    import json
    from cids import io as cids_io
    from cids.widefield import run_widefield, activity_mask
    if decomposition not in ('svd', 'local'):
        raise ValueError(f"decomposition is 'svd' or 'local', not {decomposition!r}")
    nchannels, height, width = (int(n) for n in movie.shape[1:])
    isosbestic = (functional + 1) % nchannels if nchannels > 1 else None
    if exclude_mask is None and mask_threshold is None:
        mask = None                    # every pixel, as wfield without a mask
    elif exclude_mask is None:
        mask = 'auto'
    else:
        mask = np.asarray(exclude_mask) == 0
        if mask_threshold is not None:
            mask = mask & activity_mask(movie, float(mask_threshold), functional)[0]
    if hemo_bands is not None:         # each band edge needs its crossing (to 1.3 x the edge) below Nyquist
        hemo_bands = tuple(float(e) for e in hemo_bands if 1.3 * e < frame_rate / 2)
    out = run_widefield(movie, float(frame_rate), output=str(output), functional=functional,
                        isosbestic=isosbestic, k=int(k), register=register,
                        highpass=highpass, lowpass=lowpass, hemo_bands=hemo_bands or None,
                        hemo_delay=hemo_delay, mask=mask,
                        mask_threshold=5.0 if mask_threshold is None else float(mask_threshold),
                        decomposition=decomposition, chunk_frames=int(chunk_frames), device=device,
                        verbose=verbose)
    group = cids_io.open_results(out, mode='r')
    attrs = dict(group.attrs)
    arrays = {name: np.asarray(group[name][...]) for name in group.array_keys()}
    svt = arrays.pop('svt')                                      # (channels, K, frames)
    corrected = arrays.pop('svt_corrected', None)                # (K, frames)
    C, _, T = svt.shape
    if decomposition == 'local':
        # U is sparse (pixels, K), K in the thousands: re-compress the channels and the
        # corrected components together, onto one dense U
        from cids.widefield import widefield_components, recompress
        U_local, _ = widefield_components(out)
        joint = svt if corrected is None else np.concatenate([svt, corrected[None]])
        del svt
        U, joint, values = recompress(U_local, joint, recompress_tolerance,
                                      mask=arrays.get('mask'), shape=(height, width))
        del U_local
        svt, corrected = joint[:C], (None if corrected is None else joint[C])
        arrays['singular_values'] = values
        attrs['recompress_tolerance'] = float(recompress_tolerance)
        if verbose:
            print(f'recompression: {len(values)} components over the frame')
    else:
        U = arrays.pop('U')
    nk = U.shape[-1]
    res = dict(U=np.asarray(U, dtype=np.float32),
               SVT=np.ascontiguousarray(svt.transpose(1, 2, 0).reshape(nk, T * C)))
    del svt
    if corrected is not None:
        res['SVTcorr'] = np.asarray(corrected, dtype=np.float32)
    if 'shifts' in arrays:                                       # (frames, ny, nx, 2): (dy, dx)
        from cids.registration import field_rotation
        shifts = arrays['shifts']
        dy, dx = shifts.reshape(T, -1, 2).mean(axis=1).T         # the shift of the frame centre
        angle = field_rotation(shifts, arrays['centers_y'], arrays['centers_x'])
    else:
        dy = dx = angle = np.zeros(T)
    motion = np.stack([dx, dy, angle]).astype('float32')          # (3, frames)
    res['motion'] = np.ascontiguousarray(np.repeat(motion[:, None, :], C, axis=1))
    res['mean_proj'] = np.clip(arrays['baseline'], 0, np.iinfo(np.uint16).max).astype('uint16')
    res.update({name: value for name, value in arrays.items() if name not in res})
    res['cids_attrs'] = np.array(json.dumps(attrs, default=str))
    explained_variance = None
    r2 = arrays.get('qc_compression_r2')
    if r2 is not None:
        inside = arrays.get('mask')
        if inside is None or inside.shape != r2.shape:
            inside = np.ones(r2.shape, dtype=bool)
        explained_variance = float(np.nanmedian(r2[inside.astype(bool)]))
    return res, explained_variance
