import numpy as np

__all__ = [
    'build_alignment_transform',
    'build_atlas_transform',
    'warp_image',
    'transform_coordinates',
    'inverse_transform_coordinates',
    'transform_atlas_regions',
]


def build_alignment_transform(fw, fh, rotation, scale, ratio, origin_x, origin_y):
    """Build a 3×3 forward affine matrix mapping 2P (col,row) → reference (col,row).

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


def build_atlas_transform(bregma_xy, resolution, rotation=0.0, scale=1.0, ratio=1.0):
    """Build a 3×3 forward matrix mapping atlas mm coordinates to widefield pixel coordinates.

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

    Returns
    -------
    M_fwd : ndarray, shape (3, 3)
    """
    bregma_col, bregma_row = float(bregma_xy[0]), float(bregma_xy[1])
    theta = np.deg2rad(float(rotation))
    cs, sn = np.cos(theta), np.sin(theta)
    sx = float(scale) * float(ratio) / float(resolution)
    sy = float(scale) / float(resolution)
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


