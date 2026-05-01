import numpy as np

__all__ = [
    'build_alignment_transform',
    'warp_image',
    'transform_coordinates',
    'inverse_transform_coordinates',
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


