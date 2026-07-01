from labdata.schema import *

userschema = get_user_schema()

__all__ = ['WfieldParameters', 'WfieldStack', 'ImagingWindow',
           'ImagingReference','TwoPhotonReferenceAlignment',
           'WidefieldAtlas', 'WidefieldAtlasTransform',
           'WidefieldResponse', 'CellSegmentationAtlas']

userschema = get_user_schema()

@userschema
class WfieldParameters(dj.Manual):
    definition = '''
    -> Widefield
    wfield_analysis_id : int
    ---
    motion_correction = 'ecc'    : enum('2d','ecc','normcorr','none')    # type of motion correction
    motion_conv_kernel = NULL    : blob                                  # convolution kernel for motion correction
    decomposition = 'approx'     : enum('approx','pmd')                  # type of decomposition algorithm
    k = 200                      : int                                   # number of components for decomposition
    atlas = 'dorsal_cortex'      : varchar(24)                           # atlas to reference to
    exclude_mask = NULL          : longblob                              # manual mask to exclude before svd
    functional_channel = 0       : smallint                              # functional channel
    nframes_decimate = 15        : smallint                              # number of frames to downsample (SVD)
    chunk_size = 512             : smallint                              # size of chunks for processing
    mask_std_threshold = NULL    : int                                   # threshold for functional mask
    -> [nullable] Session.proj(match_to_session = 'session_name')        # session to match to
    '''
    def create_exclude_mask(self,widefieldkey):
        '''
        Allows selecting an exclude mask
        '''
        return
        
    
@userschema
class WfieldStack(dj.Computed):
    definition = '''
    -> WfieldParameters
    ---
    motion_correction = NULL   : longblob                  # XY motion correction vector
    mean_proj = NULL           : longblob                  # mean projection used for the SVD
    explained_variance = NULL  : float
    -> AnalysisFile
    '''
    class Projection(dj.Part):
        definition = '''
        -> master
        proj_name : varchar(64)    # name of the projection (e.g. mean, std, var..)
        ---
        proj : longblob            # projection image
        '''

    def make(self, key):
        from wfield import (motion_correct, get_std_mask,
                            chunk_indices, approximate_svd,
                            hemodynamic_correction, SVDStack)
        import string
        par = (WfieldParameters() & key).fetch1()
        
        try:
            dat = (Widefield & key).open()
        except ValueError:
            # try tif files
            files = (File & (Dataset.DataFiles & (Dataset & (Widefield & key)) & 'file_path LIKE "%.tif"' & 'file_path NOT LIKE "%ref%"')).get()
            if len(files):
                from wfield import TiffStack
                dat = TiffStack(natsorted([str(f) for f in files]),nchannels = (Widefield & key).fetch1('n_channels'))
        # work in the scratch folder
        rand = ''.join(np.random.choice([s for s in string.ascii_lowercase + string.digits],9))
        scratch_folder = Path(prefs['scratch_path'])/f'wfield_{rand}'
        scratch_folder.mkdir(exist_ok = True)
        motion = scratch_folder/'temporary.motion.bin'
        out = np.memmap(motion,mode = 'w+',dtype = np.uint16, shape = dat.shape)
        (yshifts,xshifts),rshifts = motion_correct(dat,
                                                   out=out,
                                                   mode = par['motion_correction'],
                                                   chunksize = par['chunk_size'],
                                                   diff_gaussians_filter=par['motion_conv_kernel'],
                                                   apply_shifts = True)
        from tqdm.auto import tqdm
        chunkidx = chunk_indices(len(out),chunksize=par['chunk_size'])
        frame_averages = []
        for on,off in tqdm(chunkidx,desc = 'Computing the average of the frames'):
            frame_averages.append(out[on:off].mean(axis=0))
        frames_average = np.stack(frame_averages).mean(axis = 0)
        functional_channel = par['functional_channel']
        mask = np.ones(out.shape[-2::],dtype=bool)
        if not par['exclude_mask'] is None:
            mask = par['exclude_mask'].copy() == 0
        if not par['mask_std_threshold'] is None: 
            mask = mask & get_std_mask(out[:,par['functional_channel']],
                            threshold = par['mask_std_threshold']).astype(bool)
        U,SVT = approximate_svd(out, frames_average, 
                                onsets = None, 
                                mask = mask, k = par['k'],
                                nframes_per_bin = par['nframes_decimate'])
        fs = (Widefield & key).fetch1('frame_rate')
        freq_lowpass = fs/2 - 2
        freq_highpass = 0.1
        nchannels = (Widefield & key).fetch1('n_channels')
        wfield_ins = dict(motion = np.stack([xshifts,yshifts,rshifts]).transpose(0,2,1).astype('float32'),
                          mean_proj = frames_average.astype('uint16'),
                          SVT = SVT,
                          U = U)
        if nchannels>1:
            SVTcorr,rcoeffs,T = hemodynamic_correction(U,
                                                       fs = fs,
                                                       freq_highpass = freq_highpass,
                                                       freq_lowpass = freq_lowpass,
                                                       nchunks = par['chunk_size'],
                                                       SVT_470=SVT[:,par['functional_channel']::nchannels],
                                                       SVT_405 = SVT[:,par['functional_channel']+1::nchannels])
            wfield_ins['SVTcorr'] = SVTcorr
            wfield_ins['T'] = T
            wfield_ins['rcoeffs'] = rcoeffs
            SVT = SVTcorr # use the corrected SVT
        # compute pixelwise projections from the SVD components
        svd_stack = SVDStack(U, SVT)
        projections = [
            ('var',  svd_stack.var()),
        ]
        # store the results
        dataset = dict(**key)
        dataset['dataset_name'] = f'{key["dataset_name"]}_wfield_{par["wfield_analysis_id"]:02d}'
        from labdata.schema import AnalysisFile 
        resultsfolder = AnalysisFile().generate_filepaths([''],dataset)
        resultsfolder = Path(prefs['local_paths'][0])/resultsfolder[0]
        resultsfolder.mkdir(exist_ok=True,parents=True)
        resultspath = resultsfolder/'wfield_res.npz'
        np.savez(resultspath,**wfield_ins)
        # upload to aws
        filekeys = AnalysisFile().upload_files([resultspath],dataset)
        # store on database
        key = dict(key,
                   **filekeys[0],
                   motion_correction = np.stack([xshifts,yshifts,rshifts]).transpose(0,2,1).astype('float32'),
                   mean_proj = frames_average.astype('uint16').squeeze())
        # insert results
        self.insert1(key)
        # insert projections
        self.Projection.insert([dict(key, proj_name=name, proj=proj)
                                 for name, proj in projections],ignore_extra_fields=True)
        # remove the temporary data
        from shutil import rmtree
        rmtree(scratch_folder)

    def open(self, key=None, use_corrected=True):
        from wfield import SVDStack
        if key is None:
            key = self.fetch1('KEY')
        par = (WfieldParameters & key).fetch1()
        nchannels = (Widefield & key).fetch1('n_channels')
        fs = (Widefield & key).fetch1('frame_rate')
        res = self.load(key)
        if use_corrected and 'SVTcorr' in res:
            SVT = res['SVTcorr']
        else:
            SVT = res['SVT'][:, par['functional_channel']::nchannels]
        stack = SVDStack(res['U'], SVT)
        stack.fs = fs
        stack.mean_proj = res['mean_proj']
        if 'motion' in res:
            stack.motion = res['motion']
        proj_rows = (self.Projection & key).fetch(as_dict=True)
        stack.projections = {r['proj_name']: r['proj'] for r in proj_rows}
        return stack

    def load(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        filepath = (AnalysisFile & (self & key)).get()[0]
        return np.load(filepath, allow_pickle=True)

    def save_projection(self, proj, proj_name, key=None):
        if key is None:
            key = self.fetch1('KEY')
        self.Projection.insert1(dict(key, proj_name=proj_name, proj=np.asarray(proj)),
                                replace=True)

    def delete(self, transaction=True, safemode=None, force_parts=False):
        file_paths = list(self.fetch('file_path'))
        super().delete(transaction=transaction, safemode=safemode, force_parts=force_parts)
        if len(self) == 0 and file_paths:
            (AnalysisFile() & [f'file_path = "{f}"' for f in file_paths]).delete(
                force_parts=force_parts, safemode=safemode)

@userschema
class ImagingWindow(dj.Manual):
    definition = """
    -> Widefield
    ---
    window_size                  : float         # size of the window (mm)
    resolution = NULL            : float         # resolution  (mm/pixel)
    points = NULL                : blob          # points sampled manually around the circle
    circle_parameters = NULL     : blob          # circle parameters for the window
    """

    def apply_window_mask(self, image=None):
        """Set everything outside the imaging-window circle to NaN.

        Builds the circular mask from this window's `circle_parameters`
        ([col, row, radius] in widefield pixels) and fills the area outside the
        circle with NaN. The spatial (H, W) axes are auto-detected:

        - ``H x W``                  — a single image
        - ``H x W x C`` (C in {3,4}) — a colour image (mask broadcast over channels)
        - ``N x H x W``              — a movie / channel-first stack (masked per frame)
        - ``... x H x W``            — higher-dim arrays (last two axes are spatial)

        Parameters
        ----------
        image : ndarray, optional
            Image or movie to mask, in this window's pixel space. When omitted,
            the mean projection of this session's WfieldStack (lowest
            `wfield_analysis_id`) is used.

        Returns
        -------
        masked : ndarray (float)
            A copy of the input with pixels outside the window set to NaN.
        """
        win = self.fetch1()
        cp = win.get('circle_parameters')
        if cp is None:
            raise ValueError('This ImagingWindow has no circle_parameters.')
        xc, yc, radius = np.asarray(cp, dtype=float).ravel()[:3]

        if image is None:
            key = dict(subject_name=win['subject_name'],
                       session_name=win['session_name'],
                       dataset_name=win['dataset_name'])
            aids = (WfieldStack & key).fetch('wfield_analysis_id')
            if not len(aids):
                raise ValueError('No image passed and no WfieldStack mean projection '
                                 'available for this session — pass an image explicitly.')
            mp = (WfieldStack & dict(key, wfield_analysis_id=int(min(aids)))).fetch1('mean_proj')
            mp = np.squeeze(np.asarray(mp)).astype(float)
            image = mp[0] if mp.ndim == 3 else mp   # mean_proj is channel-first
        image = np.asarray(image, dtype=float)

        def _circle(h, w):
            rr, cc = np.ogrid[:h, :w]
            return (cc - xc) ** 2 + (rr - yc) ** 2 <= radius ** 2

        masked = image.copy()
        nd = image.ndim
        if nd == 2:                                    # H x W
            masked[~_circle(*image.shape)] = np.nan
        elif nd == 3 and image.shape[-1] in (3, 4):    # H x W x C (colour)
            masked[~_circle(*image.shape[:2])] = np.nan
        elif nd == 3:                                  # N x H x W (movie / channel-first)
            masked[:, ~_circle(*image.shape[1:])] = np.nan
        else:                                          # ... x H x W
            masked[..., ~_circle(*image.shape[-2:])] = np.nan
        return masked


@userschema
class ImagingReference(dj.Manual):
    definition = """
    -> Subject
    ref_num                      : smallint
    ---
    -> Widefield.proj(ref_session = "session_name",ref_dataset = "dataset_name")
    ref_image                    : longblob                                        # the reference image (uint16)
    """

    def overlay_projections_on_reference(self, ax=None, proj_name='mean',
                                          cell_seg_params=None, alpha=0.5,
                                          fov_color='yellow'):
        """Overlay CellSegmentation projections from all aligned 2P datasets on the reference image.

        For every TwoPhotonReferenceAlignment linked to this entry the method
        locates the matching CellSegmentation entries, warps the requested
        projection into reference-image space, and draws a rectangle marking
        the 2P field of view.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; a new figure is created when None.
        proj_name : str
            CellSegmentation projection to overlay (e.g. 'mean', 'max', 'correlation').
        cell_seg_params : int or list of int, optional
            Restrict to specific CellSegmentationParams parameter_set_num(s).
            All available segmentations are shown when None.
        alpha : float
            Opacity for the projection overlay. Default 0.5.
        fov_color : color
            Matplotlib color for the FOV boundary rectangle.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        from labdata.schema import CellSegmentation
        from .utils import transform_coordinates, warp_image

        ref_img = np.squeeze(np.asarray(self.fetch1('ref_image'))).astype(np.float32)
        rh, rw = ref_img.shape[:2]

        alignments = (TwoPhotonReferenceAlignment & self).fetch(as_dict=True)
        if not alignments:
            raise ValueError('No TwoPhotonReferenceAlignment entries for this ImagingReference.')

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6 * rh / rw))

        lo_r, hi_r = np.percentile(ref_img, [1, 99])
        ax.imshow(ref_img, cmap='gray', origin='upper', aspect='equal',
                  vmin=lo_r, vmax=hi_r)

        for align in alignments:
            subject_name = align['subject_name']
            session_name = align['session_name']
            dataset_name = align['dataset_name']

            fov_offset = align.get('fov_offset')
            if fov_offset is not None:
                fov_offset = np.asarray(fov_offset).ravel()
                row_off, col_off = int(fov_offset[0]), int(fov_offset[1])
            else:
                row_off = col_off = 0

            cs_base = dict(subject_name=subject_name, session_name=session_name,
                           dataset_name=dataset_name)
            if cell_seg_params is not None:
                params = ([cell_seg_params] if isinstance(cell_seg_params, (int, np.integer))
                          else list(cell_seg_params))
                cs_restrict = [dict(**cs_base, parameter_set_num=p) for p in params]
            else:
                cs_restrict = cs_base

            param_nums = list((CellSegmentation & cs_restrict).fetch('parameter_set_num'))
            if not param_nums:
                continue

            align_key = dict(subject_name=align['subject_name'],
                             ref_num=align['ref_num'],
                             session_name=align['session_name'],
                             dataset_name=align['dataset_name'])

            for param_num in param_nums:
                plane_key_base = dict(**cs_base, parameter_set_num=param_num)
                plane_rows = (CellSegmentation.Plane & plane_key_base).fetch(
                    'plane_num', 'dims', as_dict=True)

                for plane_row in plane_rows:
                    plane_num = plane_row['plane_num']
                    plane_key = dict(**plane_key_base, plane_num=plane_num)

                    dims = plane_row.get('dims')
                    if dims is not None:
                        dims = np.asarray(dims).ravel()
                        fh_seg, fw_seg = int(dims[0]), int(dims[1])
                    else:
                        proj_fallback = (CellSegmentation.Projection & plane_key).fetch(
                            'proj_im', limit=1)
                        if not len(proj_fallback):
                            continue
                        fh_seg, fw_seg = np.squeeze(np.asarray(proj_fallback[0])).shape[:2]

                    fw_raw = fw_seg + col_off
                    fh_raw = fh_seg + row_off

                    M_fwd, transpose, _ = (TwoPhotonReferenceAlignment & align_key).get_transform(
                        fw_raw, fh_raw)

                    # Warp projection into reference space
                    proj_data = (CellSegmentation.Projection
                                 & dict(**plane_key, proj_name=proj_name)).fetch('proj_im')
                    if len(proj_data):
                        proj_im = np.squeeze(np.asarray(proj_data[0])).astype(np.float32)
                        if row_off or col_off:
                            padded = np.zeros((fh_raw, fw_raw), dtype=np.float32)
                            padded[row_off:row_off + fh_seg,
                                   col_off:col_off + fw_seg] = proj_im
                            proj_im = padded
                        if transpose:
                            proj_im = proj_im.T
                        warped = warp_image(proj_im, M_fwd, (rh, rw))
                        fov_mask = warped > 0
                        if fov_mask.any():
                            lo_p, hi_p = np.percentile(warped[fov_mask], [2, 98])
                            proj_norm = np.clip(
                                (warped - lo_p) / max(hi_p - lo_p, 1e-9), 0, 1)
                            rgba = plt.get_cmap('gray')(proj_norm)
                            rgba[..., 3] = fov_mask.astype(float) * alpha
                            ax.imshow(rgba, origin='upper', aspect='equal')

                    # FOV boundary rectangle
                    afw = fh_raw if transpose else fw_raw
                    afh = fw_raw if transpose else fh_raw
                    corners = np.array([[0, 0], [afw, 0], [afw, afh],
                                        [0, afh], [0, 0]], dtype=float)
                    corners_ref = transform_coordinates(corners, M_fwd)
                    ax.plot(corners_ref[:, 0], corners_ref[:, 1],
                            '-', color=fov_color, lw=1.5, alpha=0.8)

        ax.set_xlim(0, rw)
        ax.set_ylim(rh, 0)
        ax.axis('off')
        return ax


@userschema
class TwoPhotonReferenceAlignment(dj.Manual):
    definition = """
    # parameters to match a TwoPhoton to a reference image
    -> ImagingReference
    -> TwoPhoton
    ---
    rotation            : float   # rotation applied to the reference
    scale               : float   # how to scale the reference
    transpose           : bool    # how to transpose the reference
    ratio               : float   # aspect ratio
    origin              : blob    # origin
    fov_offset = NULL   : blob    # in case the FOV is offset (dropped columns on segmentation)
    """

    def get_transform(self, fw, fh):
        """Return (M_fwd, transpose, fov_offset) for this alignment entry.

        Parameters
        ----------
        fw, fh : float
            Width and height of the raw 2P image (before transpose).

        Returns
        -------
        M_fwd : ndarray (3, 3)
        transpose : bool
        fov_offset : ndarray (2,) or None  — (row_offset, col_offset)
        """
        from .utils import build_alignment_transform
        row = self.fetch1()
        transpose = bool(row['transpose'])
        afw, afh = (fh, fw) if transpose else (fw, fh)
        origin = np.asarray(row['origin']).ravel()
        M_fwd = build_alignment_transform(
            afw, afh,
            rotation=float(row['rotation']),
            scale=float(row['scale']),
            ratio=float(row['ratio']),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
        )
        fov_offset = row.get('fov_offset')
        if fov_offset is not None:
            fov_offset = np.asarray(fov_offset).ravel()
        return M_fwd, transpose, fov_offset

    def apply_transform(self, image, output_shape=None):
        """Warp a 2P image into the reference-image pixel space using this alignment.

        Applies the stored FOV offset (padding), transpose, and affine transform
        so the result lands in the ImagingReference image coordinates. Pixels
        outside the warped 2P field of view are 0.

        Parameters
        ----------
        image : ndarray
            2P image (H x W) or movie (N x H x W) in the 2P / segmentation pixel
            space (i.e. the cropped FOV the alignment was defined against).
        output_shape : (H, W), optional
            Shape of the reference space. Defaults to this alignment's
            ImagingReference image shape.

        Returns
        -------
        warped : ndarray (float)
            The image in reference-image coordinates; an (N, H, W) movie warps
            frame by frame.
        """
        from .utils import warp_image
        row = self.fetch1()
        fov = row.get('fov_offset')
        row_off, col_off = (int(fov[0]), int(fov[1])) if fov is not None else (0, 0)

        if output_shape is None:
            ref_img = (ImagingReference & self).fetch1('ref_image')
            output_shape = np.squeeze(np.asarray(ref_img)).shape[:2]
        rh, rw = int(output_shape[0]), int(output_shape[1])

        image = np.asarray(image, dtype=float)
        if image.ndim not in (2, 3):
            raise ValueError('image must be 2-D (H x W) or 3-D (N x H x W).')
        is_movie = image.ndim == 3
        frames = image if is_movie else image[None]

        fh_img, fw_img = frames.shape[1:]
        fh_raw, fw_raw = fh_img + row_off, fw_img + col_off
        M_fwd, transpose, _ = self.get_transform(fw_raw, fh_raw)

        out = []
        for frame in frames:
            if row_off or col_off:
                padded = np.zeros((fh_raw, fw_raw), dtype=float)
                padded[row_off:row_off + fh_img, col_off:col_off + fw_img] = frame
                frame = padded
            if transpose:
                frame = frame.T
            out.append(warp_image(frame, M_fwd, (rh, rw)))
        out = np.stack(out)
        return out if is_movie else out[0]

    def _fov_dims(self):
        """Raw 2P frame ``(width, height)`` for this (single-row) alignment.

        Matches the image the alignment was built against: the CellSegmentation
        plane dims plus the FOV offset (same as ``overlay_projections_on_reference``),
        falling back to the TwoPhoton frame size.
        """
        row = self.fetch1()
        fov = row.get('fov_offset')
        row_off, col_off = (int(fov[0]), int(fov[1])) if fov is not None else (0, 0)
        try:
            from labdata.schema import CellSegmentation
            dims = (CellSegmentation.Plane & dict(
                subject_name=row['subject_name'], session_name=row['session_name'],
                dataset_name=row['dataset_name'])).fetch('dims', limit=1)
            if len(dims) and dims[0] is not None:
                d = np.asarray(dims[0]).ravel()        # (height, width)
                return int(d[1]) + col_off, int(d[0]) + row_off
        except Exception:
            pass
        from labdata.schema import TwoPhoton
        w, h = (TwoPhoton & self).fetch1('width', 'height')
        return int(w), int(h)

    def _resolve_atlas_transform(self, atlas_name=None, atlas_transform_id=None):
        """Return the WidefieldAtlasTransform for an alignment's reference widefield
        session. Warns and uses the latest (highest ``atlas_transform_id``) when
        several match — matching the dashboard, which defaults to the most recent.
        """
        import warnings
        key = self.fetch1('KEY')
        ref_session, ref_dataset = (ImagingReference & dict(
            subject_name=key['subject_name'], ref_num=key['ref_num'])).fetch1(
            'ref_session', 'ref_dataset')
        restr = dict(subject_name=key['subject_name'],
                     session_name=ref_session, dataset_name=ref_dataset)
        if atlas_name is not None:
            restr['atlas_name'] = atlas_name
        if atlas_transform_id is not None:
            restr['atlas_transform_id'] = atlas_transform_id
        xf = WidefieldAtlasTransform & restr
        if len(xf) == 0:
            raise ValueError('No WidefieldAtlasTransform for widefield session '
                             f'{key["subject_name"]}/{ref_session}/{ref_dataset}.')
        if len(xf) > 1:
            chosen = int(max(xf.fetch('atlas_transform_id')))
            warnings.warn(f'{len(xf)} WidefieldAtlasTransforms for '
                          f'{key["subject_name"]}/{ref_session}/{ref_dataset}; using the latest '
                          f'(atlas_transform_id={chosen}). Pass atlas_transform_id=... to pick one.')
            xf = xf & dict(atlas_transform_id=chosen)
        return xf

    def points_to_atlas(self, xy, atlas_transform=None, fov_dims=None,
                        atlas_name=None, atlas_transform_id=None):
        """Map 2P pixel points to atlas mm coordinates (bregma origin).

        Composes this alignment (2P → reference pixels) with the inverse of a
        `WidefieldAtlasTransform` (atlas mm → reference pixels), so points go
        2P (col, row) → reference px → atlas mm.

        Parameters
        ----------
        xy : array-like, shape (N, 2) or (2,)
            Points as ``(col, row)`` in the **raw 2P frame** pixel space.
        atlas_transform : WidefieldAtlasTransform query / restriction, optional
            The atlas registration to use. When omitted it is resolved
            automatically from this alignment's reference widefield session
            (warning and using the first if several match).
        fov_dims : (width, height), optional
            Raw 2P frame size in pixels. Defaults to the size the alignment was
            built against (CellSegmentation plane dims + fov_offset, else TwoPhoton).
        atlas_name, atlas_transform_id : optional
            Narrow the auto-resolved transform when several exist.

        Returns
        -------
        xy_mm : ndarray, shape (N, 2)
            ``(x_mm, y_mm)`` atlas coordinates (mm from bregma).
        """
        from .utils import transform_coordinates
        if atlas_transform is None:
            atlas_transform = self._resolve_atlas_transform(atlas_name, atlas_transform_id)
        elif isinstance(atlas_transform, (dict, str, list, tuple)):
            atlas_transform = WidefieldAtlasTransform & atlas_transform

        if fov_dims is None:
            fw, fh = self._fov_dims()
        else:
            fw, fh = int(fov_dims[0]), int(fov_dims[1])

        M_fwd, transpose, _ = self.get_transform(fw, fh)
        M_atlas_inv = np.linalg.inv(np.asarray(atlas_transform.get_transform()))

        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        pts = xy[:, ::-1] if transpose else xy        # (col,row) -> (row,col) if transposed
        ref_px = transform_coordinates(pts, M_fwd)     # reference pixels
        return transform_coordinates(ref_px, M_atlas_inv)   # atlas mm

    def plot_fov_on_atlas(self, ax=None, atlas_transform=None, atlas_name=None,
                          atlas_transform_id=None, fov_dims=None,
                          cmap = 'tab10',
                          color_regions = 'k', lw_regions = 1, **kwargs):
        """Plot the imaged 2P field-of-view outline(s) in atlas (mm) coordinates.

        Iterates over every row in the query, drawing each dataset's raw 2P frame
        as a closed quadrilateral mapped to atlas mm via `points_to_atlas`. The
        atlas transform for each row is found automatically from its reference
        widefield session unless one is passed explicitly.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        atlas_transform : WidefieldAtlasTransform query / restriction, optional
            Use this transform for every row instead of auto-resolving per row.
        atlas_name, atlas_transform_id : optional
            Narrow the auto-resolved transform when several exist for a session.
        fov_dims : (width, height), optional
            Raw 2P frame size in pixels; defaults per row to the size the alignment
            was built against (CellSegmentation plane dims + fov_offset, else TwoPhoton).
        cmap : str
            Matplotlib colormap name used to colour the per-row FOV outlines.
        color_regions : color or None
            Colour of the atlas region contours; set to None to skip drawing them.
        lw_regions : float
            Line width of the atlas region contours.
        **kwargs
            Forwarded to ax.plot() for the FOV outlines (e.g. lw, linestyle).

        Returns
        -------
        ax : matplotlib Axes
        """
        import warnings
        import matplotlib.pyplot as plt
        import pandas as pd

        keys = list(self.fetch('KEY'))
        if not keys:
            raise ValueError('No TwoPhotonReferenceAlignment rows in the query.')

        if ax is None:
            ax = plt.gca()
        cmap = plt.get_cmap(cmap, max(len(keys), 1))

        first_xf = None
        plotted = 0
        for i, key in enumerate(keys):
            one = self & key
            if atlas_transform is not None:
                xf = (WidefieldAtlasTransform & atlas_transform
                      if isinstance(atlas_transform, (dict, str, list, tuple))
                      else atlas_transform)
            else:
                try:
                    xf = one._resolve_atlas_transform(atlas_name, atlas_transform_id)
                except ValueError as exc:
                    warnings.warn(str(exc))
                    continue
            if first_xf is None:
                first_xf = xf

            if fov_dims is None:
                fw, fh = one._fov_dims()
            else:
                fw, fh = int(fov_dims[0]), int(fov_dims[1])
            corners = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh], [0, 0]], dtype=float)
            corners_mm = one.points_to_atlas(corners, atlas_transform=xf, fov_dims=(fw, fh))

            kw = dict(lw=1.5, color=cmap(i)) # default parameters for the lw and color
            kw.update(kwargs)
            ax.plot(corners_mm[:, 0], corners_mm[:, 1],
                    label=f"{key['session_name']}/{key['dataset_name']}", **kw)
            plotted += 1

        if plotted == 0:
            raise ValueError('No FOVs plotted — no matching WidefieldAtlasTransform found.')

        if first_xf is not None and not color_regions is None:
            ccf_regions, _, _ = first_xf.load_reference()
            regions = pd.DataFrame(ccf_regions)
            for _, reg in regions.iterrows():
                for side in ('left', 'right'):
                    ax.plot(np.asarray(reg[f'{side}_x'], dtype=float),
                            np.asarray(reg[f'{side}_y'], dtype=float),
                            '-', color=color_regions, lw=lw_regions, zorder=0)

        ax.set_aspect('equal')
        return ax


@userschema
class WidefieldAtlas(dj.Manual):
    definition = '''
    atlas_name      : varchar(24)   # atlas reference name, e.g. 'dorsal_cortex'
    ---
    ccf_regions     : longblob      # region contours as dict (mm from bregma)
    projection      : longblob      # 2D ndarray, flattened atlas projection
    brain_outline   : longblob      # 2D ndarray, brain outline contour
    reference_point = NULL : blob   # [row, col] of bregma in the atlas projection image
    resolution = NULL      : float  # mm per atlas projection pixel
    '''

    def from_wfield(self, atlas_name):
        """Load and store atlas from local wfield reference files (~/.wfield/)."""
        from wfield import allen_load_reference
        ccf_regions, proj, brain_outline = allen_load_reference(atlas_name)
        first = ccf_regions.iloc[0]
        self.insert1(dict(atlas_name=atlas_name,
                          ccf_regions=ccf_regions.to_dict(orient='list'),
                          projection=proj,
                          brain_outline=brain_outline,
                          reference_point=list(first['reference']),
                          resolution=float(first['resolution']) / 1000.0))

    def from_allensdk(self, atlas_name, structures=None, resolution=10, reference=None):
        """Build and store the atlas directly from the Allen CCF via allensdk.

        Parameters
        ----------
        atlas_name : str
            Key to store this atlas under.
        structures : list of str, optional
            Region acronyms to include. Defaults to selection_dorsal_cortex.
        resolution : int
            CCF voxel resolution in microns (10, 25, or 50).
        reference : [row, col], optional
            Reference point in the atlas projection image (pixels).
            Defaults to [540, 570].
        """
        from wfield import allen_volume_from_structures, allen_flatten_areas, projection_outline
        from wfield.allen import selection_dorsal_cortex
        if structures is None:
            structures = selection_dorsal_cortex
        if reference is None:
            reference = [540, 570]
        mask_volume, areas = allen_volume_from_structures(structures, resolution=resolution)
        proj, ccf_regions = allen_flatten_areas(areas, mask_volume,
                                                resolution=resolution,
                                                reference=reference)
        brain_outline = projection_outline(proj, resolution=resolution, reference=reference)
        first = ccf_regions.iloc[0]
        self.insert1(dict(atlas_name=atlas_name,
                          ccf_regions=ccf_regions.to_dict(orient='list'),
                          projection=proj,
                          brain_outline=brain_outline,
                          reference_point=list(first['reference']),
                          resolution=float(first['resolution']) / 1000.0))

    def plot_atlas(self, ax=None, **kwargs):
        """Display the atlas flat projection with mm extent (bregma at origin).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Target axes; uses plt.gca() if None.
        **kwargs
            Forwarded to ax.imshow() (e.g. cmap, alpha, clim).
        """
        import matplotlib.pyplot as plt
        row = self.fetch1()
        proj = row['projection']
        ref_row, ref_col = row['reference_point']
        res = float(row['resolution'])
        H, W = proj.shape
        extent = [
            (0 - ref_col) * res,   # left  (mm, west edge)
            (W - ref_col) * res,   # right (mm, east edge)
            (H - ref_row) * res,   # bottom (mm, posterior)
            (0 - ref_row) * res,   # top   (mm, anterior)
        ]
        if ax is None:
            ax = plt.gca()
        kwargs.setdefault('cmap', 'gray')
        kwargs.setdefault('origin', 'upper')
        ax.imshow(proj, extent=extent, **kwargs)
        ax.invert_xaxis()
        return ax

    def load(self):
        """Return (ccf_regions, projection, brain_outline)."""
        row = self.fetch1()
        return row['ccf_regions'], row['projection'], row['brain_outline']

    def plot_regions(self, acronyms=None, ax=None, labels=True, **kwargs):
        """Plot atlas region contours in mm coordinates.

        Parameters
        ----------
        acronyms : list of str, optional
            Acronyms to plot; plots all regions when None.
        ax : matplotlib Axes, optional
            Target axes; uses plt.gca() if None.
        labels : bool
            Annotate each region centroid with its acronym.
        **kwargs
            Forwarded to ax.plot() for contour lines.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        ccf_regions, _, _ = self.load()
        regions = pd.DataFrame(ccf_regions)
        if acronyms is not None:
            regions = regions[regions['acronym'].isin(acronyms)]
        if ax is None:
            ax = plt.gca()
        for _, row in regions.iterrows():
            rgb = row.get('allen_rgb')
            color = kwargs.get('color', [c / 255 for c in rgb] if rgb is not None else None)
            kw = {**kwargs, 'color': color}
            kw.setdefault('lw', 1)
            ax.plot(row['left_x'],  row['left_y'],  **kw)
            ax.plot(row['right_x'], row['right_y'], **kw)
            if labels:
                for side in ('left', 'right'):
                    cx, cy = row[f'{side}_center']
                    ax.text(cx, cy, row['acronym'],
                            ha='center', va='center', fontsize=6, color=color)
        ax.invert_xaxis()
        return ax


@userschema
class WidefieldAtlasTransform(dj.Manual):
    definition = '''
    -> Widefield
    -> WidefieldAtlas
    atlas_transform_id : int           # unique transform per widefield-atlas pair
    ---
    transform_type               : enum('landmarks','manual')
    reference_point = NULL       : blob       # [col, row] of bregma in the widefield image (pixels)
    resolution = NULL            : float      # mm per widefield pixel
    landmarks = NULL             : longblob   # atlas-space landmarks (dict: x,y,name,color)
    landmarks_match = NULL       : longblob   # widefield-space landmarks (dict: x,y,name,color)
    rotation = NULL              : float      # degrees counter-clockwise
    scale = NULL                 : float      # isotropic scale factor on top of 1/resolution
    ratio = NULL                 : float      # x/y aspect ratio correction
    mirror = 0                   : tinyint    # 1 = flip atlas x-axis (for reversed imaging setups)
    transform_matrix = NULL      : longblob   # 3x3 float64, atlas mm -> widefield px
    transform_matrix_inverse = NULL : longblob
    '''

    def get_transform(self):
        """Return 3x3 ndarray mapping atlas mm coordinates to widefield pixel coordinates."""
        row = self.fetch1()
        M = row.get('transform_matrix')
        if M is not None:
            return np.asarray(M)
        return self._build_transform(row)

    def _build_transform(self, row):
        from .utils import build_atlas_transform
        t = row['transform_type']
        if t == 'manual':
            return build_atlas_transform(
                bregma_xy=np.asarray(row['reference_point']),
                resolution=float(row['resolution']),
                rotation=float(row['rotation'] or 0.0),
                scale=float(row['scale'] or 1.0),
                ratio=float(row['ratio'] or 1.0),
                mirror=bool(row.get('mirror') or False),
            )
        elif t == 'landmarks':
            import pandas as pd
            from wfield import allen_transform_from_landmarks, allen_landmarks_to_image_space
            atlas = (WidefieldAtlas & self).fetch1('reference_point', 'resolution')
            # reference_point is [row, col]; allen functions expect offset as [x, y] = [col, row]
            ref_row, ref_col = atlas['reference_point']
            ref_offset = np.array([ref_col, ref_row], dtype=float)
            resolution = float(atlas['resolution'])
            landmarks_im = allen_landmarks_to_image_space(
                pd.DataFrame(row['landmarks']).copy(), ref_offset, resolution)
            M_lm = allen_transform_from_landmarks(landmarks_im,
                                                  pd.DataFrame(row['landmarks_match']))
            T_res = np.array([[1/resolution, 0, ref_offset[0]],
                               [0, 1/resolution, ref_offset[1]],
                               [0, 0, 1]], dtype=float)
            return M_lm.params @ T_res
        raise ValueError(f'Unknown transform_type: {t!r}')

    def load_reference(self):
        """Return (ccf_regions, proj, brain_outline) for this atlas."""
        return (WidefieldAtlas & self.proj()).load()

    def _atlas_pixel_transform(self):
        """Return (M_px, projection) mapping atlas *projection pixels* to widefield pixels.

        ``get_transform()`` maps atlas mm -> widefield px; the atlas projection
        image relates its own pixels to mm through the atlas ``reference_point``
        ([row, col] of bregma) and ``resolution`` (mm/px), the same convention as
        ``WidefieldAtlas.plot_atlas``. Composing the two gives a single 3x3 matrix
        mapping atlas-projection (col, row) pixels to widefield (col, row) pixels.
        """
        ref_row, ref_col = np.asarray(
            (WidefieldAtlas & self.proj()).fetch1('reference_point'), dtype=float).ravel()[:2]
        res = float((WidefieldAtlas & self.proj()).fetch1('resolution'))
        projection = np.squeeze(np.asarray((WidefieldAtlas & self.proj()).fetch1('projection')))
        # atlas px (col, row) -> atlas mm (x, y)
        T_atlas = np.array([[res, 0.0, -res * ref_col],
                            [0.0, res, -res * ref_row],
                            [0.0, 0.0, 1.0]], dtype=float)
        M_px = np.asarray(self.get_transform()) @ T_atlas
        return M_px, projection

    def _widefield_reference(self):
        """Widefield mean projection (2-D float) for this transform's session.

        Uses the WfieldStack mean projection with the lowest ``wfield_analysis_id``.
        """
        row = self.fetch1()
        key = dict(subject_name=row['subject_name'], session_name=row['session_name'],
                   dataset_name=row['dataset_name'])
        aids = (WfieldStack & key).fetch('wfield_analysis_id')
        if not len(aids):
            raise ValueError('No WfieldStack mean projection for this session — '
                             'pass output_shape (or an image) explicitly.')
        mp = (WfieldStack & dict(key, wfield_analysis_id=int(min(aids)))).fetch1('mean_proj')
        mp = np.squeeze(np.asarray(mp)).astype(float)
        return mp[0] if mp.ndim == 3 else mp   # mean_proj is channel-first

    def image_to_atlas(self, image, output_shape=None, **kwargs):
        """Warp a widefield image into atlas-projection pixel space.

        Parameters
        ----------
        image : ndarray
            Widefield image (H x W) or movie (N x H x W) in this session's
            widefield pixel space.
        output_shape : (H, W), optional
            Shape of the atlas-space output. Defaults to the atlas projection shape.
        **kwargs
            Forwarded to `warp_image` (e.g. ``order``, ``cval``).

        Returns
        -------
        warped : ndarray (float)
            The image resampled into atlas-projection coordinates; an (N, H, W)
            movie warps frame by frame.
        """
        from .utils import warp_image
        M_px, projection = self._atlas_pixel_transform()
        if output_shape is None:
            output_shape = projection.shape[:2]
        # forward map is widefield px -> atlas px = inverse of atlas -> widefield
        M_fwd = np.linalg.inv(M_px)
        return self._warp(image, M_fwd, output_shape, warp_image, **kwargs)

    def atlas_to_image(self, projection=None, output_shape=None, **kwargs):
        """Warp the atlas projection into widefield image pixel space.

        Parameters
        ----------
        projection : ndarray, optional
            Image in atlas-projection pixel space to warp. Defaults to this
            atlas's stored projection.
        output_shape : (H, W), optional
            Shape of the widefield-space output. Defaults to this session's
            widefield mean-projection shape (lowest WfieldStack analysis id).
        **kwargs
            Forwarded to `warp_image` (e.g. ``order``, ``cval``).

        Returns
        -------
        warped : ndarray (float)
            The projection resampled into widefield-image coordinates.
        """
        from .utils import warp_image
        M_px, atlas_proj = self._atlas_pixel_transform()
        if projection is None:
            projection = atlas_proj
        if output_shape is None:
            output_shape = self._widefield_reference().shape[:2]
        # forward map is atlas px -> widefield px
        return self._warp(projection, M_px, output_shape, warp_image, **kwargs)

    @staticmethod
    def _warp(image, M_fwd, output_shape, warp_image, **kwargs):
        image = np.asarray(image, dtype=float)
        if image.ndim not in (2, 3):
            raise ValueError('image must be 2-D (H x W) or 3-D (N x H x W).')
        if image.ndim == 2:
            return warp_image(image, M_fwd, output_shape, **kwargs)
        return np.stack([warp_image(f, M_fwd, output_shape, **kwargs) for f in image])

    def transform_regions(self, ccf_regions=None):
        """Return ccf_regions DataFrame transformed to widefield pixel coordinates."""
        import pandas as pd
        from .utils import transform_atlas_regions
        if ccf_regions is None:
            ccf_regions, _, _ = self.load_reference()
        if isinstance(ccf_regions, dict):
            ccf_regions = pd.DataFrame(ccf_regions)
        return transform_atlas_regions(ccf_regions, self.get_transform())

    def plot_regions(self, acronyms=None, ax=None, labels=True, sides = ('left', 'right'), **kwargs):
        """Plot transformed atlas region contours.

        Parameters
        ----------
        acronyms : list of str, optional
            Acronyms to plot; plots all regions when None.
        ax : matplotlib Axes, optional
            Target axes; uses plt.gca() if None.
        labels : bool
            Annotate each region centroid with its acronym.
        **kwargs
            Forwarded to ax.plot() for contour lines.
        """
        import matplotlib.pyplot as plt
        regions = self.transform_regions()
        if acronyms is not None:
            regions = regions[regions['acronym'].isin(acronyms)]
        if ax is None:
            ax = plt.gca()
        for _, row in regions.iterrows():
            rgb = row.get('allen_rgb')
            color = kwargs.get('color', [c / 255 for c in rgb] if rgb is not None else None)
            kw = {**kwargs, 'color': color}
            kw.setdefault('lw', 1)
            if 'left' in sides:
                ax.plot(row['left_x'],  row['left_y'],  **kw)
            if 'right' in sides:
                ax.plot(row['right_x'], row['right_y'], **kw)
            if labels:
                for side in sides:
                    cx, cy = row[f'{side}_center']
                    ax.text(cx, cy, row['acronym'],
                            ha='center', va='center', fontsize=6, color=color)
        ax.invert_xaxis()
        return ax



@userschema
class WidefieldResponse(dj.Manual):
    definition = '''
    -> Widefield
    stim_name       : varchar(64)       # e.g. 'retinotopy', 'task_stim'
    ---
    wfield_analysis_id = NULL : int     # optional link to the WfieldStack used
    -> [nullable] AnalysisFile          # optional movie file (npz, tiff, etc.)
    '''

    class Projection(dj.Part):
        definition = '''
        -> master
        proj_name : varchar(64)         # e.g. 'phase', 'magnitude', 'sign_map', 'dff'
        ---
        proj      : longblob            # 2D image
        '''

    def save_projection(self, proj, proj_name, key=None):
        if key is None:
            key = self.fetch1('KEY')
        self.Projection.insert1(dict(key, proj_name=proj_name, proj=proj),
                                replace=True)

    def load_projections(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        rows = (self.Projection & key).fetch('proj_name', 'proj', as_dict=True)
        return {r['proj_name']: r['proj'] for r in rows}

    def insert1(self, row, movie=None, **kwargs):
        '''Insert a WidefieldResponse row, optionally saving and uploading a movie.

        Parameters
        ----------
        row : dict
            Row to insert. Must include the primary keys and any non-nullable attributes.
        movie : ndarray or dict, optional
            Movie data to save as an npz file and upload via AnalysisFile.
            Pass a numpy array (saved under the key ``'movie'``) or a dict of
            arrays (each key becomes an npz field). The resulting file_path/storage
            keys are merged into ``row`` before inserting.
        '''
        if movie is not None:
            dataset = {k: row[k] for k in ('subject_name', 'session_name', 'dataset_name')}
            dataset['dataset_name'] = 'wfield_' + row['stim_name']
            resultsfolder = Path(prefs['local_paths'][0]) / AnalysisFile().generate_filepaths(
                [''], dataset)[0]
            resultsfolder.mkdir(exist_ok=True, parents=True)
            movie_path = resultsfolder / f"{row['stim_name']}_movie.npz"
            arrays = movie if isinstance(movie, dict) else {'movie': movie}
            np.savez(movie_path, **arrays)
            filekeys = AnalysisFile().upload_files([movie_path], dataset)
            row = dict(row, **filekeys[0])
        super().insert1(row, **kwargs)

    def delete(self, transaction=True, safemode=None,
               force_parts=False, keep_analysis=False):
        '''Delete WidefieldResponse rows and their associated movie files.

        Removes the rows (and Projection part-table rows) from the database, then
        deletes the linked movie from AnalysisFile and S3 — unless keep_analysis=True.

        Parameters
        ----------
        keep_analysis : bool
            If True, skip deletion of AnalysisFile rows. Default is False.
        '''
        files = [f for f in self.fetch('file_path') if f is not None]
        super().delete(transaction=transaction, safemode=safemode,
                       force_parts=force_parts)
        if keep_analysis:
            return
        if len(self) == 0 and files:
            (AnalysisFile() & [f'file_path = "{f}"' for f in files]).delete(
                force_parts=force_parts, safemode=safemode)

    def load_movie(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        filepath = (AnalysisFile & (self & key)).get()[0]
        return np.load(filepath)


def _find_point_region(point, region_contours):
    '''Return ``(acronym, hemisphere, distance)`` for the first region polygon
    that contains ``point`` (x, y in atlas mm), else ``(None, None, None)``.

    ``distance`` is the cv2 signed distance to the contour edge (mm, + inside).
    Mirrors ``wfield``'s ``find_point_region``.
    '''
    import cv2
    pt = (float(point[0]), float(point[1]))
    for acronym, side, contour in region_contours:
        d = cv2.pointPolygonTest(contour, pt, True)
        if d > 0:
            return acronym, side, float(d)
    return None, None, None


@userschema
class CellSegmentationAtlas(dj.Computed):
    '''Atlas position and region assignment for every segmented ROI.

    Combines a CellSegmentation result with a TwoPhotonReferenceAlignment
    (2P -> widefield reference image) and a WidefieldAtlasTransform
    (atlas mm <-> widefield pixels) to place each ROI in Allen CCF
    coordinates (mm from bregma) and assign it to a cortical region.

    The WidefieldAtlasTransform's widefield session columns are renamed to
    ``ref_session`` / ``ref_dataset`` so they don't collide with the 2P
    session columns; ``key_source`` constrains them to the widefield that the
    alignment's ImagingReference points at.
    '''
    definition = '''
    -> CellSegmentation
    -> TwoPhotonReferenceAlignment
    -> WidefieldAtlasTransform.proj(ref_session='session_name', ref_dataset='dataset_name')
    ---
    n_rois        : int            # total ROIs placed
    n_in_atlas    : int            # ROIs assigned to a region
    '''

    class ROI(dj.Part):
        definition = '''
        -> master
        -> CellSegmentation.ROI
        ---
        atlas_x              : float                  # ML position, mm from bregma (+ = right)
        atlas_y              : float                  # AP position, mm from bregma (+ = posterior)
        hemisphere = NULL    : enum('left','right')   # side of the matched region
        acronym = NULL       : varchar(32)            # Allen region acronym (NULL if outside all regions)
        region_distance = NULL : float                # signed distance to region edge, mm (+ inside)
        '''

    @property
    def key_source(self):
        # Only combinations where the atlas transform belongs to the widefield
        # that this alignment's ImagingReference references. Project to the
        # primary key so downstream joins/antijoins don't trip over shared
        # secondary attributes (e.g. CellSegmentation.n_rois).
        align = TwoPhotonReferenceAlignment * ImagingReference.proj('ref_session', 'ref_dataset')
        atlas = WidefieldAtlasTransform.proj(ref_session='session_name',
                                             ref_dataset='dataset_name')
        return (CellSegmentation * align * atlas).proj()

    def make(self, key):
        import pandas as pd
        from labdata.schema import CellSegmentation
        from .utils import transform_coordinates

        # alignment (2P col,row -> widefield reference px) and its optional FOV crop offset
        align = TwoPhotonReferenceAlignment & key
        fov = align.fetch1('fov_offset')
        row_off, col_off = (int(fov[0]), int(fov[1])) if fov is not None else (0, 0)

        # atlas transform (atlas mm -> widefield px); invert for widefield px -> mm.
        # ref_session/ref_dataset are the widefield session the transform aligns.
        atlas_xf = WidefieldAtlasTransform & dict(
            subject_name=key['subject_name'],
            session_name=key['ref_session'], dataset_name=key['ref_dataset'],
            atlas_name=key['atlas_name'], atlas_transform_id=key['atlas_transform_id'])
        M_atlas_inv = np.linalg.inv(atlas_xf.get_transform())

        # region contours in atlas mm, left & right hemispheres
        regions = pd.DataFrame((WidefieldAtlas & key).fetch1('ccf_regions'))
        region_contours = []
        for _, reg in regions.iterrows():
            for side in ('left', 'right'):
                xy = np.column_stack([reg[f'{side}_x'], reg[f'{side}_y']]).astype(np.float32)
                if len(xy) >= 3:
                    region_contours.append((reg['acronym'], side, xy))

        # place each ROI centroid: seg (col,row) -> reference px -> atlas mm -> region
        roi_entries, n_in_atlas = [], 0
        for plane in (CellSegmentation.Plane & key).fetch('plane_num', 'dims', as_dict=True):
            if plane['dims'] is None:
                continue
            fh, fw = int(plane['dims'][0]), int(plane['dims'][1])
            M_fwd, transpose, _ = align.get_transform(fw + col_off, fh + row_off)
            rois = (CellSegmentation.ROI & dict(key, plane_num=plane['plane_num'])).fetch(
                'roi_num', 'roi_pixels', 'roi_pixels_values', as_dict=True)
            for roi in rois:
                pix = np.asarray(roi['roi_pixels']).ravel()
                if not len(pix):
                    continue
                rr, cc = np.unravel_index(pix, (fh, fw))
                w = roi['roi_pixels_values']
                w = (np.asarray(w, float).ravel()
                     if w is not None and len(np.asarray(w).ravel()) == len(pix) else None)
                cen_col, cen_row = np.average(cc, weights=w), np.average(rr, weights=w)
                # add FOV offset, swap axes if the alignment transposed the image
                pt2p = ([cen_row + row_off, cen_col + col_off] if transpose
                        else [cen_col + col_off, cen_row + row_off])
                x_mm, y_mm = transform_coordinates(
                    transform_coordinates(pt2p, M_fwd), M_atlas_inv)[0]
                acronym, hemi, dist = _find_point_region((x_mm, y_mm), region_contours)
                n_in_atlas += acronym is not None
                roi_entries.append(dict(
                    key, plane_num=plane['plane_num'], roi_num=roi['roi_num'],
                    atlas_x=float(x_mm), atlas_y=float(y_mm),
                    hemisphere=hemi, acronym=acronym, region_distance=dist))

        self.insert1(dict(key, n_rois=len(roi_entries), n_in_atlas=int(n_in_atlas)))
        self.ROI.insert(roi_entries)
