from labdata.schema import *

userschema = get_user_schema()

__all__ = ['WfieldParameters', 'WfieldStack', 'ImagingWindow',
           'ImagingReference','TwoPhotonReferenceAlignment',
           'WidefieldAtlas', 'WidefieldAtlasTransform',
           'WidefieldResponse']

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
        dataset['dataset_name'] = f'wfield_{par["wfield_analysis_id"]:02d}'
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

@userschema
class ImagingReference(dj.Manual):
    definition = """
    -> Subject
    ref_num                      : smallint
    ---
    -> Widefield.proj(ref_session = "session_name",ref_dataset = "dataset_name") 
    ref_image                    : longblob                                        # the reference image (uint16)
    """

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
    atlas_transform_id : int           # unique transform per widefield × atlas pair
    ---
    transform_type               : enum('landmarks','manual')
    reference_point = NULL       : blob       # [col, row] of bregma in the widefield image (pixels)
    resolution = NULL            : float      # mm per widefield pixel
    landmarks = NULL             : longblob   # atlas-space landmarks (dict: x,y,name,color)
    landmarks_match = NULL       : longblob   # widefield-space landmarks (dict: x,y,name,color)
    rotation = NULL              : float      # degrees counter-clockwise
    scale = NULL                 : float      # isotropic scale factor on top of 1/resolution
    ratio = NULL                 : float      # x/y aspect ratio correction
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
        return (WidefieldAtlas & self).load()

    def transform_regions(self, ccf_regions=None):
        """Return ccf_regions DataFrame transformed to widefield pixel coordinates."""
        import pandas as pd
        from .utils import transform_atlas_regions
        if ccf_regions is None:
            ccf_regions, _, _ = self.load_reference()
        if isinstance(ccf_regions, dict):
            ccf_regions = pd.DataFrame(ccf_regions)
        return transform_atlas_regions(ccf_regions, self.get_transform())

    def plot_regions(self, acronyms=None, ax=None, labels=True, **kwargs):
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
