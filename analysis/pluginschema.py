from labdata.schema import *

userschema = get_user_schema()

__all__ = ['WfieldParameters', 'WfieldStack', 'ImagingWindow',
           'ImagingReference','TwoPhotonReferenceAlignment']

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
        
        dat = (Widefield & key).open()
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


