'''Compute tasks of the plugin, run with ``labdata run <analysis>``.

``CidsCompute`` segments two-photon and miniscope recordings with cids
(https://github.com/jcouto/cids). It lives in this plugin for now and may move
to labdata. To use it, add it to the analyses of the preference file::

    "compute": {"analysis": {"cids": "labdata.pwfield.CidsCompute", ...}}

and run it like the other segmentations, with the plugin sent to the remote::

    labdata run cids -a <SUBJECT> -s <SESSION> -t <REMOTE> --plugins pwfield -- --diameter 12
'''
import json
import string
from datetime import datetime
from pathlib import Path

import numpy as np

from labdata.utils import prefs, DEFAULT_N_JOBS, parallel_insert
from labdata.compute.suite2p import Suite2pCompute
from labdata.compute.caiman import CaimanCompute

from .pluginschema import CONTAINER

__all__ = ['CidsCompute', 'StackPlane']

# projections stored for each plane and session, from the first frames of the session, registered
NFRAMES_PROJECTION = 2000


class StackPlane:
    '''One plane of the sessions' stacks, as cids reads a movie: (frames, channels, height, width).

    Needed because labdata stacks are (frames, planes, channels, height,
    width), while cids reads 5-d movies as (frames, channels, planes, height,
    width); and to join the sessions of a multisession job and crop to
    ``--roi`` without writing the movie again. ``run_cells`` reads a movie
    only as ``movie[start:stop, channel]`` (``cids.io.read_frames``), which
    is all this supports.

    Parameters
    ----------
    stacks : list of array-like
        The stacks of the sessions, in the order they are concatenated (zarr
        arrays, scanbox memmaps): two-photon stacks 5-d as above, miniscope
        stacks (frames, height, width).
    plane : int or None
        Plane of 5-d stacks; None for stacks without planes.
    crop : sequence of 4 int or None
        ``(r0, c0, r1, c1)``: rows r0:r1 and columns c0:c1, as
        ``labdata.stacks.crop_frames``.
    '''
    ndim = 4

    def __init__(self, stacks, plane=None, crop=None):
        self.stacks = list(stacks)
        shapes = [tuple(int(a) for a in s.shape) for s in self.stacks]
        if len({s[1:] for s in shapes}) != 1 or len(shapes[0]) not in ((3, 4) if plane is None else (5,)):
            raise ValueError(f'Cannot read plane {plane} of stacks of shapes {shapes} as one movie.')
        self.plane = () if plane is None else (int(plane),)
        self.channels = len(shapes[0]) > 3   # stacks without a channel axis have one channel
        h, w = shapes[0][-2:]
        r0, c0, r1, c1 = (0, 0, h, w) if crop is None else [int(a) for a in crop]
        self.rows, self.cols = slice(r0, r1), slice(c0, c1)
        self.offsets = np.cumsum([0] + [s[0] for s in shapes])
        self.shape = (int(self.offsets[-1]), shapes[0][-3] if self.channels else 1,
                      len(range(h)[self.rows]), len(range(w)[self.cols]))
        self.dtype = np.dtype(self.stacks[0].dtype)
        chunks = getattr(self.stacks[0], 'chunks', None)
        if len(self.stacks) == 1 and chunks is not None:
            self.chunks = (int(chunks[0]),) + self.shape[1:]   # cids reads along the zarr chunks of frames

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        frames, channel = key   # movie[start:stop, channel], as cids.io.read_frames reads
        start, stop, step = frames.indices(self.shape[0])
        if step != 1 or not 0 <= channel < self.shape[1]:
            raise IndexError(f'StackPlane reads movie[start:stop, channel], not movie[{key}].')
        channel = (int(channel),) if self.channels else ()
        pieces = []
        for s, first, last in zip(self.stacks, self.offsets[:-1], self.offsets[1:]):
            a, b = max(start, first), min(stop, last)
            if b > a:
                pieces.append(np.asarray(s[(slice(a - first, b - first),) + self.plane + channel + (self.rows, self.cols)]))
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


class CidsCompute(Suite2pCompute):
    '''Cell segmentation of two-photon and miniscope recordings with cids.

    Runs ``cids.run_cells`` (registration, denoising, detection and demixing,
    extraction, deconvolution and quality) on each plane, on the GPU when the
    job has one (``device='auto'``: CUDA, then an Apple GPU, then the CPU),
    in the ``labdata-wfield`` container. Miniscope recordings run with the
    one-photon defaults of cids (``modality='1p'``), two-photon recordings
    with the two-photon ones. A job submitted with ``--multisession``
    segments several sessions of a subject together, as ``CaimanCompute``
    does: the frames are concatenated in session order, one set of ROIs per
    plane is found on all of them, and the traces, projections and motion
    shifts are stored per session under the same ROI numbers.

    What is stored in ``CellSegmentation``:

    - ``ROI``: every component of cids, its footprint (``footprints``, raw
      units, peak 1); ``Selection`` says which ones passed the quality
      checks of cids (``accepted``).
    - ``RawTraces.f_trace``: the demixed, background-free trace ``C`` (raw
      units, mean about 0), as ``CaimanCompute`` stores CaImAn's ``C``.
    - ``Traces.dff``: the dF/F of cids; ``Deconvolved.deconv``: its spikes.
    - ``MotionCorrection``: the shifts of cids with the sign flipped, the
      shift applied to each frame to register it, as for suite2p: frames by
      2 when rigid, otherwise frames by grid rows by grid columns by 2 (the
      patches of piecewise registration, or the corners of the frame with
      rotation).
    - ``Projection``: mean, std, max and lcorr of the first registered
      frames of each session (``labdata.stacks.compute_projections``), and
      the PNR and event energy images that cids found the cells on.

    The parameters are stored in ``CellSegmentationParams`` (algorithm
    ``cids``): the cell diameter (-1: measured on the movie), the decay time
    (None: fitted), the channels, and every value of
    ``cids.cell_parameters`` that the job sets (``overrides``).
    '''
    container = CONTAINER
    cuda = True
    name = 'cids'
    url = 'https://github.com/jcouto/cids'
    # stored in CellSegmentationParams only when they are set (not None)
    optional_parameter_keys = ['reference_channel', 'overrides']
    # the multisession handling of CaimanCompute: the recordings of the job, checked to go together
    _recordings = CaimanCompute._recordings

    def __init__(self, job_id, project=None, allow_s3=None, **kwargs):
        super(Suite2pCompute, self).__init__(job_id, project=project, allow_s3=allow_s3)  # takes BaseCompute init
        self.file_filters = list(self.stack_file_types)   # .zarr.zip and .sbx, read by open_stack_file
        self.parameters = dict(algorithm_name='cids')
        # stored in CellSegmentationParams
        self.parameter_keys = ['diameter', 'tau', 'channel']
        self.parameter_set_num = None   # identifier in CellSegmentationParams
        self._init_job()
        if not self.job_id is None:
            self.add_parameter_key()

    def _parameter_set(self, parameter_set_num):
        '''The parameters of a stored cids parameter set, as ``_secondary_parse`` makes them.'''
        rows = (self.schema.CellSegmentationParams & dict(parameter_set_num=int(parameter_set_num))).fetch(as_dict=True)
        if not len(rows):
            raise ValueError(f'There is no parameter set {parameter_set_num} in CellSegmentationParams.')
        if rows[0]['algorithm_name'] != self.name:
            raise ValueError(f'Parameter set {parameter_set_num} is for {rows[0]["algorithm_name"]}, not {self.name}.')
        return dict(json.loads(rows[0]['parameters_dict']), description=rows[0]['parameter_description'])

    def _secondary_parse(self, arguments, parameter_number=None):
        '''
        Handles parsing the command line interface
        '''
        if not parameter_number is None:
            self.parameters = self._parameter_set(parameter_number)
            return
        import argparse
        parser = argparse.ArgumentParser(
            description='Segmentation of two-photon and miniscope datasets with cids: registration, denoising, '
                        'detection and demixing, extraction, deconvolution and quality, on the GPU when there is one.',
            usage='cids -a <SUBJECT> -s <SESSION> -- <PARAMETERS>',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''examples:
  # two-photon somas, the cell diameter measured on the movie and the decay time fitted
  labdata run cids -a <SUBJECT> -s <SESSION> -t <REMOTE> --plugins pwfield

  # two-photon, cells of about 12 pixels, GCaMP6s, registered on a structural channel
  labdata run cids -a <SUBJECT> -s <SESSION> -t <REMOTE> --plugins pwfield -- --diameter 12 --tau 1.25 \\
      --channel 0 --reference-channel 1

  # miniscope (the one-photon defaults of cids are used for miniscope recordings), fewer seeds
  labdata run cids -a <SUBJECT> -s <SESSION> -t <REMOTE> --plugins pwfield -- --diameter 15 --min-pnr 10

  # any other value of cids.cell_parameters (`cids 2p --help` lists them), read as JSON
  labdata run cids -a <SUBJECT> -s <SESSION> -t <REMOTE> --plugins pwfield -- --diameter 12 \\
      --set max_shift=20 bidi=false bad_frames=\\"nan\\" --description "large shifts"

Every size of cids follows from --diameter and the frame rate. Every component is stored;
the quality checks of cids set the selection. The job runs on the GPU when the remote gives
it one: add a "cids" entry with a GPU queue to the analysis_options of the remote.''')
        parser.add_argument('-i', '--parameter-set', default=None, type=int,
                            help='use the stored parameter set with this number (the other parameters are ignored, '
                                 'except --roi)')
        parser.add_argument('--diameter', default=-1.0, type=float,
                            help='typical cell diameter in pixels; -1 measures it on the movie (default)')
        parser.add_argument('--tau', default=None, type=float,
                            help='decay time of the indicator in seconds; by default fitted to the decays of the '
                                 'accepted cells')
        parser.add_argument('--channel', default=0, type=int, help='functional channel (default 0)')
        parser.add_argument('--reference-channel', default=None, type=int,
                            help='channel to register on, e.g. a structural channel (default: the functional one)')
        parser.add_argument('--registration-patch', default=None, type=int,
                            help='[registration] piecewise rigid registration in patches of this many pixels; '
                                 '0 for rigid (default: 128 for frames of 256 pixels or more, otherwise rigid)')
        parser.add_argument('--min-pnr', default=None, type=float,
                            help='[detection] peak-to-noise ratio of the pixels where cells are looked for '
                                 '(default 8)')
        parser.add_argument('--min-corr', default=None, type=float,
                            help='[detection] correlation with the neighbours of the pixels where cells are '
                                 'looked for (default 0.8)')
        parser.add_argument('--set', default=[], nargs='+', metavar='NAME=VALUE', dest='values',
                            help='any other value of cids.cell_parameters; the values are read as JSON '
                                 '(true, null, 20, "nan"), otherwise as text')
        parser.add_argument('--description', default=None, type=str,
                            help='description stored with a new parameter set (up to 256 characters)')
        parser.add_argument('--roi', default=None, type=int, nargs=4,
                            help='crop the frames to rows r0:r1 and columns c0:c1 (r0 c0 r1 c1); the ROIs are stored '
                                 'in the cropped frame and the crop in CellSegmentation.crop_region')
        args = parser.parse_args(arguments[1:])
        if args.parameter_set is not None:
            self.parameters = dict(self._parameter_set(args.parameter_set), roi=args.roi)
            return
        if args.description is not None and len(args.description) > 256:
            parser.error(f'--description has {len(args.description)} characters; the limit is 256.')
        overrides = {}
        for item in args.values:
            name, sep, value = item.partition('=')
            if not sep or not name:
                parser.error(f'--set takes NAME=VALUE, not {item!r}.')
            try:
                overrides[name] = json.loads(value)
            except ValueError:
                overrides[name] = value
        for name, value in (('min_pnr', args.min_pnr), ('min_corr', args.min_corr)):
            if value is not None:
                overrides[name] = value
        if args.registration_patch is not None:
            overrides['registration_patch'] = args.registration_patch or None   # 0: rigid
        try:   # cids checks the names in the job; here too when it is installed
            from cids import cell_parameters
            unknown = sorted(set(overrides) - set(cell_parameters('2p', (512, 512), 10.0, 30.0)))
            if unknown:
                parser.error(f'cids has no parameters {unknown}; `cids 2p --help` lists them.')
        except ImportError:
            pass
        self.parameters = dict(diameter=args.diameter,
                               tau=args.tau,
                               channel=args.channel,
                               reference_channel=args.reference_channel,
                               overrides=overrides or None,
                               roi=args.roi,
                               description=args.description)

    def _open_recording(self, recording):
        '''Open the stack of one recording, its .zarr.zip or else its .sbx, as Suite2pCompute reads them.'''
        files = self.schema.File & recording
        self.track_downloads(files)  # remove_scratch deletes only the files downloaded here
        files = self.select_stack_files(files.get())
        if len(files) != 1:
            raise ValueError(f'{recording.proj().fetch1()} needs one .zarr.zip or .sbx stack; found {files}.')
        print(f'[cids] Reading {files[0]}', flush=True)
        return self.open_stack_file(files[0])

    def _segment_plane(self, movie, plane_num, seskeys, offsets, params, modality, frame_rate, folder):
        '''Run cids on one plane of the sessions (``movie``, a ``StackPlane``) and return the rows to store.

        The result is a dict with ``n_rois`` and the lists of rows for the
        plane, motion correction, projection, ROI, trace, raw trace,
        deconvolved and selection tables, one entry per session with that
        session's frames of the traces and shifts.
        '''
        from cids import run_cells, registered_movie
        from cids.io import load_arrays, load_sparse, load_attrs, has_arrays
        from cids.device import resolve
        from labdata.stacks import compute_projections
        device = resolve('auto')
        out = run_cells(movie, frame_rate, params.get('diameter', -1), modality=modality,
                        output=str(Path(folder)/f'plane{plane_num}.cids.zarr'),
                        channel=int(params.get('channel') or 0), reference_channel=params.get('reference_channel'),
                        tau=params.get('tau'), device=device, **(params.get('overrides') or {}))
        footprints = load_sparse(out, 'footprints').tocsc()   # (pixels, components), pixels y * width + x
        C, dff, spikes, accepted = load_arrays(out, 'C', 'dff', 'spikes', 'accepted')
        shifts, centers_y, centers_x = load_arrays(out, 'shifts', 'centers_y', 'centers_x')
        used = load_attrs(out).get('parameters') or {}
        dims = [int(a) for a in movie.shape[-2:]]
        n_frames = int(offsets[-1])
        if C.shape[1] != n_frames:
            raise ValueError(f'cids returned traces of {C.shape[1]} frames for {n_frames} frames.')
        # the shift applied to each frame to register it: cids samples each frame at p + shift (suite2p's sign)
        displacement = -np.asarray(shifts, dtype=np.float32)
        if displacement.shape[1:3] == (1, 1):
            displacement, motion_block_size = displacement[:, 0, 0], 0
        else:   # the patches, or the corners of the frame when the rigid motion includes a rotation
            motion_block_size = int(used.get('registration_patch') or max(dims))
        if len(displacement) != n_frames:
            print(f'[cids] Plane {plane_num}: {len(displacement)} motion shifts for {n_frames} frames; not storing them.')
            displacement = None
        rois = []
        for icell in range(footprints.shape[1]):
            on, off = footprints.indptr[icell], footprints.indptr[icell + 1]
            rois.append(dict(roi_num=icell,
                             roi_pixels=footprints.indices[on:off].astype(np.int64),
                             roi_pixels_values=footprints.data[on:off].astype(np.float32)))
        print(f'[cids] Plane {plane_num}: {int(np.sum(accepted))} of {len(rois)} components accepted.', flush=True)
        images = [(name, load_arrays(out, array)) for name, array in (('pnr', 'pnr_image'), ('energy', 'energy_image'))
                  if has_arrays(out, array)]
        registered = registered_movie(out, movie=movie, device=device)
        rows = dict(n_rois=len(rois), planedicts=[], motiondicts=[], projdicts=[], roidicts=[], tracesdicts=[],
                    rawtracesdicts=[], deconvdicts=[], selectiondicts=[])
        for key, on, off in zip(seskeys, offsets[:-1], offsets[1:]):
            on, off = int(on), int(off)
            planekey = dict(key, parameter_set_num=self.parameter_set_num, plane_num=int(plane_num))
            rows['planedicts'].append(dict(planekey, plane_n_rois=len(rois), dims=dims))
            if displacement is not None:
                rows['motiondicts'].append(dict(planekey, motion_block_size=motion_block_size,
                                                displacement=displacement[on:off]))
            projections = compute_projections(registered[on:min(off, on + NFRAMES_PROJECTION)])
            rows['projdicts'] += [dict(planekey, proj_name=name, proj_im=im)
                                  for name, im in zip(['mean', 'std', 'max', 'lcorr'], projections)]
            rows['projdicts'] += [dict(planekey, proj_name=name, proj_im=np.asarray(im, dtype=np.float32))
                                  for name, im in images]
            for roi in rois:
                icell = roi['roi_num']
                roikey = dict(planekey, roi_num=icell)
                rows['roidicts'].append(dict(roikey, roi_pixels=roi['roi_pixels'],
                                             roi_pixels_values=roi['roi_pixels_values']))
                rows['tracesdicts'].append(dict(roikey, dff=dff[icell, on:off].astype(np.float32)))
                rows['rawtracesdicts'].append(dict(roikey, f_trace=C[icell, on:off].astype(np.float32)))
                rows['deconvdicts'].append(dict(roikey, deconv=spikes[icell, on:off].astype(np.float32)))
                rows['selectiondicts'].append(dict(roikey, selection_method='auto',
                                                   selection=int(bool(accepted[icell]))))
        return rows

    def _compute(self):
        import cids
        kind, seskeys, recordings, frame_rate, n_planes, n_channels = self._recordings()
        params = json.loads((self.schema.CellSegmentationParams &
                             dict(parameter_set_num=self.parameter_set_num)).fetch1('parameters_dict'))
        for name in ('channel', 'reference_channel'):
            if params.get(name) is not None and not 0 <= int(params[name]) < n_channels:
                raise ValueError(f'The recording has {n_channels} channel(s); {name} {params[name]} is out of range.')
        crop = self.parameters.get('roi')
        modality = '1p' if kind == 'miniscope' else '2p'
        rand = ''.join(np.random.choice([s for s in string.ascii_lowercase + string.digits], 9))
        folder = Path(prefs['scratch_path'])/f'cids_temporary_{rand}'
        self.scratch_folders.append(folder)
        folder.mkdir(parents=True, exist_ok=True)
        print(f'[cids] {kind} ({modality}): {len(seskeys)} session(s), {n_planes} plane(s), {n_channels} channel(s), '
              f'{frame_rate} Hz; cids {cids.VERSION}; parameters {params}.', flush=True)
        stacks = [self._open_recording(r) for r in recordings]
        offsets = np.cumsum([0, *[int(s.shape[0]) for s in stacks]])
        if len(seskeys) > 1:
            print(f'[cids] Segmenting {len(seskeys)} sessions together, {offsets[-1]} frames: ' +
                  ', '.join(f"{k['session_name']} ({s.shape[0]})" for k, s in zip(seskeys, stacks)), flush=True)
        totals = [0 for k in seskeys]
        collected = dict(planedicts=[], motiondicts=[], projdicts=[], roidicts=[], tracesdicts=[],
                         rawtracesdicts=[], deconvdicts=[], selectiondicts=[])
        for iplane in range(n_planes):
            movie = StackPlane(stacks, plane=iplane if kind == 'two-photon' else None, crop=crop)
            rows = self._segment_plane(movie, iplane, seskeys, offsets, params, modality, frame_rate, folder)
            totals = [t + rows['n_rois'] for t in totals]
            for name in collected:
                collected[name] += rows[name]
        # frees the space of the raw files the job downloaded (kept with --keep-files); files that were local stay
        self.remove_scratch(keep_folders=True)
        segmentation_datetime = datetime.now()
        self.schema.CellSegmentation.insert([dict(key,
                                                  parameter_set_num=self.parameter_set_num,
                                                  algorithm_version=f'cids {cids.VERSION}',
                                                  n_rois=n_rois,
                                                  crop_region=crop,   # the ROIs and projections are in the cropped frame
                                                  segmentation_datetime=segmentation_datetime)
                                             for key, n_rois in zip(seskeys, totals)], allow_direct_insert=True)
        self.schema.CellSegmentation.Plane.insert(collected['planedicts'], allow_direct_insert=True)
        if collected['motiondicts']:
            self.schema.CellSegmentation.MotionCorrection.insert(collected['motiondicts'], allow_direct_insert=True)
        self.schema.CellSegmentation.Projection.insert(collected['projdicts'], allow_direct_insert=True)
        self.schema.CellSegmentation.ROI.insert(collected['roidicts'], allow_direct_insert=True)
        # the traces in parallel, to keep each insert short for mysql
        for table, name in (('Traces', 'tracesdicts'), ('RawTraces', 'rawtracesdicts'),
                            ('Deconvolved', 'deconvdicts'), ('Selection', 'selectiondicts')):
            parallel_insert(self.schema.schema_project, f'CellSegmentation.{table}', collected[name],
                            n_jobs=DEFAULT_N_JOBS, skip_duplicates=True, ignore_extra_fields=True)
        if self.keep_intermediate:
            print(f'[{self.name} job] Kept the temporary folder {folder} (the results of cids: `cids view`).')
        self.remove_scratch(keep_folders=self.keep_intermediate)
        for key in seskeys:
            self.schema.CellSegmentationMetrics().populate(key, display_progress=True)
