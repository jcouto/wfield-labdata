import numpy as np
import pandas as pd
import streamlit as st

from .common import (_normalize_image, _to_base64, _altair_image,
                     _tab_cache_factory, _refresh_button)


@st.fragment
def _imaging_reference_tab(schema, WfieldParameters, WfieldStack):
    import altair as alt
    from ..pluginschema import ImagingWindow, ImagingReference, TwoPhotonReferenceAlignment

    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    subject_name = sel_key['subject_name']


    st.subheader('Widefield Reference')
    cache = _tab_cache_factory('refresh_imaging_ref')
    _refresh_button('refresh_imaging_ref')

    @cache
    def get_imaging_refs(subject_name):
        rows = (ImagingReference & dict(subject_name=subject_name)).fetch(
            'ref_num', 'ref_session', 'ref_dataset', as_dict=True)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values('ref_num').reset_index(drop=True)

    @cache
    def get_stored_ref_image(subject_name, ref_num):
        rows = (ImagingReference & dict(subject_name=subject_name,
                                         ref_num=ref_num)).fetch(as_dict=True)
        if not rows:
            return None
        return np.squeeze(np.asarray(rows[0]['ref_image'])).astype(np.float32)

    @cache
    def get_wf_stacks(subject_name):
        return list((WfieldStack & dict(subject_name=subject_name)).fetch(
            'session_name', 'dataset_name', 'wfield_analysis_id', as_dict=True))

    @cache
    def get_wf_mean_for_ref(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        mp = (WfieldStack & key).fetch1('mean_proj')
        arr = np.squeeze(mp)
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32)

    @cache
    def get_window_for_session(subject_name, session_name, dataset_name):
        rows = (ImagingWindow & dict(subject_name=subject_name,
                                     session_name=session_name,
                                     dataset_name=dataset_name)).fetch(as_dict=True)
        return rows[0] if rows else None

    @cache
    def get_session_ref_files(subject_name, session_name):
        rows = (schema.Dataset.DataFiles
                & dict(subject_name=subject_name, session_name=session_name)
                ).fetch('dataset_name', 'file_path', as_dict=True)
        return [(r['dataset_name'], r['file_path'])
                for r in rows
                if r['file_path'].lower().endswith(('.tif', '.tiff', '.mat'))]

    @cache
    def load_ref_file(file_path):
        local = (schema.File & dict(file_path=file_path)).get()[0]
        if str(local).lower().endswith('.mat'):
            from scipy.io import loadmat
            arr = loadmat(local)['img'].astype(np.float32)
        else:
            import tifffile
            arr = tifffile.imread(local).astype(np.float32)
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = np.take(arr, 0, axis=int(np.argmin(arr.shape)))
        return arr

    existing_refs = get_imaging_refs(subject_name)
    if not existing_refs.empty:
        st.dataframe(existing_refs, hide_index=True)
    else:
        st.info('No ImagingReference entries yet for this subject.')

    wf_stacks = get_wf_stacks(subject_name)
    if not wf_stacks:
        st.info('No WfieldStack results available for this subject — run analysis first.')
    else:
        next_num = 0 if existing_refs.empty else int(existing_refs['ref_num'].max()) + 1
        with st.expander('Add ImagingReference', expanded=existing_refs.empty):
            col1, col2 = st.columns([3, 1])
            ref_num = col2.number_input('ref_num', value=next_num, step=1, min_value=0,
                                         key='ir_ref_num')
            stack_labels = [
                f"{r['session_name']} / {r['dataset_name']} (id={r['wfield_analysis_id']})"
                for r in wf_stacks]
            sel_stack = col1.selectbox('Widefield session (with WfieldStack)', stack_labels,
                                        key='ir_stack_sel')
            sr = wf_stacks[stack_labels.index(sel_stack)]
            rsess = sr['session_name']
            rdset = sr['dataset_name']
            raid  = int(sr['wfield_analysis_id'])

            with st.spinner('Loading image…'):
                raw_ref = get_wf_mean_for_ref(subject_name, rsess, rdset, raid)
            win = get_window_for_session(subject_name, rsess, rdset)

            wf_ref_files = get_session_ref_files(subject_name, rsess)
            if wf_ref_files:
                from pathlib import Path
                _DEFAULT = 'WfieldStack mean_proj (default)'
                ref_src = {_DEFAULT: None,
                           **{f"{ds}:  {Path(fp).name}": fp for ds, fp in wf_ref_files}}
                sel_src = st.selectbox('ref_image source', list(ref_src.keys()),
                                        key='ir_ref_src')
                if ref_src[sel_src] is not None:
                    with st.spinner('Loading file…'):
                        raw_ref = load_ref_file(ref_src[sel_src])

            rh0, rw0 = raw_ref.shape
            disp_arr = np.stack([_normalize_image(raw_ref)] * 3, axis=-1)
            if win and win.get('circle_parameters') is not None:
                from PIL import Image as PILImage, ImageDraw
                cp = np.asarray(win['circle_parameters'])
                xc0, yc0, rad0 = cp
                pil_img = PILImage.fromarray(disp_arr)
                draw = ImageDraw.Draw(pil_img)
                draw.ellipse([xc0 - rad0, yc0 - rad0, xc0 + rad0, yc0 + rad0],
                             outline=(255, 60, 60), width=3)
                disp_arr = np.array(pil_img)
                res0 = win.get('resolution')
                st.caption(f'ImagingWindow: center=({xc0:.0f}, {yc0:.0f}), '
                           f'radius={rad0:.1f} px'
                           + (f', resolution={res0:.4f} mm/px' if res0 else ''))

            cw0 = 350
            st.altair_chart(
                _altair_image(_to_base64(disp_arr), rw0, rh0,
                              title='Reference image (stored as uint16)', width=cw0),
                width='content')

            if st.button('Save as ImagingReference', type='primary', key='ir_save'):
                try:
                    ImagingReference.insert1(dict(
                        subject_name=subject_name,
                        ref_num=int(ref_num),
                        ref_session=rsess,
                        ref_dataset=rdset,
                        ref_image=raw_ref.astype(np.uint16),
                    ), replace=True)
                    get_imaging_refs.clear()
                    get_stored_ref_image.clear()
                    st.success(f'Saved ImagingReference ref_num={ref_num} '
                               f'({rsess} / {rdset})')
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


    st.divider()
    st.subheader('TwoPhoton Alignment')

    if existing_refs.empty:
        st.info('Create an ImagingReference above first.')
        return

    @cache
    def get_2p_alignments(subject_name):
        rows = (TwoPhotonReferenceAlignment & dict(subject_name=subject_name)).fetch(
            'ref_num', 'session_name', 'dataset_name',
            'rotation', 'scale', 'transpose', 'ratio', as_dict=True)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @cache
    def get_tp_sessions(subject_name):
        try:
            from labdata.schema import TwoPhoton
            return list((TwoPhoton & dict(subject_name=subject_name)).fetch(
                'session_name', 'dataset_name', 'n_planes', 'n_channels', as_dict=True))
        except Exception:
            return []

    alignments = get_2p_alignments(subject_name)
    if not alignments.empty:
        st.dataframe(alignments, hide_index=True)

    tp_sessions = get_tp_sessions(subject_name)
    if not tp_sessions:
        st.info('No TwoPhoton datasets found for this subject.')
        return

    col_ref, col_tp = st.columns(2)

    ref_opts = {
        f"ref {int(r['ref_num'])}: {r['ref_session']}/{r['ref_dataset']}": int(r['ref_num'])
        for _, r in existing_refs.iterrows()
    }
    sel_ref_lbl = col_ref.selectbox('ImagingReference', list(ref_opts.keys()),
                                     key='ir_ref_sel')
    sel_ref_num = ref_opts[sel_ref_lbl]

    tp_labels = [f"{r['session_name']} / {r['dataset_name']}" for r in tp_sessions]
    sel_tp_lbl = col_tp.selectbox('TwoPhoton dataset', tp_labels, key='ir_tp_sel')
    tp_row = tp_sessions[tp_labels.index(sel_tp_lbl)]

    @cache
    def get_dataset_files(subject_name, session_name):
        rows = (schema.Dataset.DataFiles
                & dict(subject_name=subject_name, session_name=session_name)
                ).fetch('dataset_name', 'file_path', as_dict=True)
        return [(r['dataset_name'], r['file_path']) for r in rows]

    @cache
    def load_datafile_raw(file_path):
        """Load without collapsing channels — returns squeezed array of any ndim."""
        local = (schema.File & dict(file_path=file_path)).get()[0]
        lname = str(local).lower()
        if lname.endswith(('.tif', '.tiff')):
            import tifffile
            arr = tifffile.imread(local)
        elif lname.endswith('.npy'):
            arr = np.load(local)
        elif lname.endswith('.mat'):
            from scipy.io import loadmat
            arr = loadmat(local)['img']
        else:
            from PIL import Image as PILImage
            arr = np.array(PILImage.open(local).convert('L'))
        return np.squeeze(arr).astype(np.float32)

    @cache
    def get_saved_alignment(subject_name, ref_num, session_name, dataset_name):
        rows = (TwoPhotonReferenceAlignment & dict(
            subject_name=subject_name, ref_num=ref_num,
            session_name=session_name, dataset_name=dataset_name,
        )).fetch(as_dict=True)
        return rows[0] if rows else None

    @cache
    def get_cell_seg_entries(subject_name, session_name, dataset_name):
        try:
            from labdata.schema import CellSegmentation, CellSegmentationParams
            rows = list((CellSegmentation & dict(subject_name=subject_name,
                                                 session_name=session_name,
                                                 dataset_name=dataset_name)).fetch(
                'parameter_set_num', 'n_rois', as_dict=True))
            if not rows:
                return []
            param_keys = [dict(parameter_set_num=r['parameter_set_num']) for r in rows]
            params = {p['parameter_set_num']: p for p in
                      (CellSegmentationParams & param_keys).fetch(as_dict=True)}
            for r in rows:
                p = params.get(r['parameter_set_num'], {})
                r['label'] = (f"params={r['parameter_set_num']} "
                              f"({p.get('algorithm_name', '?')}, {r['n_rois']} ROIs)")
            return rows
        except Exception:
            return []

    @cache
    def get_cell_seg_proj_list(subject_name, session_name, dataset_name, parameter_set_num):
        try:
            from labdata.schema import CellSegmentation
            key = dict(subject_name=subject_name, session_name=session_name,
                       dataset_name=dataset_name, parameter_set_num=parameter_set_num)
            return list((CellSegmentation.Projection & key).fetch(
                'plane_num', 'proj_name', as_dict=True))
        except Exception:
            return []

    @cache
    def load_cell_seg_proj(subject_name, session_name, dataset_name,
                           parameter_set_num, plane_num, proj_name):
        from labdata.schema import CellSegmentation
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, parameter_set_num=parameter_set_num,
                   plane_num=plane_num, proj_name=proj_name)
        return np.squeeze(np.asarray(
            (CellSegmentation.Projection & key).fetch1('proj_im')
        )).astype(np.float32)

    src_type = st.radio('Image source', ['File from DataFiles', 'CellSegmentation projection'],
                        horizontal=True, key='ir_src_type')

    if src_type == 'File from DataFiles':
        ds_files = get_dataset_files(subject_name, tp_row['session_name'])
        if not ds_files:
            st.info('No files in Dataset.DataFiles for this session.')
            return

        from pathlib import Path
        file_map = {f"{ds}:  {Path(fp).name}": fp for ds, fp in ds_files}
        _NO_SEL = '— select a file —'
        sel_file = st.selectbox('File from DataFiles', [_NO_SEL] + list(file_map.keys()),
                                 key='ir_ds_file')
        if sel_file == _NO_SEL:
            st.caption('Select a file to use as the 2P reference image.')
            return

        fkey = f'ir_fit_{sel_file}'
        if st.button('↺ Reload file', key='ir_reload_file') and fkey in st.session_state:
            del st.session_state[fkey]
        if fkey not in st.session_state:
            try:
                with st.spinner('Loading…'):
                    st.session_state[fkey] = load_datafile_raw(file_map[sel_file])
            except Exception as exc:
                st.error(f'Failed to load {sel_file}: {exc}')
                return

        raw_img = st.session_state[fkey]

        if raw_img.ndim > 2:
            ch_ax = int(np.argmin(raw_img.shape))
            n_ch  = raw_img.shape[ch_ax]
            sel_ch = st.selectbox(f'Channel (axis {ch_ax}, {n_ch} available)',
                                  list(range(n_ch)), key='ir_ch_sel')
            fit_img = np.take(raw_img, sel_ch, axis=ch_ax)
        else:
            fit_img = raw_img

        st.caption(f'2P image: {raw_img.shape}  →  using {fit_img.shape[0]} rows × {fit_img.shape[1]} cols')

    else:  # CellSegmentation projection
        cs_entries = get_cell_seg_entries(subject_name, tp_row['session_name'],
                                          tp_row['dataset_name'])
        if not cs_entries:
            st.info('No CellSegmentation entries found for this dataset.')
            return

        cs_labels = [r['label'] for r in cs_entries]
        sel_cs_lbl = st.selectbox('CellSegmentation entry', cs_labels, key='ir_cs_entry')
        cs_row = cs_entries[cs_labels.index(sel_cs_lbl)]

        proj_rows = get_cell_seg_proj_list(subject_name, tp_row['session_name'],
                                           tp_row['dataset_name'],
                                           cs_row['parameter_set_num'])
        if not proj_rows:
            st.info('No projections found for this segmentation entry.')
            return

        planes     = sorted(set(r['plane_num'] for r in proj_rows))
        proj_names = sorted(set(r['proj_name'] for r in proj_rows))

        pc1, pc2 = st.columns(2)
        sel_plane = pc1.selectbox('Plane', planes, key='ir_cs_plane')
        sel_proj  = pc2.selectbox('Projection', proj_names, key='ir_cs_proj')

        cs_img_key = f'ir_cs_{cs_row["parameter_set_num"]}_{sel_plane}_{sel_proj}'
        if st.button('↺ Reload projection', key='ir_cs_reload') and cs_img_key in st.session_state:
            del st.session_state[cs_img_key]
        if cs_img_key not in st.session_state:
            try:
                with st.spinner('Loading projection…'):
                    st.session_state[cs_img_key] = load_cell_seg_proj(
                        subject_name, tp_row['session_name'], tp_row['dataset_name'],
                        cs_row['parameter_set_num'], sel_plane, sel_proj)
            except Exception as exc:
                st.error(f'Failed to load projection: {exc}')
                return

        fit_img = st.session_state[cs_img_key]
        st.caption(f'CellSegmentation projection: plane={sel_plane}, proj={sel_proj}, '
                   f'shape={fit_img.shape[0]} rows × {fit_img.shape[1]} cols')

    ref_img_full = get_stored_ref_image(subject_name, sel_ref_num)
    if ref_img_full is None:
        st.error('Could not load ImagingReference image.')
        return
    full_rh, full_rw = ref_img_full.shape

    ref_img = ref_img_full
    rh, rw  = ref_img.shape

    # Load existing alignment when the scope changes
    # For CellSegmentation source, include the selected CS entry so switching entries
    # also triggers a reload (alignment is per session/dataset but this ensures the
    # DB is always re-read when the user picks a new CS entry).
    if src_type == 'CellSegmentation projection':
        align_scope = (f'{sel_ref_num}|cs|{tp_row["session_name"]}|{tp_row["dataset_name"]}'
                       f'|{st.session_state.get("ir_cs_entry", "")}')
    else:
        align_scope = f'{sel_ref_num}|{tp_row["session_name"]}|{tp_row["dataset_name"]}'
    if st.session_state.get('ir_align_scope') != align_scope:
        st.session_state['ir_align_scope'] = align_scope
        st.session_state.pop('ir_align_last', None)
        saved = get_saved_alignment(subject_name, sel_ref_num,
                                    tp_row['session_name'], tp_row['dataset_name'])
        if saved:
            st.session_state['ir_rot']       = float(saved['rotation'])
            st.session_state['ir_scale']     = float(saved['scale'])
            st.session_state['ir_ratio']     = float(saved['ratio'])
            st.session_state['ir_transpose'] = bool(saved['transpose'])
            origin = np.asarray(saved['origin']).ravel()
            st.session_state['ir_ox_slider'] = int(np.clip(origin[0], 0, rw))
            st.session_state['ir_oy_slider'] = int(np.clip(origin[1], 0, rh))
            if saved.get('fov_offset') is not None:
                off = np.asarray(saved['fov_offset']).ravel()
                st.session_state['ir_fov_row'] = int(off[0])
                st.session_state['ir_fov_col'] = int(off[1])
            st.session_state['ir_loaded_align'] = True
        else:
            st.session_state['ir_ox_slider']  = rw // 2
            st.session_state['ir_oy_slider']  = rh // 2
            st.session_state['ir_loaded_align'] = False

    if st.session_state.get('ir_loaded_align'):
        st.info('Existing alignment loaded — adjust and re-save to update.')

    # Drain any pending click update into the slider keys before the widgets render
    if 'ir_ox_pending' in st.session_state:
        st.session_state['ir_ox_slider'] = st.session_state.pop('ir_ox_pending')
    if 'ir_oy_pending' in st.session_state:
        st.session_state['ir_oy_slider'] = st.session_state.pop('ir_oy_pending')

    # FOV offset — rows/cols dropped at the edge of the 2P segmentation output
    with st.expander('FOV offset (optional)'):
        st.caption('Row and column offset if the segmentation output is cropped relative to the raw 2P image.')
        fo1, fo2 = st.columns(2)
        fov_row_off = int(fo1.number_input('Row offset', value=0, min_value=0, key='ir_fov_row'))
        fov_col_off = int(fo2.number_input('Col offset', value=0, min_value=0, key='ir_fov_col'))
    fov_offset = (np.array([fov_row_off, fov_col_off], dtype=float)
                  if (fov_row_off or fov_col_off) else None)

    st.write('**Alignment parameters**')
    ac1, ac2, ac3 = st.columns(3)
    ac4, ac5, ac6 = st.columns(3)
    ac7, ac8, _   = st.columns(3)
    rotation    = ac1.slider('Rotation (°)', -180.0, 180.0, 0.0, step=0.5, key='ir_rot')
    scale       = ac2.slider('Scale (px/px)', 0.01, 1.0, 0.1, step=0.005, key='ir_scale')
    ratio       = ac3.slider('X/Y aspect ratio', 0.5, 2.0, 1.0, step=0.005, key='ir_ratio')
    origin_x    = float(ac4.slider('Origin X (col)', 0, rw, key='ir_ox_slider'))
    origin_y    = float(ac5.slider('Origin Y (row)', 0, rh, key='ir_oy_slider'))
    transpose   = ac6.checkbox('Transpose 2P image', value=False, key='ir_transpose')
    alpha_blend = ac7.slider('2P overlay alpha', 0.0, 1.0, 0.5, step=0.05, key='ir_alpha')
    color_mode  = ac8.checkbox('Color (WF=green, 2P=red)', value=False, key='ir_color_mode')
    st.caption(
        f'Scale: 1 2P pixel → {float(scale):.3f} ref pixels · '
        f'X/Y aspect: {float(ratio):.3f} (1.0 = square pixels, >1 stretches x)')

    # Build forward affine: fit (col, row) to ref (col, row)
    from ..utils import build_alignment_transform, warp_image
    from skimage.measure import find_contours
    from PIL import Image as PILImage, ImageDraw
    fit = fit_img.T if transpose else fit_img
    fh, fw = fit.shape
    M_fwd = build_alignment_transform(fw, fh, rotation, scale, ratio, origin_x, origin_y)
    warped_mask = warp_image(np.ones_like(fit, dtype=float), M_fwd, (rh, rw))
    warped_fit  = warp_image(fit, M_fwd, (rh, rw))

    # Normalize both images
    mask_bool = warped_mask > 0.5
    ref_f     = _normalize_image(ref_img).astype(float)
    fit_norm  = np.zeros((rh, rw), dtype=float)
    if mask_bool.any():
        lo = np.percentile(warped_fit[mask_bool], 2)
        hi = np.percentile(warped_fit[mask_bool], 98)
        if hi > lo:
            fit_norm = np.clip((warped_fit - lo) / (hi - lo) * 255, 0, 255)

    if color_mode:
        # Color channels WF=green, 2P=red
        # Alpha blends the color image against grayscale (opacity control).
        color_r = np.zeros((rh, rw), dtype=float)
        color_g = ref_f.copy()
        color_b = np.zeros((rh, rw), dtype=float)
        if mask_bool.any():
            color_r[mask_bool] = fit_norm[mask_bool]
        ch_r = np.clip((1.0 - alpha_blend) * ref_f + alpha_blend * color_r, 0, 255)
        ch_g = ref_f                                   # green = WF always; same in gray and color
        ch_b = np.clip((1.0 - alpha_blend) * ref_f,   0, 255)
    else:
        # Grayscale: blend ref and 2P uniformly inside FOV
        ch = ref_f.copy()
        if mask_bool.any():
            ch[mask_bool] = np.clip(
                (1.0 - alpha_blend) * ref_f[mask_bool] +
                alpha_blend * fit_norm[mask_bool], 0, 255)
        ch_r = ch_g = ch_b = ch

    overlay_arr = np.stack([ch_r.astype(np.uint8),
                            ch_g.astype(np.uint8),
                            ch_b.astype(np.uint8)], axis=-1)
    for contour in find_contours(warped_mask, 0.5):
        pts = [(float(c[1]), float(c[0])) for c in contour]
        if len(pts) > 1:
            pil_ov = PILImage.fromarray(overlay_arr)
            ImageDraw.Draw(pil_ov).line(pts + [pts[0]], fill=(255, 140, 0), width=1)
            overlay_arr = np.array(pil_ov)

    # Clickable full-reference chart + zoomed FOV panel side by side
    ov_col1, ov_col2 = st.columns([3, 2])

    url_ov = _to_base64(overlay_arr)
    step_g = max(4, min(rw, rh) // 64)
    xs_g, ys_g = np.arange(0, rw, step_g, dtype=float), np.arange(0, rh, step_g, dtype=float)
    xx_g, yy_g = np.meshgrid(xs_g, ys_g)
    xsc_ov = alt.Scale(domain=[0, rw])
    ysc_ov = alt.Scale(domain=[rh, 0])
    img_l_ov = (
        alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': rw, 'y2': rh, 'url': url_ov}]))
        .mark_image(aspect=False)
        .encode(x=alt.X('x:Q', scale=xsc_ov, axis=None),
                y=alt.Y('y:Q', scale=ysc_ov, axis=None),
                x2='x2:Q', y2='y2:Q', url='url:N', tooltip=alt.value(None))
    )
    click_sel_ov = alt.selection_point(name='align_click', on='click',
                                       nearest=True, encodings=['x', 'y'])
    grid_l_ov = (
        alt.Chart(pd.DataFrame({'x': xx_g.ravel(), 'y': yy_g.ravel()}))
        .mark_point(opacity=0, size=step_g * step_g * 4)
        .encode(x=alt.X('x:Q', scale=xsc_ov, axis=None),
                y=alt.Y('y:Q', scale=ysc_ov, axis=None))
        .add_params(click_sel_ov)
    )
    with ov_col1:
        event_ov = st.altair_chart(
            alt.layer(img_l_ov, grid_l_ov)
            .properties(width=380, height=max(200, int(380 * rh / rw)),
                        title='Click to set origin · orange = 2P FOV boundary')
            .interactive(),
            on_select='rerun', key='wf_align_chart', width='content')
        st.caption('Click on the reference image to place the 2P origin.')

    with ov_col2:
        if mask_bool.any():
            rs_b, cs_b = np.where(mask_bool)
            fov_h = rs_b.max() - rs_b.min()
            fov_w = cs_b.max() - cs_b.min()
            pad_px = max(20, int(max(fov_h, fov_w) * 0.4))
            r0b = max(0, rs_b.min() - pad_px)
            r1b = min(rh, rs_b.max() + pad_px)
            c0b = max(0, cs_b.min() - pad_px)
            c1b = min(rw, cs_b.max() + pad_px)
            zoom_arr = overlay_arr[r0b:r1b, c0b:c1b].copy()
            zh, zw = zoom_arr.shape[:2]
            st.altair_chart(
                _altair_image(_to_base64(zoom_arr), zw, zh,
                              title='zoomed', width=300),
                width='content')
        else:
            st.caption('Move origin or adjust scale to place the FOV inside the reference.')

    sel_ov = (event_ov.selection or {}).get('align_click', [])
    if sel_ov:
        xc_click, yc_click = sel_ov[0].get('x'), sel_ov[0].get('y')
        if xc_click is not None and yc_click is not None:
            new_orig = [float(xc_click), float(yc_click)]
            if new_orig != st.session_state.get('ir_align_last'):
                st.session_state['ir_align_last'] = new_orig
                st.session_state['ir_ox_pending'] = int(np.clip(xc_click, 0, rw))
                st.session_state['ir_oy_pending'] = int(np.clip(yc_click, 0, rh))
                st.rerun()

    if st.button('Save TwoPhotonReferenceAlignment', type='primary', key='ir_2p_save'):
        try:
            TwoPhotonReferenceAlignment.insert1(dict(
                subject_name=subject_name,
                ref_num=int(sel_ref_num),
                session_name=tp_row['session_name'],
                dataset_name=tp_row['dataset_name'],
                rotation=float(rotation),
                scale=float(scale),
                transpose=bool(transpose),
                ratio=float(ratio),
                origin=np.array([origin_x, origin_y]),
                fov_offset=fov_offset,
            ), replace=True)
            get_2p_alignments.clear()
            get_saved_alignment.clear()
            st.session_state['ir_loaded_align'] = True
            st.success(f"Saved alignment: {tp_row['session_name']}/{tp_row['dataset_name']}"
                       f" → ref_num={sel_ref_num}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
