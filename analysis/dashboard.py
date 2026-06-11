import base64
import io
import numpy as np
import pandas as pd
import streamlit as st

dashboard_name = 'Widefield'

_ANALYSED_COLOR = '#d4f5d4'  # light green for sessions with completed WfieldStack


def _normalize_image(img):
    img = np.asarray(img, dtype=float)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return (img * 255).clip(0, 255).astype(np.uint8)


def _normalize_with_pct(img, lo_pct=2.0, hi_pct=99.0):
    img = np.asarray(img, dtype=float)
    lo = np.percentile(img, lo_pct)
    hi = np.percentile(img, hi_pct)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _to_base64(arr):
    from PIL import Image as PILImage
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        pass  # already (H, W, 3) RGB — use as-is
    elif arr.ndim == 3:
        arr = arr[0]   # multi-channel widefield: take first channel
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _altair_image(url, w, h, title='', width=400, height=None):
    import altair as alt
    if height is None:
        height = max(1, int(width * h / w))
    img_df = pd.DataFrame([{'x': 0, 'y': 0, 'x2': w, 'y2': h, 'url': url}])
    return (
        alt.Chart(img_df).mark_image(aspect=False).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[0, w]), axis=None),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[h, 0]), axis=None),
            x2=alt.X2('x2'), y2=alt.Y2('y2'), url='url:N',
            tooltip=alt.value(None),
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def _tab_cache_factory(refresh_key):
    """Return a decorator like ``st.cache_data`` that also wires a cache into a
    per-tab refresh button. When the tab's refresh flag is set (the button was
    clicked on the previous run), each cache is cleared as it is recreated — so
    only the current tab's data reloads, leaving other tabs' caches warm."""
    do_clear = st.session_state.pop(refresh_key, False)

    def cache(func):
        cached = st.cache_data(func)
        if do_clear:
            cached.clear()
        return cached
    return cache


def _refresh_button(refresh_key, label='↻ Refresh'):
    """Render a per-tab refresh button that reloads this tab's data on next run."""
    if st.button(label, key=f'{refresh_key}_btn',
                 help="Reload this tab's data from the database"):
        st.session_state[refresh_key] = True
        st.rerun()


def dashboard_function(schema=None):
    from .pluginschema import WfieldParameters, WfieldStack
    st.write('## Widefield')
    (sessions_tab, results_tab, explorer_tab, mask_tab,
     ref_tab, atlas_tab, cell_atlas_tab) = st.tabs(
        ['Sessions & Parameters', 'Projections', 'Stack explorer', 'Window mask',
         'Imaging Reference', 'Atlas Alignment', 'Cell Atlas'])
    with sessions_tab:
        _sessions_params_tab(schema, WfieldParameters, WfieldStack)
    with results_tab:
        _results_tab(schema, WfieldParameters, WfieldStack)
    with explorer_tab:
        _stack_explorer_tab(schema, WfieldParameters, WfieldStack)
    with mask_tab:
        _window_mask_tab(schema, WfieldParameters, WfieldStack)
    with ref_tab:
        _imaging_reference_tab(schema, WfieldParameters, WfieldStack)
    with atlas_tab:
        _atlas_alignment_tab(schema, WfieldParameters, WfieldStack)
    with cell_atlas_tab:
        _cell_atlas_tab(schema)




def _sessions_params_tab(schema, WfieldParameters, WfieldStack):
    import altair as alt
    cache = _tab_cache_factory('refresh_sessions')
    _refresh_button('refresh_sessions')

    @cache
    def get_table_counts():
        counts = {}
        for name, tbl in [
            ('Widefield recordings', schema.Widefield()),
            ('WfieldParameters',     WfieldParameters()),
            ('WfieldStack',          WfieldStack()),
            ('Subjects',             schema.Subject & schema.Widefield),
        ]:
            try:
                counts[name] = len(tbl)
            except Exception:
                counts[name] = None
        return counts

    counts = get_table_counts()
    m_cols = st.columns(len(counts))
    for col, (name, val) in zip(m_cols, counts.items()):
        col.metric(name, '—' if val is None else val)

    @cache
    def get_subjects():
        return list((schema.Subject & schema.Widefield).fetch('subject_name'))

    @cache
    def get_subject_counts():
        rows = (schema.Widefield * schema.Dataset * schema.Session).fetch(
            'subject_name', as_dict=True)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.groupby('subject_name').size().reset_index(name='n_recordings')

    subjects = get_subjects()
    if not subjects:
        st.info('No widefield recordings found.')
        return

    bar_data = get_subject_counts()
    if not bar_data.empty:
        bar = (
            alt.Chart(bar_data).mark_bar().encode(
                x=alt.X('subject_name:N', sort='-y', title='Subject'),
                y=alt.Y('n_recordings:Q', title='Recordings'),
                tooltip=['subject_name:N', 'n_recordings:Q'],
            )
            .properties(height=180)
        )
        click_sel = alt.selection_point(name='subj_click', fields=['subject_name'], on='click')
        bar_event = st.altair_chart(
            bar.add_params(click_sel), on_select='rerun', key='wf_bar', width='stretch')
        pts = (bar_event.selection or {}).get('subj_click', [])
        if pts and pts[0].get('subject_name') in subjects:
            st.session_state['wf_subject'] = pts[0]['subject_name']

    subject = st.selectbox('Subject', subjects, index=None, key='wf_subject')
    if not subject:
        return

    @cache
    def get_sessions(subject_name):
        rows = (schema.Widefield * schema.Dataset * schema.Session
                & dict(subject_name=subject_name)).fetch(
            'session_name', 'dataset_name', 'n_channels', 'n_frames', 'frame_rate',
            as_dict=True)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        param_counts, stack_counts = [], []
        for r in rows:
            wkey = dict(subject_name=subject_name,
                        session_name=r['session_name'],
                        dataset_name=r['dataset_name'])
            param_counts.append(len(WfieldParameters & wkey))
            stack_counts.append(len(WfieldStack & wkey))
        df['n_params'] = param_counts
        df['n_stacks'] = stack_counts
        # other datasets in the same session
        all_ds = (schema.Dataset & dict(subject_name=subject_name)).fetch(
            'session_name', 'dataset_name', as_dict=True)
        from collections import defaultdict
        session_datasets = defaultdict(list)
        for d in all_ds:
            session_datasets[d['session_name']].append(d['dataset_name'])
        df['other_datasets'] = df.apply(
            lambda r: ', '.join(
                n for n in session_datasets[r['session_name']] if n != r['dataset_name']
            ), axis=1)
        return df.sort_values('session_name', ascending=False).reset_index(drop=True)

    sessions = get_sessions(subject)
    if sessions.empty:
        st.write('No widefield sessions found.')
        return

    def _highlight_analysed(row):
        color = _ANALYSED_COLOR if row['n_stacks'] > 0 else ''
        return [f'background-color: {color}'] * len(row)

    styled = sessions.style.apply(_highlight_analysed, axis=1)

    event = st.dataframe(
        styled, hide_index=True, width='stretch',
        on_select='rerun', selection_mode='single-row', key='wf_sessions_table',
    )
    rows_sel = (event.selection or {}).get('rows', [])
    if not rows_sel or rows_sel[0] >= len(sessions):
        st.caption('Click a row to select a session. Green rows have completed analyses.')
        return

    row = sessions.iloc[rows_sel[0]]
    sel_key = dict(subject_name=subject,
                   session_name=row['session_name'],
                   dataset_name=row['dataset_name'])
    st.session_state['wf_selected_key'] = sel_key

    st.divider()
    st.write(f"**{row['session_name']}** / {row['dataset_name']} — "
             f"{int(row['n_channels'])} ch, {int(row['n_frames'])} frames, "
             f"{float(row['frame_rate']):.1f} Hz")



    @cache
    def get_params(subject_name, session_name, dataset_name):
        rows = (WfieldParameters & dict(
            subject_name=subject_name,
            session_name=session_name,
            dataset_name=dataset_name,
        )).fetch(as_dict=True)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        stack_ids = set((WfieldStack & dict(
            subject_name=subject_name,
            session_name=session_name,
            dataset_name=dataset_name,
        )).fetch('wfield_analysis_id'))
        df['populated'] = df['wfield_analysis_id'].isin(stack_ids)
        display_cols = ['wfield_analysis_id', 'motion_correction', 'motion_conv_kernel',
                        'decomposition', 'k', 'atlas', 'functional_channel',
                        'mask_std_threshold', 'populated']
        return df[[c for c in display_cols if c in df.columns]]

    existing = get_params(sel_key['subject_name'], sel_key['session_name'],
                          sel_key['dataset_name'])

    if not existing.empty:
        def _highlight_populated(row):
            color = _ANALYSED_COLOR if row['populated'] else ''
            return [f'background-color: {color}'] * len(row)

        st.dataframe(existing.style.apply(_highlight_populated, axis=1),
                     hide_index=True, width='stretch')

        unpopulated = existing[~existing['populated']]
        if not unpopulated.empty:
            st.write('**Populate**')
            pop_cols = st.columns(min(4, len(unpopulated)))
            for col, (_, prow) in zip(pop_cols, unpopulated.iterrows()):
                aid = int(prow['wfield_analysis_id'])
                if col.button(f'Run analysis_id={aid}', key=f'wf_pop_{aid}'):
                    pop_key = dict(**sel_key, wfield_analysis_id=aid)
                    with st.spinner(f'Populating wfield_analysis_id={aid}…'):
                        try:
                            WfieldStack.populate(pop_key, display_progress=False)
                            get_params.clear()
                            get_sessions.clear()
                            get_table_counts.clear()
                            st.success('Done.')
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
    else:
        st.info('No WfieldParameters entries yet for this session.')

    next_id = 0 if existing.empty else int(existing['wfield_analysis_id'].max()) + 1
    with st.expander('Add WfieldParameters', expanded=existing.empty):
        col1, col2 = st.columns(2)
        analysis_id   = col1.number_input('wfield_analysis_id', value=next_id, step=1, min_value=0)
        motion_corr   = col1.selectbox('motion_correction', ['ecc', '2d', 'normcorr', 'none'])
        decomposition = col1.selectbox('decomposition', ['approx', 'pmd'])
        atlas         = col1.text_input('atlas', value='dorsal_cortex')
        k             = col2.number_input('k (SVD components)', value=200, step=10, min_value=1)
        func_ch       = col2.number_input('functional_channel', value=0, step=1, min_value=0)
        nframes_dec   = col2.number_input('nframes_decimate', value=15, step=1, min_value=1)
        chunk_size    = col2.number_input('chunk_size', value=512, step=64, min_value=64)
        mask_thresh   = col2.number_input('mask_std_threshold (0 = NULL)', value=0, step=1,
                                          min_value=0)
        st.caption('motion_conv_kernel — leave both at 0 for none')
        kc1, kc2 = st.columns(2)
        conv_k0 = kc1.number_input('kernel[0]', value=0.0, step=0.5, format='%.2f',
                                    label_visibility='visible')
        conv_k1 = kc2.number_input('kernel[1]', value=0.0, step=0.5, format='%.2f',
                                    label_visibility='visible')
        if st.button('Add WfieldParameters', type='primary'):
            new_row = dict(
                **sel_key,
                wfield_analysis_id=int(analysis_id),
                motion_correction=motion_corr,
                decomposition=decomposition,
                atlas=atlas,
                k=int(k),
                functional_channel=int(func_ch),
                nframes_decimate=int(nframes_dec),
                chunk_size=int(chunk_size),
            )
            if mask_thresh > 0:
                new_row['mask_std_threshold'] = int(mask_thresh)
            if conv_k0 != 0.0 or conv_k1 != 0.0:
                new_row['motion_conv_kernel'] = np.array([conv_k0, conv_k1])
            try:
                WfieldParameters.insert1(new_row)
                get_params.clear()
                get_sessions.clear()
                get_table_counts.clear()
                st.success(f'Inserted wfield_analysis_id={analysis_id}.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))




@st.fragment
def _results_tab(schema, WfieldParameters, WfieldStack):
    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    st.write(f"**{sel_key['session_name']}** / {sel_key['dataset_name']}")
    cache = _tab_cache_factory('refresh_results')
    _refresh_button('refresh_results')

    @cache
    def get_stack_entries(subject_name, session_name, dataset_name):
        return (WfieldStack & dict(
            subject_name=subject_name,
            session_name=session_name,
            dataset_name=dataset_name,
        )).fetch('wfield_analysis_id', as_dict=False)

    analysis_ids = get_stack_entries(
        sel_key['subject_name'], sel_key['session_name'], sel_key['dataset_name'])

    if not len(analysis_ids):
        st.info('No WfieldStack results yet. Add parameters and run the analysis first.')
        return

    analysis_id = st.selectbox('wfield_analysis_id', analysis_ids)
    stack_key = dict(**sel_key, wfield_analysis_id=int(analysis_id))

    @cache
    def get_mean_proj(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        return (WfieldStack & key).fetch1('mean_proj')

    @cache
    def get_projections_raw(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        rows = (WfieldStack.Projection & key).fetch('proj_name', 'proj', as_dict=True)
        return [(r['proj_name'], np.asarray(r['proj'])) for r in rows]

    @cache
    def get_motion(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        return (WfieldStack & key).fetch1('motion_correction')

    cl1, cl2 = st.columns(2)
    lo_pct = cl1.slider('Display min (percentile)', 0.0, 49.0, 2.0, step=0.5,
                         key='res_lo_pct')
    hi_pct = cl2.slider('Display max (percentile)', 51.0, 100.0, 99.0, step=0.5,
                         key='res_hi_pct')

    with st.spinner('Loading projections…'):
        mean_proj = get_mean_proj(
            sel_key['subject_name'], sel_key['session_name'],
            sel_key['dataset_name'], int(analysis_id))
        proj_rows_raw = get_projections_raw(
            sel_key['subject_name'], sel_key['session_name'],
            sel_key['dataset_name'], int(analysis_id))

    proj_names = sorted({pn for pn, _ in proj_rows_raw})
    proj_tabs = st.tabs(['mean_proj'] + proj_names)

    with proj_tabs[0]:
        arr = _normalize_with_pct(mean_proj.squeeze(), lo_pct, hi_pct)
        h, w = arr.shape[:2]
        st.altair_chart(_altair_image(_to_base64(arr), w, h, title='mean projection'),
                        width='content')

    for tab, pname in zip(proj_tabs[1:], proj_names):
        with tab:
            raw = next(p for n, p in proj_rows_raw if n == pname)
            arr = _normalize_with_pct(raw, lo_pct, hi_pct)
            h, w = arr.shape[:2]
            st.altair_chart(_altair_image(_to_base64(arr), w, h, title=pname),
                            width='content')

    with st.expander('Motion correction shifts'):
        try:
            motion = get_motion(
                sel_key['subject_name'], sel_key['session_name'],
                sel_key['dataset_name'], int(analysis_id))
            if motion is not None:
                import altair as alt
                mot = np.asarray(motion)
                # shape from make(): (3, nchannels, nframes) — [x/y/r, channel, frame]
                nframes_mot = mot.shape[-1]
                frames = np.arange(nframes_mot)
                df_mot = pd.DataFrame({
                    'frame': np.concatenate([frames, frames]),
                    'shift (px)': np.concatenate([mot[0, 0, :], mot[1, 0, :]]),
                    'axis': ['x'] * nframes_mot + ['y'] * nframes_mot,
                })
                st.altair_chart(
                    alt.Chart(df_mot).mark_line(opacity=0.8).encode(
                        x=alt.X('frame:Q', title='Frame'),
                        y=alt.Y('shift (px):Q'),
                        color=alt.Color('axis:N', scale=alt.Scale(
                            domain=['x', 'y'], range=['#1f77b4', '#ff7f0e'])),
                        tooltip=['frame:Q', 'shift (px):Q', 'axis:N'],
                    ).properties(height=200).interactive(),
                    width='stretch')
            else:
                st.info('No motion correction data (motion_correction = none).')
        except Exception as exc:
            st.warning(f'Could not load motion data: {exc}')


@st.cache_data(show_spinner='Loading SVD components…')
def _get_svd_arrays(subject_name, session_name, dataset_name, analysis_id,
                    WfieldParameters, WfieldStack, Widefield):
    key = dict(subject_name=subject_name, session_name=session_name,
               dataset_name=dataset_name, wfield_analysis_id=analysis_id)
    par = (WfieldParameters & key).fetch1()
    nchannels = (Widefield & key).fetch1('n_channels')
    res = (WfieldStack & key).load()
    if 'SVTcorr' in res:
        SVT = np.array(res['SVTcorr'])
    else:
        SVT = np.array(res['SVT'])[:, par['functional_channel']::nchannels]
    return np.array(res['U']), SVT


@st.fragment
def _frame_explorer(U, SVT):
    import time
    h, w = U.shape[:2]
    nframes = SVT.shape[1]
    Uflat = U.reshape(-1, U.shape[-1])

    def _render_url(idx, scale):
        import matplotlib.cm as cm
        frame = (Uflat @ SVT[:, idx]).reshape(h, w).astype(np.float32)
        norm = np.clip((frame + scale) / (2 * scale), 0, 1)
        rgb = (cm.RdBu_r(norm)[:, :, :3] * 255).astype(np.uint8)
        return _to_base64(rgb)

    playing   = st.session_state.get('wf_playing', False)
    frame_idx = min(st.session_state.get('wf_frame_idx', 0), nframes - 1)

    # Keep slider widget in sync with the current frame during playback
    if playing:
        st.session_state['wf_frame_slider'] = frame_idx

    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    slider_val = c1.slider('Frame', 0, nframes - 1, frame_idx, key='wf_frame_slider')
    fps_val    = c2.number_input('FPS', min_value=1, max_value=60, value=10, key='wf_fps')
    scale_val  = c3.number_input('Scale', min_value=0.01, max_value=5.0, value=0.15,
                                  step=0.01, format='%.2f', key='wf_scale')

    # Stable key lets React reconcile in-place instead of clearing + remounting
    st.altair_chart(
        _altair_image(_render_url(frame_idx, scale_val), w, h,
                      title=f'Frame {frame_idx}', width=500,
                      height=max(200, int(500 * h / w))),
        key='wf_frame_chart',
        width='content',
    )

    if not playing:
        if c4.button('▶ Play', key='wf_play_btn'):
            st.session_state['wf_playing'] = True
            st.session_state['wf_frame_idx'] = slider_val
            st.rerun(scope='fragment')
        else:
            st.session_state['wf_frame_idx'] = slider_val
    else:
        c4.button('▶ Play', key='wf_play_btn', disabled=True)
        if c5.button('■ Stop', key='wf_stop_btn'):
            st.session_state['wf_playing'] = False
        else:
            time.sleep(1.0 / fps_val)
            nxt = frame_idx + 1
            if nxt < nframes:
                st.session_state['wf_frame_idx'] = nxt
                st.rerun(scope='fragment')
            else:
                st.session_state['wf_playing'] = False




@st.fragment
def _stack_explorer_tab(schema, WfieldParameters, WfieldStack):
    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return
    cache = _tab_cache_factory('refresh_explorer')
    _refresh_button('refresh_explorer')

    @cache
    def get_stack_ids_exp(subject_name, session_name, dataset_name):
        return list((WfieldStack & dict(
            subject_name=subject_name, session_name=session_name,
            dataset_name=dataset_name,
        )).fetch('wfield_analysis_id', as_dict=False))

    analysis_ids = get_stack_ids_exp(
        sel_key['subject_name'], sel_key['session_name'], sel_key['dataset_name'])
    if not analysis_ids:
        st.info('No WfieldStack results yet.')
        return

    analysis_id = st.selectbox('wfield_analysis_id', analysis_ids, key='wf_exp_aid')
    U, SVT = _get_svd_arrays(
        sel_key['subject_name'], sel_key['session_name'],
        sel_key['dataset_name'], int(analysis_id),
        WfieldParameters, WfieldStack, schema.Widefield)
    _frame_explorer(U, SVT)




@st.fragment
def _window_mask_tab(schema, WfieldParameters, WfieldStack):
    from .pluginschema import ImagingWindow
    import altair as alt
    from skimage.measure import CircleModel

    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    widefield_key = dict(subject_name=subject_name,
                         session_name=session_name,
                         dataset_name=dataset_name)
    cache = _tab_cache_factory('refresh_window')
    _refresh_button('refresh_window')

    @cache
    def get_imaging_window(subject_name, session_name, dataset_name):
        rows = (ImagingWindow & widefield_key).fetch(as_dict=True)
        return rows[0] if rows else None

    existing = get_imaging_window(subject_name, session_name, dataset_name)
    if existing:
        cp = existing.get('circle_parameters')
        res = existing.get('resolution')
        info_parts = []
        if cp is not None:
            cp = np.asarray(cp)
            info_parts.append(f'center=({cp[0]:.0f}, {cp[1]:.0f})  radius={cp[2]:.1f} px')
        if res is not None:
            info_parts.append(f'resolution={res:.4f} mm/px')
        st.success('ImagingWindow entry exists — ' + ('  |  '.join(info_parts) if info_parts else ''))


    @cache
    def get_stack_ids(subject_name, session_name, dataset_name):
        return list((WfieldStack & widefield_key).fetch('wfield_analysis_id', as_dict=False))

    analysis_ids = get_stack_ids(subject_name, session_name, dataset_name)

    col_sel, col_win = st.columns([2, 1])
    windowsize = col_win.number_input('Window size (mm)', value=5.0, step=0.5, min_value=0.1)

    if analysis_ids:
        analysis_id = col_sel.selectbox('Reference image (wfield_analysis_id)', analysis_ids,
                                        key='wf_mask_aid')
    else:
        col_sel.info('No WfieldStack results — cannot display a reference image.')
        return

    @cache
    def get_image(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        mp = (WfieldStack & key).fetch1('mean_proj')
        arr = np.squeeze(mp)
        if arr.ndim == 3:
            arr = arr[0]
        return _normalize_image(arr)

    with st.spinner('Loading image…'):
        arr = get_image(subject_name, session_name, dataset_name, int(analysis_id))

    h, w = arr.shape


    scope    = f'{subject_name}|{session_name}|{dataset_name}'
    pts_key  = f'wf_mask_pts_{scope}'
    last_key = f'wf_mask_last_{scope}'
    if pts_key not in st.session_state:
        # Pre-populate from existing DB entry so the circle is shown immediately
        if existing and existing.get('points') is not None:
            st.session_state[pts_key] = np.asarray(existing['points']).tolist()
        else:
            st.session_state[pts_key] = []

    points = st.session_state[pts_key]

    # Downsample PNG for display — domain stays [0,w]×[0,h]
    _display_size = 256
    _scale = _display_size / max(h, w)
    if _scale < 1.0:
        dh, dw = max(1, int(h * _scale)), max(1, int(w * _scale))
        display_arr = arr[::max(1, h // dh), ::max(1, w // dw)]
    else:
        display_arr = arr
    # Fit circle whenever we have ≥ 3 points
    circlepar = None
    if len(points) >= 3:
        cmod = CircleModel()
        cmod.estimate(np.array(points, dtype=float))
        circlepar = cmod.params  # (xc, yc, r)

    # Bake circle outline into display image using PIL
    dh_actual, dw_actual = display_arr.shape
    scale_x_d = dw_actual / w
    scale_y_d = dh_actual / h
    disp_rgb = np.stack([display_arr] * 3, axis=-1)
    if circlepar is not None:
        from PIL import Image as PILImage, ImageDraw
        xc_d = circlepar[0] * scale_x_d
        yc_d = circlepar[1] * scale_y_d
        r_d = circlepar[2] * (scale_x_d + scale_y_d) / 2
        pil_d = PILImage.fromarray(disp_rgb)
        ImageDraw.Draw(pil_d).ellipse(
            [xc_d - r_d, yc_d - r_d, xc_d + r_d, yc_d + r_d],
            outline=(255, 60, 60), width=2)
        disp_rgb = np.array(pil_d)
    url = _to_base64(disp_rgb)


    x_scale = alt.Scale(domain=[0, w])
    y_scale = alt.Scale(domain=[h, 0])

    img_layer = (
        alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': w, 'y2': h, 'url': url}]))
        .mark_image(aspect=False)
        .encode(
            x=alt.X('x:Q', scale=x_scale, axis=None),
            y=alt.Y('y:Q', scale=y_scale, axis=None),
            x2='x2:Q', y2='y2:Q', url='url:N',
            tooltip=alt.value(None),
        )
    )

    step = max(4, min(w, h) // 64)
    xs, ys = np.arange(0, w, step, dtype=float), np.arange(0, h, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    grid_df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})
    click_sel = alt.selection_point(name='img_click', on='click', nearest=True,
                                    encodings=['x', 'y'])
    grid_layer = (
        alt.Chart(grid_df)
        .mark_point(opacity=0, size=step * step * 4)
        .encode(x=alt.X('x:Q', scale=x_scale, axis=None),
                y=alt.Y('y:Q', scale=y_scale, axis=None))
        .add_params(click_sel)
    )

    layers = [img_layer]

    if points:
        pts_df = pd.DataFrame(points, columns=['x', 'y'])
        layers.append(
            alt.Chart(pts_df).mark_point(color='lime', size=80, filled=True,
                                         strokeWidth=1.5, stroke='black').encode(
                x=alt.X('x:Q', scale=x_scale, axis=None),
                y=alt.Y('y:Q', scale=y_scale, axis=None),
                tooltip=alt.value(None),
            )
        )

    layers.append(grid_layer)
    chart = (alt.layer(*layers)
             .properties(width=400, height=400,
                         title='Click the boundary of the imaging window')
             .interactive())

    event = st.altair_chart(chart, on_select='rerun', key='wf_circle_chart', width='content')

    sel = (event.selection or {}).get('img_click', [])
    if sel:
        x_click = sel[0].get('x')
        y_click = sel[0].get('y')
        if x_click is not None and y_click is not None:
            new_pt = [float(x_click), float(y_click)]
            if new_pt != st.session_state.get(last_key):
                st.session_state[last_key] = new_pt
                points.append(new_pt)
                st.session_state[pts_key] = points
                st.rerun()


    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button('Clear all', key='wf_mask_clear'):
        st.session_state[pts_key] = []
        st.session_state.pop(last_key, None)
        st.rerun()

    if c2.button('Undo last', key='wf_mask_undo', disabled=len(points) == 0):
        st.session_state[pts_key] = points[:-1]
        st.session_state.pop(last_key, None)
        st.rerun()

    if circlepar is not None:
        xc, yc, r = circlepar
        resolution = windowsize / (2 * r) if r > 0 else None
        c3.metric('Center (col, row)', f'({xc:.0f}, {yc:.0f})')
        c4.metric('Radius (px)', f'{r:.1f}')
        if resolution:
            c5.metric('Resolution (mm/px)', f'{resolution:.4f}')

        if st.button('Save to ImagingWindow', type='primary', key='wf_mask_save'):
            ImagingWindow.insert1(dict(
                **widefield_key,
                window_size=float(windowsize),
                resolution=float(resolution) if resolution else None,
                points=np.array(points, dtype=float),
                circle_parameters=np.array([xc, yc, r], dtype=float),
            ), replace=True)
            get_imaging_window.clear()
            st.success(f'Saved ImagingWindow — center=({xc:.0f}, {yc:.0f}), '
                       f'radius={r:.1f} px, resolution={resolution:.4f} mm/px.')
            st.rerun()
    else:
        st.caption(f'Click at least 3 points on the window boundary ({len(points)} so far).')

    if existing:
        with st.expander('Delete ImagingWindow entry'):
            st.warning('Permanently removes this ImagingWindow from the database.')
            if st.button('Confirm delete', type='primary', key='wf_mask_delete'):
                (ImagingWindow & widefield_key).delete(safemode=False)
                get_imaging_window.clear()
                st.session_state[pts_key] = []
                st.session_state.pop(last_key, None)
                st.rerun()




@st.fragment
def _imaging_reference_tab(schema, WfieldParameters, WfieldStack):
    import altair as alt
    from .pluginschema import ImagingWindow, ImagingReference, TwoPhotonReferenceAlignment

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
    from .utils import build_alignment_transform, warp_image
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


# Atlas alignment helpers

def _render_atlas_overlay(mean_proj, ccf_regions_dict, reference_point,
                          resolution, rotation=0., scale=1., ratio=1.,
                          circle_params=None, mirror=False):
    """Render atlas contours on mean_proj. Returns base64 PNG URL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd_
    from .utils import build_atlas_transform, transform_atlas_regions

    regions = pd_.DataFrame(ccf_regions_dict)
    M = build_atlas_transform(
        bregma_xy=reference_point,
        resolution=float(resolution),
        rotation=float(rotation),
        scale=float(scale),
        ratio=float(ratio),
        mirror=bool(mirror),
    )
    transformed = transform_atlas_regions(regions, M)

    mean_proj = np.squeeze(mean_proj)
    img_h, img_w = mean_proj.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5 * img_h / img_w), dpi=100, facecolor='k')
    ax.set_facecolor('k')
    is_rgb = mean_proj.ndim == 3 and mean_proj.shape[2] == 3
    if is_rgb:
        ax.imshow(mean_proj, origin='upper', aspect='auto')
    else:
        ax.imshow(mean_proj, cmap='gray', origin='upper', aspect='auto')
    for _, row in transformed.iterrows():
        ax.plot(np.asarray(row['left_x']),  np.asarray(row['left_y']),  '-', color='orange', lw=0.8)
        ax.plot(np.asarray(row['right_x']), np.asarray(row['right_y']), '-', color='orange', lw=0.8)
    ax.plot(reference_point[0], reference_point[1], 'r+', markersize=14, markeredgewidth=2)
    if circle_params is not None:
        import matplotlib.patches as mpatches
        xc, yc, r = circle_params
        ax.add_patch(mpatches.Circle((xc, yc), r, fill=False,
                                      edgecolor='black', linewidth=2))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    fig.tight_layout(pad=0.05)

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='k')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _render_img_with_landmarks(arr, points, dot_color=(0, 140, 255)):
    """Render float/uint image as base64 PNG with coloured landmark dots."""
    from PIL import Image as PILImage, ImageDraw
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr if arr.dtype == np.uint8 else _normalize_with_pct(arr).reshape(arr.shape[:2] + (3,)).astype(np.uint8)
    else:
        norm = _normalize_with_pct(arr.astype(float))
        rgb = np.stack([norm] * 3, axis=-1).astype(np.uint8)
    pil = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for x, y in points:
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=dot_color, outline=(255, 255, 255), width=1)
    import io
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _altair_clickable(url, img_w, img_h, width=380, title=''):
    """Altair image chart with a transparent grid overlay for click detection."""
    import altair as alt
    x_sc = alt.Scale(domain=[0, img_w])
    y_sc = alt.Scale(domain=[img_h, 0])
    step = max(4, min(img_w, img_h) // 64)
    xs, ys = np.arange(0, img_w, step, dtype=float), np.arange(0, img_h, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    height = max(200, int(width * img_h / img_w))
    img_layer = (
        alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': img_w, 'y2': img_h, 'url': url}]))
        .mark_image(aspect=False)
        .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                y=alt.Y('y:Q', scale=y_sc, axis=None),
                x2='x2:Q', y2='y2:Q', url='url:N',
                tooltip=alt.value(None))
    )
    click_sel = alt.selection_point(name='pt_click', on='click',
                                    nearest=True, encodings=['x', 'y'])
    grid_layer = (
        alt.Chart(pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()}))
        .mark_point(opacity=0, size=step * step * 4)
        .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                y=alt.Y('y:Q', scale=y_sc, axis=None))
        .add_params(click_sel)
    )
    return (alt.layer(img_layer, grid_layer)
            .properties(width=width, height=height, title=title)
            .interactive())


def _extract_click(event):
    """Return (x, y) from an Altair on_select event, or None."""
    sel = (event.selection or {}).get('pt_click', [])
    if sel:
        x, y = sel[0].get('x'), sel[0].get('y')
        if x is not None and y is not None:
            return float(x), float(y)
    return None


# Atlas alignment tab 

@st.fragment
def _atlas_alignment_tab(schema, WfieldParameters, WfieldStack):
    from .pluginschema import WidefieldAtlas, WidefieldAtlasTransform, WidefieldResponse

    sel_key = st.session_state.get('wf_selected_key')
    if not sel_key:
        st.info('Select a session in the Sessions & Parameters tab first.')
        return

    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    cache = _tab_cache_factory('refresh_atlas')
    _refresh_button('refresh_atlas')

    @cache
    def get_atlases():
        return list((WidefieldAtlas()).fetch('atlas_name'))

    @cache
    def get_stack_ids_at(subject_name, session_name, dataset_name):
        return list((WfieldStack & dict(subject_name=subject_name,
                                        session_name=session_name,
                                        dataset_name=dataset_name)
                     ).fetch('wfield_analysis_id'))

    @cache
    def get_mean_proj_at(subject_name, session_name, dataset_name, analysis_id):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, wfield_analysis_id=analysis_id)
        mp = np.squeeze((WfieldStack & key).fetch1('mean_proj'))
        return (mp[0] if mp.ndim == 3 else mp).astype(np.float32)

    @cache
    def get_response_projections(subject_name, session_name, dataset_name):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name)
        rows = (WidefieldResponse.Projection & key).fetch(
            'stim_name', 'proj_name', as_dict=True)
        return [(r['stim_name'], r['proj_name']) for r in rows]

    @cache
    def get_response_proj_image(subject_name, session_name, dataset_name,
                                response_name, proj_name):
        key = dict(subject_name=subject_name, session_name=session_name,
                   dataset_name=dataset_name, stim_name=response_name,
                   proj_name=proj_name)
        arr = np.squeeze(np.asarray((WidefieldResponse.Projection & key).fetch1('proj')))
        return arr  # preserve original dtype (uint8 RGB or float)

    @cache
    def get_atlas_row(atlas_name):
        return (WidefieldAtlas & dict(atlas_name=atlas_name)).fetch1()

    @cache
    def get_existing_transforms(subject_name, session_name, dataset_name, atlas_name):
        return (WidefieldAtlasTransform & dict(
            subject_name=subject_name, session_name=session_name,
            dataset_name=dataset_name, atlas_name=atlas_name,
        )).fetch(as_dict=True)

    @cache
    def get_imaging_window(subject_name, session_name, dataset_name):
        from .pluginschema import ImagingWindow
        rows = (ImagingWindow & dict(subject_name=subject_name,
                                     session_name=session_name,
                                     dataset_name=dataset_name)).fetch(as_dict=True)
        return rows[0] if rows else None

    atlas_names = get_atlases()
    if not atlas_names:
        st.warning('No WidefieldAtlas entries. '
                   'Create one with `WidefieldAtlas().from_wfield(name)` '
                   'or `.from_allensdk(name)`.')
        return

    stack_ids = get_stack_ids_at(subject_name, session_name, dataset_name)

    hc1, hc2, hc3 = st.columns(3)
    atlas_name   = hc1.selectbox('Atlas', atlas_names, key='at_atlas')
    analysis_id  = hc2.selectbox('WfieldStack', stack_ids if stack_ids else ['—'],
                                  disabled=not stack_ids, key='at_aid')
    existing     = get_existing_transforms(subject_name, session_name, dataset_name, atlas_name)
    # Default to the latest saved transform so its parameters load on open;
    # bump the number to create a new one.
    default_id   = max((r['atlas_transform_id'] for r in existing), default=0) if existing else 0
    transform_id = int(hc3.number_input('atlas_transform_id', value=default_id,
                                         step=1, min_value=0, key='at_tid'))
    if existing:
        disp_cols = [k for k in existing[0]
                     if k not in ('landmarks', 'landmarks_match',
                                  'transform_matrix', 'transform_matrix_inverse')]
        with st.expander(f'{len(existing)} existing transform(s)'):
            st.dataframe(pd.DataFrame(existing)[disp_cols], hide_index=True)

    # Reference image: mean_proj (if stack available) or any WidefieldResponse projection
    _DEFAULT_IMG = 'mean_proj (WfieldStack)'
    response_projs = get_response_projections(subject_name, session_name, dataset_name)
    img_options = ([_DEFAULT_IMG] if stack_ids else []) + [
        f'{rn} → {pn}' for rn, pn in response_projs]
    if not img_options:
        st.info('No reference image available. Run a WfieldStack analysis or add a '
                'WidefieldResponse projection first.')
        return
    sel_img = st.selectbox('Reference image', img_options, key='at_ref_img',
                            help='Use a WidefieldResponse projection (e.g. retinotopy phase map) '
                                 'as the background for atlas alignment.')

    with st.spinner('Loading…'):
        if sel_img == _DEFAULT_IMG:
            ref_image = get_mean_proj_at(subject_name, session_name, dataset_name,
                                         int(analysis_id))
        else:
            rn, pn = response_projs[img_options.index(sel_img) - (1 if stack_ids else 0)]
            ref_image = get_response_proj_image(subject_name, session_name, dataset_name,
                                                rn, pn)
        atlas_row      = get_atlas_row(atlas_name)
        imaging_window = get_imaging_window(subject_name, session_name, dataset_name)

    manual_tab, landmarks_tab = st.tabs(['Manual', 'Landmarks'])
    with manual_tab:
        _atlas_manual_subtab(sel_key, atlas_name, transform_id,
                              ref_image, atlas_row, imaging_window,
                              WidefieldAtlasTransform, get_existing_transforms)
    with landmarks_tab:
        _atlas_landmarks_subtab(sel_key, atlas_name, transform_id,
                                ref_image, atlas_row,
                                WidefieldAtlasTransform, get_existing_transforms)


def _atlas_manual_subtab(sel_key, atlas_name, transform_id,
                          mean_proj, atlas_row, imaging_window,
                          WidefieldAtlasTransform, get_existing_transforms):
    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    img_h, img_w = mean_proj.shape[:2]
    # transform_id is part of the scope so picking a saved transform reloads its parameters
    scope = f'{subject_name}|{session_name}|{dataset_name}|{atlas_name}|{transform_id}'

    # Drain pending bregma click into slider state before widgets render
    for k in ('rx', 'ry'):
        pkey = f'at_m_{scope}_{k}_pend'
        if pkey in st.session_state:
            st.session_state[f'at_m_{scope}_{k}'] = st.session_state.pop(pkey)

    # Default resolution: from ImagingWindow if available, else 0.025
    iw_res = float(imaging_window['resolution']) if (
        imaging_window and imaging_window.get('resolution') is not None) else 0.025
    iw_circle = (np.asarray(imaging_window['circle_parameters']).ravel()
                 if imaging_window and imaging_window.get('circle_parameters') is not None
                 else None)

    # Pre-populate from DB on first load
    if f'at_m_{scope}_init' not in st.session_state:
        saved = next((r for r in get_existing_transforms(
                          subject_name, session_name, dataset_name, atlas_name)
                      if r['atlas_transform_id'] == transform_id
                      and r['transform_type'] == 'manual'), None)
        if saved:
            rp = np.asarray(saved['reference_point']).ravel()
            st.session_state[f'at_m_{scope}_rx']    = float(rp[0])
            st.session_state[f'at_m_{scope}_ry']    = float(rp[1])
            st.session_state[f'at_m_{scope}_rot']    = float(saved.get('rotation') or 0.)
            st.session_state[f'at_m_{scope}_scale']  = float(saved.get('scale')    or 1.)
            st.session_state[f'at_m_{scope}_ratio']  = float(saved.get('ratio')    or 1.)
            st.session_state[f'at_m_{scope}_res']    = float(saved.get('resolution') or iw_res)
            st.session_state[f'at_m_{scope}_mirror'] = bool(saved.get('mirror') or False)
        else:
            st.session_state[f'at_m_{scope}_res'] = iw_res
        st.session_state[f'at_m_{scope}_init'] = True

    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    resolution = c1.number_input('Resolution (mm/px)', value=iw_res, step=0.001,
                                  format='%.4f', key=f'at_m_{scope}_res',
                                  help='mm per widefield pixel')
    rotation   = c2.slider('Rotation (°)', -180., 180., 0., step=0.5,
                             key=f'at_m_{scope}_rot')
    scale      = c3.slider('Scale',          0.1,   5.0, 1., step=0.05,
                             key=f'at_m_{scope}_scale')
    ratio      = c4.slider('X/Y ratio',      0.5,   2.0, 1., step=0.01,
                             key=f'at_m_{scope}_ratio')
    mirror     = c5.checkbox('Mirror X', value=False, key=f'at_m_{scope}_mirror',
                              help='Flip the atlas left/right (for imaging setups where the x-axis is reversed)')

    ref_x = float(st.session_state.get(f'at_m_{scope}_rx', img_w / 2))
    ref_y = float(st.session_state.get(f'at_m_{scope}_ry', img_h / 2))
    st.caption(f'Bregma: ({ref_x:.0f}, {ref_y:.0f}) px — click the image to reposition')

    url = _render_atlas_overlay(mean_proj, atlas_row['ccf_regions'],
                                 [ref_x, ref_y], resolution, rotation, scale, ratio,
                                 circle_params=iw_circle, mirror=mirror)
    chart = _altair_clickable(url, img_w, img_h, width=460,
                               title='Click to set bregma · atlas contours overlaid')
    last_key = f'at_m_{scope}_last'
    event = st.altair_chart(chart, on_select='rerun',
                             key=f'at_m_{scope}_chart', width='content')
    click = _extract_click(event)
    if click and list(click) != st.session_state.get(last_key):
        st.session_state[last_key] = list(click)
        st.session_state[f'at_m_{scope}_rx_pend'] = click[0]
        st.session_state[f'at_m_{scope}_ry_pend'] = click[1]
        st.rerun(scope='fragment')

    if st.button('Save manual transform', type='primary', key=f'at_m_{scope}_save'):
        WidefieldAtlasTransform.insert1(dict(
            **sel_key,
            atlas_name=atlas_name,
            atlas_transform_id=transform_id,
            transform_type='manual',
            reference_point=[ref_x, ref_y],
            resolution=float(resolution),
            rotation=float(rotation),
            scale=float(scale),
            ratio=float(ratio),
            mirror=int(mirror),
        ), replace=True)
        get_existing_transforms.clear()
        st.success(f'Saved manual transform id={transform_id}.')
        st.rerun(scope='fragment')


def _atlas_landmarks_subtab(sel_key, atlas_name, transform_id,
                              mean_proj, atlas_row,
                              WidefieldAtlasTransform, get_existing_transforms):
    subject_name = sel_key['subject_name']
    session_name = sel_key['session_name']
    dataset_name = sel_key['dataset_name']
    img_h, img_w = mean_proj.shape[:2]
    proj         = atlas_row['projection']
    atlas_res    = float(atlas_row['resolution'])          # mm / atlas pixel
    ref_row_a, ref_col_a = atlas_row['reference_point']   # [row, col]
    atlas_h, atlas_w = proj.shape
    scope = f'{subject_name}|{session_name}|{dataset_name}|{atlas_name}|{transform_id}'

    wf_key   = f'at_l_{scope}_wf'
    at_key   = f'at_l_{scope}_at'
    wf_last  = f'at_l_{scope}_wf_last'
    at_last  = f'at_l_{scope}_at_last'

    # On first load of this transform, pre-populate points from a saved landmark transform.
    if f'{at_key}_init' not in st.session_state:
        st.session_state[f'{at_key}_init'] = True
        saved = next((r for r in get_existing_transforms(
                          subject_name, session_name, dataset_name, atlas_name)
                      if r['atlas_transform_id'] == transform_id
                      and r['transform_type'] == 'landmarks'), None)
        wf_init, at_init = [], []
        if saved and saved.get('landmarks') is not None \
                and saved.get('landmarks_match') is not None:
            lm  = pd.DataFrame(saved['landmarks'])        # atlas-space, mm from bregma (x, y)
            lmm = pd.DataFrame(saved['landmarks_match'])  # widefield pixels (x=col, y=row)
            at_init = [[float(x) / atlas_res + ref_col_a, float(y) / atlas_res + ref_row_a]
                       for x, y in zip(lm['x'], lm['y'])]
            wf_init = [[float(x), float(y)] for x, y in zip(lmm['x'], lmm['y'])]
        st.session_state[wf_key] = wf_init
        st.session_state[at_key] = at_init
    st.session_state.setdefault(wf_key, [])
    st.session_state.setdefault(at_key, [])

    wf_pts    = st.session_state[wf_key]    # [[col, row], …] in widefield pixels
    atlas_pts = st.session_state[at_key]    # [[col, row], …] in atlas pixels
    n_wf, n_at = len(wf_pts), len(atlas_pts)
    n_pairs   = min(n_wf, n_at)

    st.caption(f'Image landmarks: **{n_wf}**  ·  Atlas landmarks: **{n_at}**  ·  '
               f'Pairs: **{n_pairs}** (need ≥ 3)')

    lm_col1, lm_col2 = st.columns(2)

    with lm_col1:
        st.write('**Widefield image** — click to add landmark')
        url_wf = _render_img_with_landmarks(mean_proj, wf_pts, dot_color=(0, 140, 255))
        ev_wf = st.altair_chart(
            _altair_clickable(url_wf, img_w, img_h, width=340,
                              title='Widefield landmarks (blue)'),
            on_select='rerun', key=f'at_l_{scope}_wf_chart', width='content')
        click_wf = _extract_click(ev_wf)
        if click_wf and list(click_wf) != st.session_state.get(wf_last):
            st.session_state[wf_last] = list(click_wf)
            wf_pts.append(list(click_wf))
            st.session_state[wf_key] = wf_pts
            st.rerun(scope='fragment')

    with lm_col2:
        st.write('**Atlas projection** — click to add landmark')
        url_at = _render_img_with_landmarks(proj, atlas_pts, dot_color=(255, 140, 0))
        ev_at = st.altair_chart(
            _altair_clickable(url_at, atlas_w, atlas_h, width=340,
                              title='Atlas landmarks (orange)'),
            on_select='rerun', key=f'at_l_{scope}_at_chart', width='content')
        click_at = _extract_click(ev_at)
        if click_at and list(click_at) != st.session_state.get(at_last):
            st.session_state[at_last] = list(click_at)
            atlas_pts.append(list(click_at))
            st.session_state[at_key] = atlas_pts
            st.rerun(scope='fragment')

    bc1, bc2, bc3, bc4 = st.columns(4)
    if bc1.button('Clear image pts',  key=f'at_l_{scope}_clwf'):
        st.session_state[wf_key] = []; st.session_state.pop(wf_last, None)
        st.rerun(scope='fragment')
    if bc2.button('Undo image pt',    key=f'at_l_{scope}_unwf',  disabled=n_wf == 0):
        st.session_state[wf_key] = wf_pts[:-1]; st.rerun(scope='fragment')
    if bc3.button('Clear atlas pts',  key=f'at_l_{scope}_clat'):
        st.session_state[at_key] = []; st.session_state.pop(at_last, None)
        st.rerun(scope='fragment')
    if bc4.button('Undo atlas pt',    key=f'at_l_{scope}_unat',  disabled=n_at == 0):
        st.session_state[at_key] = atlas_pts[:-1]; st.rerun(scope='fragment')

    if n_pairs > 0:
        pairs_df = pd.DataFrame({
            '#':           range(1, n_pairs + 1),
            'wf_col':      [p[0] for p in wf_pts[:n_pairs]],
            'wf_row':      [p[1] for p in wf_pts[:n_pairs]],
            'atlas_col':   [p[0] for p in atlas_pts[:n_pairs]],
            'atlas_row':   [p[1] for p in atlas_pts[:n_pairs]],
            'atlas_x_mm':  [(p[0] - ref_col_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'atlas_y_mm':  [(p[1] - ref_row_a) * atlas_res for p in atlas_pts[:n_pairs]],
        })
        st.dataframe(pairs_df, hide_index=True)

    if n_pairs < 3:
        st.info(f'Add at least 3 matched pairs to preview the transform ({n_pairs} so far).')
        return

    # Build transform and render live preview
    try:
        from wfield import allen_transform_from_landmarks, allen_landmarks_to_image_space
        from .utils import transform_atlas_regions
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        lm_df = pd.DataFrame({
            'x':     [(p[0] - ref_col_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'y':     [(p[1] - ref_row_a) * atlas_res for p in atlas_pts[:n_pairs]],
            'name':  [f'lm{i}' for i in range(n_pairs)],
            'color': ['k'] * n_pairs,
        })
        lm_match_df = pd.DataFrame({
            'x':     [p[0] for p in wf_pts[:n_pairs]],
            'y':     [p[1] for p in wf_pts[:n_pairs]],
            'name':  [f'lm{i}' for i in range(n_pairs)],
            'color': ['k'] * n_pairs,
        })
        ref_offset = np.array([ref_col_a, ref_row_a], dtype=float)  # [x=col, y=row]
        lm_im  = allen_landmarks_to_image_space(lm_df.copy(), ref_offset, atlas_res)
        M_lm   = allen_transform_from_landmarks(lm_im, lm_match_df)
        T_res  = np.array([[1/atlas_res, 0, ref_col_a],
                            [0, 1/atlas_res, ref_row_a],
                            [0, 0, 1]], dtype=float)
        M_full = M_lm.params @ T_res
        transformed = transform_atlas_regions(pd.DataFrame(atlas_row['ccf_regions']), M_full)

        import io as _io
        fig, ax = plt.subplots(figsize=(5, 5 * img_h / img_w), dpi=100, facecolor='k')
        ax.set_facecolor('k')
        ax.imshow(mean_proj, cmap='gray', origin='upper', aspect='auto')
        for _, row in transformed.iterrows():
            rgb = row.get('allen_rgb')
            c = [v / 255 for v in rgb] if rgb is not None else 'cyan'
            ax.plot(np.asarray(row['left_x']),  np.asarray(row['left_y']),  '-', color=c, lw=0.8)
            ax.plot(np.asarray(row['right_x']), np.asarray(row['right_y']), '-', color=c, lw=0.8)
        for p in wf_pts[:n_pairs]:
            ax.plot(p[0], p[1], 'bo', markersize=8, markeredgecolor='w', markeredgewidth=1)
        ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0); ax.axis('off')
        fig.tight_layout(pad=0.05)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='k')
        plt.close(fig)
        buf.seek(0)
        prev_url = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
        st.altair_chart(_altair_image(prev_url, img_w, img_h,
                                      title='Transform preview', width=460),
                        width='content')

        if st.button('Save landmark transform', type='primary',
                      key=f'at_l_{scope}_save'):
            WidefieldAtlasTransform.insert1(dict(
                **sel_key,
                atlas_name=atlas_name,
                atlas_transform_id=transform_id,
                transform_type='landmarks',
                landmarks=lm_df.to_dict(orient='list'),
                landmarks_match=lm_match_df.to_dict(orient='list'),
            ), replace=True)
            get_existing_transforms.clear()
            st.success(f'Saved landmark transform id={transform_id} '
                       f'with {n_pairs} pairs.')
            st.rerun(scope='fragment')

    except Exception as exc:
        st.error(f'Transform preview failed: {exc}')


# Cell atlas tab

@st.fragment
def _cell_atlas_tab(schema):
    """Populate CellSegmentationAtlas and scatter ROI atlas positions across
    one or more sessions / subjects on the atlas contours."""
    from .pluginschema import CellSegmentationAtlas, WidefieldAtlas

    st.subheader('CellSegmentation Atlas')
    cache = _tab_cache_factory('refresh_cellatlas')
    _refresh_button('refresh_cellatlas')

    PK = ['subject_name', 'session_name', 'dataset_name', 'parameter_set_num',
          'ref_num', 'ref_session', 'ref_dataset', 'atlas_name', 'atlas_transform_id']

    @cache
    def get_pending_count():
        return len(CellSegmentationAtlas().key_source - CellSegmentationAtlas())

    @cache
    def get_populated_masters():
        rows = CellSegmentationAtlas().fetch(*PK, 'n_rois', 'n_in_atlas', as_dict=True)
        return pd.DataFrame(rows)

    @cache
    def get_rois(keys_tuple):
        keys = [dict(zip(PK, vals)) for vals in keys_tuple]
        rows = (CellSegmentationAtlas.ROI & keys).fetch(
            'subject_name', 'session_name', 'dataset_name', 'parameter_set_num',
            'plane_num', 'roi_num', 'atlas_x', 'atlas_y',
            'acronym', 'hemisphere', as_dict=True)
        return pd.DataFrame(rows)

    # CellSegmentation key columns shared with the Selection part table
    SEG_COLS = ['subject_name', 'session_name', 'dataset_name', 'parameter_set_num']
    ROI_COLS = SEG_COLS + ['plane_num', 'roi_num']

    @cache
    def get_selection_methods(seg_keys_tuple):
        try:
            from labdata.schema import CellSegmentation
            keys = [dict(zip(SEG_COLS, v)) for v in seg_keys_tuple]
            return sorted(set((CellSegmentation.Selection & keys).fetch('selection_method')))
        except Exception:
            return []

    @cache
    def get_selected_rois(seg_keys_tuple, method):
        from labdata.schema import CellSegmentation
        keys = [dict(zip(SEG_COLS, v)) for v in seg_keys_tuple]
        rows = (CellSegmentation.Selection & keys
                & f'selection_method="{method}"' & 'selection=1').fetch(*ROI_COLS, as_dict=True)
        return pd.DataFrame(rows, columns=ROI_COLS)

    # Populate
    try:
        pending = get_pending_count()
    except Exception as exc:
        st.error(f'Could not query key_source: {exc}')
        pending = 0
    masters = get_populated_masters()

    pc1, pc2 = st.columns([3, 1])
    pc1.caption(f'{len(masters)} populated · {pending} pending')
    if pc2.button('Populate', disabled=(pending == 0), key='ca_populate'):
        with st.spinner('Populating CellSegmentationAtlas…'):
            try:
                CellSegmentationAtlas.populate(display_progress=False)
                get_pending_count.clear()
                get_populated_masters.clear()
                get_rois.clear()
                st.success('Done.')
                st.rerun(scope='fragment')
            except Exception as exc:
                st.error(str(exc))

    if masters.empty:
        st.info('No CellSegmentationAtlas entries yet — populate above.')
        return

    # Selection
    subjects = sorted(masters['subject_name'].unique())
    # Default to a single animal on first load (user can add more).
    sel_subjects = st.multiselect('Subjects', subjects, default=subjects[:1], key='ca_subjects')
    sub_df = masters[masters['subject_name'].isin(sel_subjects)].reset_index(drop=True)
    if sub_df.empty:
        st.info('Select at least one subject.')
        return

    def _lbl(r):
        return (f"{r['subject_name']} / {r['session_name']} / {r['dataset_name']} "
                f"(p{r['parameter_set_num']}, ref{r['ref_num']}, "
                f"{r['atlas_name']}#{r['atlas_transform_id']}) "
                f"— {int(r['n_rois'])} ROIs")
    labels = [_lbl(r) for _, r in sub_df.iterrows()]
    sel_labels = st.multiselect('Sessions', labels, default=labels, key='ca_sessions')
    if not sel_labels:
        st.info('Select at least one session.')
        return
    sel_rows = sub_df.iloc[[labels.index(l) for l in sel_labels]]
    seg_keys_tuple = tuple(map(tuple, sel_rows[SEG_COLS].drop_duplicates().values))

    # Options
    sel_methods = get_selection_methods(seg_keys_tuple)
    o1, o2, o3 = st.columns(3)
    color_by    = o1.selectbox('Color by', ['session', 'subject', 'region'], key='ca_colorby')
    sel_method  = o2.selectbox('Selection', ['(all ROIs)'] + sel_methods, key='ca_selmethod',
                               help='Keep only ROIs marked as cells (selection=1) by this '
                                    'CellSegmentation.Selection method')
    only_region = o3.checkbox('Only ROIs in a region', value=False, key='ca_onlyreg')
    o4, o5 = st.columns(2)
    psize       = o4.slider('Point size', 1, 40, 3, key='ca_psize')
    palpha      = o5.slider('Point alpha', 0.05, 1.0, 0.4, step=0.05, key='ca_palpha')

    keys_tuple = tuple(tuple(r[c] for c in PK) for _, r in sel_rows.iterrows())
    with st.spinner('Loading ROIs…'):
        df = get_rois(keys_tuple)
    if df.empty:
        st.warning('No ROIs found for the selected sessions.')
        return
    if sel_method != '(all ROIs)':
        sel_df = get_selected_rois(seg_keys_tuple, sel_method)
        if sel_df.empty:
            st.warning(f'No ROIs selected by method "{sel_method}" in the selection.')
            return
        df = df.merge(sel_df, on=ROI_COLS, how='inner')
        if df.empty:
            st.warning(f'No atlas-placed ROIs selected by method "{sel_method}".')
            return
    if only_region:
        df = df[df['acronym'].notna()]
        if df.empty:
            st.warning('No ROIs assigned to a region in the selection.')
            return

    if color_by == 'subject':
        df['grp'] = df['subject_name']
    elif color_by == 'session':
        df['grp'] = df['subject_name'] + ' / ' + df['session_name']
    else:
        df['grp'] = df['acronym'].fillna('(none)')
    groups = sorted(df['grp'].unique())

    # Figure
    import plotly.graph_objects as go
    import plotly.express as px

    atlas_names = sel_rows['atlas_name'].unique()
    if len(atlas_names) > 1:
        st.warning(f'Multiple atlases selected ({", ".join(atlas_names)}); '
                   f'drawing contours from "{atlas_names[0]}".')

    fig = go.Figure()
    try:
        regions = pd.DataFrame(
            (WidefieldAtlas & dict(atlas_name=atlas_names[0])).fetch1('ccf_regions'))

        # Decide which hemispheres to draw by assigning each cell to the nearer
        # hemisphere using the region centers (robust to the ML sign convention).
        def _centers_xy(col):
            c = np.array([np.asarray(v, float).ravel()[:2] for v in regions[col]], float)
            return c[np.all(np.isfinite(c), axis=1)]
        left_c, right_c = _centers_xy('left_center'), _centers_xy('right_center')
        cells = df[['atlas_x', 'atlas_y']].to_numpy(float)

        def _min_dist(pts, centers):  # nearest-center distance per point
            return np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(axis=1)

        if len(left_c) and len(right_c) and len(cells):
            to_left = _min_dist(cells, left_c) <= _min_dist(cells, right_c)
            sides = [s for s, present in (('left', bool(to_left.any())),
                                          ('right', bool((~to_left).any()))) if present]
        else:
            sides = ['left', 'right']

        lbl_x, lbl_y, lbl_t = [], [], []
        for _, reg in regions.iterrows():
            for side in sides:
                xs = np.asarray(reg[f'{side}_x'], dtype=float)
                ys = np.asarray(reg[f'{side}_y'], dtype=float)
                if len(xs) > 1:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode='lines',
                        line=dict(color='black', width=1),
                        hoverinfo='skip', showlegend=False))
                    c = np.asarray(reg[f'{side}_center'], dtype=float).ravel()
                    if c.size >= 2 and np.all(np.isfinite(c)):
                        lbl_x.append(float(c[0])); lbl_y.append(float(c[1]))
                        lbl_t.append(str(reg['acronym']))
        if lbl_x:
            fig.add_trace(go.Scatter(
                x=lbl_x, y=lbl_y, mode='text', text=lbl_t,
                textfont=dict(color='black', size=9),
                hoverinfo='skip', showlegend=False))
    except Exception:
        pass

    palette = px.colors.qualitative.Dark24
    for i, g in enumerate(groups):
        s = df[df['grp'] == g]
        fig.add_trace(go.Scattergl(
            x=s['atlas_x'], y=s['atlas_y'], mode='markers',
            name=f'{g} ({len(s)})',
            marker=dict(size=psize, opacity=palpha,
                        color=palette[i % len(palette)]),
            customdata=s['acronym'].fillna('(none)'),
            hovertemplate='ML %{x:.2f} mm<br>AP %{y:.2f} mm<br>%{customdata}<extra></extra>'))

    # Orientation: imaged hemisphere on the left of the screen, anterior (-AP) at the top
    axis_style = dict(
        title_font=dict(color='black'), tickfont=dict(color='black'),
        ticks='outside', ticklen=6, tickwidth=1, tickcolor='black',
        showline=True, linecolor='black', linewidth=1,
        showgrid=False, zeroline=False,
    )
    fig.update_xaxes(title='ML (mm from bregma)', **axis_style)
    fig.update_yaxes(title='AP (mm from bregma)', autorange='reversed',
                     scaleanchor='x', scaleratio=1, **axis_style)
    fig.update_layout(height=650, title=f'{len(df)} ROIs · {len(sel_rows)} session(s)',
                      legend=dict(font=dict(size=10)), plot_bgcolor='white',
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')

    # Region summary
    with st.expander('ROIs per region'):
        summary = (df.assign(acronym=df['acronym'].fillna('(none)'))
                     .groupby('acronym').size()
                     .reset_index(name='n_rois').sort_values('n_rois', ascending=False))
        st.dataframe(summary, hide_index=True)
