import numpy as np
import pandas as pd
import streamlit as st

from .common import (_normalize_with_pct, _to_base64, _altair_image,
                     _tab_cache_factory, _refresh_button)


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
