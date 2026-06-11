import numpy as np
import pandas as pd
import streamlit as st

from .common import _to_base64, _altair_image, _tab_cache_factory, _refresh_button


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
