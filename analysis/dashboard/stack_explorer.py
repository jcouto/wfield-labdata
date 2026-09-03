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
    # float32 halves the memory traffic of the per-frame U @ SVT[:, i] and avoids
    # numpy silently upcasting (copying) U on every frame when dtypes differ.
    U = np.ascontiguousarray(res['U'], dtype=np.float32)
    return U, np.ascontiguousarray(SVT, dtype=np.float32)


def _colormap_lut(name='RdBu_r'):
    """256-entry uint8 RGB lookup table for a matplotlib colormap."""
    import matplotlib
    cmap = matplotlib.colormaps[name]
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


_RDBU_LUT = _colormap_lut()


def _rerun_pending():
    """True when Streamlit has queued a rerun for this session, i.e. the user
    interacted with a widget while the script is running.

    A widget interaction inside a fragment does *not* preempt a running script
    (Streamlit only interrupts for full-app reruns and ``st.rerun``), so a loop
    that animates within one fragment run must poll this and return to let the
    queued rerun through. Falls back to False if Streamlit's internals change."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner_utils.script_requests import ScriptRequestType
        reqs = get_script_run_ctx().script_requests
        return reqs is not None and reqs._state != ScriptRequestType.CONTINUE
    except Exception:
        return False


@st.fragment
def _frame_explorer(U, SVT):
    import time
    h, w = U.shape[:2]
    nframes = SVT.shape[1]
    Uflat = U.reshape(-1, U.shape[-1])
    img_w = 500
    img_h = max(200, int(img_w * h / w))

    def _render_rgb(idx, scale):
        frame = (Uflat @ SVT[:, idx]).reshape(h, w)
        # 8-bit lookup table instead of calling the matplotlib colormap per frame
        lut_idx = np.clip((frame + scale) * (255.0 / (2 * scale)), 0, 255).astype(np.uint8)
        return _RDBU_LUT[lut_idx]

    playing   = st.session_state.get('wf_playing', False)
    frame_idx = min(st.session_state.get('wf_frame_idx', 0), nframes - 1)

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    # One button toggles playback; its label shows the action it will take.
    # It is created before the slider because a widget's session-state value
    # can only be assigned before the widget is instantiated, and stopping must
    # move the slider to the frame where playback ended.
    was_playing = playing
    toggled = c4.button('■ Stop' if was_playing else '▶ Play', key='wf_play_btn')
    if (toggled and was_playing) or st.session_state.pop('wf_sync_slider', False):
        playing = False
        st.session_state['wf_playing'] = False
        st.session_state['wf_frame_slider'] = frame_idx

    slider_val = c1.slider('Frame', 0, nframes - 1, frame_idx, key='wf_frame_slider')
    fps_val    = c2.number_input('FPS', min_value=1, max_value=120, value=10, key='wf_fps',
                                  help='Playback speed in data frames per second; '
                                       'frames are skipped when drawing cannot keep up')
    scale_val  = c3.number_input('Scale', min_value=0.01, max_value=5.0, value=0.15,
                                  step=0.01, format='%.2f', key='wf_scale')

    if toggled:
        if not was_playing:
            st.session_state['wf_playing'] = True
            st.session_state['wf_frame_idx'] = slider_val
            st.session_state['wf_play_from'] = slider_val
        # The button was already drawn with the old label on this run; rerun
        # so it shows the new one (and the slider its new value) before playing.
        st.rerun(scope='fragment')
    if playing:
        # Dragging the slider mid-play seeks; otherwise resume where we were
        # (e.g. after an FPS / Scale change reran the fragment).
        if slider_val != st.session_state.get('wf_play_from'):
            frame_idx = slider_val
            st.session_state['wf_play_from'] = slider_val
    else:
        frame_idx = slider_val
    st.session_state['wf_frame_idx'] = frame_idx

    display = st.empty()
    if not playing:
        display.altair_chart(
            _altair_image(_to_base64(_render_rgb(frame_idx, scale_val)), w, h,
                          title=f'Frame {frame_idx}', width=img_w, height=img_h),
            key='wf_frame_chart',
            width='content',
        )
        return

    # Playback runs inside this single fragment run: only the image element is
    # replaced in place each frame, so the slider, inputs and buttons are not
    # re-rendered and the picture does not flicker. Any interaction (Stop,
    # slider drag, FPS / Scale change) queues a fragment rerun; the loop polls
    # for it and returns so that rerun can act on the new widget values.
    # FPS is the playback speed in data frames per second. Frames are scheduled
    # by wall clock, so when drawing is slower than that (each frame costs a
    # matrix product, JPEG encode, websocket transfer and browser decode),
    # intermediate frames are skipped rather than slowing the playback down.
    start_idx, idx, drawn = frame_idx, frame_idx, 0
    t_start = time.perf_counter()
    while idx < nframes:
        if _rerun_pending():
            return
        st.session_state['wf_frame_idx'] = idx
        elapsed = time.perf_counter() - t_start
        url = _to_base64(_render_rgb(idx, scale_val), fmt='JPEG', quality=85)
        stats = (f' · drawing {drawn / elapsed:.0f}/s · {len(url) // 1024} KB/frame'
                 if elapsed > 1 else '')
        display.image(url, caption=f'Frame {idx}{stats}', width=img_w)
        drawn += 1
        elapsed = time.perf_counter() - t_start
        nxt = max(idx + 1, start_idx + int(elapsed * fps_val))
        due = (nxt - start_idx) / fps_val
        # Always yield briefly so the server's event loop can flush the frame
        # to the browser and pick up incoming widget events.
        time.sleep(max(0.002, due - elapsed))
        idx = nxt
    # Reached the last frame: stop and move the slider there on the next run.
    st.session_state['wf_playing'] = False
    st.session_state['wf_sync_slider'] = True
    st.rerun(scope='fragment')


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
