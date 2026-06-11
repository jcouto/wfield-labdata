import numpy as np
import pandas as pd
import streamlit as st

from .common import _ANALYSED_COLOR, _tab_cache_factory, _refresh_button


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
            alt.Chart(bar_data).mark_bar(color='black').encode(
                x=alt.X('subject_name:N', sort='ascending', title='Subject'),
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
