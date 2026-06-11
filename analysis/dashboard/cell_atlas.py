import numpy as np
import pandas as pd
import streamlit as st

from .common import _tab_cache_factory, _refresh_button


@st.fragment
def _cell_atlas_tab(schema):
    """Populate CellSegmentationAtlas and scatter ROI atlas positions across
    one or more sessions / subjects on the atlas contours."""
    from ..pluginschema import CellSegmentationAtlas, WidefieldAtlas

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
