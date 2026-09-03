import streamlit as st

from .common import _schema_reference
from .sessions import _sessions_params_tab
from .results import _results_tab
from .stack_explorer import _stack_explorer_tab
from .window_mask import _window_mask_tab
from .imaging_reference import _imaging_reference_tab
from .atlas_alignment import _atlas_alignment_tab
from .cell_atlas import _cell_atlas_tab

dashboard_name = 'Widefield'


def dashboard_function(schema=None):
    from ..pluginschema import WfieldParameters, WfieldStack
    st.write('## Widefield')

    # Views are functions dispatched by a segmented control so that only the
    # active one runs on each rerun; with st.tabs every body executes (and every
    # tab queries the database / renders its images) even when hidden.
    def _sessions_view():
        _sessions_params_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldParameters', 'WfieldStack')

    def _results_view():
        _results_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldStack')

    def _explorer_view():
        _stack_explorer_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldStack')

    def _mask_view():
        _window_mask_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('ImagingWindow')

    def _reference_view():
        _imaging_reference_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('ImagingReference', 'TwoPhotonReferenceAlignment')

    def _atlas_view():
        _atlas_alignment_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WidefieldAtlas', 'WidefieldAtlasTransform', 'WidefieldResponse')

    def _cell_atlas_view():
        _cell_atlas_tab(schema)
        _schema_reference('CellSegmentationAtlas')

    views = {'Sessions & Parameters': _sessions_view,
             'Projections': _results_view,
             'Stack explorer': _explorer_view,
             'Window mask': _mask_view,
             'Imaging Reference': _reference_view,
             'Atlas Alignment': _atlas_view,
             'Cell Atlas': _cell_atlas_view}
    default = 'Sessions & Parameters'
    active = st.segmented_control('view', list(views), default=default,
                                  selection_mode='single', key='wf_active_view',
                                  label_visibility='collapsed')
    views.get(active, views[default])()   # segmented_control can return None
