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
    (sessions_tab, results_tab, explorer_tab, mask_tab,
     ref_tab, atlas_tab, cell_atlas_tab) = st.tabs(
        ['Sessions & Parameters', 'Projections', 'Stack explorer', 'Window mask',
         'Imaging Reference', 'Atlas Alignment', 'Cell Atlas'])
    with sessions_tab:
        _sessions_params_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldParameters', 'WfieldStack')
    with results_tab:
        _results_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldStack')
    with explorer_tab:
        _stack_explorer_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WfieldStack')
    with mask_tab:
        _window_mask_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('ImagingWindow')
    with ref_tab:
        _imaging_reference_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('ImagingReference', 'TwoPhotonReferenceAlignment')
    with atlas_tab:
        _atlas_alignment_tab(schema, WfieldParameters, WfieldStack)
        _schema_reference('WidefieldAtlas', 'WidefieldAtlasTransform', 'WidefieldResponse')
    with cell_atlas_tab:
        _cell_atlas_tab(schema)
        _schema_reference('CellSegmentationAtlas')
