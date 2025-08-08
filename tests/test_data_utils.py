import pytest
from eyedentify3d.utils.data_utils import DataObject
from eyedentify3d.data_parsers.htc_vive_pro_data import HtcViveProData


def test_data_object_type_alias():
    """Test that DataObject is a type alias for HtcViveProData."""
    assert DataObject is HtcViveProData
