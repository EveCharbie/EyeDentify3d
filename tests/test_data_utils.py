from eyedentify3d.utils.data_utils import DataObject
from eyedentify3d import HtcViveProData, ReducedData


def test_data_object_type_alias():
    """Test that DataObject is a type alias for ReducedData | HtcViveProData."""
    data_types = ReducedData | HtcViveProData
    assert DataObject == data_types
