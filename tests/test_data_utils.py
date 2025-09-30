from eyedentify3d.utils.data_utils import DataObject
from eyedentify3d import HtcViveProData, ReducedData, PupilInvisibleData


def test_data_object_type_alias():
    """Test that DataObject is a type alias for data classes"""
    data_types = ReducedData | HtcViveProData | PupilInvisibleData
    assert DataObject == data_types
