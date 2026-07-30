from eyedentify3d.utils.data_utils import DataObject
from eyedentify3d import HtcViveProData, ReducedData, PupilInvisibleData, PicoNeoData, TobiiProGlassesData
from eyedentify3d.data_parsers.abstract_data import EmptyData


def test_data_object_type_alias():
    """Test that DataObject is a type alias for data classes"""
    data_types = EmptyData | ReducedData | HtcViveProData | PupilInvisibleData | PicoNeoData | TobiiProGlassesData
    assert DataObject == data_types
