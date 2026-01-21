from typing import TypeAlias


from ..data_parsers.reduced_data import ReducedData
from ..data_parsers.htc_vive_pro_data import HtcViveProData
from ..data_parsers.pupil_invisible_data import PupilInvisibleData
from ..data_parsers.pico_neo_data import PicoNeoData

DataObject: TypeAlias = ReducedData | HtcViveProData | PupilInvisibleData | PicoNeoData
