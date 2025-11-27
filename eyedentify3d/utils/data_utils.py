from typing import TypeAlias


from ..data_parsers.reduced_data import ReducedData
from ..data_parsers.htc_vive_pro_data import HtcViveProData
from ..data_parsers.pupil_invisible_data import PupilInvisibleData
from ..data_parsers.pico_neo_data import PicoNeoData
from ..data_parsers.tobii_pro_glasses_data import TobiiProGlassesData


DataObject: TypeAlias = ReducedData | HtcViveProData | PupilInvisibleData | PicoNeoData | TobiiProGlassesData
