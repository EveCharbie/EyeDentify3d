from typing import TypeAlias


from ..data_parsers.reduced_data import ReducedData
from ..data_parsers.htc_vive_pro_data import HtcViveProData


DataObject: TypeAlias = ReducedData | HtcViveProData
