from .CAD120.cad120 import CAD120
from .HICO.hico import HICO
from .VCOCO.vcoco import VCOCO

import utils

#import CAD120.metadata as cad_metadata
#import HICO.metadata as hico_metadata
#import VCOCO.metadata as vcoco_metadata


from .CAD120 import metadata as cad_metadata
from .HICO import metadata as hico_metadata
from .VCOCO import metadata as vcoco_metadata

__all__ = ('CAD120', 'HICO', 'VCOCO', 'utils', 'cad_metadata', 'hico_metadata','vcoco_metadata')
