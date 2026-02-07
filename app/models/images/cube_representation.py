"""
Cube Tensor Representations
"""

from enum import Enum
from typing import Dict, List


class CubeRepresentation(str, Enum):
    """
    Different ways in which a cube can be represented
    """

    BIL = "bil"
    BIP = "bip"
    BSQ = "bsq"


# Map each representation to its dimensional mapping
# Which dimension maps to which slice - Hight, Width and Channel
DIMENSION_MAPPING: Dict[CubeRepresentation, Dict[str, int]] = {
    CubeRepresentation.BIL: {
        "H": 0,
        "W": 2,
        "C": 1,
    },
    CubeRepresentation.BIP: {
        "H": 0,
        "W": 1,
        "C": 2,
    },
    CubeRepresentation.BSQ: {
        "H": 1,
        "W": 2,
        "C": 0,
    },
}

# Dimensional arrangements which show the order of arrangement for each
# cube representations.
DIMENSIONAL_ARRANGEMENTS: Dict[CubeRepresentation, List[str]] = {
    CubeRepresentation.BIL: ["H", "C", "W"],
    CubeRepresentation.BIP: ["H", "W", "C"],
    CubeRepresentation.BSQ: ["C", "H", "W"],
}
